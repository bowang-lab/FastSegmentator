"""Robustness / regression tests for the FastSegmentator GPU fast-path.

Two kinds of test:
  * GPU-op parity regressions — the optimized GPU ops (cubic-B-spline resample, crop,
    connected-component postprocess) must stay bit-/numerically-identical to the scipy
    references they replaced. These need CUDA + cupy and skip otherwise.
  * Input-validation / error-handling — the CLI must fail fast and clearly on bad input
    (no folder, no .nii.gz). These run on CPU via subprocess.

Run:  .venv/bin/python -m pytest tests/ -q
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

torch = pytest.importorskip("torch")
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


# ---------------------------------------------------------------------------
# GPU cubic-B-spline resample == scipy order-3 (resample_data_or_seg)
# ---------------------------------------------------------------------------
@cuda
def test_cubic_resample_matches_scipy_order3():
    from nnunetv2.preprocessing.resampling.default_resampling import (
        torch_resample_data_or_seg_to_shape as TR, resample_data_or_seg_to_shape as SCI)
    from scipy.ndimage import gaussian_filter
    np.random.seed(0)
    vol = (gaussian_filter(np.random.rand(64, 70, 60).astype(np.float32), 2) * 2000 - 1000)[None]
    for cur, tgt, out in [([0.66, 0.66, 1.25], [1.5, 1.5, 1.5], (57, 57, 53)),
                          ([1.5, 1.5, 1.5], [0.7, 0.9, 1.2], (130, 96, 77))]:
        sci = SCI(vol, list(out), cur, tgt, is_seg=False, order=3, order_z=0, force_separate_z=None)
        gpu = TR(torch.as_tensor(vol, device="cuda"), out, cur, tgt,
                 is_seg=False, order=3, order_z=0, force_separate_z=None).cpu().numpy()
        assert np.abs(sci - gpu).max() < 1e-2, f"{out}: max|d|={np.abs(sci-gpu).max()}"
        assert np.corrcoef(sci.ravel(), gpu.ravel())[0, 1] > 0.99999


@cuda
def test_cubic_resample_deterministic():
    from nnunetv2.preprocessing.resampling.default_resampling import torch_resample_data_or_seg_to_shape as TR
    a = torch.rand(1, 50, 60, 40, device="cuda")
    r1 = TR(a, (40, 40, 40), [1, 1, 2], [1.5, 1.5, 1.5], order=3, force_separate_z=None)
    r2 = TR(a, (40, 40, 40), [1, 1, 2], [1.5, 1.5, 1.5], order=3, force_separate_z=None)
    assert torch.equal(r1, r2)


# ---------------------------------------------------------------------------
# GPU crop == official totalsegmentator.cropping (bit-identical bbox + data)
# ---------------------------------------------------------------------------
@cuda
def test_crop_to_mask_gpu_matches_official():
    import nibabel as nib
    from nnunetv2.preprocessing.cropping.cropping import crop_to_mask_gpu, undo_crop_gpu
    from totalsegmentator.cropping import crop_to_mask, undo_crop
    np.random.seed(1)
    aff = np.diag([0.7, 0.7, 1.5, 1.0]); aff[:3, 3] = [10, 20, 30]
    img = nib.Nifti1Image((np.random.rand(120, 140, 90) * 2000 - 1000).astype(np.int16), aff)
    m = np.zeros((120, 140, 90), np.uint8); m[30:80, 40:100, 20:60] = 1
    mask = nib.Nifti1Image(m, aff)
    for addon in [(0, 0, 0), (20, 20, 20), (10, 10, 10)]:
        co, bo = crop_to_mask(img, mask, addon=list(addon), dtype=np.int32)
        cg, bg = crop_to_mask_gpu(img, mask, addon=addon, dtype=np.int32)
        assert bo == bg
        assert np.array_equal(co.get_fdata(), cg.get_fdata())
        assert np.allclose(co.affine, cg.affine)
    _, b = crop_to_mask(img, mask, addon=[20, 20, 20])
    cropped = nib.Nifti1Image(np.random.rand(*[h - l for l, h in b]), np.eye(4))
    assert np.array_equal(undo_crop(cropped, img, b).get_fdata(),
                          undo_crop_gpu(cropped, img, b).get_fdata())


@cuda
def test_crop_handles_negative_strides():
    """nibabel canonical/undo views can have negative strides; crop must not crash."""
    import nibabel as nib
    from nnunetv2.preprocessing.cropping.cropping import crop_to_mask_gpu
    arr = np.ascontiguousarray((np.random.rand(40, 40, 30) * 1000).astype(np.float32))[::-1]  # negative stride
    img = nib.Nifti1Image(arr, np.eye(4))
    m = np.zeros((40, 40, 30), np.uint8); m[10:30, 10:30, 5:25] = 1
    cg, bg = crop_to_mask_gpu(img, nib.Nifti1Image(m, np.eye(4)), addon=(3, 3, 3), dtype=np.int32)
    assert cg.get_fdata().size > 0


# ---------------------------------------------------------------------------
# GPU connected-component postprocess == scipy reference
# ---------------------------------------------------------------------------
@cuda
def test_gpu_postprocess_matches_scipy():
    pytest.importorskip("cupy")
    from nnunetv2.postprocessing.gpu_postprocessing import (
        keep_largest_blob_multilabel_gpu, remove_small_blobs_multilabel_gpu, remove_outside_of_mask_gpu)
    from totalsegmentator.postprocessing import (
        keep_largest_blob_multilabel, remove_small_blobs_multilabel, remove_outside_of_mask)
    cm = {1: "a", 2: "b"}
    data = np.zeros((80, 80, 80), np.uint8)
    data[10:40, 10:40, 10:40] = 1; data[5:8, 5:8, 5:8] = 1
    data[50:70, 50:70, 50:70] = 2; data[20:22, 60:62, 60:62] = 2
    assert np.array_equal(
        keep_largest_blob_multilabel(data.copy(), cm, ["a", "b"], quiet=True),
        keep_largest_blob_multilabel_gpu(data.copy(), cm, ["a", "b"]))
    assert np.array_equal(
        remove_small_blobs_multilabel(data.copy(), cm, ["a", "b"], interval=[100, 1e10], quiet=True),
        remove_small_blobs_multilabel_gpu(data.copy(), cm, ["a", "b"], interval=[100, 1e10]))
    mask = np.zeros((80, 80, 80), np.uint8); mask[20:60, 20:60, 20:60] = 1
    assert np.array_equal(
        remove_outside_of_mask(data.copy(), mask.copy(), addon=5),
        remove_outside_of_mask_gpu(data.copy(), mask.copy(), addon=5))


# ---------------------------------------------------------------------------
# Input validation / error handling (CPU, via the CLI)
# ---------------------------------------------------------------------------
def _run(args):
    return subprocess.run([sys.executable, "-m", "fastsegmentator.totalseg_infer", *args],
                          capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=str(REPO))


def test_cli_errors_on_missing_input_dir(tmp_path):
    r = _run(["-i", str(tmp_path / "nope"), "-o", str(tmp_path / "o"), "--task", "total"])
    assert r.returncode == 1
    assert "not a directory" in (r.stdout + r.stderr)


def test_cli_errors_on_empty_input_dir(tmp_path):
    (tmp_path / "in").mkdir()
    r = _run(["-i", str(tmp_path / "in"), "-o", str(tmp_path / "o"), "--task", "total"])
    assert r.returncode == 1
    assert "no .nii.gz" in (r.stdout + r.stderr)
