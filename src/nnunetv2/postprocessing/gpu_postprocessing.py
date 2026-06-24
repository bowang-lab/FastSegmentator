"""GPU/CuPy ports of the connected-component postprocessing in
totalsegmentator.postprocessing. Bit-identical to the scipy versions
(cupyx.scipy.ndimage.label / binary_dilation match scipy.ndimage), but run on GPU
instead of CPU (scipy CC on a 768x768x90 multilabel seg costs ~0.7-1.3s per op).

Used by the `body` (largest-blob / small-blob) and `remove_outside` (heartchambers)
mode postprocess paths.
"""
import numpy as np
import cupy as cp
from cupyx.scipy.ndimage import label as _gpu_label, binary_dilation as _gpu_dilation


def keep_largest_blob_gpu(data_cp):
    """cupy port of postprocessing.keep_largest_blob (largest connected component)."""
    blob_map, n = _gpu_label(data_cp)
    if n == 0:
        return data_cp
    # bincount over labels 1..n; ignore background (index 0)
    counts = cp.bincount(blob_map.ravel())
    counts[0] = 0
    largest = int(cp.argmax(counts))
    return (blob_map == largest).astype(cp.uint8)


def remove_small_blobs_gpu(img_cp, interval=(10, 30)):
    """cupy port of postprocessing.remove_small_blobs (drop blobs outside [lo, hi])."""
    mask, n = _gpu_label(img_cp)
    if n == 0:
        return img_cp.astype(cp.uint8)
    counts = cp.bincount(mask.ravel())
    if counts.size <= 1:
        return img_cp.astype(cp.uint8)
    remove = (counts <= interval[0]) | (counts > interval[1])
    remove_idx = cp.nonzero(remove)[0]
    mask[cp.isin(mask, remove_idx)] = 0
    mask[mask > 0] = 1
    return mask.astype(cp.uint8)


def keep_largest_blob_multilabel_gpu(data, class_map, rois, quiet=True):
    """GPU port of postprocessing.keep_largest_blob_multilabel.
    data: multilabel np.ndarray; processed per-roi on GPU, returned as np.ndarray."""
    class_map_inv = {v: k for k, v in class_map.items()}
    d = cp.asarray(data)
    for roi in rois:
        idx = class_map_inv[roi]
        roi_mask = d == idx
        cleaned = keep_largest_blob_gpu(roi_mask.astype(cp.uint8)) > 0.5
        d[roi_mask] = 0
        d[cleaned] = idx
    return cp.asnumpy(d)


def remove_small_blobs_multilabel_gpu(data, class_map, rois, interval=(10, 30), quiet=True):
    """GPU port of postprocessing.remove_small_blobs_multilabel."""
    class_map_inv = {v: k for k, v in class_map.items()}
    d = cp.asarray(data)
    for roi in rois:
        idx = class_map_inv[roi]
        roi_mask = d == idx
        cleaned = remove_small_blobs_gpu(roi_mask.astype(cp.uint8), interval) > 0.5
        d[roi_mask] = 0
        d[cleaned] = idx
    return cp.asnumpy(d)


def remove_outside_of_mask_gpu(seg, mask, addon=1):
    """GPU port of postprocessing.remove_outside_of_mask (binary_dilation + mask-out).
    seg, mask: np.ndarray; returns np.ndarray (uint8)."""
    seg_cp = cp.asarray(seg)
    mask_cp = _gpu_dilation(cp.asarray(mask) > 0, iterations=addon, brute_force=True)
    seg_cp[mask_cp == 0] = 0
    return cp.asnumpy(seg_cp.astype(cp.uint8))
