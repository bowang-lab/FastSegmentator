# FastSegmentator

Fast inference pipeline for [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) and
[TotalSegmentator](https://github.com/wasserth/TotalSegmentator), the most
popular frameworks for medical image segmentation. This project provides a
clean, minimal inference module with only the necessary components, plus an
**end-to-end GPU fast-path**: every stage — resampling (cucim + a GPU
cubic-B-spline that matches scipy `order=3` to ~1e-13), normalization, the
sliding-window forward pass, logits→label conversion, cropping, and
connected-component postprocessing — runs on the GPU. Across **24 parity-validated
modes** it reproduces official TotalSegmentator output at **≥0.999 DSC on the headline
modes** (≥0.995 on 21 of 24) while running **2–9× faster** (forward-pass-bound; the rest
is fixed import/model-load overhead amortized in batch).

## Requirements

- **Python**: 3.10
- **CUDA**: 12.4
- [**uv**](https://docs.astral.sh/uv/) for environment management

## Installation

1. Clone this repo. The vendored `nnunetv2` lives in `src/`, and
   `TotalSegmentator` is expected as a sibling checkout (see
   `[tool.uv.sources]` in `pyproject.toml`):

```bash
git clone https://github.com/JunMa11/FastSegmentator.git
git clone https://github.com/wasserth/TotalSegmentator.git   # sibling of FastSegmentator
cd FastSegmentator
```

2. Create the environment and install everything (including the
   `FastSegmentator` command) in one step:

```bash
uv sync
```

`uv sync` builds the editable `nnunetv2` package from `src/`, installs the
pinned CUDA 12.1 torch wheels, `cupy`/`cucim`, and registers the
`FastSegmentator` console script into `.venv/`.

3. Activate the environment so the `FastSegmentator` command is on your PATH:

```bash
source .venv/bin/activate
```

## Data and Model Weights

Download the dataset and model weights from the
[Google Drive link](https://drive.google.com/drive/folders/1WRu2v3Mr67mkf1lB_ZPyvRztvGu-htL8?usp=sharing).

- Place the dataset in `FastSegmentator/nnUNet_data/`
- Place the model weights in `FastSegmentator/model_weights/`

TotalSegmentator weights default to `~/.totalsegmentator/nnunet/results`
(override with `--weights_dir`).

## Running Inference

With the environment activated, the `FastSegmentator` command dispatches to one
of two backends:

```bash
FastSegmentator <command> [options]
```

> Without activating, you can equivalently run `uv run FastSegmentator ...` or
> `.venv/bin/FastSegmentator ...`.

### `totalseg` — TotalSegmentator modes (config-driven)

```bash
FastSegmentator totalseg \
    -i <path_to_input_images> \
    -o <path_to_output_segmentations> \
    --task total
```

| Flag | Default | Description |
|------|---------|-------------|
| `-i`, `--input_path` | (required) | Folder of `*.nii.gz` input images |
| `-o`, `--output_path` | (required) | Folder to write multilabel output NIfTIs |
| `--task` | `total` | Mode (e.g. `total`, `total_mr`, `body_mr`, …) |
| `--weights_dir` | `~/.totalsegmentator/nnunet/results` | TotalSegmentator weights path |
| `--device` | `cuda` | Device (`cuda` or `cpu`) |

Run `FastSegmentator totalseg --help` for the full list of `--task` modes.

### `nnunet` — generic nnU-Net model folder

```bash
FastSegmentator nnunet \
    -i <path_to_input_images> \
    -o <path_to_output_segmentations> \
    --model_path <path_to_model_weights>
```

| Flag | Default | Description |
|------|---------|-------------|
| `-i`, `--input_path` | (required) | Path to the input image folder |
| `-o`, `--output_path` | (required) | Path to save output segmentations |
| `--model_path` | (required) | Path to the trained model directory |
| `--fold` | `all` | Fold to use for inference |
| `--checkpoint` | `checkpoint_final.pth` | Checkpoint filename |
| `--use_softmax` | `False` | Apply softmax to output probabilities |
| `--device` | `cuda` | Device (`cuda` or `cpu`) |

> **Trainers.** By design, the `nnunet` branch resolves only the standard
> nnU-Net trainers (`nnUNetTrainer`, `nnUNetTrainerNoMirroring`,
> `nnUNetTrainerTopkLoss`). To use a model trained with a custom trainer, point
> the `nnUNet_extTrainer` environment variable at the directory containing your
> trainer class so it can be resolved at checkpoint load:
>
> ```bash
> export nnUNet_extTrainer=/path/to/your/trainers
> ```

### Example

```bash
FastSegmentator nnunet \
    -i ./nnUNet_data/Dataset701_AbdomenCT/imagesVal \
    -o ./seg \
    --model_path ./model_weights/701/nnUNetTrainerMICCAI_repvgg__nnUNetPlans__3d_fullres
```

## Parity with official TotalSegmentator

The fast-path is validated to match official TotalSegmentator on the **same
input** (parity, not vs. ground truth) across **24 modes**.
Overview + interactive figures: [`report/index.html`](report/index.html); full
per-mode report: [`report/validation_report.html`](report/validation_report.html).

Of the 24 validated modes, **21 reach ≥0.995 DSC** — every previously-failing
pathology mode is now **≥0.999** — and 3 thin/sparse modes carry small,
characterized caveats, all at **2–9× speedup**:

| Mode | Task | DSC vs official | Speedup |
|------|------|-----------------|---------|
| `total` (CT) | 291–295 | 1.0000 | 9.6× |
| `total_mr` | 850,851 | 0.9999 | 9.5× |
| `lung_vessels` | 117 | 0.99998 | 2.8× |
| `lung_vessels_LEGACY` | 258 | 0.99989 | 4.2× |
| `lung_nodules` | 913 | 0.9999 | 7.5× |
| `liver_lesions` | 591 | 1.0000 | 4.9× |
| `liver_lesions_mr` | 589 | 1.0000¹ | 6.2× |
| `liver_segments_mr` | 576 | 1.0000 | 6.7× |
| `pleural_pericard_effusion` | 315 | 0.9990 | 9.3× |
| `craniofacial_structures`, `head_muscles`, `liver_segments`, `body`, … | — | ≥0.996 | 2–5× |

¹ 86-voxel lesion on the crop boundary — nondeterministic on *both* pipelines
(official itself flips 86/52 voxels across runs); our deterministic output
matches official's same-draw at DSC 1.0.

**Three fixes brought the harder modes to parity** (each isolated by bisecting
against official's per-function intermediates):

1. **GPU cubic-B-spline input resample** — replaced order-1 trilinear
   (`F.interpolate`) with a separable order-3 cubic B-spline matching nnU-Net's
   `skimage.resize(order=3)` to ~1e-13. *(pleural, lung_nodules)*
2. **`dtype=np.int32` on the cucim input resample** — matches official's
   pre-model int truncation. *(total_mr, liver_segments_mr, liver crops)*
3. **Per-mode softmax→argmax convert** for low-confidence lesion modes.
   *(liver_lesions, liver_lesions_mr)*

Plus GPU-ported crop + connected-component postprocess (bit-identical to the
scipy originals) and cuDNN-deterministic forward for reproducibility.
