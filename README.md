# FastSegmentator

Fast inference pipeline for [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) and
[TotalSegmentator](https://github.com/wasserth/TotalSegmentator), the most
popular frameworks for medical image segmentation. This project provides a
clean, minimal inference module with only the necessary components, plus a
GPU fast-path (cucim resampling + logit-level thresholding).

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
