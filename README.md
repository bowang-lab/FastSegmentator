# FastUNet

Fast inference pipeline for [nnU-Net](https://github.com/MIC-DKFZ/nnUNet), the most popular framework for medical image segmentation. This project provides a clean, minimal inference module with only the necessary components.

## Requirements

- **Python**: 3.10
- **CUDA**: 12.4

## Installation

1. Create the environment and clone the repo:

```bash
conda create -n fast_unet python==3.10
conda activate fast_unet
git clone https://github.com/JunMa11/FastUNet.git
cd FastUNet
```

2. Install dependencies:

```bash
pip install torch torchvision torchaudio
cd nnUNet
pip install -e .
pip install cupy-cuda12x
cd ..
```

## Data and Model Weights

Download the dataset and model weights from the [Google Drive link](https://drive.google.com/drive/folders/1WRu2v3Mr67mkf1lB_ZPyvRztvGu-htL8?usp=sharing).

- Place the dataset in `FastUNet/nnUNet_data/`
- Place the model weights in `FastUNet/model_weights/`

## Running Inference

```bash
python nnunet_infer_nii.py \
    -i <path_to_input_images> \
    -o <path_to_output_segmentations> \
    --model_path <path_to_model_weights>
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `-i`, `--input_path` | (required) | Path to the input image folder |
| `-o`, `--output_path` | (required) | Path to save output segmentations |
| `--model_path` | (required) | Path to the trained model directory |
| `--fold` | `all` | Fold to use for inference |
| `--checkpoint` | `checkpoint_final.pth` | Checkpoint filename |
| `--use_softmax` | `False` | Apply softmax to output probabilities |
| `--device` | `cuda` | Device (`cuda` or `cpu`) |

### Example

```bash
python nnunet_infer_nii.py \
    -i ./nnUNet_data/Dataset701_AbdomenCT/imagesVal \
    -o ./seg \
    --model_path ./model_weights/701/nnUNetTrainerMICCAI_repvgg__nnUNetPlans__3d_fullres
```
