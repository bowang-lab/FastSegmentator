"""
TotalSegmentator MRI inference — task: total_mr (multilabel output)

Runs the 2 sub-models (850-851) sequentially and merges them into one
multilabel NIfTI where each voxel value is the global class index from
class_map["total_mr"].

Usage:
  python totalseg_mr_infer.py -i /input_folder -o /output_folder
  python totalseg_mr_infer.py -i /input_folder -o /output_folder \\
      --weights_dir /custom/weights --device cpu
"""
import argparse
import glob
import os
import sys
from pathlib import Path
from time import time

import numpy as np
import SimpleITK as sitk
import torch
from tqdm import tqdm

_TOTALSEG_SRC = Path(__file__).parent.parent / "TotalSegmentator"
if _TOTALSEG_SRC.exists():
    sys.path.insert(0, str(_TOTALSEG_SRC))

# Monkey-patch custom trainers before any nnunetv2 inference import.
# Stubs only need build_network_architecture (inherited) for inference.
import nnunetv2
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerNoMirroring import nnUNetTrainerNoMirroring
from nnunetv2.training.nnUNetTrainer.variants.loss.nnUNetTrainerTopkLoss import nnUNetTrainerDiceTopK10Loss

class nnUNetTrainer_MOSAIC_1k_QuarterLR_NoMirroring(nnUNetTrainerNoMirroring):
    pass

class nnUNetTrainerDiceTopK10Loss_2000epochs(nnUNetTrainerDiceTopK10Loss):
    pass

class nnUNetTrainerSkeletonRecall(nnUNetTrainer):
    pass

def _find_class(folder, class_name, current_module):
    if class_name == "nnUNetTrainer_MOSAIC_1k_QuarterLR_NoMirroring":
        return nnUNetTrainer_MOSAIC_1k_QuarterLR_NoMirroring
    if class_name == "nnUNetTrainerDiceTopK10Loss_2000epochs":
        return nnUNetTrainerDiceTopK10Loss_2000epochs
    if class_name == "nnUNetTrainerSkeletonRecall":
        return nnUNetTrainerSkeletonRecall
    return recursive_find_python_class(folder, class_name, current_module)

nnunetv2.inference.predict_from_raw_data.recursive_find_python_class = _find_class

from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from totalsegmentator.config import get_weights_dir
from totalsegmentator.map_to_binary import class_map, class_map_parts_mr, map_taskid_to_partname_mr
from nnunet_infer_nii import SimplePredictor

CHECKPOINT = "checkpoint_final.pth"

_TRAINER_DIR = "nnUNetTrainer_2000epochs_NoMirroring__nnUNetPlans__3d_fullres"

MR_MODELS = {
    850: f"Dataset850_TotalSegMRI_part1_organs_1088subj/{_TRAINER_DIR}",
    851: f"Dataset851_TotalSegMRI_part2_muscles_1088subj/{_TRAINER_DIR}",
}

_GLOBAL_IDX = {name: idx for idx, name in class_map["total_mr"].items()}


def build_predictors(weights_dir, device):
    predictors = {}
    for task_id, folder in MR_MODELS.items():
        model_path = Path(weights_dir) / folder
        if not model_path.exists():
            raise FileNotFoundError(
                f"Weights not found: {model_path}\n"
                f"Run: python download_totalseg_weights.py --full"
            )
        p = SimplePredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=False,
            perform_everything_on_device=device.type != "cpu",
            device=device,
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=False,
        )
        p.initialize_from_trained_model_folder(
            str(model_path), use_folds=None, checkpoint_name=CHECKPOINT
        )
        p.network.to(device)
        predictors[task_id] = p
    return predictors


def infer_file(file_path, predictors, output_path):
    image, props = SimpleITKIO().read_images([file_path])
    seg_combined = None

    for task_id, predictor in predictors.items():
        t0 = time()
        part_seg = predictor.inference(image, props, use_softmax=False).numpy()
        print(f"  task {task_id}: {time() - t0:.2f}s")

        if seg_combined is None:
            seg_combined = np.zeros(part_seg.shape, dtype=np.uint8)

        part_name = map_taskid_to_partname_mr[task_id]
        for local_idx, class_name in class_map_parts_mr[part_name].items():
            seg_combined[part_seg == local_idx] = _GLOBAL_IDX[class_name]

    sitk_img = sitk.GetImageFromArray(seg_combined)
    sitk_img.SetSpacing(props["sitk_stuff"]["spacing"])
    sitk_img.SetOrigin(props["sitk_stuff"]["origin"])
    sitk_img.SetDirection(props["sitk_stuff"]["direction"])

    out_name = os.path.basename(file_path).replace("_0000.nii.gz", ".nii.gz")
    sitk.WriteImage(sitk_img, os.path.join(output_path, out_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TotalSegmentator MRI inference (multilabel NIfTI)"
    )
    parser.add_argument("-i", "--input_path", required=True,
                        help="Folder containing *.nii.gz input images")
    parser.add_argument("-o", "--output_path", required=True,
                        help="Folder to write multilabel output NIfTIs")
    parser.add_argument("--weights_dir", default=None,
                        help="Path to TotalSegmentator weights (default: ~/.totalsegmentator/nnunet/results)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    weights_dir = args.weights_dir or str(get_weights_dir())
    device = torch.device(args.device) if args.device == "cpu" else torch.device(args.device, 0)

    os.makedirs(args.output_path, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.input_path, "*.nii.gz")))
    if not files:
        raise FileNotFoundError(f"No .nii.gz files found in {args.input_path}")

    print(f"Loading {len(MR_MODELS)} MR sub-models from {weights_dir} ...")
    predictors = build_predictors(weights_dir, device)

    for file in tqdm(files, desc="Inference"):
        print(f"\n{os.path.basename(file)}")
        infer_file(file, predictors, args.output_path)
