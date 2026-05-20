"""
TotalSegmentator CT inference — task: total (multilabel output)

Custom cucim-optimized pipeline (ported from infer_ct.py):
  - cucim GPU change_spacing for canonical 1.5 mm pre/post resample
  - FastPreprocessor: torch GPU trilinear inside run_case_npy (no scipy)
  - logits-to-segmentation via threshold (max_logit >= 0.5) — no softmax

Runs the 5 sub-models (291-295) sequentially and merges them into one
multilabel NIfTI where each voxel value is the global class index from
class_map["total"].

Usage:
  python totalseg_ct_infer.py -i /input_folder -o /output_folder
  python totalseg_ct_infer.py -i /input_folder -o /output_folder \\
      --weights_dir /custom/weights --device cpu
"""
import argparse
import glob
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from tqdm import tqdm

_TOTALSEG_SRC = Path(__file__).parent.parent / "TotalSegmentator"
if _TOTALSEG_SRC.exists():
    sys.path.insert(0, str(_TOTALSEG_SRC))

from nnunetv2.preprocessing.cropping.cropping import crop_to_nonzero
from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from nnunetv2.preprocessing.resampling.default_resampling import (
    compute_new_shape,
    fast_resample_data_or_seg_to_shape,
)
from nnunetv2.utilities.utils import log_runtime
from totalsegmentator.alignment import undo_canonical as tseg_undo_canonical
from totalsegmentator.config import get_weights_dir
from totalsegmentator.map_to_binary import class_map, class_map_5_parts, map_taskid_to_partname_ct
from totalsegmentator.resampling import change_spacing as tseg_change_spacing

from nnunet_infer_nii import SimplePredictor

CHECKPOINT = "checkpoint_final.pth"
RESAMPLE   = [1.5, 1.5, 1.5]

_TRAINER_DIR = "nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres"

CT_MODELS = {
    291: f"Dataset291_TotalSegmentator_part1_organs_1559subj/{_TRAINER_DIR}",
    292: f"Dataset292_TotalSegmentator_part2_vertebrae_1532subj/{_TRAINER_DIR}",
    293: f"Dataset293_TotalSegmentator_part3_cardiac_1559subj/{_TRAINER_DIR}",
    294: f"Dataset294_TotalSegmentator_part4_muscles_1559subj/{_TRAINER_DIR}",
    295: f"Dataset295_TotalSegmentator_part5_ribs_1559subj/{_TRAINER_DIR}",
}

_GLOBAL_IDX = {name: idx for idx, name in class_map["total"].items()}


class FastPreprocessor(DefaultPreprocessor):
    """run_case_npy with torch GPU trilinear resample to model spacing."""

    def run_case_npy(self, data, seg, properties, plans_manager, configuration_manager, dataset_json):
        data = data.clone()
        data = data.permute([0, *[i + 1 for i in plans_manager.transpose_forward]])
        original_spacing = [properties['spacing'][i] for i in plans_manager.transpose_forward]

        properties['shape_before_cropping'] = data.shape[1:]
        data, seg, bbox = crop_to_nonzero(data, seg)
        properties['bbox_used_for_cropping'] = bbox
        properties['shape_after_cropping_and_before_resampling'] = data.shape[1:]

        target_spacing = configuration_manager.spacing
        if len(target_spacing) < len(data.shape[1:]):
            target_spacing = [original_spacing[0]] + target_spacing
        new_shape = compute_new_shape(data.shape[1:], original_spacing, target_spacing)

        data = self._normalize(
            data, seg, configuration_manager,
            plans_manager.foreground_intensity_properties_per_channel,
        )

        print(f"  [FastPreprocessor] shape={tuple(data.shape[1:])}  new_shape={tuple(new_shape)}")
        return fast_resample_data_or_seg_to_shape(data, new_shape, original_spacing, target_spacing)


class FastPredictor(SimplePredictor):
    """SimplePredictor with FastPreprocessor (torch trilinear) replacing the scipy default."""

    def preprocess(self, image, props):
        preprocessor = FastPreprocessor(verbose=False)
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)
        image = image.to(self.device, dtype=torch.float32)
        return preprocessor.run_case_npy(
            image, None, props,
            self.plans_manager, self.configuration_manager, self.dataset_json,
        )


def build_predictors(weights_dir, device, step_size):
    predictors = {}
    for task_id, folder in CT_MODELS.items():
        model_path = Path(weights_dir) / folder
        if not model_path.exists():
            raise FileNotFoundError(
                f"Weights not found: {model_path}\n"
                f"Run: python download_totalseg_weights.py --full"
            )
        p = FastPredictor(
            tile_step_size=step_size,
            use_gaussian=True,
            use_mirroring=False,
            perform_everything_on_device=device.type != "cpu",
            device=device,
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=False,
        )
        p.initialize_from_trained_model_folder(
            str(model_path), use_folds=None, checkpoint_name=CHECKPOINT,
        )
        p.network.to(device)
        predictors[task_id] = p
    return predictors


@log_runtime
def infer_file(file_path, predictors, output_path, resample=RESAMPLE):
    # T2 forward: as_closest_canonical + cucim GPU change_spacing to 1.5 mm
    img_orig    = nib.load(file_path)
    img_can     = nib.as_closest_canonical(img_orig)
    img_can_shape = img_can.shape
    img_1p5     = tseg_change_spacing(img_can, resample, order=3, use_gpu=True)
    arr_xyz     = img_1p5.get_fdata().astype(np.float32)
    image       = arr_xyz.transpose(2, 1, 0)[np.newaxis]   # (1, Z, Y, X) at 1.5 mm canonical

    # Input is already canonical+resampled — leave _can_* unset so the predictor skips its hooks.
    props = {'spacing': list(resample)}

    seg_combined = None
    for task_id, predictor in predictors.items():
        part_seg = predictor.inference(image, props, use_softmax=False).numpy()
        if seg_combined is None:
            seg_combined = np.zeros(part_seg.shape, dtype=np.uint8)
        part_name = map_taskid_to_partname_ct[task_id]
        for local_idx, class_name in class_map_5_parts[part_name].items():
            seg_combined[part_seg == local_idx] = _GLOBAL_IDX[class_name]

    # T2 inverse: cucim GPU change_spacing back (order=0) + undo_canonical
    seg_xyz = seg_combined.transpose(2, 1, 0).astype(np.uint8)
    seg_nib = nib.Nifti1Image(seg_xyz, img_1p5.affine)
    seg_nib = tseg_change_spacing(
        seg_nib, resample, img_can_shape,
        order=0, force_affine=img_can.affine, use_gpu=True,
    )
    seg_nib = tseg_undo_canonical(seg_nib, img_orig)
    seg_combined = seg_nib.get_fdata().transpose(2, 1, 0).astype(np.uint8)

    out_name = os.path.basename(file_path).replace("_0000.nii.gz", "").replace(".nii.gz", "")
    sitk_img = sitk.GetImageFromArray(seg_combined)
    sitk_img.CopyInformation(sitk.ReadImage(file_path))
    sitk.WriteImage(sitk_img, os.path.join(output_path, f"{out_name}.nii.gz"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TotalSegmentator CT inference (multilabel NIfTI, custom cucim pipeline)"
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
    device      = torch.device(args.device)

    # Adaptive tile_step_size (mirrors official TotalSegmentator):
    #   resample[0] < 3.0 mm → step=0.8 (≈11% GPU speedup, ~0.001 DSC trade-off)
    #   otherwise           → step=0.5
    resample  = RESAMPLE
    step_size = 0.8 if (resample is not None and resample[0] < 3.0) else 0.5

    os.makedirs(args.output_path, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.input_path, "*.nii.gz")))
    if not files:
        raise FileNotFoundError(f"No .nii.gz files found in {args.input_path}")

    print(f"Loading {len(CT_MODELS)} CT sub-models from {weights_dir} (step_size={step_size}) ...")
    predictors = build_predictors(weights_dir, device, step_size=step_size)

    for file in tqdm(files, desc="Inference"):
        print(f"\n{os.path.basename(file)}")
        infer_file(file, predictors, args.output_path, resample=resample)
