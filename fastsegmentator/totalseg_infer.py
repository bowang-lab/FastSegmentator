"""
TotalSegmentator unified inference (CT + MR), config-driven.

Behaviour is driven entirely by the per-mode `TaskConfig` (mirrors the official
TotalSegmentator dispatch — no "stage" control flow). A mode runs if its weights
are present locally; the fields decide the pipeline:
  - resample        : cucim GPU change_spacing to model spacing (None = native)
  - crop            : ROI names → rough whole-body seg builds a crop bbox
  - mode_postprocess: "aux" / "body" / "remove_outside" branches
  - crop_model      : recursive crop (only `teeth`; runs total→craniofacial→teeth)

Fast-path applied to every mode (whole pipeline on GPU):
  - cucim GPU change_spacing for canonical resample (skipped when resample=None)
  - FastPreprocessor: torch GPU order-3 cubic-B-spline resample (matches nnU-Net's
    scipy order-3 to ~1e-13) inside run_case_npy
  - GPU crop (crop_to_mask_gpu) + GPU connected-component postprocess
  - logits-to-segmentation via threshold (max_logit >= 0.5) by default; per-mode
    softmax-argmax (use_softmax) for low-confidence lesion modes

Usage:
  python totalseg_infer.py -i <in_folder> -o <out_folder> --task total
  python totalseg_infer.py -i <in_folder> -o <out_folder> --task total_mr
  python totalseg_infer.py -i <in_folder> -o <out_folder> --task cerebral_bleed
"""
import argparse
import glob
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from tqdm import tqdm

# --- Custom trainer registration ---------------------------------------------
# SimplePredictor.initialize_from_trained_model_folder() resolves the trainer
# class by name via `nnunet_infer_nii.recursive_find_python_class`. Patch that
# module-level reference so our stubs are reachable at checkpoint-load time.

from nnunetv2.utilities.find_class_by_name import recursive_find_python_class  # noqa: E402
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer  # noqa: E402
from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerNoMirroring import nnUNetTrainerNoMirroring  # noqa: E402


class nnUNetTrainer_DASegOrd0(nnUNetTrainer): pass
class nnUNetTrainer_DASegOrd0_NoMirroring(nnUNetTrainerNoMirroring): pass
class nnUNetTrainer_2000epochs_NoMirroring(nnUNetTrainerNoMirroring): pass
class nnUNetTrainer_4000epochs_NoMirroring(nnUNetTrainerNoMirroring): pass
# Extra trainers: loss/augmentation variants only differ at training time;
# the network built at inference is identical, so a `pass` stub is sufficient.
class nnUNetTrainerDiceTopK10Loss_2000epochs(nnUNetTrainer): pass
class nnUNetTrainerSkeletonRecall(nnUNetTrainer): pass
class nnUNetTrainer_MOSAIC_1k_QuarterLR_NoMirroring(nnUNetTrainerNoMirroring): pass
class nnUNetTrainer_onlyMirror01(nnUNetTrainer): pass


_CUSTOM_TRAINERS = {
    "nnUNetTrainer_DASegOrd0":              nnUNetTrainer_DASegOrd0,
    "nnUNetTrainer_DASegOrd0_NoMirroring":  nnUNetTrainer_DASegOrd0_NoMirroring,
    "nnUNetTrainer_2000epochs_NoMirroring": nnUNetTrainer_2000epochs_NoMirroring,
    "nnUNetTrainer_4000epochs_NoMirroring": nnUNetTrainer_4000epochs_NoMirroring,
    "nnUNetTrainerDiceTopK10Loss_2000epochs":          nnUNetTrainerDiceTopK10Loss_2000epochs,
    "nnUNetTrainerSkeletonRecall":                     nnUNetTrainerSkeletonRecall,
    "nnUNetTrainer_MOSAIC_1k_QuarterLR_NoMirroring":   nnUNetTrainer_MOSAIC_1k_QuarterLR_NoMirroring,
    "nnUNetTrainer_onlyMirror01":                      nnUNetTrainer_onlyMirror01,
}


def _custom_find_class(folder, class_name, current_module):
    if class_name in _CUSTOM_TRAINERS:
        return _CUSTOM_TRAINERS[class_name]
    return recursive_find_python_class(folder, class_name, current_module)


from . import nnunet_infer_nii  # noqa: E402
nnunet_infer_nii.recursive_find_python_class = _custom_find_class

# --- Project imports ---------------------------------------------------------

from nnunetv2.preprocessing.cropping.cropping import (  # noqa: E402
    crop_to_nonzero, crop_to_mask_gpu, undo_crop_gpu,
)
from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor  # noqa: E402
from nnunetv2.preprocessing.resampling.default_resampling import (  # noqa: E402
    compute_new_shape, torch_resample_data_or_seg_to_shape,
)
from nnunetv2.utilities.helpers import empty_cache  # noqa: E402
from nnunetv2.utilities.utils import log_runtime  # noqa: E402
from totalsegmentator.alignment import undo_canonical as tseg_undo_canonical  # noqa: E402
from totalsegmentator.config import get_weights_dir  # noqa: E402
# Crop + connected-component postprocess: GPU/CuPy ports (bit-identical to the official
# numpy/scipy versions) keep the whole pipeline on GPU. extract_skin / remove_auxiliary_labels
# stay official (no-op / negligible).
from totalsegmentator.postprocessing import extract_skin, remove_auxiliary_labels  # noqa: E402
from nnunetv2.postprocessing.gpu_postprocessing import (  # noqa: E402
    keep_largest_blob_multilabel_gpu, remove_small_blobs_multilabel_gpu, remove_outside_of_mask_gpu,
)
from totalsegmentator.map_to_binary import (  # noqa: E402
    class_map, class_map_5_parts, class_map_parts_mr, class_map_parts_headneck_muscles,
    map_taskid_to_partname_ct, map_taskid_to_partname_mr, map_taskid_to_partname_headneck_muscles,
)
from totalsegmentator.resampling import change_spacing as tseg_change_spacing  # noqa: E402
from totalsegmentator.libs import reorder_multilabel_like_v1  # noqa: E402

from .nnunet_infer_nii import SimplePredictor  # noqa: E402


CHECKPOINT = "checkpoint_final.pth"

_CLASS_MAP_PARTS = {
    "class_map_5_parts":               class_map_5_parts,
    "class_map_parts_mr":              class_map_parts_mr,
    "class_map_parts_headneck_muscles": class_map_parts_headneck_muscles,
}
_PARTNAME_MAPS = {
    "map_taskid_to_partname_ct":               map_taskid_to_partname_ct,
    "map_taskid_to_partname_mr":               map_taskid_to_partname_mr,
    "map_taskid_to_partname_headneck_muscles": map_taskid_to_partname_headneck_muscles,
}


# --- Task config -------------------------------------------------------------

@dataclass(frozen=True)
class TaskConfig:
    task_ids: tuple                          # single-model: (tid,); multi: (tid1, tid2, ...)
    resample: Optional[tuple]                # XYZ order (matches nibabel & official dispatcher); None = native
    trainer: str
    class_map_key: str
    plans: str = "nnUNetPlans"
    model_config: str = "3d_fullres"
    folds: Optional[tuple] = (0,)            # None = auto-detect all folds (ensemble)
    class_map_parts_key: Optional[str] = None
    partname_map_key: Optional[str] = None
    modality: str = "ct"
    licensed: bool = False
    use_softmax: bool = False                # default: max_logit>=0.5 threshold (matches official for confident classes). Set True only where the low-confidence argmax-vs-threshold gap matters (e.g. liver_lesions).
    # --- Crop pre-pass + mode postprocess ---
    crop: tuple = ()                         # ROI names; non-empty triggers the crop pre-pass
    crop_addon: tuple = (3, 3, 3)            # NOTE: overridden to 20mm for non-recursive crops
    crop_model: Optional[str] = None         # recursive crop (only `teeth` uses this)
    robust_crop: bool = False                # use 3mm rough crop model (T297) instead of 6mm (T298)
    mode_postprocess: tuple = ()             # tags: "body", "remove_outside", "aux"
    remove_outside: tuple = ()
    remove_outside_dilation_mm: Optional[int] = None


TASK_CONFIGS: dict[str, TaskConfig] = {

    # === Uncropped — multi-model totals ===
    "total": TaskConfig(
        task_ids=(291, 292, 293, 294, 295),
        resample=(1.5, 1.5, 1.5),
        trainer="nnUNetTrainerNoMirroring",
        class_map_key="total",
        class_map_parts_key="class_map_5_parts",
        partname_map_key="map_taskid_to_partname_ct",
    ),
    "total_mr": TaskConfig(
        task_ids=(850, 851),
        resample=(1.5, 1.5, 1.5),
        trainer="nnUNetTrainer_2000epochs_NoMirroring",
        class_map_key="total_mr",
        class_map_parts_key="class_map_parts_mr",
        partname_map_key="map_taskid_to_partname_mr",
        modality="mr",
    ),

    # === Uncropped — single-model open CT ===
    "breasts": TaskConfig(
        task_ids=(527,),
        resample=(1.5, 1.5, 1.5),
        trainer="nnUNetTrainer_DASegOrd0_NoMirroring",
        class_map_key="breasts",
    ),
    "trunk_cavities": TaskConfig(
        task_ids=(343,),
        resample=(1.5, 1.5, 1.5),
        trainer="nnUNetTrainer",
        class_map_key="trunk_cavities",
    ),

    # === Uncropped — single-model open MR ===
    "body_mr": TaskConfig(
        task_ids=(597,),
        resample=(1.5, 1.5, 1.5),
        trainer="nnUNetTrainer_DASegOrd0",
        class_map_key="body_mr",
        modality="mr",
    ),
    "vertebrae_mr": TaskConfig(
        task_ids=(756,),
        resample=(1.5, 1.5, 1.5),
        trainer="nnUNetTrainer_DASegOrd0_NoMirroring",
        class_map_key="vertebrae_mr",
        modality="mr",
    ),

    # === Uncropped — licensed CT ===
    "vertebrae_body": TaskConfig(
        task_ids=(305,),
        resample=(1.5, 1.5, 1.5),
        trainer="nnUNetTrainer_DASegOrd0",
        class_map_key="vertebrae_body",
        licensed=True,
    ),
    "appendicular_bones": TaskConfig(
        # Auxiliary-label removal applied inline (see _remove_auxiliary_labels).
        task_ids=(304,),
        resample=(1.5, 1.5, 1.5),
        trainer="nnUNetTrainerNoMirroring",
        class_map_key="appendicular_bones",
        mode_postprocess=("aux",),
        licensed=True,
    ),
    "tissue_types": TaskConfig(
        task_ids=(481,),
        resample=(1.5, 1.5, 1.5),
        trainer="nnUNetTrainer",
        class_map_key="tissue_types",
        licensed=True,
    ),
    "tissue_4_types": TaskConfig(
        task_ids=(485,),
        resample=(1.5, 1.5, 1.5),
        trainer="nnUNetTrainer",
        class_map_key="tissue_4_types",
        licensed=True,
    ),
    "face": TaskConfig(
        task_ids=(303,),
        resample=(1.5, 1.5, 1.5),
        trainer="nnUNetTrainerNoMirroring",
        class_map_key="face",
        licensed=True,
    ),
    "thigh_shoulder_muscles": TaskConfig(
        task_ids=(857,),
        resample=(1.5, 1.5, 1.5),
        trainer="nnUNetTrainer_2000epochs_NoMirroring",
        class_map_key="thigh_shoulder_muscles",
        licensed=True,
    ),

    # === Uncropped — licensed MR ===
    "appendicular_bones_mr": TaskConfig(
        task_ids=(855,),
        resample=(1.5, 1.5, 1.5),
        trainer="nnUNetTrainer_2000epochs_NoMirroring",
        class_map_key="appendicular_bones_mr",
        modality="mr",
        licensed=True,
    ),
    "tissue_types_mr": TaskConfig(
        task_ids=(925,),
        resample=(1.5, 1.5, 1.5),
        trainer="nnUNetTrainer_DASegOrd0_NoMirroring",
        class_map_key="tissue_types_mr",
        modality="mr",
        licensed=True,
    ),
    "thigh_shoulder_muscles_mr": TaskConfig(
        task_ids=(857,),
        resample=(1.5, 1.5, 1.5),
        trainer="nnUNetTrainer_2000epochs_NoMirroring",
        class_map_key="thigh_shoulder_muscles_mr",
        modality="mr",
        licensed=True,
    ),

    # === Cropped modes (rough-seg crop pre-pass + mode postprocess) ===
    "body":                     TaskConfig((299,),  (1.5, 1.5, 1.5),     "nnUNetTrainer",                          "body",                     mode_postprocess=("body",)),
    "face_mr":                  TaskConfig((856,),  (1.5, 1.5, 1.5),     "nnUNetTrainer_2000epochs_NoMirroring",   "face_mr",                  modality="mr", mode_postprocess=("aux",), licensed=True),
    "brain_aneurysm":           TaskConfig((615,),  (0.390625, 0.390625, 0.5000016391277313), "nnUNetTrainerDiceTopK10Loss_2000epochs", "brain_aneurysm", folds=None),
    "cerebral_bleed":           TaskConfig((150,),  None,                "nnUNetTrainer",                          "cerebral_bleed",           crop=("brain",)),
    "hip_implant":              TaskConfig((260,),  None,                "nnUNetTrainer",                          "hip_implant",              crop=("femur_left","femur_right","hip_left","hip_right")),
    "liver_vessels":            TaskConfig((8,),    None,                "nnUNetTrainer",                          "liver_vessels",            crop=("liver",), crop_addon=(20,20,20)),
    "lung_vessels":             TaskConfig((117,),  (0.703125, 0.703125, 1.0), "nnUNetTrainerSkeletonRecall",      "lung_vessels",             crop=("lung_upper_lobe_left","lung_lower_lobe_left","lung_upper_lobe_right","lung_middle_lobe_right","lung_lower_lobe_right"), robust_crop=True),
    "lung_vessels_LEGACY":      TaskConfig((258,),  None,                "nnUNetTrainer",                          "lung_vessels",             crop=("lung_upper_lobe_left","lung_lower_lobe_left","lung_upper_lobe_right","lung_middle_lobe_right","lung_lower_lobe_right")),
    "lung_nodules":             TaskConfig((913,),  (1.5, 1.5, 1.5),     "nnUNetTrainer_MOSAIC_1k_QuarterLR_NoMirroring", "lung_nodules",      crop=("lung_upper_lobe_left","lung_lower_lobe_left","lung_upper_lobe_right","lung_middle_lobe_right","lung_lower_lobe_right"), crop_addon=(10,10,10)),
    "pleural_pericard_effusion":TaskConfig((315,),  None,                "nnUNetTrainer",                          "pleural_pericard_effusion",crop=("lung_upper_lobe_left","lung_lower_lobe_left","lung_upper_lobe_right","lung_middle_lobe_right","lung_lower_lobe_right"), crop_addon=(50,50,50), folds=None),
    "kidney_cysts":             TaskConfig((789,),  (1.5, 1.5, 1.5),     "nnUNetTrainer_DASegOrd0_NoMirroring",    "kidney_cysts",             crop=("kidney_left","kidney_right","liver","spleen","colon"), crop_addon=(10,10,10), mode_postprocess=("aux",)),
    "liver_segments":           TaskConfig((570,),  (0.8046879768371582, 0.8046879768371582, 1.5), "nnUNetTrainerNoMirroring", "liver_segments", crop=("liver",), crop_addon=(10,10,10)),
    "liver_segments_mr":        TaskConfig((576,),  (1.1250001788139343, 1.1875, 3.0), "nnUNetTrainer_DASegOrd0_NoMirroring", "liver_segments_mr", modality="mr", crop=("liver",), crop_addon=(10,10,10)),
    "liver_lesions":            TaskConfig((591,),  (0.75, 0.75, 1.0),   "nnUNetTrainer",                          "liver_lesions",            model_config="3d_fullres_high", crop=("liver",), crop_addon=(10,10,10), robust_crop=True, use_softmax=True),
    "liver_lesions_mr":         TaskConfig((589,),  (0.8603515625, 0.857421875, 1.0), "nnUNetTrainer_DASegOrd0",   "liver_lesions_mr",         modality="mr", crop=("liver",), crop_addon=(3,3,3), robust_crop=True, use_softmax=True),
    "head_glands_cavities":     TaskConfig((775,),  (0.75, 0.75, 1.0),   "nnUNetTrainer_DASegOrd0_NoMirroring",    "head_glands_cavities",     model_config="3d_fullres_high", crop=("skull",), crop_addon=(10,10,10)),
    "head_muscles":             TaskConfig((777,),  (0.75, 0.75, 1.0),   "nnUNetTrainer_DASegOrd0_NoMirroring",    "head_muscles",             model_config="3d_fullres_high", crop=("skull",), crop_addon=(10,10,10)),
    "headneck_bones_vessels":   TaskConfig((776,),  (0.75, 0.75, 1.0),   "nnUNetTrainer_DASegOrd0_NoMirroring",    "headneck_bones_vessels",   model_config="3d_fullres_high", crop=("clavicula_left","clavicula_right","vertebrae_C1","vertebrae_C5","vertebrae_T1","vertebrae_T4"), crop_addon=(40,40,40)),
    "headneck_muscles":         TaskConfig((778, 779), (0.75, 0.75, 1.0),"nnUNetTrainer_DASegOrd0_NoMirroring",    "headneck_muscles",         model_config="3d_fullres_high", class_map_parts_key="class_map_parts_headneck_muscles", partname_map_key="map_taskid_to_partname_headneck_muscles", crop=("clavicula_left","clavicula_right","vertebrae_C1","vertebrae_C5","vertebrae_T1","vertebrae_T4"), crop_addon=(40,40,40)),
    "craniofacial_structures":  TaskConfig((115,),  (0.5, 0.5, 0.5),     "nnUNetTrainer_DASegOrd0_NoMirroring",    "craniofacial_structures",  crop=("skull",), crop_addon=(20,20,20)),
    "oculomotor_muscles":       TaskConfig((351,),  (0.47251562774181366, 0.47251562774181366, 0.8500002026557922), "nnUNetTrainer_DASegOrd0_NoMirroring", "oculomotor_muscles", crop=("skull",), crop_addon=(20,20,20)),
    "ventricle_parts":          TaskConfig((552,),  (0.4384765625, 0.4345703125, 1.0), "nnUNetTrainerNoMirroring", "ventricle_parts",          crop=("brain",), crop_addon=(0,0,0)),
    "abdominal_muscles":        TaskConfig((952,),  (0.75, 0.75, 1.0),   "nnUNetTrainer_DASegOrd0_NoMirroring",    "abdominal_muscles",        model_config="3d_fullres_high", crop=("body_trunc",), crop_addon=(5,5,5)),
    "brain_structures":         TaskConfig((409,),  (0.5, 0.5, 1.0),     "nnUNetTrainer_DASegOrd0",                "brain_structures",         model_config="3d_fullres_high", crop=("brain",), crop_addon=(10,10,10), licensed=True),
    "heartchambers_highres":    TaskConfig((301,),  None,                "nnUNetTrainer",                          "heartchambers_highres",    crop=("heart",), crop_addon=(5,5,5), mode_postprocess=("remove_outside",), remove_outside=("heart","aorta","inferior_vena_cava"), remove_outside_dilation_mm=10, licensed=True, robust_crop=True),
    "coronary_arteries":        TaskConfig((509,),  (0.7, 0.7, 0.7),     "nnUNetTrainerSkeletonRecall",            "coronary_arteries",        model_config="3d_fullres_high", crop=("heart",), crop_addon=(20,20,20), licensed=True),
    "coronary_arteries_LEGACY": TaskConfig((507,),  (0.7, 0.7, 0.7),     "nnUNetTrainer_DASegOrd0_NoMirroring",    "coronary_arteries",        model_config="3d_fullres_high", crop=("heart",), crop_addon=(20,20,20), licensed=True),
    "aortic_sinuses":           TaskConfig((920,),  (0.7, 0.7, 0.7),     "nnUNetTrainer_DASegOrd0_NoMirroring",    "aortic_sinuses",           model_config="3d_fullres_high", crop=("heart",), crop_addon=(0,0,0), licensed=True),

    # === Recursive-crop modes (crop_model set; validated on ToothFairy3, DSC 0.9999) ===
    "teeth": TaskConfig(
        task_ids=(113,),
        resample=(0.5, 0.5, 0.5),
        trainer="nnUNetTrainer_onlyMirror01",
        model_config="3d_lowres_high",
        class_map_key="teeth",
        crop=("teeth_lower", "teeth_upper"),
        crop_addon=(10, 10, 10),
        crop_model="craniofacial_structures",
    ),
}


# --- Fast-path preprocessor + predictor --------------------------------------

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

        # GPU order-3 cubic-B-spline input resample (separate-z auto-detected by anisotropy),
        # matching nnU-Net's resampling_fn_data (scipy order-3) to ~1e-13.
        return torch_resample_data_or_seg_to_shape(
            data, new_shape, original_spacing, target_spacing, is_seg=False, order=3, order_z=0,
            force_separate_z=None)


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


# --- Helpers -----------------------------------------------------------------

def resolve_model_dir(weights_dir: Path, task_id: int, trainer: str, plans: str, model_config: str) -> Path:
    pattern = f"Dataset{task_id:03d}_*"
    candidates = sorted(weights_dir.glob(pattern))
    if not candidates:
        raise FileNotFoundError(
            f"No dataset folder matching {pattern} in {weights_dir}.\n"
            f"Run: python download_totalseg_weights.py --full (or fetch Task{task_id:03d})"
        )
    if len(candidates) > 1:
        raise RuntimeError(f"Multiple dataset folders for Task{task_id:03d}: {candidates}")
    return candidates[0] / f"{trainer}__{plans}__{model_config}"


def build_predictors(cfg: TaskConfig, weights_dir: Path, device: torch.device, step_size: float,
                     use_fast: bool = True) -> dict:
    predictor_cls = FastPredictor if use_fast else SimplePredictor  # SimplePredictor = scipy separate-z preprocessing
    predictors = {}
    for task_id in cfg.task_ids:
        model_path = resolve_model_dir(weights_dir, task_id, cfg.trainer, cfg.plans, cfg.model_config)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Weights not found: {model_path}\n"
                f"Expected layout: {{weights_dir}}/Dataset{task_id:03d}_*/"
                f"{cfg.trainer}__{cfg.plans}__{cfg.model_config}/"
            )
        p = predictor_cls(
            tile_step_size=step_size,
            use_gaussian=True,
            use_mirroring=False,
            perform_everything_on_device=device.type != "cpu",
            device=device,
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=False,
        )
        folds = list(cfg.folds) if cfg.folds is not None else None
        p.initialize_from_trained_model_folder(
            str(model_path), use_folds=folds, checkpoint_name=CHECKPOINT,
        )
        p.network.to(device)
        # Sanity check: cfg.resample is the TS-level (cucim) spacing. For most modes
        # it equals the model's plans spacing, so FastPreprocessor's internal resample
        # is a no-op. A few modes (e.g. lung_nodules: cfg 1.5mm vs plans 0.742mm) match
        # official by double-resampling (cucim → 1.5, then FastPreprocessor → plans), so
        # a mismatch here is informational, NOT an error.
        if cfg.resample is not None:
            plan_spacing = list(p.configuration_manager.spacing)  # ZYX
            cfg_zyx = [cfg.resample[2], cfg.resample[1], cfg.resample[0]]
            if not np.allclose(plan_spacing, cfg_zyx, atol=1e-4):
                print(f"  NOTE: Task{task_id} plans-spacing {plan_spacing} (ZYX) != "
                      f"cfg.resample {cfg_zyx} (ZYX) — FastPreprocessor will resample "
                      f"cfg.resample → plans spacing (matches official double-resample).")
        predictors[task_id] = p
    return predictors


def _merge_part(part_seg: np.ndarray, task_id: int, cfg: TaskConfig,
                seg_combined: np.ndarray, global_idx: dict) -> np.ndarray:
    """In-place: write `part_seg` labels into `seg_combined` in the mode's global label space."""
    if cfg.class_map_parts_key is None:
        # Single-model: model labels are already global labels.
        np.copyto(seg_combined, part_seg.astype(np.uint8))
        return seg_combined
    parts = _CLASS_MAP_PARTS[cfg.class_map_parts_key]
    partname_map = _PARTNAME_MAPS[cfg.partname_map_key]
    part_name = partname_map[task_id]
    for local_idx, class_name in parts[part_name].items():
        seg_combined[part_seg == local_idx] = global_idx[class_name]
    return seg_combined


# --- Rough segmentation + crop mask --------------------------------

# Rough crop models, keyed by crop_task (mirrors python_api.py rough-seg block).
_ROUGH_CONFIGS = {
    "total":     dict(task_ids=(298,), resample=(6.0, 6.0, 6.0),
                      trainer="nnUNetTrainer_4000epochs_NoMirroring", class_map_key="total"),
    "total_3mm": dict(task_ids=(297,), resample=(3.0, 3.0, 3.0),
                      trainer="nnUNetTrainer_4000epochs_NoMirroring", class_map_key="total"),
    "body":      dict(task_ids=(300,), resample=(6.0, 6.0, 6.0),
                      trainer="nnUNetTrainer", class_map_key="body"),
    "total_mr":  dict(task_ids=(852,), resample=(3.0, 3.0, 3.0),
                      trainer="nnUNetTrainer_2000epochs_NoMirroring", class_map_key="total_mr",
                      modality="mr"),
}


def rough_cfg_for(cfg: TaskConfig) -> tuple[str, TaskConfig]:
    """Pick the rough crop model + its config (mirrors official crop dispatch).

    robust_crop modes use the 3mm total model (T297) instead of 6mm (T298); for MR
    the rough model is already 3mm (T852), so robust_crop is a no-op there.
    """
    if cfg.modality == "mr":
        crop_task = "total_mr"
    elif set(cfg.crop) & {"body_trunc", "body_extremities"}:
        crop_task = "body"
    elif cfg.robust_crop:
        crop_task = "total_3mm"
    else:
        crop_task = "total"
    spec  = _ROUGH_CONFIGS[crop_task]
    rough = TaskConfig(
        task_ids=spec["task_ids"], resample=spec["resample"], trainer=spec["trainer"],
        class_map_key=spec["class_map_key"], modality=spec.get("modality", "ct"),
    )
    return crop_task, rough


def _mask_from_seg(organ_seg: nib.Nifti1Image, class_map_inv: dict, rois) -> nib.Nifti1Image:
    data = np.asarray(organ_seg.dataobj)   # uint8 label map; avoids a full float64 get_fdata
    mask = np.zeros(organ_seg.shape, dtype=np.uint8)
    for roi in rois:
        if roi not in class_map_inv:
            raise KeyError(f"crop ROI '{roi}' not in the rough-seg class map "
                           f"{sorted(class_map_inv)[:8]}... — check the crop config / rough model")
        mask[data == class_map_inv[roi]] = 1
    return nib.Nifti1Image(mask, organ_seg.affine)


def _build_predictors_cached(cfg: TaskConfig, weights_dir, device, step, cache, key, use_fast=True):
    if key not in cache:
        cache[key] = build_predictors(cfg, weights_dir, device, step, use_fast=use_fast)
    return cache[key]


def run_mode_to_original_space(file_path: str, mode_name: str, weights_dir, device, cache):
    """Run a full mode pipeline (incl. its own crop pre-pass, recursively if the mode
    has a crop_model) and return its multilabel seg as a nib image in original space.
    Used as the crop source for recursive-crop modes (e.g. teeth → craniofacial_structures)."""
    cfg = TASK_CONFIGS[mode_name]
    preds = _build_predictors_cached(cfg, weights_dir, device,
                                     _step_size_for(mode_name, cfg), cache, mode_name)
    crop_mask = None
    if cfg.crop:
        crop_mask, _ = build_crop_mask(file_path, cfg, weights_dir, device, cache)
    return _predict_to_original_space(file_path, preds, cfg, mode_name, crop_mask)


def build_crop_mask(file_path: str, cfg: TaskConfig, weights_dir, device, cache):
    """Build (crop_mask, remove_outside_mask|None) from a rough crop source.

    Source is either a recursive full-mode run (cfg.crop_model set, e.g. teeth →
    craniofacial_structures) or a rough whole-body seg (6mm total / 3mm robust / body / MR).
    """
    if cfg.crop_model is not None:
        organ_seg = run_mode_to_original_space(file_path, cfg.crop_model, weights_dir, device, cache)
        cmap_key  = cfg.crop_model
    else:
        crop_task, rough_cfg = rough_cfg_for(cfg)
        rough_preds = _build_predictors_cached(
            rough_cfg, weights_dir, device, _step_size_for(crop_task, rough_cfg),
            cache, "rough:" + crop_task)
        organ_seg = _predict_to_original_space(file_path, rough_preds, rough_cfg, crop_task)
        # rough_cfg.class_map_key is the real class map; crop_task may be an alias ("total_3mm").
        cmap_key  = rough_cfg.class_map_key
    class_map_inv = {v: k for k, v in class_map[cmap_key].items()}
    crop_mask   = _mask_from_seg(organ_seg, class_map_inv, cfg.crop)
    remove_mask = (_mask_from_seg(organ_seg, class_map_inv, cfg.remove_outside)
                   if cfg.remove_outside else None)
    return crop_mask, remove_mask


# --- Inference ---------------------------------------------------------------

def _predict_to_original_space(file_path: str, predictors: dict, cfg: TaskConfig,
                               task_name: str, crop_mask_nib=None) -> nib.Nifti1Image:
    """Predict and return a multilabel seg in the input image's original space.

    Order mirrors official nnunet.py: crop (original orientation) → canonical →
    resample → predict → resample-back → undo canonical → undo crop.
    """
    img_orig = nib.load(file_path)

    bbox = None
    if crop_mask_nib is not None:
        if not np.asarray(crop_mask_nib.dataobj).any():   # dataobj (uint8) avoids a full float64 get_fdata
            # Empty crop mask → empty segmentation (matches nnunet.py:455-458).
            return nib.Nifti1Image(np.zeros(img_orig.shape, dtype=np.uint8), img_orig.affine)
        # Official forces crop_addon=[20,20,20] for every crop_model=None mode: the crop block
        # at python_api.py:767 runs for standard crops too (roi_subset_crop = crop if crop is
        # not None ...) and overrides the per-task value. Replicate for parity.
        addon = [20, 20, 20] if cfg.crop_model is None else list(cfg.crop_addon)
        # Crop img_orig directly (crop_to_mask_gpu slices, never mutates) — avoids an
        # extra full-volume get_fdata() copy.
        img_work, bbox = crop_to_mask_gpu(img_orig, crop_mask_nib, addon=addon, dtype=np.int32)
    else:
        img_work = img_orig

    img_can       = nib.as_closest_canonical(img_work)
    img_can_shape = img_can.shape

    if cfg.resample is not None:
        # dtype=np.int32 matches official (nnunet.py:485-486): the resampled image is
        # truncated to int BEFORE the model. Float here diverges on low-signal MR edges
        # (e.g. total_mr scapula) and shifts the rough-seg liver mask → wrong crop bbox.
        img_rsp     = tseg_change_spacing(img_can, list(cfg.resample), order=3, dtype=np.int32, use_gpu=True)
        arr_xyz     = img_rsp.get_fdata().astype(np.float32)
        spacing_zyx = [float(cfg.resample[2]), float(cfg.resample[1]), float(cfg.resample[0])]
    else:
        img_rsp     = img_can
        arr_xyz     = img_can.get_fdata().astype(np.float32)
        zooms_xyz   = img_can.header.get_zooms()[:3]
        spacing_zyx = [float(zooms_xyz[2]), float(zooms_xyz[1]), float(zooms_xyz[0])]

    image = arr_xyz.transpose(2, 1, 0)[np.newaxis]   # (1, Z, Y, X)
    props = {'spacing': spacing_zyx}

    seg_combined = None
    global_idx   = {name: idx for idx, name in class_map[cfg.class_map_key].items()}
    for task_id, predictor in predictors.items():
        part_seg = predictor.inference(image, props, use_softmax=cfg.use_softmax).numpy()
        if seg_combined is None:
            seg_combined = np.zeros(part_seg.shape, dtype=np.uint8)
        _merge_part(part_seg, task_id, cfg, seg_combined, global_idx)

    # Postprocess at resample resolution (matches nnunet.py:622-637).
    seg_xyz = seg_combined.transpose(2, 1, 0).astype(np.uint8)
    seg_nib = nib.Nifti1Image(seg_xyz, img_rsp.affine)
    seg_nib = remove_auxiliary_labels(seg_nib, task_name)   # no-op without `{task}_auxiliary`
    if "body" in cfg.mode_postprocess:
        data = seg_nib.get_fdata().astype(np.uint8)
        data = keep_largest_blob_multilabel_gpu(data, class_map[task_name], ["body_trunc"])
        vox_vol = float(np.prod(seg_nib.header.get_zooms()))
        data = remove_small_blobs_multilabel_gpu(
            data, class_map[task_name], ["body_extremities"],
            interval=[50000 / vox_vol, 1e10],
        )
        seg_nib = nib.Nifti1Image(data, seg_nib.affine)

    # Undo resample → canonical → crop (matches nnunet.py:706-728).
    if cfg.resample is not None:
        seg_nib = tseg_change_spacing(
            seg_nib, list(cfg.resample), img_can_shape,
            order=0, force_affine=img_can.affine, use_gpu=True,
        )
    seg_nib = tseg_undo_canonical(seg_nib, img_orig)
    if bbox is not None:
        seg_nib = undo_crop_gpu(seg_nib, img_orig, bbox)
    return seg_nib


@log_runtime
def infer_file(file_path: str, predictors: dict, cfg: TaskConfig, task_name: str,
               output_path: str, weights_dir=None, device=None, cache: Optional[dict] = None,
               roi_subset: tuple = (), v1_order: bool = False):
    crop_mask = remove_mask = None
    if cfg.crop:
        crop_mask, remove_mask = build_crop_mask(file_path, cfg, weights_dir, device, cache)

    seg_nib  = _predict_to_original_space(file_path, predictors, cfg, task_name, crop_mask)
    img_data = seg_nib.get_fdata().astype(np.uint8)

    # v1 label reorder for `total` (matches nnunet.py:738), then roi_subset filter (nnunet.py:740).
    if v1_order and task_name == "total":
        img_data = reorder_multilabel_like_v1(img_data, class_map["total"], class_map["total_v1"])
    if roi_subset:
        cmap = class_map["total_v1"] if (v1_order and task_name == "total") else class_map[cfg.class_map_key]
        keep = [idx for idx, name in cmap.items() if name in roi_subset]
        img_data = (img_data * np.isin(img_data, keep)).astype(np.uint8)

    # remove_outside postprocess in ORIGINAL space (matches nnunet.py:744-749).
    if "remove_outside" in cfg.mode_postprocess and remove_mask is not None:
        img_orig    = nib.load(file_path)
        dilation_vx = int(cfg.remove_outside_dilation_mm
                          / float(np.mean(img_orig.header.get_zooms()[:3])))
        img_data    = remove_outside_of_mask_gpu(img_data, remove_mask.get_fdata(), addon=dilation_vx)

    out_name = os.path.basename(file_path).replace("_0000.nii.gz", "").replace(".nii.gz", "")
    seg_out  = img_data.transpose(2, 1, 0).astype(np.uint8)   # XYZ → ZYX for sitk
    sitk_img = sitk.GetImageFromArray(seg_out)
    sitk_img.CopyInformation(sitk.ReadImage(file_path))
    sitk.WriteImage(sitk_img, os.path.join(output_path, f"{out_name}.nii.gz"))

    # Derived body/skin masks. Official only emits these in split-file mode; we add
    # them in multilabel mode as a convenience (parity is judged on the multilabel array).
    if "body" in cfg.mode_postprocess:
        img_orig = nib.load(file_path)
        body_nib = nib.Nifti1Image((img_data > 0).astype(np.uint8), img_orig.affine)
        nib.save(body_nib, os.path.join(output_path, f"{out_name}_body.nii.gz"))
        nib.save(extract_skin(img_orig, body_nib),
                 os.path.join(output_path, f"{out_name}_skin.nii.gz"))


def _step_size_for(task_name: str, cfg: TaskConfig) -> float:
    # Mirrors official: tile_step=0.8 only when task=="total" AND resample[0]<3.0 mm.
    # `total_mr` always uses 0.5.
    if task_name == "total" and cfg.resample is not None and cfg.resample[0] < 3.0:
        return 0.8
    return 0.5


def main():
    parser = argparse.ArgumentParser(
        description="TotalSegmentator unified inference (CT + MR, config-driven)"
    )
    parser.add_argument("-i", "--input_path", required=True,
                        help="Folder containing *.nii.gz input images")
    parser.add_argument("-o", "--output_path", required=True,
                        help="Folder to write multilabel output NIfTIs")
    parser.add_argument("--task", default="total", choices=sorted(TASK_CONFIGS.keys()),
                        help="Mode (default: total)")
    parser.add_argument("--weights_dir", default=None,
                        help="Path to TotalSegmentator weights "
                             "(default: ~/.totalsegmentator/nnunet/results)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use_softmax", action="store_true", default=False,
                        help="Override: softmax+argmax instead of the 0.5 logit threshold "
                             "(matches official for low-confidence sparse classes)")
    parser.add_argument("--roi_subset", nargs="+", default=None,
                        help="Restrict output to these ROI names (total* modes only); "
                             "prunes multi-model parts to those containing the subset")
    parser.add_argument("--v1_order", action="store_true", default=False,
                        help="Reorder `total` labels to the v1 label scheme (class_map total_v1)")
    args = parser.parse_args()

    cfg = TASK_CONFIGS[args.task]
    if args.use_softmax:
        cfg = replace(cfg, use_softmax=True)

    roi_subset = tuple(args.roi_subset) if args.roi_subset else ()
    if roi_subset and not args.task.startswith("total"):
        sys.exit(f"ERROR: --roi_subset is only supported for total* modes, not '{args.task}'.")
    # Prune multi-model total parts to those whose part-map intersects the subset (nnunet.py:539-550).
    if roi_subset and cfg.class_map_parts_key:
        parts = _CLASS_MAP_PARTS[cfg.class_map_parts_key]
        partname_to_taskid = {v: k for k, v in _PARTNAME_MAPS[cfg.partname_map_key].items()}
        new_ids = tuple(partname_to_taskid[pn] for pn, pm in parts.items()
                        if any(o in roi_subset for o in pm.values()))
        if new_ids:
            print(f"[roi_subset] pruning models {cfg.task_ids} → {new_ids}")
            cfg = replace(cfg, task_ids=new_ids)
    if cfg.licensed:
        print(f"NOTE: '{args.task}' is a commercial TotalSegmentator mode. "
              f"This script skips the license check — make sure you have the weights.")

    # Deterministic forward: the fp16 sliding-window pass otherwise selects cuDNN algorithms
    # nondeterministically across processes, occasionally shifting the rough-seg crop mask by
    # ~1 coarse voxel (e.g. liver_lesions crop 175↔172 → DSC 1.0↔0.95). Force reproducibility.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    weights_dir = Path(args.weights_dir) if args.weights_dir else Path(get_weights_dir())
    device      = torch.device(args.device)

    step_size = _step_size_for(args.task, cfg)

    # --- Input validation -----------------------------------------------------
    if not os.path.isdir(args.input_path):
        sys.exit(f"ERROR: input path is not a directory: {args.input_path}")
    files = sorted(glob.glob(os.path.join(args.input_path, "*.nii.gz")))
    if not files:
        sys.exit(f"ERROR: no .nii.gz files found in {args.input_path}")
    os.makedirs(args.output_path, exist_ok=True)

    # Predictor cache: main model + any rough/recursive crop-source models are built
    # once and reused across files (build_crop_mask / run_mode_to_original_space populate it).
    cache = {}
    try:
        predictors = _build_predictors_cached(cfg, weights_dir, device, step_size, cache, args.task)
    except FileNotFoundError as e:
        sys.exit(f"ERROR: could not load weights for task '{args.task}' "
                 f"(models {cfg.task_ids}) under {weights_dir}: {e}\n"
                 f"Download the TotalSegmentator weights or pass --weights_dir.")
    if cfg.crop:
        src = cfg.crop_model if cfg.crop_model else rough_cfg_for(cfg)[0]
        print(f"[crop] source={src} (crop_model={cfg.crop_model}, crop={cfg.crop})")

    print(f"[task={args.task}] models={cfg.task_ids}  resample={cfg.resample}  step={step_size}  modality={cfg.modality}")
    print(f"Loading sub-model(s) from {weights_dir} ...")

    # Per-case error handling: a single bad case (corrupt NIfTI, OOM, rough-seg failure)
    # is logged and skipped — it never aborts the batch (mirrors nnunet_infer_nii.py).
    failures = []
    for file in tqdm(files, desc=f"Inference ({args.task})"):
        print(f"\n{os.path.basename(file)}")
        try:
            infer_file(file, predictors, cfg, args.task, args.output_path,
                       weights_dir=weights_dir, device=device, cache=cache,
                       roi_subset=roi_subset, v1_order=args.v1_order)
        except Exception as e:
            failures.append(os.path.basename(file))
            print(f"FAILED {os.path.basename(file)}: {type(e).__name__}: {e}")
            empty_cache(device)

    if failures:
        print(f"\n{len(failures)} of {len(files)} case(s) FAILED and were skipped:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)


if __name__ == "__main__":
    main()
