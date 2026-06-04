"""
TotalSegmentator unified inference (CT + MR), config-driven.

Stage 1: all uncropped, single-/multi-model modes.
  CT:  total, trunk_cavities, breasts,
       (licensed:) vertebrae_body, appendicular_bones, tissue_types,
       tissue_4_types, face, thigh_shoulder_muscles
  MR:  total_mr, body_mr, vertebrae_mr,
       (licensed:) appendicular_bones_mr, tissue_types_mr,
       thigh_shoulder_muscles_mr

Fast-path applied to every mode:
  - cucim GPU change_spacing for canonical resample (skipped when resample=None)
  - FastPreprocessor: torch GPU trilinear inside run_case_npy
  - logits-to-segmentation via threshold (max_logit >= 0.5) — no softmax

Stages 2-3 (crop pre-pass, postprocess, recursion, ROI subset) are reserved by
config but not yet implemented; invoking those modes exits with a clear error.

Usage:
  python totalseg_infer.py -i <in_folder> -o <out_folder> --task total
  python totalseg_infer.py -i <in_folder> -o <out_folder> --task total_mr
  python totalseg_infer.py -i <in_folder> -o <out_folder> --task trunk_cavities
"""
import argparse
import glob
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from tqdm import tqdm

_TOTALSEG_SRC = Path(__file__).parent.parent / "TotalSegmentator"
if _TOTALSEG_SRC.exists():
    sys.path.insert(0, str(_TOTALSEG_SRC))

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


_CUSTOM_TRAINERS = {
    "nnUNetTrainer_DASegOrd0":              nnUNetTrainer_DASegOrd0,
    "nnUNetTrainer_DASegOrd0_NoMirroring":  nnUNetTrainer_DASegOrd0_NoMirroring,
    "nnUNetTrainer_2000epochs_NoMirroring": nnUNetTrainer_2000epochs_NoMirroring,
    "nnUNetTrainer_4000epochs_NoMirroring": nnUNetTrainer_4000epochs_NoMirroring,
}


def _custom_find_class(folder, class_name, current_module):
    if class_name in _CUSTOM_TRAINERS:
        return _CUSTOM_TRAINERS[class_name]
    return recursive_find_python_class(folder, class_name, current_module)


import nnunet_infer_nii  # noqa: E402
nnunet_infer_nii.recursive_find_python_class = _custom_find_class

# --- Project imports ---------------------------------------------------------

from nnunetv2.preprocessing.cropping.cropping import crop_to_nonzero  # noqa: E402
from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor  # noqa: E402
from nnunetv2.preprocessing.resampling.default_resampling import (  # noqa: E402
    compute_new_shape, fast_resample_data_or_seg_to_shape,
)
from nnunetv2.utilities.utils import log_runtime  # noqa: E402
from totalsegmentator.alignment import undo_canonical as tseg_undo_canonical  # noqa: E402
from totalsegmentator.config import get_weights_dir  # noqa: E402
from totalsegmentator.map_to_binary import (  # noqa: E402
    class_map, class_map_5_parts, class_map_parts_mr, class_map_parts_headneck_muscles,
    map_taskid_to_partname_ct, map_taskid_to_partname_mr, map_taskid_to_partname_headneck_muscles,
)
from totalsegmentator.resampling import change_spacing as tseg_change_spacing  # noqa: E402

from nnunet_infer_nii import SimplePredictor  # noqa: E402


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
    # --- Reserved for later stages (defined here so configs are stable) ---
    crop: tuple = ()                         # ROI names; non-empty triggers stage-2 crop pre-pass
    crop_addon: tuple = (3, 3, 3)
    crop_model: Optional[str] = None         # recursive crop (only `teeth` uses this)
    mode_postprocess: tuple = ()             # tags: "body", "remove_outside", "aux"
    remove_outside: tuple = ()
    remove_outside_dilation_mm: Optional[int] = None
    stage: int = 1


TASK_CONFIGS: dict[str, TaskConfig] = {

    # === Stage 1 — multi-model totals ===
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

    # === Stage 1 — single-model open CT ===
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

    # === Stage 1 — single-model open MR ===
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

    # === Stage 1 — licensed, uncropped CT ===
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

    # === Stage 1 — licensed, uncropped MR ===
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

    # === Stage 2 — placeholders (crop pre-pass / postprocess required) ===
    "body":                     TaskConfig((299,),  (1.5, 1.5, 1.5),     "nnUNetTrainer",                          "body",                     mode_postprocess=("body",),                  stage=2),
    "face_mr":                  TaskConfig((856,),  (1.5, 1.5, 1.5),     "nnUNetTrainer_2000epochs_NoMirroring",   "face_mr",                  modality="mr", mode_postprocess=("aux",), licensed=True, stage=2),
    "brain_aneurysm":           TaskConfig((615,),  (0.390625, 0.390625, 0.5000016391277313), "nnUNetTrainerDiceTopK10Loss_2000epochs", "brain_aneurysm", folds=None, stage=2),
    "cerebral_bleed":           TaskConfig((150,),  None,                "nnUNetTrainer",                          "cerebral_bleed",           crop=("brain",), stage=2),
    "hip_implant":              TaskConfig((260,),  None,                "nnUNetTrainer",                          "hip_implant",              crop=("femur_left","femur_right","hip_left","hip_right"), stage=2),
    "liver_vessels":            TaskConfig((8,),    None,                "nnUNetTrainer",                          "liver_vessels",            crop=("liver",), crop_addon=(20,20,20), stage=2),
    "lung_vessels":             TaskConfig((117,),  (0.703125, 0.703125, 1.0), "nnUNetTrainerSkeletonRecall",      "lung_vessels",             crop=("lung_upper_lobe_left","lung_lower_lobe_left","lung_upper_lobe_right","lung_middle_lobe_right","lung_lower_lobe_right"), stage=2),
    "lung_vessels_LEGACY":      TaskConfig((258,),  None,                "nnUNetTrainer",                          "lung_vessels",             crop=("lung_upper_lobe_left","lung_lower_lobe_left","lung_upper_lobe_right","lung_middle_lobe_right","lung_lower_lobe_right"), stage=2),
    "lung_nodules":             TaskConfig((913,),  (1.5, 1.5, 1.5),     "nnUNetTrainer_MOSAIC_1k_QuarterLR_NoMirroring", "lung_nodules",      crop=("lung_upper_lobe_left","lung_lower_lobe_left","lung_upper_lobe_right","lung_middle_lobe_right","lung_lower_lobe_right"), crop_addon=(10,10,10), stage=2),
    "pleural_pericard_effusion":TaskConfig((315,),  None,                "nnUNetTrainer",                          "pleural_pericard_effusion",crop=("lung_upper_lobe_left","lung_lower_lobe_left","lung_upper_lobe_right","lung_middle_lobe_right","lung_lower_lobe_right"), crop_addon=(50,50,50), folds=None, stage=2),
    "kidney_cysts":             TaskConfig((789,),  (1.5, 1.5, 1.5),     "nnUNetTrainer_DASegOrd0_NoMirroring",    "kidney_cysts",             crop=("kidney_left","kidney_right","liver","spleen","colon"), crop_addon=(10,10,10), mode_postprocess=("aux",), stage=2),
    "liver_segments":           TaskConfig((570,),  (0.8046879768371582, 0.8046879768371582, 1.5), "nnUNetTrainerNoMirroring", "liver_segments", crop=("liver",), crop_addon=(10,10,10), stage=2),
    "liver_segments_mr":        TaskConfig((576,),  (1.1250001788139343, 1.1875, 3.0), "nnUNetTrainer_DASegOrd0_NoMirroring", "liver_segments_mr", modality="mr", crop=("liver",), crop_addon=(10,10,10), stage=2),
    "liver_lesions":            TaskConfig((591,),  (0.75, 0.75, 1.0),   "nnUNetTrainer",                          "liver_lesions",            model_config="3d_fullres_high", crop=("liver",), crop_addon=(10,10,10), stage=2),
    "liver_lesions_mr":         TaskConfig((589,),  (0.8603515625, 0.857421875, 1.0), "nnUNetTrainer_DASegOrd0",   "liver_lesions_mr",         modality="mr", crop=("liver",), crop_addon=(3,3,3), stage=2),
    "head_glands_cavities":     TaskConfig((775,),  (0.75, 0.75, 1.0),   "nnUNetTrainer_DASegOrd0_NoMirroring",    "head_glands_cavities",     model_config="3d_fullres_high", crop=("skull",), crop_addon=(10,10,10), stage=2),
    "head_muscles":             TaskConfig((777,),  (0.75, 0.75, 1.0),   "nnUNetTrainer_DASegOrd0_NoMirroring",    "head_muscles",             model_config="3d_fullres_high", crop=("skull",), crop_addon=(10,10,10), stage=2),
    "headneck_bones_vessels":   TaskConfig((776,),  (0.75, 0.75, 1.0),   "nnUNetTrainer_DASegOrd0_NoMirroring",    "headneck_bones_vessels",   model_config="3d_fullres_high", crop=("clavicula_left","clavicula_right","vertebrae_C1","vertebrae_C5","vertebrae_T1","vertebrae_T4"), crop_addon=(40,40,40), stage=2),
    "headneck_muscles":         TaskConfig((778, 779), (0.75, 0.75, 1.0),"nnUNetTrainer_DASegOrd0_NoMirroring",    "headneck_muscles",         model_config="3d_fullres_high", class_map_parts_key="class_map_parts_headneck_muscles", partname_map_key="map_taskid_to_partname_headneck_muscles", crop=("clavicula_left","clavicula_right","vertebrae_C1","vertebrae_C5","vertebrae_T1","vertebrae_T4"), crop_addon=(40,40,40), stage=2),
    "craniofacial_structures":  TaskConfig((115,),  (0.5, 0.5, 0.5),     "nnUNetTrainer_DASegOrd0_NoMirroring",    "craniofacial_structures",  crop=("skull",), crop_addon=(20,20,20), stage=2),
    "oculomotor_muscles":       TaskConfig((351,),  (0.47251562774181366, 0.47251562774181366, 0.8500002026557922), "nnUNetTrainer_DASegOrd0_NoMirroring", "oculomotor_muscles", crop=("skull",), crop_addon=(20,20,20), stage=2),
    "ventricle_parts":          TaskConfig((552,),  (0.4384765625, 0.4345703125, 1.0), "nnUNetTrainerNoMirroring", "ventricle_parts",          crop=("brain",), crop_addon=(0,0,0), stage=2),
    "abdominal_muscles":        TaskConfig((952,),  (0.75, 0.75, 1.0),   "nnUNetTrainer_DASegOrd0_NoMirroring",    "abdominal_muscles",        model_config="3d_fullres_high", crop=("body_trunc",), crop_addon=(5,5,5), stage=2),
    "brain_structures":         TaskConfig((409,),  (0.5, 0.5, 1.0),     "nnUNetTrainer_DASegOrd0",                "brain_structures",         model_config="3d_fullres_high", crop=("brain",), crop_addon=(10,10,10), licensed=True, stage=2),
    "heartchambers_highres":    TaskConfig((301,),  None,                "nnUNetTrainer",                          "heartchambers_highres",    crop=("heart",), crop_addon=(5,5,5), mode_postprocess=("remove_outside",), remove_outside=("heart","aorta","inferior_vena_cava"), remove_outside_dilation_mm=10, licensed=True, stage=2),
    "coronary_arteries":        TaskConfig((509,),  (0.7, 0.7, 0.7),     "nnUNetTrainerSkeletonRecall",            "coronary_arteries",        model_config="3d_fullres_high", crop=("heart",), crop_addon=(20,20,20), licensed=True, stage=2),
    "coronary_arteries_LEGACY": TaskConfig((507,),  (0.7, 0.7, 0.7),     "nnUNetTrainer_DASegOrd0_NoMirroring",    "coronary_arteries",        model_config="3d_fullres_high", crop=("heart",), crop_addon=(20,20,20), licensed=True, stage=2),
    "aortic_sinuses":           TaskConfig((920,),  (0.7, 0.7, 0.7),     "nnUNetTrainer_DASegOrd0_NoMirroring",    "aortic_sinuses",           model_config="3d_fullres_high", crop=("heart",), crop_addon=(0,0,0), licensed=True, stage=2),

    # === Stage 3 — recursive crop ===
    "teeth": TaskConfig(
        task_ids=(113,),
        resample=(0.5, 0.5, 0.5),
        trainer="nnUNetTrainer_onlyMirror01",
        model_config="3d_lowres_high",
        class_map_key="teeth",
        crop=("teeth_lower", "teeth_upper"),
        crop_addon=(10, 10, 10),
        crop_model="craniofacial_structures",
        stage=3,
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


def build_predictors(cfg: TaskConfig, weights_dir: Path, device: torch.device, step_size: float) -> dict:
    predictors = {}
    for task_id in cfg.task_ids:
        model_path = resolve_model_dir(weights_dir, task_id, cfg.trainer, cfg.plans, cfg.model_config)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Weights not found: {model_path}\n"
                f"Expected layout: {{weights_dir}}/Dataset{task_id:03d}_*/"
                f"{cfg.trainer}__{cfg.plans}__{cfg.model_config}/"
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
        folds = list(cfg.folds) if cfg.folds is not None else None
        p.initialize_from_trained_model_folder(
            str(model_path), use_folds=folds, checkpoint_name=CHECKPOINT,
        )
        p.network.to(device)
        # Assert model's plans-spacing matches cfg.resample when cfg.resample is set
        if cfg.resample is not None:
            plan_spacing = list(p.configuration_manager.spacing)  # ZYX
            cfg_zyx = [cfg.resample[2], cfg.resample[1], cfg.resample[0]]
            if not np.allclose(plan_spacing, cfg_zyx, atol=1e-4):
                raise RuntimeError(
                    f"Task{task_id} plans-spacing {plan_spacing} (ZYX) != cfg.resample {cfg_zyx} (ZYX). "
                    f"Likely wrong trainer/plans/model_config combo in TASK_CONFIGS."
                )
        predictors[task_id] = p
    return predictors


def _remove_auxiliary_labels(seg: np.ndarray, task_name: str) -> np.ndarray:
    aux_key = f"{task_name}_auxiliary"
    if aux_key in class_map:
        for idx in class_map[aux_key].keys():
            seg[seg == idx] = 0
    return seg


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


# --- Inference ---------------------------------------------------------------

@log_runtime
def infer_file(file_path: str, predictors: dict, cfg: TaskConfig, task_name: str, output_path: str):
    img_orig      = nib.load(file_path)
    img_can       = nib.as_closest_canonical(img_orig)
    img_can_shape = img_can.shape

    if cfg.resample is not None:
        # T2 forward: cucim GPU change_spacing to cfg.resample (XYZ).
        img_rsp     = tseg_change_spacing(img_can, list(cfg.resample), order=3, use_gpu=True)
        arr_xyz     = img_rsp.get_fdata().astype(np.float32)
        # nnU-Net props['spacing'] is ZYX; cfg.resample is XYZ → reverse.
        spacing_zyx = [float(cfg.resample[2]), float(cfg.resample[1]), float(cfg.resample[0])]
    else:
        # Native spacing: no T2 resample; spacing comes from the image itself.
        img_rsp     = img_can
        arr_xyz     = img_can.get_fdata().astype(np.float32)
        zooms_xyz   = img_can.header.get_zooms()[:3]
        spacing_zyx = [float(zooms_xyz[2]), float(zooms_xyz[1]), float(zooms_xyz[0])]

    image = arr_xyz.transpose(2, 1, 0)[np.newaxis]   # (1, Z, Y, X)
    props = {'spacing': spacing_zyx}

    seg_combined = None
    global_idx   = {name: idx for idx, name in class_map[cfg.class_map_key].items()}
    for task_id, predictor in predictors.items():
        part_seg = predictor.inference(image, props, use_softmax=False).numpy()
        if seg_combined is None:
            seg_combined = np.zeros(part_seg.shape, dtype=np.uint8)
        _merge_part(part_seg, task_id, cfg, seg_combined, global_idx)

    # Mode-specific postprocess (stage-1 subset: auxiliary-label removal only)
    if "aux" in cfg.mode_postprocess:
        seg_combined = _remove_auxiliary_labels(seg_combined, task_name)

    # T2 inverse (only when we forward-resampled) + undo canonical
    seg_xyz = seg_combined.transpose(2, 1, 0).astype(np.uint8)
    seg_nib = nib.Nifti1Image(seg_xyz, img_rsp.affine)
    if cfg.resample is not None:
        seg_nib = tseg_change_spacing(
            seg_nib, list(cfg.resample), img_can_shape,
            order=0, force_affine=img_can.affine, use_gpu=True,
        )
    seg_nib = tseg_undo_canonical(seg_nib, img_orig)

    seg_out = seg_nib.get_fdata().transpose(2, 1, 0).astype(np.uint8)
    out_name = os.path.basename(file_path).replace("_0000.nii.gz", "").replace(".nii.gz", "")
    sitk_img = sitk.GetImageFromArray(seg_out)
    sitk_img.CopyInformation(sitk.ReadImage(file_path))
    sitk.WriteImage(sitk_img, os.path.join(output_path, f"{out_name}.nii.gz"))


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
    args = parser.parse_args()

    cfg = TASK_CONFIGS[args.task]
    if cfg.stage > 1:
        sys.exit(
            f"ERROR: mode '{args.task}' requires stage {cfg.stage} "
            f"(crop pre-pass / mode postprocess / recursive crop), which is not yet "
            f"implemented in totalseg_infer.py. See PLAN.md."
        )
    if cfg.licensed:
        print(f"NOTE: '{args.task}' is a commercial TotalSegmentator mode. "
              f"This script skips the license check — make sure you have the weights.")

    weights_dir = Path(args.weights_dir) if args.weights_dir else Path(get_weights_dir())
    device      = torch.device(args.device)

    step_size = _step_size_for(args.task, cfg)
    os.makedirs(args.output_path, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.input_path, "*.nii.gz")))
    if not files:
        raise FileNotFoundError(f"No .nii.gz files found in {args.input_path}")

    print(f"[task={args.task}] models={cfg.task_ids}  resample={cfg.resample}  step={step_size}  modality={cfg.modality}")
    print(f"Loading {len(cfg.task_ids)} sub-model(s) from {weights_dir} ...")
    predictors = build_predictors(cfg, weights_dir, device, step_size)

    for file in tqdm(files, desc=f"Inference ({args.task})"):
        print(f"\n{os.path.basename(file)}")
        infer_file(file, predictors, cfg, args.task, args.output_path)


if __name__ == "__main__":
    main()
