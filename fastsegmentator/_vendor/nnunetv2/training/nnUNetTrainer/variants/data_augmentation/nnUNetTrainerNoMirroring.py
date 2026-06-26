"""Inference-only stub.

Original training-time overrides (mirror-axis configuration, augmentation
pipelines) have been removed — at inference, mirroring is read from the
checkpoint's `inference_allowed_mirroring_axes` field, not from this class.
"""
from fastsegmentator._vendor.nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerNoMirroring(nnUNetTrainer):
    pass


class nnUNetTrainer_onlyMirror01(nnUNetTrainer):
    pass


class nnUNetTrainer_onlyMirror01_1500ep(nnUNetTrainer_onlyMirror01):
    pass


class nnUNetTrainer_onlyMirror01_DASegOrd0(nnUNetTrainer_onlyMirror01):
    pass
