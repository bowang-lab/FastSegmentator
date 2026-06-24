"""Inference-only stub for the custom `nnUNetTrainerDA5` trainer.

DA5 differs from the base trainer only in (training-time) data augmentation, so
the network it builds is identical to nnUNetTrainer's. This stub lets the
`nnunet` pipeline resolve DA5 checkpoints via the `nnUNet_extTrainer` env var
without touching the vendored nnunetv2 package.
"""
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerDA5(nnUNetTrainer):
    pass
