"""Inference-only stub. `_build_loss` overrides removed — loss is training-only."""
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerTopk10Loss(nnUNetTrainer):
    pass


class nnUNetTrainerTopk10LossLS01(nnUNetTrainer):
    pass


class nnUNetTrainerDiceTopK10Loss(nnUNetTrainer):
    pass
