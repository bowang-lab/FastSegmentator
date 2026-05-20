"""Inference-only minimal nnUNetTrainer.

Only `build_network_architecture` is implemented — the rest of the training
pipeline (dataloaders, optimizer, loss, logger, augmentation) has been stripped.
Used at checkpoint load to construct the network. Cannot train.
"""
from typing import List, Tuple, Union

from torch import nn

from nnunetv2.utilities.get_network_from_plans import get_network_from_plans


class nnUNetTrainer(object):
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        return get_network_from_plans(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            allow_init=True,
            deep_supervision=enable_deep_supervision,
        )
