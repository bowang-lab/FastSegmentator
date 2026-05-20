"""Inference-only port of nnUNetCLSTrainerMTL.

Only `build_network_architecture` and the network classes it constructs are
implemented. Training methods (loss, dataloaders, optimizer, validation, etc.)
have been stripped. At inference, the network's classification branch is
ignored — `forward` returns only the segmentation output.
"""
from typing import List, Tuple, Union

import torch.nn as nn

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class FeaturePyramidNetwork(nn.Module):
    def __init__(self, in_channels_list, strides, out_channels, fusion_layers=3):
        super().__init__()
        if fusion_layers == 3:
            self.conv1x1_1 = nn.Conv3d(in_channels_list[0], out_channels, kernel_size=1)
            self.deconv1 = nn.ConvTranspose3d(out_channels, out_channels, kernel_size=strides[0], stride=strides[0])
            self.smooth1 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv1x1_2 = nn.Conv3d(in_channels_list[1], out_channels, kernel_size=1)
        self.conv1x1_3 = nn.Conv3d(in_channels_list[2], out_channels, kernel_size=1)
        self.deconv2 = nn.ConvTranspose3d(out_channels, out_channels, kernel_size=strides[1], stride=strides[1])
        self.smooth2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.fusion_layers = fusion_layers

    def forward(self, x1, x2, x3):
        p3 = self.conv1x1_3(x3)
        p2 = self.conv1x1_2(x2)
        p3_up = self.deconv2(p3)
        p2 = self.relu(self.norm(p2 + p3_up))
        p2 = self.smooth2(p2)
        if self.fusion_layers == 3:
            p1 = self.conv1x1_1(x1)
            p2_up = self.deconv1(p2)
            p1 = self.relu(self.norm(p1 + p2_up))
            p1 = self.smooth1(p1)
            return p1
        return p2


class SegmentationNetworkFusionClassificationHead(nn.Module):
    """Full MTL network kept so the checkpoint state_dict loads exactly.
    At inference, forward() runs only the seg path."""

    def __init__(self, seg_network: nn.Module, features_per_stage: List[int], strides: List[tuple],
                 num_hidden_features: int, num_classes: int, fusion_layers: int = 3):
        super().__init__()
        self.seg_network = seg_network
        self.encoder = self.seg_network.encoder
        self.decoder = self.seg_network.decoder
        self.feature_fusion_block = FeaturePyramidNetwork(
            features_per_stage[-3:], strides[-2:], num_hidden_features, fusion_layers,
        )
        self.conv_block = nn.Sequential(
            nn.Conv3d(num_hidden_features, num_hidden_features * 2, kernel_size=3, padding=1),
            nn.BatchNorm3d(num_hidden_features * 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(num_hidden_features * 2, num_hidden_features * 2, kernel_size=3, padding=1),
            nn.BatchNorm3d(num_hidden_features * 2),
            nn.ReLU(inplace=True),
        )
        self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(num_hidden_features * 2, num_hidden_features),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(num_hidden_features, num_classes),
        )

    def forward(self, x):
        skips = self.seg_network.encoder(x)
        return self.seg_network.decoder(skips)


class nnUNetCLSTrainerMTL(nnUNetTrainer):
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        seg_net = nnUNetTrainer.build_network_architecture(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision,
        )
        # emb_dim and cls_class_num derived as in original initialize():
        #   emb_dim = features_per_stage[-1]; cls_class_num = 1 for binary, else n_classes.
        # Binary (the LUNA25 case) is what's needed here.
        emb_dim = arch_init_kwargs["features_per_stage"][-1]
        cls_class_num = 1
        return SegmentationNetworkFusionClassificationHead(
            seg_net,
            arch_init_kwargs["features_per_stage"],
            arch_init_kwargs["strides"],
            emb_dim,
            cls_class_num,
        )
