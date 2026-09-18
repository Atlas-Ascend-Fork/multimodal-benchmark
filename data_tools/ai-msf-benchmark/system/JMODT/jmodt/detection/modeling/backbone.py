# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import torch
import torch.nn as nn
import torch.nn.functional as F

from jmodt.config import cfg
from jmodt.ops.pointnet2.pointnet2_modules import PointnetFPModule, PointnetSAModuleMSG
from shared.pointnet_backbone import (
    AttentionFusionBase,
    ImageAttentionLayer,
    build_fp_modules,
    build_sa_modules,
)
from shared.pointnet_backbone import forward_pointnet


def conv3x3(in_channels, out_channels, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
    )


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels, 2 * stride)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)

        return out


IALayer = ImageAttentionLayer


class AttentionFusion(AttentionFusionBase):
    def __init__(self, img_in_channels, pc_in_channels, out_channels):
        super().__init__(img_in_channels, pc_in_channels, out_channels)


def feature_gather(feature_map, xy):
    """
    :param xy:(B,N,2)  normalize to [-1,1]
    :param feature_map:(B,C,H,W)
    :return:
    """
    xy = xy.unsqueeze(1)  # xy(B,N,2)->(B,1,N,2)
    # use grid_sample for this.
    interpolate_feature = F.grid_sample(
        feature_map.float(), xy, align_corners=True
    )  # (B,C,1,N)

    return interpolate_feature.squeeze(2)  # (B,C,N)


def _pointnet2msg_initialize_part_2(_state):
    if cfg.LI_FUSION.ENABLED:
        _state["self"].img_block = nn.ModuleList()
        _state["self"].fusion_conv = nn.ModuleList()
        _state["self"].de_conv = nn.ModuleList()
        for _state["i"] in range(len(cfg.LI_FUSION.IMG_CHANNELS) - 1):
            _state["self"].img_block.append(
                BasicBlock(
                    cfg.LI_FUSION.IMG_CHANNELS[_state["i"]],
                    cfg.LI_FUSION.IMG_CHANNELS[_state["i"] + 1],
                    stride=1,
                )
            )
            _state["self"].fusion_conv.append(
                AttentionFusion(
                    cfg.LI_FUSION.IMG_CHANNELS[_state["i"] + 1],
                    cfg.LI_FUSION.POINT_CHANNELS[_state["i"]],
                    cfg.LI_FUSION.POINT_CHANNELS[_state["i"]],
                )
            )
            _state["self"].de_conv.append(
                nn.ConvTranspose2d(
                    cfg.LI_FUSION.IMG_CHANNELS[_state["i"] + 1],
                    cfg.LI_FUSION.DeConv_Reduce[_state["i"]],
                    kernel_size=cfg.LI_FUSION.DeConv_Kernels[_state["i"]],
                    stride=cfg.LI_FUSION.DeConv_Kernels[_state["i"]],
                )
            )
        _state["self"].image_fusion_conv = nn.Conv2d(
            sum(cfg.LI_FUSION.DeConv_Reduce),
            cfg.LI_FUSION.IMG_FEATURES_CHANNEL // 4,
            kernel_size=1,
        )
        _state["self"].image_fusion_bn = torch.nn.BatchNorm2d(
            cfg.LI_FUSION.IMG_FEATURES_CHANNEL // 4
        )
        _state["self"].final_fusion_img_point = AttentionFusion(
            cfg.LI_FUSION.IMG_FEATURES_CHANNEL // 4,
            cfg.LI_FUSION.IMG_FEATURES_CHANNEL,
            cfg.LI_FUSION.IMG_FEATURES_CHANNEL,
        )


class PointNet2MSG(nn.Module):
    def __init__(self, input_channels=6, use_xyz=True):
        super().__init__()
        self.sa_modules, skip_channels, channel_out = build_sa_modules(
            cfg, PointnetSAModuleMSG, input_channels, use_xyz
        )
        _state = {"self": self}
        _pointnet2msg_initialize_part_2(_state)
        self.fp_modules = build_fp_modules(
            cfg, PointnetFPModule, skip_channels, channel_out
        )

    @staticmethod
    def break_up_pc(pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, pc, image=None, xy=None):
        return forward_pointnet(self, pc, image, xy, cfg, feature_gather)
