# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import logging

from lib.config import cfg
from pointnet2_lib.pointnet2.pointnet2_modules import (
    PointnetFPModule,
    PointnetSAModuleMSG,
)
from torch.nn.functional import grid_sample
import torch
import torch.nn as nn
import torch.nn.functional as F
from shared.pointnet_backbone import (
    AttentionFusionBase,
    ImageAttentionLayer,
    build_fp_modules,
    build_sa_modules,
)
from shared.pointnet_backbone import forward_pointnet

LOGGER = logging.getLogger(__name__)

BatchNorm2d = nn.BatchNorm2d


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class BasicBlock(nn.Module):
    def __init__(self, inplanes, outplanes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, outplanes, stride)
        self.bn1 = BatchNorm2d(outplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(outplanes, outplanes, 2 * stride)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)

        return out


class FusionConv(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(FusionConv, self).__init__()

        self.conv1 = torch.nn.Conv1d(inplanes, outplanes, 1)
        self.bn1 = torch.nn.BatchNorm1d(outplanes)

    def forward(self, point_features, img_features):
        fusion_features = torch.cat([point_features, img_features], dim=1)
        fusion_features = F.relu(self.bn1(self.conv1(fusion_features)))

        return fusion_features


# ================addition attention (add)=======================#
class IaLayer(ImageAttentionLayer):
    def __init__(self, channels):
        LOGGER.info("%s", "##############ADDITION ATTENTION(ADD)#########")
        super().__init__(channels)


class AttenFusionConv(AttentionFusionBase):
    attention_layer_class = IaLayer

    def __init__(self, inplanes_i, inplanes_p, outplanes):
        super().__init__(inplanes_i, inplanes_p, outplanes)


def feature_gather(feature_map, xy):
    """
    :param xy:(B,N,2)  normalize to [-1,1]
    :param feature_map:(B,C,H,W)
    :return:
    """

    # use grid_sample for this.
    # xy(B,N,2)->(B,1,N,2)
    xy = xy.unsqueeze(1)

    interpolate_feature = grid_sample(feature_map, xy)  # (B,C,1,N)

    return interpolate_feature.squeeze(2)  # (B,C,N)


def get_model(input_channels=6, use_xyz=True):
    return Pointnet2MSG(input_channels=input_channels, use_xyz=use_xyz)


def _initialize_image_fusion(model):
    model.img_block = nn.ModuleList()
    model.fusion_conv = nn.ModuleList()
    model.de_conv = nn.ModuleList()
    fusion = cfg.LI_FUSION
    for index in range(len(fusion.IMG_CHANNELS) - 1):
        model.img_block.append(BasicBlock(
            fusion.IMG_CHANNELS[index], fusion.IMG_CHANNELS[index + 1], stride=1
        ))
        if fusion.ADD_Image_Attention:
            layer = AttenFusionConv(
                fusion.IMG_CHANNELS[index + 1], fusion.POINT_CHANNELS[index],
                fusion.POINT_CHANNELS[index],
            )
        else:
            layer = FusionConv(
                fusion.IMG_CHANNELS[index + 1] + fusion.POINT_CHANNELS[index],
                fusion.POINT_CHANNELS[index],
            )
        model.fusion_conv.append(layer)
        model.de_conv.append(nn.ConvTranspose2d(
            fusion.IMG_CHANNELS[index + 1], fusion.DeConv_Reduce[index],
            kernel_size=fusion.DeConv_Kernels[index],
            stride=fusion.DeConv_Kernels[index],
        ))
    model.image_fusion_conv = nn.Conv2d(
        sum(fusion.DeConv_Reduce), fusion.IMG_FEATURES_CHANNEL // 4, kernel_size=1
    )
    model.image_fusion_bn = torch.nn.BatchNorm2d(fusion.IMG_FEATURES_CHANNEL // 4)
    fusion_layer = AttenFusionConv if fusion.ADD_Image_Attention else FusionConv
    first_channel = fusion.IMG_FEATURES_CHANNEL // 4
    if not fusion.ADD_Image_Attention:
        first_channel += fusion.IMG_FEATURES_CHANNEL
    model.final_fusion_img_point = fusion_layer(
        first_channel, fusion.IMG_FEATURES_CHANNEL, fusion.IMG_FEATURES_CHANNEL
    ) if fusion.ADD_Image_Attention else fusion_layer(
        first_channel, fusion.IMG_FEATURES_CHANNEL
    )


class Pointnet2MSG(nn.Module):
    def __init__(self, input_channels=6, use_xyz=True):
        super().__init__()
        self.sa_modules, skip_channels, channel_out = build_sa_modules(
            cfg, PointnetSAModuleMSG, input_channels, use_xyz
        )
        if cfg.LI_FUSION.ENABLED:
            _initialize_image_fusion(self)
        self.fp_modules = build_fp_modules(
            cfg, PointnetFPModule, skip_channels, channel_out
        )

    def forward(self, pointcloud: torch.cuda.FloatTensor, image=None, xy=None):
        return forward_pointnet(
            self, pointcloud, image, xy, cfg, feature_gather, (1280.0, 384.0)
        )

    def break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()  # mark
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def _break_up_pc2(self, pc):
        xyz = pc[..., 0:3]  # mark
        features = pc[..., 3:].transpose(1, 2) if pc.size(-1) > 3 else None

        return xyz, features


class Pointnet2MsgReturnMiddleStages(Pointnet2MSG):
    def __init__(self, input_channels=6, use_xyz=True):
        super().__init__(input_channels, use_xyz)

    def forward(self, pointcloud: torch.cuda.FloatTensor):
        xyz, features = self.break_up_pc(pointcloud)

        l_xyz, l_features = [xyz], [features]
        idxs = []
        for i, _ in enumerate(self.SA_modules):
            li_xyz, li_features, idx = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)
            idxs.append(idx)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        return l_xyz, l_features, idxs
