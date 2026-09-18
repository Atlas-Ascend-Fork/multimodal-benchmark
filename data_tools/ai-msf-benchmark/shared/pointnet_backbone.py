# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""PointNet++ set-abstraction and feature-propagation builders."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from shared.parameters import bind_named as _bind_named


class ImageAttentionLayer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.ic, self.pc = channels
        reduced_channels = self.pc // 4
        self.conv1 = nn.Sequential(
            nn.Conv1d(self.ic, self.pc, 1), nn.BatchNorm1d(self.pc), nn.ReLU()
        )
        self.fc1 = nn.Linear(self.ic, reduced_channels)
        self.fc2 = nn.Linear(self.pc, reduced_channels)
        self.fc3 = nn.Linear(reduced_channels, 1)

    def forward(self, image_features, point_features):
        batch_size = image_features.size(0)
        image_flat = image_features.transpose(1, 2).contiguous().view(-1, self.ic)
        point_flat = point_features.transpose(1, 2).contiguous().view(-1, self.pc)
        attention = torch.sigmoid(
            self.fc3(torch.tanh(self.fc1(image_flat) + self.fc2(point_flat)))
        )
        attention = attention.squeeze(1).view(batch_size, 1, -1)
        return self.conv1(image_features) * attention


class AttentionFusionBase(nn.Module):
    attention_layer_class = ImageAttentionLayer

    def __init__(self, image_channels, point_channels, output_channels):
        super().__init__()
        self.ia_layer = type(self).attention_layer_class(
            channels=[image_channels, point_channels]
        )
        self.conv1 = nn.Conv1d(point_channels * 2, output_channels, 1)
        self.bn1 = nn.BatchNorm1d(output_channels)

    def forward(self, point_features, image_features):
        image_features = self.ia_layer(image_features, point_features)
        fused = torch.cat([point_features, image_features], dim=1)
        return F.relu(self.bn1(self.conv1(fused)))


def build_sa_modules(config, module_class, input_channels, use_xyz):
    modules = nn.ModuleList()
    skip_channels = [input_channels]
    channel_in = input_channels
    channel_out = input_channels
    for index, npoint in enumerate(config.RPN.SA_CONFIG.NPOINTS):
        mlps = config.RPN.SA_CONFIG.MLPS[index].copy()
        channel_out = 0
        for mlp_index, mlp in enumerate(mlps):
            mlps[mlp_index] = [channel_in] + mlp
            channel_out += mlp[-1]
        modules.append(
            module_class(
                npoint=npoint,
                radii=config.RPN.SA_CONFIG.RADIUS[index],
                nsamples=config.RPN.SA_CONFIG.NSAMPLE[index],
                mlps=mlps,
                use_xyz=use_xyz,
                bn=config.RPN.USE_BN,
            )
        )
        skip_channels.append(channel_out)
        channel_in = channel_out
    return modules, skip_channels, channel_out


def build_fp_modules(config, module_class, skip_channels, channel_out):
    modules = nn.ModuleList()
    for index, mlp in enumerate(config.RPN.FP_MLPS):
        previous = (
            config.RPN.FP_MLPS[index + 1][-1]
            if index + 1 < len(config.RPN.FP_MLPS)
            else channel_out
        )
        modules.append(
            module_class(mlp=[previous + skip_channels[index]] + mlp)
        )
    return modules


def _fuse_final_image(model, image_levels, feature_levels, fusion_inputs):
    xy, config, gather_feature = fusion_inputs
    if not config.LI_FUSION.ENABLED:
        return
    upsampled = [
        model.de_conv[index](image_levels[index + 1])
        for index in range(len(config.LI_FUSION.IMG_CHANNELS) - 1)
    ]
    fused_image = F.relu(
        model.image_fusion_bn(model.image_fusion_conv(torch.cat(upsampled, dim=1)))
    )
    gathered = gather_feature(fused_image, xy)
    feature_levels[0] = model.final_fusion_img_point(feature_levels[0], gathered)


def forward_pointnet(*call_args, **call_kwargs):
    """Run shared abstraction, propagation, and optional image fusion."""
    (
        model,
        point_cloud,
        image,
        xy,
        config,
        gather_feature,
        normalize_dimensions,
    ) = _bind_named(
        call_args,
        call_kwargs,
        ('model', 'point_cloud', 'image', 'xy', 'config', 'gather_feature', 'normalize_dimensions'),
        {'normalize_dimensions': None},
    )
    xyz, features = model.break_up_pc(point_cloud)
    xyz_levels, feature_levels = [xyz], [features]
    if config.LI_FUSION.ENABLED and normalize_dimensions is not None:
        width, height = normalize_dimensions
        xy[:, :, 0] = xy[:, :, 0] / (width - 1.0) * 2.0 - 1.0
        xy[:, :, 1] = xy[:, :, 1] / (height - 1.0) * 2.0 - 1.0
    coordinate_levels, image_levels = [xy], [image]

    for index, module in enumerate(model.sa_modules):
        next_xyz, next_features, selected = module(
            xyz_levels[index], feature_levels[index]
        )
        if config.LI_FUSION.ENABLED:
            selected = selected.long().unsqueeze(-1).repeat(1, 1, 2)
            next_coordinates = torch.gather(coordinate_levels[index], 1, selected)
            next_image = model.img_block[index](image_levels[index])
            image_features = gather_feature(next_image, next_coordinates)
            next_features = model.fusion_conv[index](next_features, image_features)
            coordinate_levels.append(next_coordinates)
            image_levels.append(next_image)
        xyz_levels.append(next_xyz)
        feature_levels.append(next_features)

    for index in range(-1, -(len(model.fp_modules) + 1), -1):
        feature_levels[index - 1] = model.fp_modules[index](
            xyz_levels[index - 1],
            xyz_levels[index],
            feature_levels[index - 1],
            feature_levels[index],
        )
    _fuse_final_image(model, image_levels, feature_levels, (xy, config, gather_feature))
    return xyz_levels[0], feature_levels[0]
