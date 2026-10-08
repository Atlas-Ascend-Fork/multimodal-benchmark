# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Network construction for compatible RCNN heads."""

from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RcnnDependencies(NamedTuple):
    config: object
    proposal_target_class: object
    pointnet_sa_module: object
    pointnet_utils: object
    loss_utils: object
    proposal_uses_mode: bool


class RcnnBase(nn.Module):
    """Shared point encoder and detection heads."""

    dependencies = None

    def __init__(self, num_classes, input_channels=0, use_xyz=True, mode="TRAIN"):
        super().__init__()
        self.mode = mode
        self._initialize_encoder(input_channels, use_xyz)
        channel_in = self._encoder_output_channels(input_channels)
        self._initialize_detection_heads(channel_in, num_classes)
        proposal_class = self.dependencies.proposal_target_class
        self.proposal_target_layer = (
            proposal_class(mode=mode)
            if self.dependencies.proposal_uses_mode
            else proposal_class()
        )

    @staticmethod
    def break_up_pc(point_cloud):
        xyz = point_cloud[..., :3].contiguous()
        features = (
            point_cloud[..., 3:].transpose(1, 2).contiguous()
            if point_cloud.size(-1) > 3
            else None
        )
        return xyz, features

    def init_weights(self, weight_init="xavier"):
        initializers = {
            "kaiming": nn.init.kaiming_normal_,
            "xavier": nn.init.xavier_normal_,
            "normal": nn.init.normal_,
        }
        if weight_init not in initializers:
            raise NotImplementedError
        initializer = initializers[weight_init]
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Conv1d)):
                if weight_init == "normal":
                    initializer(module.weight, mean=0, std=0.001)
                else:
                    initializer(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        nn.init.normal_(self.reg_layer[-1].conv.weight, mean=0, std=0.001)

    def _initialize_encoder(self, input_channels, use_xyz):
        config = self.dependencies.config
        point_utils = self.dependencies.pointnet_utils
        self.sa_modules = nn.ModuleList()
        if config.RCNN.USE_RPN_FEATURES:
            self.rcnn_input_channel = (
                3
                + int(config.RCNN.USE_INTENSITY)
                + int(config.RCNN.USE_MASK)
                + int(config.RCNN.USE_DEPTH)
            )
            self.xyz_up_layer = point_utils.SharedMLP(
                [self.rcnn_input_channel] + config.RCNN.XYZ_UP_LAYER,
                bn=config.RCNN.USE_BN,
            )
            output_channels = config.RCNN.XYZ_UP_LAYER[-1]
            self.merge_down_layer = point_utils.SharedMLP(
                [output_channels * 2, output_channels], bn=config.RCNN.USE_BN
            )

        channel_in = input_channels
        sa_config = config.RCNN.SA_CONFIG
        for index, configured_points in enumerate(sa_config.NPOINTS):
            mlp = [channel_in] + sa_config.MLPS[index]
            npoint = configured_points if configured_points != -1 else None
            self.sa_modules.append(
                self.dependencies.pointnet_sa_module(
                    npoint=npoint,
                    radius=sa_config.RADIUS[index],
                    nsample=sa_config.NSAMPLE[index],
                    mlp=mlp,
                    use_xyz=use_xyz,
                    bn=config.RCNN.USE_BN,
                )
            )
            channel_in = mlp[-1]
        self._sa_output_channels = channel_in

    def _encoder_output_channels(self, input_channels):
        del input_channels
        return self._sa_output_channels

    def _head(self, channel_in, hidden_channels, output_channels, batch_norm):
        point_utils = self.dependencies.pointnet_utils
        layers = []
        previous = channel_in
        for channels in hidden_channels:
            layers.append(point_utils.Conv1d(previous, channels, bn=batch_norm))
            previous = channels
        layers.append(point_utils.Conv1d(previous, output_channels, activation=None))
        return layers

    def _initialize_detection_heads(self, channel_in, num_classes):
        config = self.dependencies.config
        class_channels = 1 if num_classes == 2 else num_classes
        class_layers = self._head(
            channel_in, config.RCNN.CLS_FC, class_channels, config.RCNN.USE_BN
        )
        if config.RCNN.DP_RATIO >= 0:
            class_layers.insert(1, nn.Dropout(config.RCNN.DP_RATIO))
        self.cls_layer = nn.Sequential(*class_layers)
        self._initialize_classification_loss()

        if config.USE_IOU_BRANCH:
            iou_layers = self._head(
                channel_in, config.RCNN.REG_FC[:2], 1, config.RCNN.USE_BN
            )
            if config.RCNN.DP_RATIO >= 0:
                iou_layers.insert(1, nn.Dropout(config.RCNN.DP_RATIO))
            self.iou_branch = nn.Sequential(*iou_layers)

        location_bins = int(config.RCNN.LOC_SCOPE / config.RCNN.LOC_BIN_SIZE) * 2
        y_bins = int(config.RCNN.LOC_Y_SCOPE / config.RCNN.LOC_Y_BIN_SIZE) * 2
        regression_channels = location_bins * 4 + config.RCNN.NUM_HEAD_BIN * 2 + 3
        regression_channels += 1 if not config.RCNN.LOC_Y_BY_BIN else y_bins * 2
        regression_layers = self._head(
            channel_in,
            config.RCNN.REG_FC,
            regression_channels,
            config.RCNN.USE_BN,
        )
        if config.RCNN.DP_RATIO >= 0:
            regression_layers.insert(1, nn.Dropout(config.RCNN.DP_RATIO))
        self.reg_layer = nn.Sequential(*regression_layers)

    def _initialize_classification_loss(self):
        config = self.dependencies.config
        if config.RCNN.LOSS_CLS == "SigmoidFocalLoss":
            self.cls_loss_func = (
                self.dependencies.loss_utils.SigmoidFocalClassificationLoss(
                    alpha=config.RCNN.FOCAL_ALPHA[0], gamma=config.RCNN.FOCAL_GAMMA
                )
            )
        elif config.RCNN.LOSS_CLS == "BinaryCrossEntropy":
            self.cls_loss_func = F.binary_cross_entropy
        elif config.RCNN.LOSS_CLS == "CrossEntropy":
            class_weights = torch.from_numpy(config.RCNN.CLS_WEIGHT).float()
            self.cls_loss_func = nn.CrossEntropyLoss(
                ignore_index=-1, reduce=False, weight=class_weights
            )
        else:
            raise NotImplementedError

    def _encoded_features(self, points):
        config = self.dependencies.config
        xyz, features = self.break_up_pc(points)
        if config.RCNN.USE_RPN_FEATURES:
            xyz_input = points[..., :self.rcnn_input_channel].transpose(1, 2)
            xyz_features = self.xyz_up_layer(xyz_input.contiguous().unsqueeze(3))
            rpn_features = points[..., self.rcnn_input_channel:].transpose(1, 2)
            merged = torch.cat(
                (xyz_features, rpn_features.contiguous().unsqueeze(3)), dim=1
            )
            features = self.merge_down_layer(merged).squeeze(3)
        xyz_levels, feature_levels = [xyz], [features]
        for index, module in enumerate(self.sa_modules):
            next_xyz, next_features, _ = module(
                xyz_levels[index], feature_levels[index]
            )
            xyz_levels.append(next_xyz)
            feature_levels.append(next_features)
        return feature_levels[-1]

    def _detection_outputs(self, features):
        config = self.dependencies.config
        result = {
            "rcnn_cls": self.cls_layer(features).squeeze(-1),
            "rcnn_reg": self.reg_layer(features).squeeze(-1),
        }
        if config.USE_IOU_BRANCH:
            result["rcnn_iou_branch"] = self.iou_branch(features).squeeze(-1)
        return result
