# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Construction and inference for compatible RPN heads."""

from typing import NamedTuple

import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class RpnDependencies(NamedTuple):
    config: object
    backbone_class: object
    proposal_class: object
    conv1d: object
    loss_utils: object
    xy_key: str


def _head(dependencies, layer_sizes, input_channels, output_channels):
    config = dependencies.config
    layers = []
    previous = input_channels
    for channels in layer_sizes:
        layers.append(dependencies.conv1d(previous, channels, bn=config.RPN.USE_BN))
        previous = channels
    layers.append(dependencies.conv1d(previous, output_channels, activation=None))
    if config.RPN.DP_RATIO >= 0:
        layers.insert(1, nn.Dropout(config.RPN.DP_RATIO))
    return nn.Sequential(*layers)


def initialize_rpn(module, dependencies, use_xyz, mode):
    config = dependencies.config
    module.training_mode = mode == "TRAIN"
    input_channels = int(config.RPN.USE_INTENSITY) + 3 * int(config.RPN.USE_RGB)
    module.backbone_net = dependencies.backbone_class(
        input_channels=input_channels, use_xyz=use_xyz
    )
    feature_channels = config.RPN.FP_MLPS[0][-1]
    module.rpn_cls_layer = _head(
        dependencies, config.RPN.CLS_FC, feature_channels, 1
    )
    location_bins = int(config.RPN.LOC_SCOPE / config.RPN.LOC_BIN_SIZE) * 2
    location_channels = location_bins * (4 if config.RPN.LOC_XZ_FINE else 2)
    regression_channels = location_channels + config.RPN.NUM_HEAD_BIN * 2 + 4
    module.rpn_reg_layer = _head(
        dependencies, config.RPN.REG_FC, feature_channels, regression_channels
    )
    if config.RPN.LOSS_CLS == "DiceLoss":
        module.rpn_cls_loss_func = dependencies.loss_utils.DiceLoss(ignore_target=-1)
    elif config.RPN.LOSS_CLS == "SigmoidFocalLoss":
        module.rpn_cls_loss_func = dependencies.loss_utils.SigmoidFocalClassificationLoss(
            alpha=config.RPN.FOCAL_ALPHA[0], gamma=config.RPN.FOCAL_GAMMA
        )
    elif config.RPN.LOSS_CLS == "BinaryCrossEntropy":
        module.rpn_cls_loss_func = F.binary_cross_entropy
    else:
        raise NotImplementedError
    module.proposal_layer = dependencies.proposal_class(mode=mode)
    initialize_rpn_weights(module, config)


def initialize_rpn_weights(module, config):
    if config.RPN.LOSS_CLS == "SigmoidFocalLoss":
        prior = 0.01
        bias = -np.log((1 - prior) / prior)
        nn.init.constant_(module.rpn_cls_layer[2].conv.bias, bias)
    nn.init.normal_(module.rpn_reg_layer[-1].conv.weight, mean=0, std=0.001)


def forward_rpn(module, input_data, dependencies):
    points = input_data["pts_input"]
    if dependencies.config.LI_FUSION.ENABLED:
        backbone_output = module.backbone_net(
            points, input_data["img"], input_data[dependencies.xy_key]
        )
    else:
        backbone_output = module.backbone_net(points)
    xyz, features = backbone_output
    classification = module.rpn_cls_layer(features).transpose(1, 2).contiguous()
    regression = module.rpn_reg_layer(features).transpose(1, 2).contiguous()
    return {
        "rpn_cls": classification,
        "rpn_reg": regression,
        "backbone_xyz": xyz,
        "backbone_features": features,
    }
