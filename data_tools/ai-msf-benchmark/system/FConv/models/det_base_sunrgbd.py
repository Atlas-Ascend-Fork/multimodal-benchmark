# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from typing import NamedTuple


from configs.config import cfg
from ops.pybind11.box_ops_cc import rbbox_iou_3d_pair
import torch
from utils.model_helpers import (
    forward_conv_features,
    initialize_conv_feature_weights,
    initialize_detector_forward,
    initialize_pointnet_detector,
    pooled_point_features,
)
import torch.nn as nn

from datasets.dataset_info import DATASET_INFO
from models.box_transform import (
    size_decode,
    size_encode,
    center_decode,
    center_encode,
    angle_decode,
    angle_encode,
)
from models.common import conv1d, de_conv1d
from models.common import softmax_focal_loss_ignore, get_accuracy
from models.det_base import (
    HELPER_CONTINUE,
    PointNetDet as BasePointNetDet,
    PointNetModule,
    collect_detector_results,
    compute_detector_losses,
    compute_detector_metrics,
    prepare_detector_outputs,
)
from shared.parameters import ParameterSchema as _ParameterSchema, bind_parameters as _bind_parameters


class ConvfeatnetForwardParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    x1: object
    x2: object
    x3: object
    x4: object
    x5: object


# single scale PointNet module

_CONVFEATNET_FORWARD_PARAMETERS_SCHEMA = _ParameterSchema(
    ConvfeatnetForwardParameters,
    ("x1", "x2", "x3", "x4", "x5"),
    {},
    frozenset(()),
    5,
    None,
    None,
)


# multi-scale PointNet module


class PointNetFeat(nn.Module):

    def __init__(self, input_channel=3, num_vec=0):
        super(PointNetFeat, self).__init__()

        self.num_vec = num_vec
        u = cfg.DATA.HEIGHT_HALF
        if not (len(u) == 5):
            raise AssertionError
        self.pointnet1 = PointNetModule(
            input_channel - 3, [64, 64, 128], u[0], 128, use_xyz=True, use_feature=False
        )

        self.pointnet2 = PointNetModule(
            input_channel - 3, [64, 64, 128], u[1], 128, use_xyz=True, use_feature=False
        )

        self.pointnet3 = PointNetModule(
            input_channel - 3,
            [128, 128, 256],
            u[2],
            256,
            use_xyz=True,
            use_feature=False,
        )

        self.pointnet4 = PointNetModule(
            input_channel - 3,
            [256, 256, 512],
            u[3],
            256,
            use_xyz=True,
            use_feature=False,
        )

        self.pointnet5 = PointNetModule(
            input_channel - 3,
            [256, 256, 512],
            u[4],
            256,
            use_xyz=True,
            use_feature=False,
        )

    def forward(self, point_cloud, sample_pc, feat=None, one_hot_vec=None):
        return pooled_point_features(self, point_cloud, sample_pc, feat, one_hot_vec)


# FCN


class ConvFeatNet(nn.Module):

    def __init__(self, i_c=128, num_vec=10):
        super(ConvFeatNet, self).__init__()

        self.block1_conv1 = conv1d(i_c + num_vec, 64, 3, 1, 1)

        self.block2_conv1 = conv1d(64, 128, 3, 2, 1)
        self.block2_conv2 = conv1d(128, 128, 3, 1, 1)
        self.block2_merge = conv1d(128 + 128 + num_vec, 128, 1, 1)

        self.block3_conv1 = conv1d(128, 256, 3, 2, 1)
        self.block3_conv2 = conv1d(256, 256, 3, 1, 1)
        self.block3_merge = conv1d(256 + 256 + num_vec, 256, 1, 1)

        self.block4_conv1 = conv1d(256, 512, 3, 2, 1)
        self.block4_conv2 = conv1d(512, 512, 3, 1, 1)
        self.block4_merge = conv1d(512 + 512 + num_vec, 512, 1, 1)

        self.block5_conv1 = conv1d(512, 512, 3, 2, 1)
        self.block5_conv2 = conv1d(512, 512, 3, 1, 1)
        self.block5_merge = conv1d(512 + 512 + num_vec, 512, 1, 1)

        self.block5_deconv = de_conv1d(512, 256, 8, 8, 0)
        self.block4_deconv = de_conv1d(512, 256, 4, 4, 0)
        self.block3_deconv = de_conv1d(256, 256, 2, 2, 0)
        self.block2_deconv = de_conv1d(128, 256, 1, 1, 0)

        initialize_conv_feature_weights(self, nn)

    def forward(self, *call_args, **call_kwargs):
        bound_parameters = _bind_parameters(
            _CONVFEATNET_FORWARD_PARAMETERS_SCHEMA, call_args, call_kwargs
        )
        x1, x2, x3, x4, x5 = bound_parameters
        return forward_conv_features(self, (x1, x2, x3, x4, x5))


# the whole pipeline


def _pointnetdet_forward_410_part_1_chunk_1(_state):
    initialize_detector_forward(_state, 5)


def _pointnetdet_forward_410_part_1_chunk_2(_state):
    return None


def _pointnetdet_forward_410_part_1(_state):
    _pointnetdet_forward_410_part_1_chunk_1(_state)
    _pointnetdet_forward_410_part_1_chunk_2(_state)



class PointNetDet(BasePointNetDet):

    def __init__(self, input_channel=3, num_vec=0, num_classes=2):
        super().__init__(input_channel, num_vec, num_classes)

        initialize_pointnet_detector(
            self, cfg, DATASET_INFO, PointNetFeat, ConvFeatNet,
            input_channel, num_vec, num_classes, 1024, nn,
        )

        nn.init.kaiming_uniform_(self.cls_out.weight, mode="fan_in")
        nn.init.kaiming_uniform_(self.reg_out.weight, mode="fan_in")

        self.cls_out.bias.data.zero_()
        self.reg_out.bias.data.zero_()

    def forward(self, data_dicts):
        _state = locals()
        _pointnetdet_forward_410_part_1(_state)
        _helper_result = prepare_detector_outputs(_state)
        if _helper_result is not HELPER_CONTINUE:
            return _helper_result
        compute_detector_losses(_state)
        compute_detector_metrics(_state)
        collect_detector_results(_state)

        return (_state["losses"], _state["metrics"])
