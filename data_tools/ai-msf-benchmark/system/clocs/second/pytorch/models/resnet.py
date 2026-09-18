# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import NamedTuple
from torch import nn
import spconv


from shared.parameters import ParameterSchema as _ParameterSchema, bind_parameters as _bind_parameters


class SparsebasicblockInitParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    inplanes: object
    planes: object
    stride: object
    downsample: object
    indice_key: object


class SparsebottleneckInitParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    inplanes: object
    planes: object
    stride: object
    downsample: object
    indice_key: object


def conv3x3(in_planes, out_planes, stride=1, indice_key=None):
    """3x3 convolution with padding"""
    return spconv.SubMConv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
        indice_key=indice_key,
    )


def conv1x1(in_planes, out_planes, stride=1, indice_key=None):
    """1x1 convolution"""
    return spconv.SubMConv3d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        padding=1,
        bias=False,
        indice_key=indice_key,
    )


_SPARSEBASICBLOCK_INIT_PARAMETERS_SCHEMA = _ParameterSchema(
    SparsebasicblockInitParameters,
    ("inplanes", "planes", "stride", "downsample", "indice_key"),
    {"stride": 1, "downsample": None, "indice_key": None},
    frozenset(()),
    5,
    None,
    None,
)
_SPARSEBOTTLENECK_INIT_PARAMETERS_SCHEMA = _ParameterSchema(
    SparsebottleneckInitParameters,
    ("inplanes", "planes", "stride", "downsample", "indice_key"),
    {"stride": 1, "downsample": None, "indice_key": None},
    frozenset(()),
    5,
    None,
    None,
)


class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, *call_args, **call_kwargs):
        bound_parameters = _bind_parameters(
            _SPARSEBASICBLOCK_INIT_PARAMETERS_SCHEMA, call_args, call_kwargs
        )
        inplanes, planes, stride, downsample, indice_key = bound_parameters
        super(SparseBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, indice_key=indice_key)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes, indice_key=indice_key)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out.features = self.bn1(out.features)
        out.features = self.relu(out.features)

        out = self.conv2(out)
        out.features = self.bn2(out.features)

        if self.downsample is not None:
            identity = self.downsample(x)

        out.features += identity.features
        out.features = self.relu(out.features)

        return out


class SparseBottleneck(spconv.SparseModule):
    expansion = 4

    def __init__(self, *call_args, **call_kwargs):
        bound_parameters = _bind_parameters(
            _SPARSEBOTTLENECK_INIT_PARAMETERS_SCHEMA, call_args, call_kwargs
        )
        inplanes, planes, stride, downsample, indice_key = bound_parameters
        super(SparseBottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes, indice_key=indice_key)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = conv3x3(planes, planes, stride, indice_key=indice_key)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion, indice_key=indice_key)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out.features = self.bn1(out.features)
        out.features = self.relu(out.features)

        out = self.conv2(out)
        out.features = self.bn2(out.features)
        out.features = self.relu(out.features)

        out = self.conv3(out)
        out.features = self.bn3(out.features)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out.features = self.relu(out.features)

        return out
