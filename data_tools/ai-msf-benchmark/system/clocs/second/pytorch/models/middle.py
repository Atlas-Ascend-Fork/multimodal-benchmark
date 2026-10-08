# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import NamedTuple
import logging

from torch import nn
import numpy as np
import spconv
import torch

from second.pytorch.models.resnet import SparseBasicBlock
from torchplus.nn import Empty, Sequential
from torchplus.ops.array_ops import scatter_nd
from torchplus.tools import change_default_args
from second.pytorch.models.sparse_utils import configure_sparse_state


from shared.parameters import ParameterSchema as _ParameterSchema, bind_parameters as _bind_parameters


class MiddleInitParameters(NamedTuple):
    """Arguments shared by all middle-layer constructors."""

    output_shape: object
    use_norm: object
    num_input_features: object
    num_filters_down1: object
    num_filters_down2: object
    name: object


LOGGER = logging.getLogger(__name__)


_MIDDLE_PARAMETER_FIELDS = (
        "output_shape",
        "use_norm",
        "num_input_features",
        "num_filters_down1",
        "num_filters_down2",
        "name",
)


def _middle_schema(name, num_input_features=128):
    return _ParameterSchema(
        MiddleInitParameters,
        _MIDDLE_PARAMETER_FIELDS,
        {
        "use_norm": True,
        "num_input_features": num_input_features,
        "num_filters_down1": None,
        "num_filters_down2": None,
        "name": name,
        },
        frozenset(),
        6,
        None,
        None,
    )


_SPARSEMIDDLEEXTRACTOR_INIT_PARAMETERS_SCHEMA = _middle_schema("SparseMiddleExtractor")
_SPMIDDLED4HD_INIT_PARAMETERS_SCHEMA = _middle_schema("SpMiddleD4HD")
_SPRESNETD4HD_INIT_PARAMETERS_SCHEMA = _middle_schema("SpResNetD4HD")
_SPMIDDLED4HDLITE_INIT_PARAMETERS_SCHEMA = _middle_schema("SpMiddleD4HDLite")
_SPMIDDLED8HD_INIT_PARAMETERS_SCHEMA = _middle_schema("SpMiddleD8HD")
_SPMIDDLEFHD_INIT_PARAMETERS_SCHEMA = _middle_schema("SpMiddleFHD")
_SPMIDDLEVISION_INIT_PARAMETERS_SCHEMA = _middle_schema("SpMiddleVision", 256)
_SPMIDDLEFHDV2_INIT_PARAMETERS_SCHEMA = _middle_schema("SpMiddleFHDV2")
_SPMIDDLE2K_INIT_PARAMETERS_SCHEMA = _middle_schema("SpMiddle2K")
_SPMIDDLEFHDLITE_INIT_PARAMETERS_SCHEMA = _middle_schema("SpMiddleFHDLite")
_SPMIDDLERESNETFHD_INIT_PARAMETERS_SCHEMA = _middle_schema("SpMiddleResNetFHD")
_SPMIDDLEFHDLARGE_INIT_PARAMETERS_SCHEMA = _middle_schema("SpMiddleFHDLarge")
_MIDDLEEXTRACTOR_INIT_PARAMETERS_SCHEMA = _middle_schema("MiddleExtractor", 260)


def _bind_middle_parameters(schema, call_args, call_kwargs):
    parameters = _bind_parameters(schema, call_args, call_kwargs)
    return parameters._replace(
        num_filters_down1=(
            [64]
            if parameters.num_filters_down1 is None
            else parameters.num_filters_down1
        ),
        num_filters_down2=(
            [64, 64]
            if parameters.num_filters_down2 is None
            else parameters.num_filters_down2
        ),
    )


def _sparse_middle_stage(filters, batch_norm1d, indice_key):
    layers = []
    for input_filters, output_filters in zip(filters, filters[1:]):
        layers.extend([
            spconv.SubMConv3d(
                input_filters, output_filters, 3, bias=False, indice_key=indice_key
            ),
            batch_norm1d(output_filters), nn.ReLU(),
        ])
    layers.extend([
        spconv.SparseConv3d(
            filters[-1], filters[-1], (3, 1, 1), (2, 1, 1), bias=False
        ),
        batch_norm1d(filters[-1]), nn.ReLU(),
    ])
    return layers


class _SparseMiddleBase(nn.Module):
    flatten_output = True

    def forward(self, voxel_features, coors, batch_size):
        tensor_args = [voxel_features, coors.int(), self.sparse_shape, batch_size]
        if hasattr(self, "grid"):
            tensor_args.append(self.grid)
        output = self.middle_conv(spconv.SparseConvTensor(*tensor_args)).dense()
        if not self.flatten_output:
            return output
        batch, channels, depth, height, width = output.shape
        return output.view(batch, channels * depth, height, width)


class SparseMiddleExtractor(_SparseMiddleBase):

    def __init__(self, *call_args, **call_kwargs):
        parameters = _bind_middle_parameters(
            _SPARSEMIDDLEEXTRACTOR_INIT_PARAMETERS_SCHEMA, call_args, call_kwargs
        )
        super(SparseMiddleExtractor, self).__init__()
        self.name = parameters.name
        if parameters.use_norm:
            batch_norm1d = change_default_args(eps=1e-3, momentum=0.01)(nn.BatchNorm1d)
        else:
            batch_norm1d = Empty
        sparse_shape = np.array(parameters.output_shape[1:4]) + [1, 0, 0]
        LOGGER.info("%s", sparse_shape)
        self.sparse_shape = sparse_shape
        self.voxel_output_shape = parameters.output_shape
        first_filters = [parameters.num_input_features] + parameters.num_filters_down1
        if len(parameters.num_filters_down1) == 0:
            second_filters = [first_filters[-1]] + parameters.num_filters_down2
        else:
            second_filters = [parameters.num_filters_down1[-1]] + parameters.num_filters_down2
        middle_layers = _sparse_middle_stage(first_filters, batch_norm1d, "subm0")
        middle_layers += _sparse_middle_stage(second_filters, batch_norm1d, "subm1")
        self.middle_conv = spconv.SparseSequential(*middle_layers)


def _configure_sparse_state(_state):
    configure_sparse_state(
        _state, change_default_args, nn, spconv, Empty, LOGGER
    )


def _legacy_middle_state(module, parameters):
    state = {
        "self": module,
        "name": parameters.name,
        "use_norm": parameters.use_norm,
        "output_shape": parameters.output_shape,
        "num_input_features": parameters.num_input_features,
    }
    _configure_sparse_state(state)
    return state


class SpMiddleD4HD(_SparseMiddleBase):

    def __init__(self, *call_args, **call_kwargs):
        parameters = _bind_parameters(
            _SPMIDDLED4HD_INIT_PARAMETERS_SCHEMA, call_args, call_kwargs
        )
        super(SpMiddleD4HD, self).__init__()
        _configure_middle(self, parameters, "d4")


def _spresnetd4hd_init_682_part_2(_state):
    _state["self"].middle_conv = spconv.SparseSequential(
        _state["sub_m_conv3d"](_state["num_input_features"], 32, 3, indice_key="res0"),
        _state["batch_norm1d"](32),
        nn.ReLU(),
        SparseBasicBlock(32, 32, indice_key="res0"),
        SparseBasicBlock(32, 32, indice_key="res0"),
        _state["sp_conv3d"](32, 64, 3, 2, padding=1),
        _state["batch_norm1d"](64),
        nn.ReLU(),
        SparseBasicBlock(64, 64, indice_key="res1"),
        SparseBasicBlock(64, 64, indice_key="res1"),
        _state["sp_conv3d"](64, 64, 3, 2, padding=[0, 1, 1]),
        _state["batch_norm1d"](64),
        nn.ReLU(),
        SparseBasicBlock(64, 64, indice_key="res2"),
        SparseBasicBlock(64, 64, indice_key="res2"),
        _state["sp_conv3d"](64, 64, (3, 1, 1), (2, 1, 1)),
        _state["batch_norm1d"](64),
        nn.ReLU(),
    )


class SpResNetD4HD(_SparseMiddleBase):

    def __init__(self, *call_args, **call_kwargs):
        parameters = _bind_parameters(
            _SPRESNETD4HD_INIT_PARAMETERS_SCHEMA, call_args, call_kwargs
        )
        super(SpResNetD4HD, self).__init__()
        _spresnetd4hd_init_682_part_2(_legacy_middle_state(self, parameters))


class SpMiddleD4HDLite(_SparseMiddleBase):

    def __init__(self, *call_args, **call_kwargs):
        parameters = _bind_parameters(
            _SPMIDDLED4HDLITE_INIT_PARAMETERS_SCHEMA, call_args, call_kwargs
        )
        super(SpMiddleD4HDLite, self).__init__()
        _configure_middle(self, parameters, "d4lite")

_MIDDLE_LAYER_SPECS = {
    "d4": (
        ("sub", 32, 32, "subm0"), ("sub", 32, 32, "subm0"),
        ("sp", 32, 64, 3, 2, 1),
        ("sub", 64, 64, "subm1"), ("sub", 64, 64, "subm1"),
        ("sub", 64, 64, "subm1"), ("sp", 64, 64, 3, 2, [0, 1, 1]),
        ("sub", 64, 64, "subm2"), ("sub", 64, 64, "subm2"),
        ("sub", 64, 64, "subm2"),
        ("sp", 64, 64, (3, 1, 1), (2, 1, 1), 0),
    ),
    "d4lite": (
        ("sub", 16, 16, "subm0"), ("sub", 16, 16, "subm0"),
        ("sp", 16, 32, 3, 2, 1),
        ("sub", 32, 32, "subm1"), ("sub", 32, 32, "subm1"),
        ("sub", 32, 32, "subm1"), ("sp", 32, 64, 3, 2, [0, 1, 1]),
        ("sub", 64, 64, "subm2"), ("sub", 64, 64, "subm2"),
        ("sub", 64, 64, "subm2"),
        ("sp", 64, 64, (3, 1, 1), (2, 1, 1), 0),
    ),
    "d8": (
        ("sub", 16, 16, "subm0"), ("sub", 16, 16, "subm0"),
        ("sp", 16, 32, 3, 2, 1),
        ("sub", 32, 32, "subm1"), ("sub", 32, 32, "subm1"),
        ("sub", 32, 32, "subm1"), ("sp", 32, 64, 3, 2, 1),
        ("sub", 64, 64, "subm2"), ("sub", 64, 64, "subm2"),
        ("sub", 64, 64, "subm2"), ("sp", 64, 64, 3, 2, [0, 1, 1]),
        ("sub", 64, 64, "subm3"), ("sub", 64, 64, "subm3"),
        ("sub", 64, 64, "subm3"), ("sp", 64, 64, (3, 1, 1), (2, 1, 1), 0),
    ),
    "fhd": (
        ("sub", 16, 16, "subm0"), ("sub", 16, 16, "subm0"),
        ("sp", 16, 32, 3, 2, 1), ("sub", 32, 32, "subm1"),
        ("sub", 32, 32, "subm1"), ("sp", 32, 64, 3, 2, 1),
        ("sub", 64, 64, "subm2"), ("sub", 64, 64, "subm2"),
        ("sub", 64, 64, "subm2"), ("sp", 64, 64, 3, 2, [0, 1, 1]),
        ("sub", 64, 64, "subm3"), ("sub", 64, 64, "subm3"),
        ("sub", 64, 64, "subm3"), ("sp", 64, 64, (3, 1, 1), (2, 1, 1), 0),
    ),
    "2k": (
        ("sub", 8, 8, "subm0"), ("sub", 8, 8, "subm0"),
        ("sp", 8, 16, 3, 2, 1), ("sub", 16, 16, "subm1"),
        ("sub", 16, 16, "subm1"), ("sp", 16, 32, 3, 2, 1),
        ("sub", 32, 32, "subm2"), ("sub", 32, 32, "subm2"),
        ("sp", 32, 64, 3, 2, 1), ("sub", 64, 64, "subm3"),
        ("sub", 64, 64, "subm3"), ("sub", 64, 64, "subm3"),
        ("sp", 64, 64, 3, 2, [0, 1, 1]), ("sub", 64, 64, "subm4"),
        ("sub", 64, 64, "subm4"), ("sub", 64, 64, "subm4"),
        ("sp", 64, 64, (3, 1, 1), (2, 1, 1), 0),
    ),
    "large": (
        ("sub", 16, 16, "subm0"), ("sub", 16, 16, "subm0"),
        ("sp", 16, 32, 3, 2, 1),
        ("sub", 32, 32, "subm1"), ("sub", 32, 32, "subm1"),
        ("sp", 32, 64, 3, 2, 1),
        ("sub", 64, 64, "subm2"), ("sub", 64, 64, "subm2"),
        ("sub", 64, 64, "subm2"),
        ("sp", 64, 128, 3, 2, [0, 1, 1], 64),
        ("sub", 128, 128, "subm3"), ("sub", 128, 128, "subm3"),
        ("sub", 128, 128, "subm3"),
        ("sp", 128, 128, (3, 1, 1), (2, 1, 1), 0),
    ),
}


def _build_middle_layers(num_input_features, use_norm, variant, add_pool=False):
    bias = not use_norm
    batch_norm = (
        change_default_args(eps=1e-3, momentum=0.01)(nn.BatchNorm1d)
        if use_norm else Empty
    )
    sparse_conv = change_default_args(bias=bias)(spconv.SparseConv3d)
    submanifold_conv = change_default_args(bias=bias)(spconv.SubMConv3d)
    layers = []
    for index, spec in enumerate(_MIDDLE_LAYER_SPECS[variant]):
        kind, _, output_channels = spec[:3]
        input_channels = num_input_features if index == 0 else spec[1]
        if kind == "sub":
            layer = submanifold_conv(
                input_channels, output_channels, 3, indice_key=spec[3]
            )
        else:
            layer = sparse_conv(
                input_channels, output_channels, spec[3], spec[4], padding=spec[5]
            )
        norm_channels = spec[6] if len(spec) > 6 else output_channels
        layers.extend((layer, batch_norm(norm_channels), nn.ReLU()))
    if add_pool:
        layers.append(spconv.SparseMaxPool3d([2, 1, 1]))
    return layers


def _configure_middle(module, parameters, variant, max_batch_size=None, add_pool=False):
    module.name = parameters.name
    sparse_shape = np.array(parameters.output_shape[1:4]) + [1, 0, 0]
    LOGGER.info("%s", sparse_shape)
    module.sparse_shape = sparse_shape
    module.voxel_output_shape = parameters.output_shape
    layers = _build_middle_layers(
        parameters.num_input_features, parameters.use_norm, variant, add_pool
    )
    module.middle_conv = spconv.SparseSequential(*layers)
    if max_batch_size is not None:
        module.max_batch_size = max_batch_size
        module.grid = torch.full(
            [max_batch_size, *sparse_shape], -1, dtype=torch.int32
        ).cuda()


class SpMiddleD8HD(_SparseMiddleBase):

    def __init__(self, *call_args, **call_kwargs):
        parameters = _bind_parameters(
            _SPMIDDLED8HD_INIT_PARAMETERS_SCHEMA, call_args, call_kwargs
        )
        super(SpMiddleD8HD, self).__init__()
        _configure_middle(self, parameters, "d8")


class SpMiddleFHD(_SparseMiddleBase):

    def __init__(self, *call_args, **call_kwargs):
        parameters = _bind_parameters(
            _SPMIDDLEFHD_INIT_PARAMETERS_SCHEMA, call_args, call_kwargs
        )
        super(SpMiddleFHD, self).__init__()
        _configure_middle(self, parameters, "fhd")
        self.max_batch_size = 6


def _spmiddlevision_init_1088_part_2(_state):
    _state["self"].middle_conv = spconv.SparseSequential(
        _state["sp_conv3d"](_state["num_input_features"], 256, 3, 2, padding=1),
        _state["batch_norm1d"](256),
        nn.ReLU(),
        _state["sp_conv3d"](256, 256, 3, 2, padding=1),
        _state["batch_norm1d"](256),
        nn.ReLU(),
        _state["sp_conv3d"](256, 256, 3, 2, padding=[0, 1, 1]),
        _state["batch_norm1d"](256),
        nn.ReLU(),
    )
    _state["self"].max_batch_size = 6


class SpMiddleVision(_SparseMiddleBase):
    flatten_output = False

    def __init__(self, *call_args, **call_kwargs):
        parameters = _bind_parameters(
            _SPMIDDLEVISION_INIT_PARAMETERS_SCHEMA, call_args, call_kwargs
        )
        super(SpMiddleVision, self).__init__()
        _spmiddlevision_init_1088_part_2(_legacy_middle_state(self, parameters))


class SpMiddleFHDV2(_SparseMiddleBase):

    def __init__(self, *call_args, **call_kwargs):
        parameters = _bind_parameters(
            _SPMIDDLEFHDV2_INIT_PARAMETERS_SCHEMA, call_args, call_kwargs
        )
        super(SpMiddleFHDV2, self).__init__()
        _configure_middle(self, parameters, "fhd", max_batch_size=6, add_pool=True)


class SpMiddle2K(_SparseMiddleBase):

    def __init__(self, *call_args, **call_kwargs):
        parameters = _bind_parameters(
            _SPMIDDLE2K_INIT_PARAMETERS_SCHEMA, call_args, call_kwargs
        )
        super(SpMiddle2K, self).__init__()
        _configure_middle(self, parameters, "2k", max_batch_size=3)


def _spmiddlefhdlite_init_1376_part_2(_state):
    _state["self"].middle_conv = spconv.SparseSequential(
        _state["sp_conv3d"](_state["num_input_features"], 16, 3, 2, padding=1),
        _state["batch_norm1d"](16),
        nn.ReLU(),
        _state["sp_conv3d"](16, 32, 3, 2, padding=1),
        _state["batch_norm1d"](32),
        nn.ReLU(),
        _state["sp_conv3d"](32, 64, 3, 2, padding=[0, 1, 1]),
        _state["batch_norm1d"](64),
        nn.ReLU(),
        _state["sp_conv3d"](64, 64, (3, 1, 1), (2, 1, 1)),
        _state["batch_norm1d"](64),
        nn.ReLU(),
    )


class SpMiddleFHDLite(_SparseMiddleBase):

    def __init__(self, *call_args, **call_kwargs):
        parameters = _bind_parameters(
            _SPMIDDLEFHDLITE_INIT_PARAMETERS_SCHEMA, call_args, call_kwargs
        )
        super(SpMiddleFHDLite, self).__init__()
        _spmiddlefhdlite_init_1376_part_2(_legacy_middle_state(self, parameters))


def _spmiddleresnetfhd_initialize_part_2(_state):
    _state["self"].middle_conv = spconv.SparseSequential(
        _state["sub_m_conv3d"](_state["num_input_features"], 16, 3, indice_key="res0"),
        _state["batch_norm1d"](16),
        nn.ReLU(),
        SparseBasicBlock(16, 16, indice_key="res0"),
        SparseBasicBlock(16, 16, indice_key="res0"),
        _state["sp_conv3d"](16, 32, 3, 2, padding=1),
        _state["batch_norm1d"](32),
        nn.ReLU(),
        SparseBasicBlock(32, 32, indice_key="res1"),
        SparseBasicBlock(32, 32, indice_key="res1"),
        _state["sp_conv3d"](32, 64, 3, 2, padding=1),
        _state["batch_norm1d"](64),
        nn.ReLU(),
        SparseBasicBlock(64, 64, indice_key="res2"),
        SparseBasicBlock(64, 64, indice_key="res2"),
        _state["sp_conv3d"](64, 128, 3, 2, padding=[0, 1, 1]),
        _state["batch_norm1d"](128),
        nn.ReLU(),
        SparseBasicBlock(128, 128, indice_key="res3"),
        SparseBasicBlock(128, 128, indice_key="res3"),
        _state["sp_conv3d"](128, 128, (3, 1, 1), (2, 1, 1)),
        _state["batch_norm1d"](128),
        nn.ReLU(),
    )


class SpMiddleResNetFHD(_SparseMiddleBase):

    def __init__(self, *call_args, **call_kwargs):
        parameters = _bind_parameters(
            _SPMIDDLERESNETFHD_INIT_PARAMETERS_SCHEMA, call_args, call_kwargs
        )
        super(SpMiddleResNetFHD, self).__init__()
        _spmiddleresnetfhd_initialize_part_2(
            _legacy_middle_state(self, parameters)
        )


class SpMiddleFHDLarge(_SparseMiddleBase):

    def __init__(self, *call_args, **call_kwargs):
        parameters = _bind_parameters(
            _SPMIDDLEFHDLARGE_INIT_PARAMETERS_SCHEMA, call_args, call_kwargs
        )
        super(SpMiddleFHDLarge, self).__init__()
        _configure_middle(self, parameters, "large")


class MiddleExtractor(nn.Module):

    def __init__(self, *call_args, **call_kwargs):
        parameters = _bind_middle_parameters(
            _MIDDLEEXTRACTOR_INIT_PARAMETERS_SCHEMA, call_args, call_kwargs
        )
        super(MiddleExtractor, self).__init__()
        self.name = parameters.name
        if parameters.use_norm:
            batch_norm3d = change_default_args(eps=1e-3, momentum=0.01)(nn.BatchNorm3d)
            conv3d = change_default_args(bias=False)(nn.Conv3d)
        else:
            batch_norm3d = Empty
            conv3d = change_default_args(bias=True)(nn.Conv3d)
        self.voxel_output_shape = parameters.output_shape
        self.middle_conv = Sequential(
            ZeroPad3d(1),
            conv3d(parameters.num_input_features, 64, 3, stride=(2, 1, 1)),
            batch_norm3d(64),
            nn.ReLU(),
            ZeroPad3d([1, 1, 1, 1, 0, 0]),
            conv3d(64, 64, 3, stride=1),
            batch_norm3d(64),
            nn.ReLU(),
            ZeroPad3d(1),
            conv3d(64, 64, 3, stride=2),
            batch_norm3d(64),
            nn.ReLU(),
        )

    def forward(self, voxel_features, coors, batch_size):
        output_shape = [batch_size] + self.voxel_output_shape[1:]
        output_shape[4] = 320
        ret = scatter_nd(coors.long(), voxel_features, output_shape)
        ret = ret.permute(0, 4, 1, 2, 3)
        ret = self.middle_conv(ret)
        n, c, d, h, w = ret.shape
        ret = ret.view(n, c * d, h, w)

        return ret


class ZeroPad3d(nn.ConstantPad3d):

    def __init__(self, padding):
        super(ZeroPad3d, self).__init__(padding, 0)
