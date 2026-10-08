# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import NamedTuple

from torch import nn
from torch.nn import functional as F
import torch

from torchplus.nn import Empty
from torchplus.tools import change_default_args


from shared.parameters import ParameterSchema as _ParameterSchema, bind_parameters as _bind_parameters


class VoxelfeatureextractorInitParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    num_input_features: object
    use_norm: object
    num_filters: object
    with_distance: object
    name: object


Voxelfeatureextractorv2InitParameters = VoxelfeatureextractorInitParameters
Voxelfeatureextractorv3InitParameters = VoxelfeatureextractorInitParameters
SimplevoxelInitParameters = VoxelfeatureextractorInitParameters


def get_paddings_indicator(actual_num, max_num, axis=0):
    """Create boolean mask by actually number of a padded tensor.

    Args:
        actual_num ([type]): [description]
        max_num ([type]): [description]

    Returns:
        [type]: [description]
    """

    actual_num = torch.unsqueeze(actual_num, axis + 1)
    max_num_shape = [1] * len(actual_num.shape)
    max_num_shape[axis + 1] = -1
    max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(
        max_num_shape
    )
    paddings_indicator = actual_num.int() > max_num
    # paddings_indicator shape: [batch_size, max_num]
    return paddings_indicator


class VFELayer(nn.Module):
    def __init__(self, in_channels, out_channels, use_norm=True, name="vfe"):
        super(VFELayer, self).__init__()
        self.name = name
        self.units = int(out_channels / 2)
        if use_norm:
            batch_norm1d = change_default_args(eps=1e-3, momentum=0.01)(nn.BatchNorm1d)
            linear = change_default_args(bias=False)(nn.Linear)
        else:
            batch_norm1d = Empty
            linear = change_default_args(bias=True)(nn.Linear)
        self.linear = linear(in_channels, self.units)
        self.norm = batch_norm1d(self.units)

    def forward(self, inputs):
        voxel_count = inputs.shape[1]
        x = self.linear(inputs)
        x = self.norm(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        pointwise = F.relu(x)

        aggregated = torch.max(pointwise, dim=1, keepdim=True)[0]
        repeated = aggregated.repeat(1, voxel_count, 1)

        concatenated = torch.cat([pointwise, repeated], dim=2)
        return concatenated


_VOXEL_PARAMETER_NAMES = (
    "num_input_features", "use_norm", "num_filters", "with_distance", "name"
)


def _voxel_schema(default_name):
    return _ParameterSchema(
        VoxelfeatureextractorInitParameters,
        _VOXEL_PARAMETER_NAMES,
        {
        "num_input_features": 4,
        "use_norm": True,
        "num_filters": None,
        "with_distance": False,
        "name": default_name,
        },
        frozenset(()),
        5,
        None,
        None,
    )


_VOXELFEATUREEXTRACTOR_INIT_PARAMETERS_SCHEMA = _voxel_schema("VoxelFeatureExtractor")
_VOXELFEATUREEXTRACTORV2_INIT_PARAMETERS_SCHEMA = _VOXELFEATUREEXTRACTOR_INIT_PARAMETERS_SCHEMA
_VOXELFEATUREEXTRACTORV3_INIT_PARAMETERS_SCHEMA = _VOXELFEATUREEXTRACTOR_INIT_PARAMETERS_SCHEMA
_SIMPLEVOXEL_INIT_PARAMETERS_SCHEMA = _voxel_schema("SimpleVoxel")


def _prepare_vfe_features(module, features, num_voxels):
    points_mean = features[:, :, :3].sum(dim=1, keepdim=True) / num_voxels.type_as(
        features
    ).view(-1, 1, 1)
    features_relative = features[:, :, :3] - points_mean
    if module.with_distance:
        points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
        features = torch.cat([features, features_relative, points_dist], dim=-1)
    else:
        features = torch.cat([features, features_relative], dim=-1)
    mask = get_paddings_indicator(num_voxels, features.shape[1], axis=0)
    return features, torch.unsqueeze(mask, -1).type_as(features)


def _finish_vfe_features(module, features, mask):
    features = module.linear(features)
    features = module.norm(features.permute(0, 2, 1).contiguous())
    features = F.relu(features.permute(0, 2, 1).contiguous())
    features *= mask
    return torch.max(features, dim=1)[0]


class VoxelFeatureExtractor(nn.Module):
    def __init__(self, *call_args, **call_kwargs):
        bound_parameters = _bind_parameters(
            _VOXELFEATUREEXTRACTOR_INIT_PARAMETERS_SCHEMA, call_args, call_kwargs
        )
        num_input_features, use_norm, num_filters, with_distance, name = (
            bound_parameters
        )
        if num_filters is None:
            num_filters = [32, 128]
        super(VoxelFeatureExtractor, self).__init__()
        self.name = name
        if use_norm:
            batch_norm1d = change_default_args(eps=1e-3, momentum=0.01)(nn.BatchNorm1d)
            linear = change_default_args(bias=False)(nn.Linear)
        else:
            batch_norm1d = Empty
            linear = change_default_args(bias=True)(nn.Linear)
        if not (len(num_filters) == 2):
            raise AssertionError
        num_input_features += 3  # add mean features
        if with_distance:
            num_input_features += 1
        self.with_distance = with_distance
        self.vfe1 = VFELayer(num_input_features, num_filters[0], use_norm)
        self.vfe2 = VFELayer(num_filters[0], num_filters[1], use_norm)
        self.linear = linear(num_filters[1], num_filters[1])
        self.norm = batch_norm1d(num_filters[1])

    def forward(self, features, num_voxels):
        features, mask = _prepare_vfe_features(self, features, num_voxels)

        x = self.vfe1(features)
        x *= mask
        x = self.vfe2(x)
        x *= mask
        return _finish_vfe_features(self, x, mask)


class VoxelFeatureExtractorV2(nn.Module):
    def __init__(self, *call_args, **call_kwargs):
        bound_parameters = _bind_parameters(
            _VOXELFEATUREEXTRACTORV2_INIT_PARAMETERS_SCHEMA, call_args, call_kwargs
        )
        num_input_features, use_norm, num_filters, with_distance, name = (
            bound_parameters
        )
        if num_filters is None:
            num_filters = [32, 128]
        super(VoxelFeatureExtractorV2, self).__init__()
        self.name = name
        if use_norm:
            batch_norm1d = change_default_args(eps=1e-3, momentum=0.01)(nn.BatchNorm1d)
            linear = change_default_args(bias=False)(nn.Linear)
        else:
            batch_norm1d = Empty
            linear = change_default_args(bias=True)(nn.Linear)
        if not (len(num_filters) > 0):
            raise AssertionError
        num_input_features += 3
        if with_distance:
            num_input_features += 1
        self.with_distance = with_distance

        num_filters = [num_input_features] + num_filters
        filters_pairs = [
            [num_filters[i], num_filters[i + 1]] for i in range(len(num_filters) - 1)
        ]
        self.vfe_layers = nn.ModuleList(
            [VFELayer(i, o, use_norm) for i, o in filters_pairs]
        )
        self.linear = linear(num_filters[-1], num_filters[-1])
        self.norm = batch_norm1d(num_filters[-1])

    def forward(self, features, num_voxels):
        features, mask = _prepare_vfe_features(self, features, num_voxels)
        for vfe in self.vfe_layers:
            features = vfe(features)
            features *= mask
        return _finish_vfe_features(self, features, mask)


class _MeanVoxelBase(nn.Module):
    schema = None

    def __init__(self, *call_args, **call_kwargs):
        bound_parameters = _bind_parameters(
            type(self).schema, call_args, call_kwargs
        )
        num_input_features, use_norm, num_filters, with_distance, name = (
            bound_parameters
        )
        if num_filters is None:
            num_filters = [32, 128]
        super().__init__()
        self.name = name

    def forward(self, features, num_voxels):
        points_mean = features[:, :, :4].sum(dim=1, keepdim=False) / num_voxels.type_as(
            features
        ).view(-1, 1)
        return points_mean.contiguous()


class VoxelFeatureExtractorV3(_MeanVoxelBase):
    schema = _VOXELFEATUREEXTRACTORV3_INIT_PARAMETERS_SCHEMA


class SimpleVoxel(_MeanVoxelBase):
    """Simple voxel encoder. only keep r, z and reflection feature."""
    schema = _SIMPLEVOXEL_INIT_PARAMETERS_SCHEMA
