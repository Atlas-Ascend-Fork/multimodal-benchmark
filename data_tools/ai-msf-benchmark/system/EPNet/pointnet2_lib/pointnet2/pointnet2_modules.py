# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import NamedTuple
from typing import List

from pointnet2 import pointnet2_utils
from pointnet2 import pytorch_utils as pt_utils
import torch
import torch.nn as nn
import torch.nn.functional as F


from shared.parameters import ParameterSchema as _ParameterSchema, bind_parameters as _bind_parameters


class PointnetsamodulemsgInitParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    npoint: object
    radii: object
    nsamples: object
    mlps: object
    bn: object
    use_xyz: object
    pool_method: object
    instance_norm: object


class PointnetsamoduleInitParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    mlp: object
    npoint: object
    radius: object
    nsample: object
    bn: object
    use_xyz: object
    pool_method: object
    instance_norm: object


class _PointnetSAModuleBase(nn.Module):

    def __init__(self):
        super().__init__()
        self.npoint = None
        self.groupers = None
        self.mlps = None
        self.pool_method = "max_pool"

    def forward(
        self, xyz: torch.Tensor, features: torch.Tensor = None, new_xyz=None
    ) -> (torch.Tensor, torch.Tensor):
        r"""
        :param xyz: (B, N, 3) tensor of the xyz coordinates of the features
        :param features: (B, N, C) tensor of the descriptors of the the features
        :param new_xyz:
        :return:
            new_xyz: (B, npoint, 3) tensor of the new features' xyz
            new_features: (B, npoint, \sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        new_features_list = []

        xyz_flipped = xyz.transpose(1, 2).contiguous()
        # xyz_flipped,
        # ).transpose(1, 2).contiguous() if self.npoint is not None else None

        #

        if new_xyz is None:
            if self.npoint is not None:
                idx = pointnet2_utils.furthest_point_sample(xyz, self.npoint)
                new_xyz = (
                    pointnet2_utils.gather_operation(xyz_flipped, idx)
                    .transpose(1, 2)
                    .contiguous()
                )
            else:
                new_xyz = None
                idx = None
        else:
            idx = None

        for i, _ in enumerate(self.groupers):
            new_features = self.groupers[i](
                xyz, new_xyz, features
            )  # (B, C, npoint, nsample)

            new_features = self.mlps[i](new_features)  # (B, mlp[-1], npoint, nsample)
            if self.pool_method == "max_pool":
                new_features = F.max_pool2d(
                    new_features, kernel_size=[1, new_features.size(3)]
                )  # (B, mlp[-1], npoint, 1)
            elif self.pool_method == "avg_pool":
                new_features = F.avg_pool2d(
                    new_features, kernel_size=[1, new_features.size(3)]
                )  # (B, mlp[-1], npoint, 1)
            else:
                raise NotImplementedError

            new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)
            new_features_list.append(new_features)

        return new_xyz, torch.cat(new_features_list, dim=1), idx


_POINTNETSAMODULEMSG_INIT_PARAMETERS_SCHEMA = _ParameterSchema(
    PointnetsamodulemsgInitParameters,
    (
        "npoint",
        "radii",
        "nsamples",
        "mlps",
        "bn",
        "use_xyz",
        "pool_method",
        "instance_norm",
    ),
    {"bn": True, "use_xyz": True, "pool_method": "max_pool", "instance_norm": False},
    frozenset(()),
    0,
    None,
    None,
)
_POINTNETSAMODULE_INIT_PARAMETERS_SCHEMA = _ParameterSchema(
    PointnetsamoduleInitParameters,
    (
        "mlp",
        "npoint",
        "radius",
        "nsample",
        "bn",
        "use_xyz",
        "pool_method",
        "instance_norm",
    ),
    {
        "npoint": None,
        "radius": None,
        "nsample": None,
        "bn": True,
        "use_xyz": True,
        "pool_method": "max_pool",
        "instance_norm": False,
    },
    frozenset(()),
    0,
    None,
    None,
)


class PointnetSAModuleMSG(_PointnetSAModuleBase):
    """Pointnet set abstraction layer with multiscale grouping"""

    def __init__(self, *call_args, **call_kwargs):
        """
        :param npoint: int
        :param radii: list of float, list of radii to group with
        :param nsamples: list of int, number of samples in each ball query
        :param mlps: list of list of int, spec of the pointnet before the global pooling for each scale
        :param bn: whether to use batchnorm
        :param use_xyz:
        :param pool_method: max_pool / avg_pool
        :param instance_norm: whether to use instance_norm
        """
        bound_parameters = _bind_parameters(
            _POINTNETSAMODULEMSG_INIT_PARAMETERS_SCHEMA, call_args, call_kwargs
        )
        npoint, radii, nsamples, mlps, bn, use_xyz, pool_method, instance_norm = (
            bound_parameters
        )
        super().__init__()

        if not (len(radii) == len(nsamples) == len(mlps)):
            raise AssertionError

        self.npoint = npoint
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i, _ in enumerate(radii):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(
                pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz)
                if npoint is not None
                else pointnet2_utils.GroupAll(use_xyz)
            )
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3

            self.mlps.append(
                pt_utils.SharedMLP(mlp_spec, bn=bn, instance_norm=instance_norm)
            )
        self.pool_method = pool_method


class PointnetSAModule(PointnetSAModuleMSG):
    """Pointnet set abstraction layer"""

    def __init__(self, *call_args, **call_kwargs):
        """
        :param mlp: list of int, spec of the pointnet before the global max_pool
        :param npoint: int, number of features
        :param radius: float, radius of ball
        :param nsample: int, number of samples in the ball query
        :param bn: whether to use batchnorm
        :param use_xyz:
        :param pool_method: max_pool / avg_pool
        :param instance_norm: whether to use instance_norm
        """
        bound_parameters = _bind_parameters(
            _POINTNETSAMODULE_INIT_PARAMETERS_SCHEMA, call_args, call_kwargs
        )
        mlp, npoint, radius, nsample, bn, use_xyz, pool_method, instance_norm = (
            bound_parameters
        )
        super().__init__(
            mlps=[mlp],
            npoint=npoint,
            radii=[radius],
            nsamples=[nsample],
            bn=bn,
            use_xyz=use_xyz,
            pool_method=pool_method,
            instance_norm=instance_norm,
        )


class PointnetFPModule(nn.Module):
    r"""Propigates the features of one set to another"""

    def __init__(
        self, *, mlp: List[int], bn: bool = True, activation=nn.ReLU(inplace=True)
    ):
        """
        :param mlp: list of int
        :param bn: whether to use batchnorm
        """
        super().__init__()
        self.mlp = pt_utils.SharedMLP(mlp, bn=bn, activation=activation)

    def forward(
        self,
        unknown: torch.Tensor,
        known: torch.Tensor,
        unknow_feats: torch.Tensor,
        known_feats: torch.Tensor,
    ) -> torch.Tensor:
        """
        :param unknown: (B, n, 3) tensor of the xyz positions of the unknown features
        :param known: (B, m, 3) tensor of the xyz positions of the known features
        :param unknow_feats: (B, C1, n) tensor of the features to be propigated to
        :param known_feats: (B, C2, m) tensor of features to be propigated
        :return:
            new_features: (B, mlp[-1], n) tensor of the features of the unknown features
        """
        if known is not None:
            dist, idx = pointnet2_utils.three_nn(unknown, known)
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm

            interpolated_feats = pointnet2_utils.three_interpolate(
                known_feats, idx, weight
            )
        else:
            interpolated_feats = known_feats.expand(
                *known_feats.size()[0:2], unknown.size(1)
            )

        if unknow_feats is not None:
            new_features = torch.cat(
                [interpolated_feats, unknow_feats], dim=1
            )  # (B, C2 + C1, n)
        else:
            new_features = interpolated_feats

        new_features = new_features.unsqueeze(-1)
        new_features = self.mlp(new_features)

        return new_features.squeeze(-1)


if __name__ == "__main__":
    pass
