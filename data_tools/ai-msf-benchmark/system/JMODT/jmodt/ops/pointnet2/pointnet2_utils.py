# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""JMODT compatibility exports for the canonical PointNet2 operations."""

__all__ = [
    "FurthestPointSampling", "furthest_point_sample", "GatherOperation",
    "gather_operation", "ThreeNN", "three_nn", "ThreeInterpolate",
    "three_interpolate", "GroupingOperation", "grouping_operation",
    "BallQuery", "ball_query", "QueryAndGroup", "GroupAll",
]

from pathlib import Path
import sys

from shared.compatibility import import_from_root
from jmodt.ops.pointnet2 import pointnet2_cuda


sys.modules["pointnet2_cuda"] = pointnet2_cuda
_MODULE = import_from_root(
    "pointnet2.pointnet2_utils",
    Path(__file__).parents[4] / "EPNet/pointnet2_lib",
)
_MODULE.pointnet2 = pointnet2_cuda

FurthestPointSampling = _MODULE.FurthestPointSampling
furthest_point_sample = _MODULE.furthest_point_sample
GatherOperation = _MODULE.GatherOperation
gather_operation = _MODULE.gather_operation
ThreeNN = _MODULE.ThreeNN
three_nn = _MODULE.three_nn
ThreeInterpolate = _MODULE.ThreeInterpolate
three_interpolate = _MODULE.three_interpolate
GroupingOperation = _MODULE.GroupingOperation
grouping_operation = _MODULE.grouping_operation
BallQuery = _MODULE.BallQuery
ball_query = _MODULE.ball_query
QueryAndGroup = _MODULE.QueryAndGroup
GroupAll = _MODULE.GroupAll
