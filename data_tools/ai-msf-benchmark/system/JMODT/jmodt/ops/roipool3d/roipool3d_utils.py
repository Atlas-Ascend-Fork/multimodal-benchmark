# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""JMODT compatibility exports for the canonical ROI-pooling helpers."""

__all__ = [
    "roipool3d_gpu", "pts_in_boxes3d_cpu", "roipool_pc_cpu", "roipool3d_cpu"
]

from pathlib import Path
import sys

from shared.compatibility import import_from_root
from jmodt.ops.roipool3d import roipool3d_cuda
from jmodt.utils import kitti_utils


sys.modules["roipool3d_cuda"] = roipool3d_cuda
_MODULE = import_from_root(
    "lib.utils.roipool3d.roipool3d_utils",
    Path(__file__).parents[4] / "EPNet",
)
_MODULE.roipool3d_cuda = roipool3d_cuda
_MODULE.kitti_utils = kitti_utils

roipool3d_gpu = _MODULE.roipool3d_gpu
pts_in_boxes3d_cpu = _MODULE.pts_in_boxes3d_cpu
roipool_pc_cpu = _MODULE.roipool_pc_cpu
roipool3d_cpu = _MODULE.roipool3d_cpu
