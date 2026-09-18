# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""JMODT compatibility exports for the canonical PointNet2 modules."""

__all__ = ["PointnetSAModuleMSG", "PointnetSAModule", "PointnetFPModule"]

from pathlib import Path

from shared.compatibility import import_from_root
from . import pointnet2_utils  # noqa: F401  Ensures the JMODT CUDA backend is selected.


_MODULE = import_from_root(
    "pointnet2.pointnet2_modules",
    Path(__file__).parents[4] / "EPNet/pointnet2_lib",
)

PointnetSAModuleMSG = _MODULE.PointnetSAModuleMSG
PointnetSAModule = _MODULE.PointnetSAModule
PointnetFPModule = _MODULE.PointnetFPModule
