# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Compatibility exports for the canonical PointNet2 layer helpers."""

__all__ = ["SharedMLP", "BatchNorm1d", "BatchNorm2d", "Conv1d", "Conv2d", "FC"]

from pathlib import Path

from shared.compatibility import load_from_path


_CANONICAL_PATH = (
    Path(__file__).resolve().parents[4]
    / "EPNet/pointnet2_lib/pointnet2/pytorch_utils.py"
)
_MODULE = load_from_path("_ai_msf_pointnet2_pytorch_utils", _CANONICAL_PATH)

SharedMLP = _MODULE.SharedMLP
BatchNorm1d = _MODULE.BatchNorm1d
BatchNorm2d = _MODULE.BatchNorm2d
Conv1d = _MODULE.Conv1d
Conv2d = _MODULE.Conv2d
FC = _MODULE.FC
