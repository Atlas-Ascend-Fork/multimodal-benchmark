# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Compatibility export for the canonical EPNet rotated-IoU implementation."""

__all__ = ["rotate_iou_gpu_eval"]

from pathlib import Path

from shared.compatibility import load_from_path


_CANONICAL_PATH = (
    Path(__file__).resolve().parents[4]
    / "EPNet/tools/kitti_object_eval_python/rotate_iou.py"
)
_MODULE = load_from_path("_ai_msf_rotate_iou", _CANONICAL_PATH)

rotate_iou_gpu_eval = _MODULE.rotate_iou_gpu_eval
