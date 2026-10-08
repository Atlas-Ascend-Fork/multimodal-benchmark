# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Compatibility exports for the canonical KITTI evaluation implementation."""

__all__ = ["get_official_eval_result", "get_coco_eval_result"]

from pathlib import Path

from shared.compatibility import import_from_root


_EVAL_MODULE = import_from_root(
    "tools.kitti_object_eval_python.eval",
    Path(__file__).resolve().parents[4] / "EPNet",
)

get_official_eval_result = _EVAL_MODULE.get_official_eval_result
get_coco_eval_result = _EVAL_MODULE.get_coco_eval_result
