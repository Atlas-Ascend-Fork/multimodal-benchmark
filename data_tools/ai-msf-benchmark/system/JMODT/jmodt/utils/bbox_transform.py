# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""JMODT compatibility exports for the shared EPNet box decoder."""

__all__ = ["rotate_pc_along_y_torch", "decode_bbox_target"]

from pathlib import Path

from shared.compatibility import import_from_root
from jmodt.config import cfg


_MODULE = import_from_root("lib.utils.bbox_transform", Path(__file__).parents[3] / "EPNet")
_MODULE.cfg = cfg

rotate_pc_along_y_torch = _MODULE.rotate_pc_along_y_torch
decode_bbox_target = _MODULE.decode_bbox_target
