# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Compatibility exports for the canonical mixed-precision optimizer."""

__all__ = (
    "FastAIMixedOptim", "OptimWrapper", "bn_types", "get_master", "is_tuple",
    "listify", "master2model", "model_g2master_g", "split_bn_bias",
    "trainable_params",
)

from pathlib import Path

from shared.compatibility import load_from_path


_CANONICAL = load_from_path(
    "_canonical_fastai_optim",
    Path(__file__).resolve().parents[3] / "clocs/torchplus/train/fastai_optim.py",
)

FastAIMixedOptim = _CANONICAL.FastAIMixedOptim
OptimWrapper = _CANONICAL.OptimWrapper
bn_types = _CANONICAL.bn_types
get_master = _CANONICAL.get_master
is_tuple = _CANONICAL.is_tuple
listify = _CANONICAL.listify
master2model = _CANONICAL.master2model
model_g2master_g = _CANONICAL.model_g2master_g
split_bn_bias = _CANONICAL.split_bn_bias
trainable_params = _CANONICAL.trainable_params
