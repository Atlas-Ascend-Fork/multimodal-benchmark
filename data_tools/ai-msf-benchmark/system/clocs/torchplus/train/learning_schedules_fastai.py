# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""CLOCs compatibility exports for the shared one-cycle scheduler."""

__all__ = ["LRSchedulerStep", "annealing_cos", "OneCycle", "FakeOptim"]

from pathlib import Path

from shared.compatibility import load_from_path


_MODULE = load_from_path(
    "_ai_msf_learning_schedules_fastai",
    Path(__file__).parents[3] / "EPNet/tools/train_utils/learning_schedules_fastai.py",
)

LRSchedulerStep = _MODULE.LRSchedulerStep
annealing_cos = _MODULE.annealing_cos
OneCycle = _MODULE.OneCycle
FakeOptim = _MODULE.FakeOptim
