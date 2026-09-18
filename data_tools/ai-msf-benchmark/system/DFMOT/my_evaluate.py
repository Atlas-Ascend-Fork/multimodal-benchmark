# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import logging

from kitti_devkit.evaluate_tracking import evaluate

LOGGER = logging.getLogger(__name__)

if __name__ == "__main__":
    step = "train"
    part = "all"
    result_path = "results"

    X = evaluate(step, result_path, part=part)
    LOGGER.info("%s", X)
    MOTA, MOTP, recall, prec, F1, fp, fn, id_switches = X
