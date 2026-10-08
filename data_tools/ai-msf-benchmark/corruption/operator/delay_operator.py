# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""'
@Project: AI-MSF-benchmark
@Description: Please add Description
@Time:2022/9/6 15:18
@Author:NianGao

"""

import os
import shutil

from natsort import natsorted

from utils.utils import symlink


class DelayOperator(object):
    @staticmethod
    def operator_map():
        operator_map = {
            "delay": DelayOperator.delay,
        }
        return operator_map

    # delay a source of signal (camera or Lidar)
    @staticmethod
    def delay(inputbasedir, outputbasedir, severity, copy_mode=0):
        t = [1, 2, 3, 4, 5][severity - 1]
        base_input_fns = natsorted(os.listdir(inputbasedir))
        output_fns = base_input_fns.copy()
        input_fns = [base_input_fns[0]] * t + base_input_fns[:-t]
        if not (len(input_fns) == len(output_fns)):
            raise AssertionError
        for frame_i, frame_o in zip(input_fns, output_fns):
            source = os.path.join(inputbasedir, frame_i)
            target = os.path.join(outputbasedir, frame_o)
            if copy_mode == 0:
                symlink(source, target)
            else:
                shutil.copy(source, target)
