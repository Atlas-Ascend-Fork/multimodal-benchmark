# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""'
@Project: AI-MSF-benchmark
@Description:    LidarSimulation
@Time:2022/1/13 14:45
@Author:NianGao

"""
import numpy as np

from corruption.operator.lidar_operator import LidarOperator
from corruption.simulation.common import SimulationBase


class LidarSimulation(SimulationBase):
    """
    :Author:  NianGao
    :Create:  2022/1/13
    """

    @staticmethod
    def simulation_one(input_fn, output_fn, dop, sev):
        op_map = LidarOperator.operator_map()
        op = op_map[dop]
        data = np.fromfile(input_fn, dtype=np.float32, count=-1).reshape([-1, 4])
        # aug one
        aug_data = op(data, sev)
        aug_data.tofile(output_fn)
