# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""'
@Project: fusion
@Description: Please add Description
@Time:2022/1/13 17:26
@Author:NianGao

"""
import cv2
import matplotlib.pyplot as plt

from corruption.operator.image_operator import ImageOperator
from corruption.simulation.common import SimulationBase


class ImageSimulation(SimulationBase):
    """
    :Author:  NianGao
    :Create:  2022/1/13
    """

    @staticmethod
    def simulation_one(input_fn, output_fn, dop, sev):
        op_map = ImageOperator.operator_map()
        op = op_map[dop]
        image = cv2.imread(input_fn)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # aug one
        aug_image = op(image=image, severity=sev)
        plt.imsave(output_fn, aug_image)
