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
@Time:2022/9/6 13:55
@Author:NianGao

"""


import numpy as np


class LidarOperator(object):
    """
    :Author:  NianGao
    :Create:  2022/1/13
    """

    @staticmethod
    def operator_map():
        operator_map = {
            "loss_partial": LidarOperator.loss_partial,
            "gaussian_noise": LidarOperator.gaussian_noise,
            "impulse_noise": LidarOperator.impulse_noise,
            "loss_complete": LidarOperator.loss_complete,
        }
        return operator_map

    @staticmethod
    def loss_partial(pointcloud, severity=1):
        params = [0.1, 0.25, 0.5, 0.75, 0.9][severity - 1]
        n, c = pointcloud.shape
        index = np.random.permutation(n)[: int(n * params)]
        pointcloud[index] = [0, 0, 0, 0]
        return pointcloud

    @staticmethod
    def gaussian_noise(pointcloud, severity=1):
        n, c_value = pointcloud.shape
        c = [0.02, 0.04, 0.06, 0.08, 0.1][severity - 1]
        jitter = np.random.normal(size=(n, c_value)) * c
        new_pc = (pointcloud + jitter).astype("float32")
        return new_pc

    @staticmethod
    def impulse_noise(pointcloud, severity=1):
        n, c_value = pointcloud.shape
        c = [n // 30, n // 20, n // 10, n // 7, n // 5][severity - 1]
        index = np.random.choice(n, c, replace=False)
        pointcloud[index] += np.random.choice([-1, 1], size=(c, c_value)) * 0.2
        return pointcloud

    @staticmethod
    def loss_complete(pointcloud, severity=1):
        n, c = pointcloud.shape
        index = range(n)
        pointcloud[index] = [0, 0, 0, 0]
        return pointcloud
