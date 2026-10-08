# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""'
@Project: fusion_rewrite
@Description: Please add Description
@Time:2022/9/6 14:24
@Author:NianGao

"""
import functools
import math

import numpy as np
from shared.rotation import euler_angles_to_rotation_matrix
from shared.rotation import is_rotation_matrix, rotation_matrix_to_euler_angles


class CalibOperator(object):
    """
    :Author:  NianGao
    :Create:  2022/1/13
    """

    @staticmethod
    def get_dop_method_arr():
        dop_method_arr = []
        return dop_method_arr

    @staticmethod
    def operator_map():
        operator_map = {
            "rotation_x": functools.partial(CalibOperator.rotation, aixs=0),
            "rotation_y": functools.partial(CalibOperator.rotation, aixs=1),
            "rotation_z": functools.partial(CalibOperator.rotation, aixs=2),
        }
        return operator_map

    @staticmethod
    def rotation(v2_c, severity=1, aixs=None):
        num = [0.5, 1, 2][severity - 1]
        theta_xyz = [0, 0, 0]
        theta_xyz[aixs] = num
        trans_xyz = [0, 0, 0]
        v2_c_noise = CalibOperator.extrinsic_mutaion(v2_c, theta_xyz, trans_xyz)
        v2_c_noise = v2_c_noise.flatten()
        return v2_c_noise

    @staticmethod
    def extrinsic_mutaion(v2_c, theta_xyz, trans_xyz):
        v2_c_noise = np.zeros_like(v2_c)
        theta_xyz = [theta * math.pi / 180.0 for theta in theta_xyz]
        theta_xyz = np.array(theta_xyz)

        # add rotaion noise
        angles = rotation_matrix_to_euler_angles(v2_c[:, :3])
        angles += theta_xyz  # add noise
        v2_c_rotation_noise = euler_angles_to_rotation_matrix(angles)
        v2_c_noise[:, :3] = v2_c_rotation_noise

        # add tans noise
        v2_c_trans_noise = v2_c[:, 3] + trans_xyz
        v2_c_noise[:, 3] = v2_c_trans_noise
        return v2_c_noise
