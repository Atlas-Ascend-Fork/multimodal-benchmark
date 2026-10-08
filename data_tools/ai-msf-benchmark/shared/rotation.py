# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Rotation matrices, Euler angles, and rigid transforms."""

import math

import numpy as np


def is_rotation_matrix(matrix):
    identity = np.identity(3, dtype=matrix.dtype)
    return np.linalg.norm(identity - matrix.T.dot(matrix)) < 1e-6


def rotation_matrix_to_euler_angles(matrix):
    if not is_rotation_matrix(matrix):
        raise AssertionError("matrix is not a valid rotation matrix")
    sy = math.hypot(matrix[0, 0], matrix[1, 0])
    if sy >= 1e-6:
        x = math.atan2(matrix[2, 1], matrix[2, 2])
        y = math.atan2(-matrix[2, 0], sy)
        z = math.atan2(matrix[1, 0], matrix[0, 0])
    else:
        x = math.atan2(-matrix[1, 2], matrix[1, 1])
        y = math.atan2(-matrix[2, 0], sy)
        z = 0
    return np.array((x, y, z))


def euler_angles_to_rotation_matrix(theta):
    x, y, z = theta
    matrix_x = np.array(
        ((1, 0, 0), (0, math.cos(x), -math.sin(x)), (0, math.sin(x), math.cos(x)))
    )
    matrix_y = np.array(
        ((math.cos(y), 0, math.sin(y)), (0, 1, 0), (-math.sin(y), 0, math.cos(y)))
    )
    matrix_z = np.array(
        ((math.cos(z), -math.sin(z), 0), (math.sin(z), math.cos(z), 0), (0, 0, 1))
    )
    return matrix_z.dot(matrix_y.dot(matrix_x))


def rotation_x(angle):
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array(((1, 0, 0), (0, cosine, -sine), (0, sine, cosine)))


def rotation_y(angle):
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array(((cosine, 0, sine), (0, 1, 0), (-sine, 0, cosine)))


def rotation_z(angle):
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array(((cosine, -sine, 0), (sine, cosine, 0), (0, 0, 1)))


def transform_from_rotation_translation(rotation, translation):
    rotation = rotation.reshape(3, 3)
    translation = translation.reshape(3, 1)
    return np.vstack((np.hstack((rotation, translation)), (0, 0, 0, 1)))


def inverse_rigid_transform(transform):
    inverse = np.zeros_like(transform)
    inverse[:3, :3] = transform[:3, :3].T
    inverse[:3, 3] = np.dot(-transform[:3, :3].T, transform[:3, 3])
    return inverse
