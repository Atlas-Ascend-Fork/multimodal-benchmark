# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from skimage import io
import numpy as np


def read_3d_points(rgbpath, depthpath, rtilt, k):
    """
    a python implementation of SUNRGBDTOOL read3dPoints.m
    Rtilt: (3, 3)
    K: (3, 3)
    """
    depth_vis = io.imread(depthpath)
    valid = (depth_vis != 0).ravel()

    depth = (depth_vis >> 3) | (depth_vis << 13)
    depth = depth.astype(np.float32) / 1000
    depth[depthpath > 8] = 8
    height = depth.shape[0]
    width = depth.shape[1]

    cx, cy = k[0, 2], k[1, 2]
    fx, fy = k[0, 0], k[1, 1]

    x, y = np.meshgrid(np.arange(width), np.arange(height))
    x3 = (x - cx) * depth / fx
    y3 = (y - cy) * depth / fy
    z3 = depth

    points = np.stack([x3.ravel(), z3.ravel(), -y3.ravel()], 1)
    points = points[valid]

    rgb = io.imread(rgbpath)
    rgb = rgb.astype(np.float32).reshape(-1, 3)[valid] / 255
    points = np.matmul(rtilt, points.T).T

    points_rgb = np.concatenate([points, rgb], 1)

    return points_rgb
