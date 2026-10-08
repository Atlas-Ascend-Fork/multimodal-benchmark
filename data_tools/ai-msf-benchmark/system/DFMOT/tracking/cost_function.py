# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Author: wangxy

import copy
import math

import numpy as np
from shared.box_geometry import camera_box_corners

from shared.box_geometry import box3d_iou as iou3d, box3d_volume as box3d_vol
from shared.box_geometry import convex_hull_intersection, poly_area, polygon_clip


def iou_batch(box_a, box_b):

    box_a = [int(x) for x in box_a]
    box_b = [int(x) for x in box_b]

    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)

    box_a_area = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
    box_b_area = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)

    iou = inter_area / float(box_a_area + box_b_area - inter_area)

    return iou


def eucli_distance(detection, track):
    dist = math.sqrt(
        (detection[0] - track[0]) ** 2
        + (detection[1] - track[1]) ** 2
        + (detection[2] - track[2]) ** 2
    )
    return dist


def roty(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def convert_3dbox_to_8corner(bbox3d_input):
    """Takes an object's 3D box with the representation of [x,y,z,theta,l,w,h] and
    convert it to the 8 corners of the 3D box

    Returns:
        corners_3d: (8,3) array in in rect camera coord
    """
    # compute rotational matrix around yaw axis
    bbox3d = copy.copy(bbox3d_input)
    return camera_box_corners(bbox3d[:3], bbox3d[4:7], bbox3d[3])
