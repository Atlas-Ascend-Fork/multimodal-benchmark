# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

""" Helper functions for calculating 2D and 3D bounding box IoU.

Collected by Charles R. Qi
Date: September 2017
"""
import logging

from scipy.spatial import ConvexHull
import numpy as np

from shared.box_geometry import box3d_iou, box3d_volume as box3d_vol
from shared.box_geometry import convex_hull_intersection, poly_area, polygon_clip

LOGGER = logging.getLogger(__name__)


def is_clockwise(p):
    x = p[:, 0]
    y = p[:, 1]
    return np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)) > 0


def box3d_iou_pair(corners1, corners2):
    """Compute paired 3D bounding box IoU.
    /*
      camera coordinate
                7 -------- 4
               /|         /|
              6 -------- 5 .
              | |        | |
              . 3 -------- 0
              |/         |/
              2 -------- 1
     */
    Input:
        corners1: numpy array (n,8,3), assume up direction is negative Y
        corners2: numpy array (n,8,3), assume up direction is negative Y
    Output:
        ious: numpy array (n, 2)
        [:, 0] BEV box IoU
        [:, 1] 3D bounding box IoU
    """
    # corner points are in counter clockwise order
    if not (len(corners1) == len(corners2)):
        raise AssertionError("number of boxes sholud be equal")
    num_pairs = len(corners1)
    ious = np.zeros((num_pairs, 2), dtype=corners1.dtype)
    for i in range(num_pairs):
        iou_3d, iou_2d = box3d_iou(corners1[i], corners2[i])
        ious[i, 0] = iou_2d
        ious[i, 1] = iou_3d

    return ious


if __name__ == "__main__":

    # Function for polygon ploting
    from matplotlib.patches import Polygon
    import matplotlib.pyplot as plt

    def plot_polys(plist, scale=500.0):
        fig, ax = plt.subplots()
        patches = []
        for p in plist:
            poly = Polygon(np.array(p) / scale, True)
            patches.append(poly)

    # Demo on ConvexHull
    points = np.random.rand(30, 2)  # 30 random points in 2-D
    hull = ConvexHull(points)
    # **In 2D "volume" is is area, "area" is perimeter
    LOGGER.info("%s %s", "Hull area: ", hull.volume)
    for simplex in hull.simplices:
        LOGGER.info("%s", simplex)

    # Demo on convex hull overlaps
    sub_poly = [(0, 0), (300, 0), (300, 300), (0, 300)]
    clip_poly = [(150, 150), (300, 300), (150, 450), (0, 300)]
    inter_poly = polygon_clip(sub_poly, clip_poly)
    LOGGER.info("%s", poly_area(np.array(inter_poly)[:, 0], np.array(inter_poly)[:, 1]))

    # Test convex hull interaction function
    rect1 = [(50, 0), (50, 300), (300, 300), (300, 0)]
    rect2 = [(150, 150), (300, 300), (150, 450), (0, 300)]
    plot_polys([rect1, rect2])
    inter, area = convex_hull_intersection(rect1, rect2)
    LOGGER.info("%s %s", inter, area)
    if inter is not None:
        LOGGER.info("%s", poly_area(np.array(inter)[:, 0], np.array(inter)[:, 1]))

    LOGGER.info("%s", "------------------")
    rect1 = [
        (0.30026005199835404, 8.9408694211408424),
        (-1.1571105364358421, 9.4686676477075533),
        (0.1777082043006144, 13.154404877812102),
        (1.6350787927348105, 12.626606651245391),
    ]
    rect1 = [rect1[0], rect1[3], rect1[2], rect1[1]]
    rect2 = [
        (0.23908745901608636, 8.8551095691132886),
        (-1.2771419487733995, 9.4269062966181956),
        (0.13138836963152717, 13.161896351296868),
        (1.647617777421013, 12.590099623791961),
    ]
    rect2 = [rect2[0], rect2[3], rect2[2], rect2[1]]
    plot_polys([rect1, rect2])
    inter, area = convex_hull_intersection(rect1, rect2)
    LOGGER.info("%s %s", inter, area)
