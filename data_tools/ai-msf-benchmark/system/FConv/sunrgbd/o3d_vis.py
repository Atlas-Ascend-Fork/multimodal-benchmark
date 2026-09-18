# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import os
import sys

from sunrgbd_data import sunrgbd_object
import open3d as o3d
import sunrgbd_utils as utils


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)


def get_pcd(x):

    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(x[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(x[:, 3:])

    return pcd


def get_lineset(corners, color=(1, 0, 0)):
    """corners: (8, 3)
      1 -------- 0
     /|         /|
    2 -------- 3 .
    | |        | |
    . 5 -------- 4
    |/         |/
    6 -------- 7
    """
    if not (corners.shape == (8, 3)):
        raise AssertionError
    points = corners

    lines = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],
        [2, 3],
        [3, 7],
        [1, 5],
        [0, 4],
    ]

    colors = [color for (i, _) in enumerate(lines)]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


if __name__ == "__main__":
    split = "training"
    idx_filename = "sunrgbd_data/matlab/SUNRGBDtoolbox/mysunrgbd/val_data_idx.txt"
    dataset = sunrgbd_object("sunrgbd_data/matlab/SUNRGBDtoolbox/mysunrgbd", split)
    data_idx_list = [int(line.rstrip()) for line in open(idx_filename)]

    data_idx = data_idx_list[0]

    calib = dataset.get_calibration(data_idx)
    objects = dataset.get_label_objects(data_idx)
    pc_upright_depth = dataset.get_depth(data_idx)

    line_sets = []

    pcd = get_pcd(pc_upright_depth)

    for obj_idx, _ in enumerate(objects):
        obj = objects[obj_idx]
        box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, calib)
        line_set = get_lineset(box3d_pts_3d)
        line_sets.append(line_set)
        o3d.io.write_line_set("%d.ply" % obj_idx, line_set, write_ascii=True)

    o3d.io.write_point_cloud("tmp.ply", pcd)
