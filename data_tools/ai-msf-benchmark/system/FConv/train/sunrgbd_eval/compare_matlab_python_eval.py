# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

""" Compare MATLAB and Python eval code on AP computation """

import os

import numpy as np

from utils.box_util import is_clockwise
from shared.dataset_geometry import SUNRGBD_CLASSES_10

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

gt_boxes_dir = os.path.join(BASE_DIR, "gt_boxes/")


def flip_axis_to_camera(pc):
    """Flip X-right,Y-forward,Z-up to X-right,Y-down,Z-forward
    Input and output are both (N,3) array
    """
    pc2 = np.copy(pc)
    pc2[:, [0, 1, 2]] = pc2[:, [0, 2, 1]]  # cam X,Y,Z = depth X,-Z,Y
    pc2[:, 1] *= -1
    return pc2


def box_conversion(bbox):
    """In upright depth camera coord"""
    bbox3d = np.zeros((8, 3))
    # Make clockwise
    # NOTE: in box3d IoU evaluation we require the polygon vertices in
    # counter clockwise order. However, from dumped data in MATLAB
    # some of the polygons are in clockwise, some others are counter clockwise
    # so we need to inspect each box and make them consistent..
    xy = np.reshape(bbox[0:8], (4, 2))
    if is_clockwise(xy):
        bbox3d[0:4, 0:2] = xy
        bbox3d[4:, 0:2] = xy
    else:
        bbox3d[0:4, 0:2] = xy[::-1, :]
        bbox3d[4:, 0:2] = xy[::-1, :]
    bbox3d[0:4, 2] = bbox[9]  # zmax
    bbox3d[4:, 2] = bbox[8]  # zmin
    return bbox3d


def wrapper(bbox):
    bbox3d = box_conversion(bbox)
    bbox3d = flip_axis_to_camera(bbox3d)
    bbox3d_flipped = np.copy(bbox3d)
    bbox3d_flipped[0:4, :] = bbox3d[4:, :]
    bbox3d_flipped[4:, :] = bbox3d[0:4, :]
    return bbox3d_flipped


def get_gt_cls(classname):
    gt = {}
    gt_boxes = np.loadtxt(os.path.join(gt_boxes_dir, "%s_gt_boxes.dat" % (classname)))
    gt_imgids = np.loadtxt(os.path.join(gt_boxes_dir, "%s_gt_imgids.txt" % (classname)))

    for i, _ in enumerate(gt_imgids):
        imgid = gt_imgids[i]
        bbox = gt_boxes[i]
        bbox3d = wrapper(bbox)

        if imgid not in gt:
            gt[imgid] = []
        gt[imgid].append(bbox3d)
    return gt


def get_gt_all():
    gt_all = {}

    for classname in SUNRGBD_CLASSES_10:
        gt_all[classname] = get_gt_cls(classname)

    return gt_all
