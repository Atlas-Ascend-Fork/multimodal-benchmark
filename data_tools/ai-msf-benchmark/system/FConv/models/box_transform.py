# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import numpy as np
import torch


def size_decode(offset, class_mean_size, size_class_label):

    offset_select = torch.gather(
        offset, 1, size_class_label.view(-1, 1, 1).expand(-1, -1, 3)
    )
    offset_select = offset_select.squeeze(1)

    ex = class_mean_size[size_class_label]

    return offset_select * ex + ex


def size_encode(gt, class_mean_size, size_class_label):
    ex = class_mean_size[size_class_label]
    return (gt - ex) / ex


def center_decode(ex, offset):
    return ex + offset


def center_encode(gt, ex):
    return gt - ex


def angle_decode(ex_res, ex_class_id, num_bins=12, to_label_format=True):

    ex_res_select = torch.gather(ex_res, 1, ex_class_id.unsqueeze(1))
    ex_res_select = ex_res_select.squeeze(1)

    angle_per_class = 2 * np.pi / float(num_bins)

    angle = ex_class_id.float() * angle_per_class + ex_res_select * (
        angle_per_class / 2
    )

    if to_label_format:
        flag = angle > np.pi
        angle[flag] = angle[flag] - 2 * np.pi

    return angle


def angle_encode(gt_angle, num_bins=12):
    gt_angle = gt_angle % (2 * np.pi)
    if not (((gt_angle >= 0) & (gt_angle <= 2 * np.pi)).all()):
        raise AssertionError

    angle_per_class = 2 * np.pi / float(num_bins)
    shifted_angle = (gt_angle + angle_per_class / 2) % (2 * np.pi)
    gt_class_id = torch.floor(shifted_angle / angle_per_class).long()
    gt_res = shifted_angle - (
        gt_class_id.float() * angle_per_class + angle_per_class / 2
    )

    gt_res /= angle_per_class / 2
    return gt_class_id, gt_res
