# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""ROI augmentation and background sampling operations."""

import numpy as np
import torch
from shared.parameters import bind_named as _bind_named


def _scaled_box(box, position_range, size_range, angle_range):
    position = (torch.rand(3, device=box.device) * 2 - 1) * position_range
    size = (torch.rand(3, device=box.device) * 2 - 1) * size_range + 1
    angle = (torch.rand(1, device=box.device) * 2 - 1) * angle_range
    return torch.cat((box[:3] + position, box[3:6] * size, box[6:] + angle))


def random_aug_box3d(box, config):
    method = config.RCNN.REG_AUG_METHOD
    if method == "single":
        return _scaled_box(box, 0.5, 0.15, np.pi / 12)
    if method == "multiple":
        ranges = (
            (0.2, 0.1, np.pi / 12), (0.3, 0.15, np.pi / 12),
            (0.5, 0.15, np.pi / 9), (0.8, 0.15, np.pi / 6),
            (1.0, 0.15, np.pi / 3),
        )
        index = torch.randint(0, len(ranges), (1,))[0].long()
        return _scaled_box(box, *ranges[index])
    if method == "normal":
        deviations = (0.3, 0.2, 0.3, 0.25, 0.15, 0.5)
        shifts = [np.random.normal(0, deviation) for deviation in deviations]
        shifts.append(((torch.rand() - 0.5) / 0.5) * np.pi / 12)
        return torch.as_tensor(shifts).type_as(box) + box
    raise NotImplementedError


def data_augmentation(pts, rois, gt_of_rois, config, box_utils):
    batch_size, boxes_num = pts.shape[:2]
    angles = torch.rand((batch_size, boxes_num), device=pts.device) - 0.5 / 0.5
    angles *= np.pi / config.AUG_ROT_RANGE
    for index in range(batch_size):
        gt_alpha = _box_alpha(gt_of_rois[index])
        roi_alpha = _box_alpha(rois[index])
        pts[index] = box_utils.rotate_pc_along_y_torch(pts[index], angles[index])
        for boxes, alpha in ((gt_of_rois, gt_alpha), (rois, roi_alpha)):
            boxes[index] = box_utils.rotate_pc_along_y_torch(
                boxes[index].unsqueeze(dim=1), angles[index]
            ).squeeze(dim=1)
            beta = torch.atan2(boxes[index, :, 2], boxes[index, :, 0])
            boxes[index, :, 6] = torch.sign(beta) * np.pi / 2 + alpha - beta
    scales = 1 + (torch.rand((batch_size, boxes_num), device=pts.device) * 2 - 1) * 0.05
    pts *= scales.unsqueeze(2).unsqueeze(3)
    gt_of_rois[:, :, :6] *= scales.unsqueeze(2)
    rois[:, :, :6] *= scales.unsqueeze(2)
    flip = torch.sign(torch.rand((batch_size, boxes_num), device=pts.device) - 0.5)
    pts[:, :, :, 0] *= flip.unsqueeze(2)
    for boxes in (gt_of_rois, rois):
        boxes[:, :, 0] *= flip
        yaw = boxes[:, :, 6]
        boxes[:, :, 6] = (flip == 1).float() * yaw + (
            (flip == -1).float() * (torch.sign(yaw) * np.pi - yaw)
        )
    return pts, rois, gt_of_rois


def _box_alpha(boxes):
    beta = torch.atan2(boxes[:, 2], boxes[:, 0])
    return -torch.sign(beta) * np.pi / 2 + beta + boxes[:, 6]


def sample_bg_inds(hard_inds, easy_inds, count, hard_ratio):
    hard_count, easy_count = hard_inds.numel(), easy_inds.numel()
    if hard_count and easy_count:
        hard_samples = int(count * hard_ratio)
        easy_samples = count - hard_samples
        hard_order = torch.randint(0, hard_count, (hard_samples,)).long()
        easy_order = torch.randint(0, easy_count, (easy_samples,)).long()
        return torch.cat((hard_inds[hard_order], easy_inds[easy_order]))
    if hard_count:
        return hard_inds[torch.randint(0, hard_count, (count,)).long()]
    if easy_count:
        return easy_inds[torch.randint(0, easy_count, (count,)).long()]
    raise RuntimeError("no background regions available")


def sample_roi_indices(layer, max_overlaps, gt_boxes3d, foreground_limit, config):
    """Select foreground and background ROI indices for one image."""
    threshold = min(config.RCNN.REG_FG_THRESH, config.RCNN.CLS_FG_THRESH)
    foreground = torch.nonzero(max_overlaps >= threshold).view(-1)
    easy_background = torch.nonzero(
        max_overlaps < config.RCNN.CLS_BG_THRESH_LO
    ).view(-1)
    hard_background = torch.nonzero(
        (max_overlaps < config.RCNN.CLS_BG_THRESH)
        & (max_overlaps >= config.RCNN.CLS_BG_THRESH_LO)
    ).view(-1)
    foreground_count = foreground.numel()
    background_count = hard_background.numel() + easy_background.numel()
    if foreground_count > 0 and background_count > 0:
        foreground_per_image = min(foreground_limit, foreground_count)
        order = torch.from_numpy(np.random.permutation(foreground_count)).type_as(
            gt_boxes3d
        ).long()
        foreground = foreground[order[:foreground_per_image]]
        background_per_image = config.RCNN.ROI_PER_IMAGE - foreground_per_image
        background = layer.sample_bg_inds(
            hard_background, easy_background, background_per_image
        )
    elif foreground_count > 0:
        order = np.floor(
            np.random.rand(config.RCNN.ROI_PER_IMAGE) * foreground_count
        )
        foreground = foreground[torch.from_numpy(order).type_as(gt_boxes3d).long()]
        foreground_per_image, background_per_image = config.RCNN.ROI_PER_IMAGE, 0
        background = easy_background
    elif background_count > 0:
        foreground_per_image, background_per_image = 0, config.RCNN.ROI_PER_IMAGE
        background = layer.sample_bg_inds(
            hard_background, easy_background, background_per_image
        )
    else:
        raise RuntimeError("no foreground or background regions available")
    return foreground, background, foreground_per_image, background_per_image


def aug_roi_by_noise(*call_args, **call_kwargs):
    (
        layer,
        roi_boxes,
        gt_boxes,
        source_iou,
        aug_times,
        config,
        iou_utils,
    ) = _bind_named(
        call_args,
        call_kwargs,
        ('layer', 'roi_boxes', 'gt_boxes', 'source_iou', 'aug_times', 'config', 'iou_utils'),
    )
    output_iou = torch.zeros(roi_boxes.shape[0]).type_as(gt_boxes)
    threshold = min(config.RCNN.REG_FG_THRESH, config.RCNN.CLS_FG_THRESH)
    for index, roi_box in enumerate(roi_boxes):
        current_iou = attempts = 0
        augmented = roi_box
        keep = True
        while current_iou < threshold and attempts < aug_times:
            keep = np.random.rand() < 0.2
            augmented = roi_box if keep else layer.random_aug_box3d(roi_box)
            current_iou = iou_utils.boxes_iou3d_gpu(
                augmented.view(1, 7), gt_boxes[index].view(1, 7)
            )[0][0]
            attempts += 1
        roi_boxes[index] = augmented.view(-1)
        output_iou[index] = source_iou[index] if attempts == 0 or keep else current_iou
    return roi_boxes, output_iou
