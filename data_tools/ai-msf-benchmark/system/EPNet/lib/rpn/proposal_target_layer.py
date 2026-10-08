# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from lib.config import cfg
import lib.utils.iou3d.iou3d_utils as iou3d_utils
import lib.utils.kitti_utils as kitti_utils
import lib.utils.roipool3d.roipool3d_utils as roipool3d_utils
import numpy as np
import torch
import torch.nn as nn
from shared.proposal_target import aug_roi_by_noise, data_augmentation
from shared.proposal_target import random_aug_box3d, sample_bg_inds, sample_roi_indices
from shared.parameters import bind_named as _bind_named


def _proposaltargetlayer_forward_77_part_1_chunk_1(_state):
    _state["roi_boxes3d"], _state["gt_boxes3d"] = (
        _state["input_dict"]["roi_boxes3d"],
        _state["input_dict"]["gt_boxes3d"],
    )
    _state["batch_rois"], _state["batch_gt_of_rois"], _state["batch_roi_iou"] = _state[
        "self"
    ].sample_rois_for_rcnn(_state["roi_boxes3d"], _state["gt_boxes3d"])
    _state["rpn_xyz"], _state["rpn_features"] = (
        _state["input_dict"]["rpn_xyz"],
        _state["input_dict"]["rpn_features"],
    )
    if cfg.RCNN.USE_INTENSITY:
        _state["pts_extra_input_list"] = [
            _state["input_dict"]["rpn_intensity"].unsqueeze(dim=2),
            _state["input_dict"]["seg_mask"].unsqueeze(dim=2),
        ]
    else:
        _state["pts_extra_input_list"] = [
            _state["input_dict"]["seg_mask"].unsqueeze(dim=2)
        ]
    if cfg.RCNN.USE_DEPTH:
        _state["pts_depth"] = _state["input_dict"]["pts_depth"] / 70.0 - 0.5
        _state["pts_extra_input_list"].append(_state["pts_depth"].unsqueeze(dim=2))
    if cfg.RCNN.USE_RGB:
        _state["pts_rgb"] = _state["input_dict"]["pts_rgb"]
        _state["pts_extra_input_list"].append(_state["pts_rgb"])
    _state["pts_extra_input"] = torch.cat(_state["pts_extra_input_list"], dim=2)
    _state["pts_feature"] = torch.cat(
        (_state["pts_extra_input"], _state["rpn_features"]), dim=2
    )
    _state["pooled_features"], _state["pooled_empty_flag"] = (
        roipool3d_utils.roipool3d_gpu(
            _state["rpn_xyz"],
            _state["pts_feature"],
            _state["batch_rois"],
            cfg.RCNN.POOL_EXTRA_WIDTH,
            sampled_pt_num=cfg.RCNN.NUM_POINTS,
        )
    )
    _state["sampled_pts"], _state["sampled_features"] = (
        _state["pooled_features"][:, :, :, 0:3],
        _state["pooled_features"][:, :, :, 3:],
    )


def _proposaltargetlayer_forward_77_part_1_chunk_2(_state):
    _state["mask_score"] = (
        _state["pooled_features"][:, :, :, 3].sum(-1) / cfg.RCNN.NUM_POINTS
    )
    if cfg.AUG_DATA:
        (
            _state["sampled_pts"],
            _state["batch_rois"],
            _state["batch_gt_of_rois"],
        ) = _state["self"].data_augmentation(
            _state["sampled_pts"], _state["batch_rois"], _state["batch_gt_of_rois"]
        )
    _state["batch_size"] = _state["batch_rois"].shape[0]
    _state["roi_ry"] = _state["batch_rois"][:, :, 6] % (2 * np.pi)
    _state["roi_center"] = _state["batch_rois"][:, :, 0:3]


def _proposaltargetlayer_forward_77_part_1(_state):
    _proposaltargetlayer_forward_77_part_1_chunk_1(_state)
    _proposaltargetlayer_forward_77_part_1_chunk_2(_state)


def _proposaltargetlayer_forward_77_part_2(_state):
    _state["sampled_pts"] = _state["sampled_pts"] - _state["roi_center"].unsqueeze(
        dim=2
    )
    _state["batch_gt_of_rois"][:, :, 0:3] = (
        _state["batch_gt_of_rois"][:, :, 0:3] - _state["roi_center"]
    )
    _state["batch_gt_of_rois"][:, :, 6] = (
        _state["batch_gt_of_rois"][:, :, 6] - _state["roi_ry"]
    )
    for _state["k"] in range(_state["batch_size"]):
        _state["sampled_pts"][_state["k"]] = kitti_utils.rotate_pc_along_y_torch(
            _state["sampled_pts"][_state["k"]], _state["batch_rois"][_state["k"], :, 6]
        )
        _state["batch_gt_of_rois"][_state["k"]] = kitti_utils.rotate_pc_along_y_torch(
            _state["batch_gt_of_rois"][_state["k"]].unsqueeze(dim=1),
            _state["roi_ry"][_state["k"]],
        ).squeeze(dim=1)
    _state["valid_mask"] = _state["pooled_empty_flag"] == 0
    _state["reg_valid_mask"] = (
        (_state["batch_roi_iou"] > cfg.RCNN.REG_FG_THRESH) & _state["valid_mask"]
    ).long()
    _state["batch_cls_label"] = (
        _state["batch_roi_iou"] > cfg.RCNN.CLS_FG_THRESH
    ).long()
    _state["invalid_mask"] = (_state["batch_roi_iou"] > cfg.RCNN.CLS_BG_THRESH) & (
        _state["batch_roi_iou"] < cfg.RCNN.CLS_FG_THRESH
    )
    _state["batch_cls_label"][_state["valid_mask"] == 0] = -1
    _state["batch_cls_label"][_state["invalid_mask"] > 0] = -1
    _state["output_dict"] = {
        "sampled_pts": _state["sampled_pts"].view(-1, cfg.RCNN.NUM_POINTS, 3),
        "pts_feature": _state["sampled_features"].view(
            -1, cfg.RCNN.NUM_POINTS, _state["sampled_features"].shape[3]
        ),
        "cls_label": _state["batch_cls_label"].view(-1),
        "mask_score": _state["mask_score"].view(-1),
        "reg_valid_mask": _state["reg_valid_mask"].view(-1),
        "gt_of_rois": _state["batch_gt_of_rois"].view(-1, 7),
        "gt_iou": _state["batch_roi_iou"].view(-1),
        "roi_boxes3d": _state["batch_rois"].view(-1, 7),
    }


class ProposalTargetLayer(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def random_aug_box3d(box3d):
        return random_aug_box3d(box3d, cfg)

    def forward(self, input_dict):
        _state = locals()
        _proposaltargetlayer_forward_77_part_1(_state)
        _proposaltargetlayer_forward_77_part_2(_state)

        return _state["output_dict"]

    def sample_rois_for_rcnn(self, roi_boxes3d, gt_boxes3d):
        """Sample foreground and background ROIs for RCNN training."""
        batch_size = roi_boxes3d.size(0)
        fg_rois_per_image = int(np.round(cfg.RCNN.FG_RATIO * cfg.RCNN.ROI_PER_IMAGE))
        batch_rois = gt_boxes3d.new(batch_size, cfg.RCNN.ROI_PER_IMAGE, 7).zero_()
        batch_gt_of_rois = gt_boxes3d.new(batch_size, cfg.RCNN.ROI_PER_IMAGE, 7).zero_()
        batch_roi_iou = gt_boxes3d.new(batch_size, cfg.RCNN.ROI_PER_IMAGE).zero_()
        for idx in range(batch_size):
            cur_roi, cur_gt = roi_boxes3d[idx], gt_boxes3d[idx]
            k = cur_gt.__len__() - 1
            while cur_gt[k].sum() == 0:
                k -= 1
            cur_gt = cur_gt[: k + 1]
            iou3d = iou3d_utils.boxes_iou3d_gpu(cur_roi, cur_gt[:, 0:7])  # (M, N)
            max_overlaps, gt_assignment = torch.max(iou3d, dim=1)
            fg_inds, bg_inds, fg_count, bg_count = self._sample_roi_indices(
                max_overlaps, gt_boxes3d, fg_rois_per_image
            )
            rois, iou_of_rois, gt_of_rois = self._augment_sampled_rois(
                cur_roi, cur_gt, gt_assignment, max_overlaps,
                fg_inds, bg_inds, fg_count, bg_count,
            )
            batch_rois[idx] = rois
            batch_gt_of_rois[idx] = gt_of_rois
            batch_roi_iou[idx] = iou_of_rois
        return batch_rois, batch_gt_of_rois, batch_roi_iou

    def sample_bg_inds(self, hard_bg_inds, easy_bg_inds, bg_rois_per_this_image):
        return sample_bg_inds(
            hard_bg_inds, easy_bg_inds, bg_rois_per_this_image,
            cfg.RCNN.HARD_BG_RATIO,
        )

    def aug_roi_by_noise_torch(self, roi_boxes3d, gt_boxes3d, iou3d_src, aug_times=10):
        return aug_roi_by_noise(
            self, roi_boxes3d, gt_boxes3d, iou3d_src, aug_times, cfg, iou3d_utils
        )

    def data_augmentation(self, pts, rois, gt_of_rois):
        return data_augmentation(pts, rois, gt_of_rois, cfg, kitti_utils)

    def _sample_roi_indices(self, max_overlaps, gt_boxes3d, fg_rois_per_image):
        return sample_roi_indices(
            self, max_overlaps, gt_boxes3d, fg_rois_per_image, cfg
        )

    def _augment_sampled_rois(self, *call_args, **call_kwargs):
        (
            cur_roi,
            cur_gt,
            assignment,
            overlaps,
            fg_inds,
            bg_inds,
            fg_count,
            bg_count,
        ) = _bind_named(
            call_args,
            call_kwargs,
            ('cur_roi', 'cur_gt', 'assignment', 'overlaps', 'fg_inds', 'bg_inds', 'fg_count', 'bg_count'),
        )
        roi_list, iou_list, gt_list = [], [], []
        if fg_count > 0:
            fg_gt = cur_gt[assignment[fg_inds]]
            fg_rois, fg_iou = self.aug_roi_by_noise_torch(
                cur_roi[fg_inds], fg_gt, overlaps[fg_inds],
                aug_times=cfg.RCNN.ROI_FG_AUG_TIMES,
            )
            roi_list.append(fg_rois)
            iou_list.append(fg_iou)
            gt_list.append(fg_gt)
        if bg_count > 0:
            bg_gt = cur_gt[assignment[bg_inds]]
            aug_times = 1 if cfg.RCNN.ROI_FG_AUG_TIMES > 0 else 0
            bg_rois, bg_iou = self.aug_roi_by_noise_torch(
                cur_roi[bg_inds], bg_gt, overlaps[bg_inds], aug_times=aug_times
            )
            roi_list.append(bg_rois)
            iou_list.append(bg_iou)
            gt_list.append(bg_gt)
        return torch.cat(roi_list), torch.cat(iou_list), torch.cat(gt_list)
