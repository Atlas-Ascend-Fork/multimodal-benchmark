# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import numpy as np
import torch
import torch.nn as nn

from jmodt.config import cfg
from jmodt.ops.iou3d import iou3d_utils
from jmodt.ops.roipool3d import roipool3d_utils
from jmodt.utils import kitti_utils
from shared.proposal_target import aug_roi_by_noise, data_augmentation
from shared.proposal_target import random_aug_box3d, sample_bg_inds, sample_roi_indices
from shared.parameters import bind_named as _bind_named


class ProposalTargetLayer(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode

    @staticmethod
    def random_aug_box3d(box3d):
        return random_aug_box3d(box3d, cfg)

    @staticmethod
    def data_augmentation(pts, rois, gt_of_rois):
        return data_augmentation(pts, rois, gt_of_rois, cfg, kitti_utils)

    @staticmethod
    def _point_features(input_dict):
        rpn_features = input_dict["rpn_features"]
        if cfg.RCNN.USE_INTENSITY:
            extra_features = [
                input_dict["rpn_intensity"].unsqueeze(dim=2),
                input_dict["seg_mask"].unsqueeze(dim=2),
            ]
        else:
            extra_features = [input_dict["seg_mask"].unsqueeze(dim=2)]
        if cfg.RCNN.USE_DEPTH:
            pts_depth = input_dict["pts_depth"] / 70.0 - 0.5
            extra_features.append(pts_depth.unsqueeze(dim=2))
        if cfg.RCNN.USE_RGB:
            extra_features.append(input_dict["pts_rgb"])
        return torch.cat((torch.cat(extra_features, dim=2), rpn_features), dim=2)

    @staticmethod
    def _pool_rois(rpn_xyz, point_features, rois):
        return roipool3d_utils.roipool3d_gpu(
            rpn_xyz, point_features, rois, cfg.RCNN.POOL_EXTRA_WIDTH,
            sampled_pt_num=cfg.RCNN.NUM_POINTS,
        )

    @staticmethod
    def _canonicalize_training(sampled_pts, rois, gt_of_rois):
        roi_ry = rois[:, :, 6] % (2 * np.pi)
        roi_center = rois[:, :, 0:3]
        sampled_pts -= roi_center.unsqueeze(dim=2)
        gt_of_rois[:, :, 0:3] -= roi_center
        gt_of_rois[:, :, 6] -= roi_ry
        for batch_index in range(rois.shape[0]):
            sampled_pts[batch_index] = kitti_utils.rotate_pc_along_y_torch(
                sampled_pts[batch_index], rois[batch_index, :, 6]
            )
            gt_of_rois[batch_index] = kitti_utils.rotate_pc_along_y_torch(
                gt_of_rois[batch_index].unsqueeze(dim=1), roi_ry[batch_index]
            ).squeeze(dim=1)

    @staticmethod
    def _training_output(*call_args, **call_kwargs):
        (
            sampled_pts,
            sampled_features,
            cls_label,
            mask_score,
            reg_valid_mask,
            gt_of_rois,
            roi_iou,
            rois,
            gt_tids,
        ) = _bind_named(
            call_args,
            call_kwargs,
            (
                'sampled_pts', 'sampled_features', 'cls_label', 'mask_score',
                'reg_valid_mask', 'gt_of_rois', 'roi_iou', 'rois', 'gt_tids',
            ),
        )
        output = {
            "sampled_pts": sampled_pts.view(-1, cfg.RCNN.NUM_POINTS, 3),
            "pts_feature": sampled_features.view(
                -1, cfg.RCNN.NUM_POINTS, sampled_features.shape[3]
            ),
            "cls_label": cls_label.view(-1),
            "mask_score": mask_score.view(-1),
            "reg_valid_mask": reg_valid_mask.view(-1),
            "gt_of_rois": gt_of_rois.view(-1, 7),
            "gt_iou": roi_iou.view(-1),
            "roi_boxes3d": rois.view(-1, 7),
            "gt_tids": gt_tids,
            "max_iou": torch.max(roi_iou),
        }
        output["pts_input"] = torch.cat(
            (output["sampled_pts"], output["pts_feature"]), dim=2
        )
        return output

    def forward(self, input_dict):
        point_features = self._point_features(input_dict)
        if self.mode == "TRAIN":
            return self._forward_train(input_dict, point_features)
        return self._forward_eval(input_dict, point_features)

    def sample_rois_for_rcnn(self, roi_boxes3d, gt_boxes3d, gt_tids):
        """Sample foreground/background ROIs and their tracking identities."""
        batch_size = roi_boxes3d.size(0)
        fg_rois_per_image = int(np.round(cfg.RCNN.FG_RATIO * cfg.RCNN.ROI_PER_IMAGE))
        batch_rois = gt_boxes3d.new(batch_size, cfg.RCNN.ROI_PER_IMAGE, 7).zero_()
        batch_gt_boxes_of_rois = gt_boxes3d.new(
            batch_size, cfg.RCNN.ROI_PER_IMAGE, 7
        ).zero_()
        batch_roi_iou = gt_boxes3d.new(batch_size, cfg.RCNN.ROI_PER_IMAGE).zero_()
        batch_gt_tids_of_rois = gt_boxes3d.new(
            batch_size, cfg.RCNN.ROI_PER_IMAGE
        ).zero_()
        for idx in range(batch_size):
            cur_roi = roi_boxes3d[idx]
            cur_gt_boxes = gt_boxes3d[idx]
            cur_gt_tids = gt_tids[idx]
            k = cur_gt_boxes.__len__() - 1
            while cur_gt_boxes[k].sum() == 0:
                k -= 1
            cur_gt_boxes = cur_gt_boxes[: k + 1]
            cur_gt_tids = cur_gt_tids[: k + 1]
            iou3d = iou3d_utils.boxes_iou3d_gpu(cur_roi, cur_gt_boxes)  # (M, N)
            max_overlaps, gt_assignment = torch.max(iou3d, dim=1)
            fg_inds, bg_inds, fg_count, bg_count = self._sample_roi_indices(
                max_overlaps, gt_boxes3d, fg_rois_per_image
            )
            rois, iou_of_rois, gt_boxes_of_rois, gt_tids_of_rois = (
                self._augment_sampled_rois(
                    cur_roi, cur_gt_boxes, cur_gt_tids, gt_assignment, max_overlaps,
                    fg_inds, bg_inds, fg_count, bg_count,
                )
            )
            batch_rois[idx] = rois
            batch_gt_boxes_of_rois[idx] = gt_boxes_of_rois
            batch_roi_iou[idx] = iou_of_rois
            batch_gt_tids_of_rois[idx] = gt_tids_of_rois
        return batch_rois, batch_gt_boxes_of_rois, batch_roi_iou, batch_gt_tids_of_rois

    def sample_bg_inds(self, hard_bg_inds, easy_bg_inds, bg_rois_per_this_image):
        return sample_bg_inds(
            hard_bg_inds, easy_bg_inds, bg_rois_per_this_image,
            cfg.RCNN.HARD_BG_RATIO,
        )

    def aug_roi_by_noise_torch(self, roi_boxes3d, gt_boxes3d, iou3d_src, aug_times=10):
        return aug_roi_by_noise(
            self, roi_boxes3d, gt_boxes3d, iou3d_src, aug_times, cfg, iou3d_utils
        )

    def _forward_train(self, input_dict, point_features):
        rois, gt_of_rois, roi_iou, gt_tids = self.sample_rois_for_rcnn(
            input_dict["roi_boxes3d"], input_dict["gt_boxes3d"], input_dict["gt_tids"]
        )
        pooled, empty_flag = self._pool_rois(input_dict["rpn_xyz"], point_features, rois)
        sampled_pts, sampled_features = pooled[:, :, :, :3], pooled[:, :, :, 3:]
        mask_score = pooled[:, :, :, 3].sum(-1) / cfg.RCNN.NUM_POINTS
        if self.training and cfg.AUG_DATA:
            sampled_pts, rois, gt_of_rois = self.data_augmentation(
                sampled_pts, rois, gt_of_rois
            )
        self._canonicalize_training(sampled_pts, rois, gt_of_rois)
        valid_mask = empty_flag == 0
        reg_valid_mask = ((roi_iou > cfg.RCNN.REG_FG_THRESH) & valid_mask).long()
        cls_label = (roi_iou > cfg.RCNN.CLS_FG_THRESH).long()
        invalid = (roi_iou > cfg.RCNN.CLS_BG_THRESH) & (roi_iou < cfg.RCNN.CLS_FG_THRESH)
        cls_label[valid_mask == 0] = -1
        cls_label[invalid > 0] = -1
        gt_tids[roi_iou < cfg.REID.FG_THRESH] = 0
        output = self._training_output(
            sampled_pts, sampled_features, cls_label, mask_score, reg_valid_mask,
            gt_of_rois, roi_iou, rois, gt_tids,
        )
        return output["pts_input"], output

    def _forward_eval(self, input_dict, point_features):
        rois = input_dict["roi_boxes3d"]
        pooled, _ = self._pool_rois(input_dict["rpn_xyz"], point_features, rois)
        pooled[:, :, :, :3] -= rois[:, :, :3].unsqueeze(dim=2)
        for batch_index in range(rois.shape[0]):
            pooled[batch_index, :, :, :3] = kitti_utils.rotate_pc_along_y_torch(
                pooled[batch_index, :, :, :3], rois[batch_index, :, 6]
            )
        return pooled.view(-1, pooled.shape[2], pooled.shape[3]), None

    def _sample_roi_indices(self, max_overlaps, gt_boxes3d, fg_rois_per_image):
        return sample_roi_indices(
            self, max_overlaps, gt_boxes3d, fg_rois_per_image, cfg
        )

    def _augment_sampled_rois(self, *call_args, **call_kwargs):
        (
            cur_roi,
            cur_gt,
            cur_tids,
            assignment,
            overlaps,
            fg_inds,
            bg_inds,
            fg_count,
            bg_count,
        ) = _bind_named(
            call_args,
            call_kwargs,
            ('cur_roi', 'cur_gt', 'cur_tids', 'assignment', 'overlaps', 'fg_inds', 'bg_inds', 'fg_count', 'bg_count'),
        )
        rois, ious, boxes, tids = [], [], [], []
        for indices, count, aug_times in (
            (fg_inds, fg_count, cfg.RCNN.ROI_FG_AUG_TIMES),
            (bg_inds, bg_count, 1 if cfg.RCNN.ROI_FG_AUG_TIMES > 0 else 0),
        ):
            if count == 0:
                continue
            assigned = assignment[indices]
            gt_boxes = cur_gt[assigned]
            aug_rois, aug_iou = self.aug_roi_by_noise_torch(
                cur_roi[indices], gt_boxes, overlaps[indices], aug_times=aug_times
            )
            rois.append(aug_rois)
            ious.append(aug_iou)
            boxes.append(gt_boxes)
            tids.append(cur_tids[assigned])
        return tuple(torch.cat(values, dim=0) for values in (rois, ious, boxes, tids))
