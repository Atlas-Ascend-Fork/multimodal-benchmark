# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Proposal selection for compatible RPN implementations."""

from typing import NamedTuple

import torch
import torch.nn as nn


class ProposalDependencies(NamedTuple):
    config: object
    decode_bbox_target: object
    iou3d_utils: object
    kitti_utils: object
    distance_config: str
    skip_empty_near: bool


class ProposalLayerBase(nn.Module):
    """Decode RPN output and apply the configured non-maximum suppression."""

    dependencies = None

    def __init__(self, mode="TRAIN"):
        super().__init__()
        self.mode = mode
        config = self.dependencies.config
        self.mean_size = torch.from_numpy(config.CLS_MEAN_SIZE[0]).cuda()

    def forward(self, rpn_scores, rpn_reg, xyz):
        config = self.dependencies.config
        batch_size = xyz.shape[0]
        proposals = self.dependencies.decode_bbox_target(
            xyz.view(-1, 3),
            rpn_reg.view(-1, rpn_reg.shape[-1]),
            anchor_size=self.mean_size,
            loc_scope=config.RPN.LOC_SCOPE,
            loc_bin_size=config.RPN.LOC_BIN_SIZE,
            num_head_bin=config.RPN.NUM_HEAD_BIN,
            get_xz_fine=config.RPN.LOC_XZ_FINE,
            get_y_by_bin=False,
            get_ry_fine=False,
        )
        proposals[:, 1] += proposals[:, 3] / 2
        proposals = proposals.view(batch_size, -1, 7)
        _, sorted_indices = torch.sort(rpn_scores, dim=1, descending=True)

        post_nms = config[self.mode].RPN_POST_NMS_TOP_N
        result_boxes = rpn_scores.new(batch_size, post_nms, 7).zero_()
        result_scores = rpn_scores.new(batch_size, post_nms).zero_()
        use_distance = getattr(
            config, self.dependencies.distance_config
        ).RPN_DISTANCE_BASED_PROPOSE
        proposal_method = (
            self.distance_based_proposal if use_distance else self.score_based_proposal
        )
        for batch_index in range(batch_size):
            scores, boxes = proposal_method(
                rpn_scores[batch_index],
                proposals[batch_index],
                sorted_indices[batch_index],
            )
            proposal_count = boxes.size(0)
            result_boxes[batch_index, :proposal_count] = boxes
            result_scores[batch_index, :proposal_count] = scores
        return result_boxes, result_scores

    def distance_based_proposal(self, scores, proposals, order):
        """Propose ROIs in near and far distance ranges."""
        config = self.dependencies.config
        pre_total = config[self.mode].RPN_PRE_NMS_TOP_N
        post_total = config[self.mode].RPN_POST_NMS_TOP_N
        pre_limits = (0, int(pre_total * 0.7), pre_total - int(pre_total * 0.7))
        post_limits = (
            0,
            int(post_total * 0.7),
            post_total - int(post_total * 0.7),
        )
        ranges = (0, 40.0, 80.0)
        ordered_scores = scores[order]
        ordered_proposals = proposals[order]
        distances = ordered_proposals[:, 2]
        near_mask = (distances > ranges[0]) & (distances <= ranges[1])
        selected_scores, selected_proposals = [], []

        def range_candidates(index, mask):
            if mask.sum() != 0:
                return (
                    ordered_scores[mask][:pre_limits[index]],
                    ordered_proposals[mask][:pre_limits[index]],
                )
            if index == 1:
                if self.dependencies.skip_empty_near:
                    return None, None
                raise AssertionError("%d" % index)
            start = pre_limits[index - 1]
            return (
                ordered_scores[near_mask][start:][:pre_limits[index]],
                ordered_proposals[near_mask][start:][:pre_limits[index]],
            )

        for index in range(1, len(ranges)):
            mask = (distances > ranges[index - 1]) & (distances <= ranges[index])
            current_scores, current_proposals = range_candidates(index, mask)
            if current_scores is None:
                continue
            keep = self._nms(current_proposals, current_scores)[:post_limits[index]]
            selected_scores.append(current_scores[keep])
            selected_proposals.append(current_proposals[keep])
        return torch.cat(selected_scores, dim=0), torch.cat(selected_proposals, dim=0)

    def score_based_proposal(self, scores, proposals, order):
        config = self.dependencies.config
        pre_nms = config[self.mode].RPN_PRE_NMS_TOP_N
        ordered_scores = scores[order][:pre_nms]
        ordered_proposals = proposals[order][:pre_nms]
        boxes_bev = self.dependencies.kitti_utils.boxes3d_to_bev_torch(
            ordered_proposals
        )
        keep = self.dependencies.iou3d_utils.nms_gpu(
            boxes_bev, ordered_scores, config[self.mode].RPN_NMS_THRESH
        )
        keep = keep[:config[self.mode].RPN_POST_NMS_TOP_N]
        return ordered_scores[keep], ordered_proposals[keep]

    def _nms(self, proposals, scores):
        config = self.dependencies.config
        boxes_bev = self.dependencies.kitti_utils.boxes3d_to_bev_torch(proposals)
        if config.RPN.NMS_TYPE == "rotate":
            operation = self.dependencies.iou3d_utils.nms_gpu
        elif config.RPN.NMS_TYPE == "normal":
            operation = self.dependencies.iou3d_utils.nms_normal_gpu
        else:
            raise NotImplementedError
        return operation(boxes_bev, scores, config[self.mode].RPN_NMS_THRESH)
