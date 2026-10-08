# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import torch
import torch.nn as nn
import torch.nn.functional as F

from jmodt.config import cfg
from jmodt.detection.layers.proposal_target_layer import ProposalTargetLayer
from jmodt.ops.pointnet2.pointnet2_modules import PointnetSAModule
from jmodt.utils import loss_utils
import jmodt.ops.pointnet2.pytorch_utils as pt_utils
from shared.rcnn import RcnnBase, RcnnDependencies


class RCNN(RcnnBase):
    dependencies = RcnnDependencies(
        cfg, ProposalTargetLayer, PointnetSAModule, pt_utils, loss_utils, True
    )

    def __init__(self, num_classes, input_channels=0, use_xyz=True, mode="TRAIN"):
        super().__init__(num_classes, input_channels, use_xyz, mode)
        self.link_layer = self._reid_head(cfg.REID.LINK_FC)
        self.se_layer = self._reid_head(cfg.REID.SE_FC)
        self.init_weights(weight_init="xavier")

    @staticmethod
    def get_unique_tid_feature(prev_fg_tid: torch.Tensor, prev_fg_feat: torch.Tensor):
        prev_tid_diff = torch.min(prev_fg_tid)
        prev_fg_tid_clip = (prev_fg_tid - prev_tid_diff).long()
        matrix = prev_fg_tid.new_zeros(
            torch.max(prev_fg_tid_clip) + 1, len(prev_fg_tid)
        )
        matrix[prev_fg_tid_clip, torch.arange(len(prev_fg_tid))] = 1
        matrix = F.normalize(matrix, p=1, dim=1)
        prev_tid_feat_mean = torch.mm(matrix, prev_fg_feat)
        unique_clipped_ids = torch.unique(prev_fg_tid_clip)
        unique_features = prev_tid_feat_mean[unique_clipped_ids]
        return unique_clipped_ids + prev_tid_diff, unique_features

    @staticmethod
    def _reid_frame_values(gt_tids, input_features):
        num_frames = gt_tids.shape[0]
        tids = (
            gt_tids[range(0, num_frames, 2)],
            gt_tids[range(1, num_frames, 2)],
        )
        feature_length = input_features.shape[-2]
        features = input_features.view(num_frames, -1, feature_length)
        features = (
            features[range(0, num_frames, 2)],
            features[range(1, num_frames, 2)],
        )
        return num_frames, feature_length, tids, features

    def forward(self, input_data):
        """Run RCNN detection and optional training-time re-identification."""
        points, target_dict = self._prepare_targets(input_data)
        features = self._encoded_features(points)
        result = self._detection_outputs(features)
        if self.mode == "TRAIN":
            result.update(target_dict)
            self._add_reid_outputs(result, target_dict.get("gt_tids"), features)
        else:
            result["rcnn_feat"] = features
        return result

    def _reid_head(self, hidden_channels):
        layers = self._head(
            self._sa_output_channels, hidden_channels, 1, cfg.REID.USE_BN
        )
        if cfg.REID.DP_RATIO >= 0:
            layers.insert(1, nn.Dropout(cfg.RCNN.DP_RATIO))
        return nn.Sequential(*layers)

    def _prepare_targets(self, input_data):
        if cfg.RCNN.ROI_SAMPLE_JIT:
            with torch.no_grad():
                return self.proposal_target_layer(input_data)
        points = input_data.get("pts_input")
        target_dict = {
            "pts_input": points,
            "roi_boxes3d": input_data.get("roi_boxes3d"),
        }
        if self.training:
            target_dict.update(
                {
                    "cls_label": input_data.get("cls_label"),
                    "reg_valid_mask": input_data.get("reg_valid_mask"),
                    "gt_of_rois": input_data.get("gt_boxes3d"),
                }
            )
        return points, target_dict

    def _append_reid_pair(self, prev_tid, next_tid, prev_feat, next_feat, result):
        prev_tid, prev_feat = self.get_unique_tid_feature(prev_tid, prev_feat)
        next_tid, next_feat = self.get_unique_tid_feature(next_tid, next_feat)
        link = (prev_tid.unsqueeze(1) == next_tid).float()
        result["gt_links"].append(link.view(-1))
        differences = torch.abs(
            prev_feat.unsqueeze(1).repeat(1, len(next_tid), 1)
            - next_feat.unsqueeze(0).repeat(len(prev_tid), 1, 1)
        )
        link_features = differences.view(len(prev_tid) * len(next_tid), -1, 1)
        link_scores = self.link_layer(link_features).view(len(prev_tid), len(next_tid))
        link_scores = (
            torch.softmax(link_scores, dim=1) + torch.softmax(link_scores, dim=0)
        ) / 2
        result["rcnn_link"].append(link_scores.view(-1, 1))
        result["gt_starts"].append(1 - link.sum(0))
        result["gt_ends"].append(1 - link.sum(1))
        result["start_features"].append(differences.mean(dim=0))
        result["end_features"].append(differences.mean(dim=1))

    def _store_reid_results(self, result, gt_tids, values):
        for label_key, score_key, feature_key in (
            ("gt_links", "rcnn_link", None),
            ("gt_starts", "rcnn_start", "start_features"),
            ("gt_ends", "rcnn_end", "end_features"),
        ):
            if values[label_key]:
                result[label_key] = torch.cat(values[label_key])
                if feature_key is None:
                    result[score_key] = torch.cat(values[score_key])
                else:
                    features = torch.cat(values[feature_key]).unsqueeze(-1)
                    result[score_key] = self.se_layer(features).squeeze(-1)
            else:
                result[label_key] = gt_tids.new(0)
                result[score_key] = gt_tids.new(0, 1)

    def _add_reid_outputs(self, result, gt_tids, input_features):
        num_frames, _, tids, features = self._reid_frame_values(
            gt_tids, input_features
        )
        value_names = (
            "gt_links", "rcnn_link", "gt_starts", "gt_ends",
            "start_features", "end_features",
        )
        values = {key: [] for key in value_names}
        for index in range(num_frames // 2):
            prev_mask, next_mask = tids[0][index] > 0, tids[1][index] > 0
            prev_tid, next_tid = tids[0][index][prev_mask], tids[1][index][next_mask]
            prev_feat = features[0][index][prev_mask]
            next_feat = features[1][index][next_mask]
            if len(prev_feat) > 0 and len(next_feat) > 0:
                self._append_reid_pair(
                    prev_tid, next_tid, prev_feat, next_feat, values
                )
        self._store_reid_results(result, gt_tids, values)
