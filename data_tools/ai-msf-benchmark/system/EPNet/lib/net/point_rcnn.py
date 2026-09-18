# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import logging

from lib.config import cfg
from lib.net.rcnn_net import RCNNNet
from lib.net.rpn import RPN
import torch
import torch.nn as nn

LOGGER = logging.getLogger(__name__)


class PointRCNN(nn.Module):
    def __init__(self, num_classes, use_xyz=True, mode="TRAIN"):
        super().__init__()

        if not (cfg.RPN.ENABLED or cfg.RCNN.ENABLED):
            raise AssertionError

        if cfg.RPN.ENABLED:
            self.rpn = RPN(use_xyz=use_xyz, mode=mode)

        if cfg.RCNN.ENABLED:
            rcnn_input_channels = 128  # channels of rpn features
            if cfg.RCNN.BACKBONE == "pointnet":
                self.rcnn_net = RCNNNet(
                    num_classes=num_classes,
                    input_channels=rcnn_input_channels,
                    use_xyz=use_xyz,
                )
            elif cfg.RCNN.BACKBONE == "pointsift":
                pass
            else:
                raise NotImplementedError

    def run_rcnn_stage(
        self, input_data, output, rpn_output, backbone_xyz, backbone_features
    ):
        with torch.no_grad():
            rpn_cls, rpn_reg = rpn_output["rpn_cls"], rpn_output["rpn_reg"]
            rpn_scores_raw = rpn_cls[:, :, 0]
            rpn_scores_norm = torch.sigmoid(rpn_scores_raw)
            seg_mask = (rpn_scores_norm > cfg.RPN.SCORE_THRESH).float()
            pts_depth = torch.norm(backbone_xyz, p=2, dim=2)
            rois, roi_scores_raw = self.rpn.proposal_layer(
                rpn_scores_raw, rpn_reg, backbone_xyz
            )
            output.update({
                "rois": rois,
                "roi_scores_raw": roi_scores_raw,
                "seg_result": seg_mask,
            })
        rcnn_input_info = {
            "rpn_xyz": backbone_xyz,
            "rpn_features": backbone_features.permute((0, 2, 1)),
            "seg_mask": seg_mask,
            "roi_boxes3d": rois,
            "pts_depth": pts_depth,
        }
        if self.training:
            rcnn_input_info["gt_boxes3d"] = input_data["gt_boxes3d"]
        output.update(self.rcnn_net(rcnn_input_info))

    def forward(self, input_data):
        LOGGER.info("%s", "==========%%%%%%%%================")
        if cfg.RPN.ENABLED:
            output = {}
            # rpn inference
            with torch.set_grad_enabled((not cfg.RPN.FIXED) and self.training):
                if cfg.RPN.FIXED:
                    self.rpn.eval()
                rpn_output = self.rpn(input_data)
                output.update(rpn_output)
                backbone_xyz = rpn_output["backbone_xyz"]
                backbone_features = rpn_output["backbone_features"]
                LOGGER.info(
                    "%s %s",
                    "==========%%%%%%%%================",
                    backbone_features.shape,
                )
            # rcnn inference
            if cfg.RCNN.ENABLED:
                self.run_rcnn_stage(
                    input_data, output, rpn_output, backbone_xyz, backbone_features
                )

        elif cfg.RCNN.ENABLED:
            output = self.rcnn_net(input_data)
        else:
            raise NotImplementedError

        return output

    def forward_features(self, input_data):
        output = {}
        # rpn inference
        with torch.set_grad_enabled((not cfg.RPN.FIXED) and self.training):
            if cfg.RPN.FIXED:
                self.rpn.eval()
            rpn_output = self.rpn(input_data)
            output.update(rpn_output)
            _ = rpn_output["backbone_xyz"]
            backbone_features = rpn_output["backbone_features"]
            return backbone_features

    def forward_with_grad(self, input_data):
        if cfg.RPN.ENABLED:
            output = {}
            # rpn inference
            if cfg.RPN.FIXED:
                self.rpn.eval()
            with torch.no_grad():
                rpn_output = self.rpn(input_data)
                output.update(rpn_output)
                backbone_xyz = rpn_output["backbone_xyz"]
                backbone_features = rpn_output["backbone_features"]

            # rcnn inference
            if cfg.RCNN.ENABLED:
                self.run_rcnn_stage(
                    input_data, output, rpn_output, backbone_xyz, backbone_features
                )

        elif cfg.RCNN.ENABLED:
            output = self.rcnn_net(input_data)
        else:
            raise NotImplementedError

        return output


if __name__ == "__main__":
    pass
