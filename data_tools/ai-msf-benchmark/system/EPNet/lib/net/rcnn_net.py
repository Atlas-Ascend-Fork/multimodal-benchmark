# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import torch

from lib.config import cfg
from lib.rpn.proposal_target_layer import ProposalTargetLayer
from pointnet2_lib.pointnet2.pointnet2_modules import PointnetSAModule
import lib.utils.kitti_utils as kitti_utils
import lib.utils.loss_utils as loss_utils
import lib.utils.roipool3d.roipool3d_utils as roipool3d_utils
import pointnet2_lib.pointnet2.pytorch_utils as pt_utils
from shared.rcnn import RcnnBase, RcnnDependencies


class RCNNNet(RcnnBase):
    dependencies = RcnnDependencies(
        cfg, ProposalTargetLayer, PointnetSAModule, pt_utils, loss_utils, False
    )

    def __init__(self, num_classes, input_channels=0, use_xyz=True):
        super().__init__(num_classes, input_channels, use_xyz)
        self.init_weights(weight_init="xavier")

    def forward(self, input_data):
        """Run RCNN prediction and attach training targets when applicable."""
        points, target_dict = self._prepare_inputs(input_data)
        result = self._detection_outputs(self._encoded_features(points))
        if self.training:
            result.update(target_dict)
        return result

    def _jit_inputs(self, input_data):
        if self.training:
            with torch.no_grad():
                target_dict = self.proposal_target_layer(input_data)
            points = torch.cat(
                (target_dict["sampled_pts"], target_dict["pts_feature"]), dim=2
            )
            target_dict["pts_input"] = points
            return points, target_dict
        rpn_xyz = input_data["rpn_xyz"]
        extra_features = [input_data["seg_mask"].unsqueeze(dim=2)]
        if cfg.RCNN.USE_INTENSITY:
            extra_features.insert(0, input_data["rpn_intensity"].unsqueeze(dim=2))
        if cfg.RCNN.USE_DEPTH:
            extra_features.append((input_data["pts_depth"] / 70.0 - 0.5).unsqueeze(2))
        point_features = torch.cat(
            (torch.cat(extra_features, dim=2), input_data["rpn_features"]), dim=2
        )
        rois = input_data["roi_boxes3d"]
        pooled, _ = roipool3d_utils.roipool3d_gpu(
            rpn_xyz,
            point_features,
            rois,
            cfg.RCNN.POOL_EXTRA_WIDTH,
            sampled_pt_num=cfg.RCNN.NUM_POINTS,
        )
        pooled[:, :, :, :3] -= rois[:, :, :3].unsqueeze(dim=2)
        for index in range(rois.shape[0]):
            pooled[index, :, :, :3] = kitti_utils.rotate_pc_along_y_torch(
                pooled[index, :, :, :3], rois[index, :, 6]
            )
        return pooled.view(-1, pooled.shape[2], pooled.shape[3]), None

    def _prepare_inputs(self, input_data):
        if cfg.RCNN.ROI_SAMPLE_JIT:
            return self._jit_inputs(input_data)
        target_dict = {
            "pts_input": input_data["pts_input"],
            "roi_boxes3d": input_data["roi_boxes3d"],
        }
        if self.training:
            target_dict.update(
                {
                    "cls_label": input_data["cls_label"],
                    "reg_valid_mask": input_data["reg_valid_mask"],
                    "gt_of_rois": input_data["gt_boxes3d_ct"],
                }
            )
        return input_data["pts_input"], target_dict
