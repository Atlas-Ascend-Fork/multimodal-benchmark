# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import torch.nn as nn

from jmodt.config import cfg
from jmodt.detection.layers.proposal_layer import ProposalLayer
from jmodt.detection.modeling.backbone import PointNet2MSG
import jmodt.ops.pointnet2.pytorch_utils as pt_utils
import jmodt.utils.loss_utils as loss_utils
from shared.rpn import RpnDependencies, forward_rpn
from shared.rpn import initialize_rpn, initialize_rpn_weights


_DEPENDENCIES = RpnDependencies(
    cfg, PointNet2MSG, ProposalLayer, pt_utils.Conv1d, loss_utils, "pts_xy"
)


class RPN(nn.Module):
    def __init__(self, use_xyz=True, mode="TRAIN"):
        super().__init__()
        initialize_rpn(self, _DEPENDENCIES, use_xyz, mode)

    def init_weights(self):
        initialize_rpn_weights(self, cfg)

    def forward(self, input_data):
        return forward_rpn(self, input_data, _DEPENDENCIES)
