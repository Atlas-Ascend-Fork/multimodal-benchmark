# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from lib.config import cfg
from lib.utils.bbox_transform import decode_bbox_target
import lib.utils.iou3d.iou3d_utils as iou3d_utils
import lib.utils.kitti_utils as kitti_utils
from shared.proposal_layer import ProposalDependencies, ProposalLayerBase


class ProposalLayer(ProposalLayerBase):
    dependencies = ProposalDependencies(
        cfg, decode_bbox_target, iou3d_utils, kitti_utils, "TEST", False
    )
