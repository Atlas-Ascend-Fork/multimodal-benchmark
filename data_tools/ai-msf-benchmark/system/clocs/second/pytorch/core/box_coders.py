# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


import torch

from second.core.box_coders import GroundBox3dCoder, BevBoxCoder
from second.pytorch.core import box_torch_ops


class GroundBox3dCoderTorch(GroundBox3dCoder):
    def encode_torch(self, boxes, anchors):
        return box_torch_ops.second_box_encode(
            boxes, anchors, self.vec_encode, self.linear_dim
        )

    def decode_torch(self, boxes, anchors):
        return box_torch_ops.second_box_decode(
            boxes, anchors, self.vec_encode, self.linear_dim
        )


class BevBoxCoderTorch(BevBoxCoder):
    def encode_torch(self, boxes, anchors):
        anchors = anchors[..., [0, 1, 3, 4, 6]]
        boxes = boxes[..., [0, 1, 3, 4, 6]]
        return box_torch_ops.bev_box_encode(
            boxes, anchors, self.vec_encode, self.linear_dim
        )

    def decode_torch(self, encodings, anchors):
        anchors = anchors[..., [0, 1, 3, 4, 6]]
        ret = box_torch_ops.bev_box_decode(
            encodings, anchors, self.vec_encode, self.linear_dim
        )
        z_fixed = torch.full(
            [*ret.shape[:-1], 1], self.z_fixed, dtype=ret.dtype, device=ret.device
        )
        h_fixed = torch.full(
            [*ret.shape[:-1], 1], self.h_fixed, dtype=ret.dtype, device=ret.device
        )
        return torch.cat(
            [ret[..., :2], z_fixed, ret[..., 2:4], h_fixed, ret[..., 4:]], dim=-1
        )
