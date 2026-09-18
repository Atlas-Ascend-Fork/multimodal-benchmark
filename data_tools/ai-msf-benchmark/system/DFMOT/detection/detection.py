# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Author: wangxy

import numpy as np


class Detection2D(object):
    def __init__(self, tlwh):
        """
        :param tlwh:  top_left x   top_left y    width   height
        :param additional_info:
        """
        self.tlwh = np.asarray(tlwh, dtype=np.float)

    def to_x1y1x2y2(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def to_xyah(self):
        """
        Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret


class Detection3DFusion(object):
    def __init__(self, b_box_3_d, additional_info):
        self.bbox = np.asarray(b_box_3_d, dtype=np.float)
        self.additional_info = np.asarray(additional_info, dtype=np.float32)


class Detection3DOnly(object):
    def __init__(self, b_box_3_d, additional_info):
        self.bbox = np.asarray(b_box_3_d, dtype=np.float)
        self.additional_info = np.asarray(additional_info, dtype=np.float32)
