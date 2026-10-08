# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Center-view accessors for FConv provider datasets."""

import numpy as np


class CenterViewMixin:
    rotate_points = None

    def get_center_view_rot_angle(self, index):
        return np.pi / 2 + self.frustum_angle_list[index]

    def get_box3d_center(self, index):
        corners = self.box3d_list[index]
        return (corners[0] + corners[6]) / 2

    def get_center_view_box3d_center(self, index):
        center = self.get_box3d_center(index)
        return self.rotate_points(
            np.expand_dims(center, 0), self.get_center_view_rot_angle(index)
        ).squeeze()

    def get_center_view_box3d(self, index):
        return self.rotate_points(
            np.copy(self.box3d_list[index]), self.get_center_view_rot_angle(index)
        )

    def get_center_view_point_set(self, index):
        return self.get_center_view(self.input_list[index], index)

    def get_center_view(self, point_set, index):
        return self.rotate_points(
            np.copy(point_set), self.get_center_view_rot_angle(index)
        )
