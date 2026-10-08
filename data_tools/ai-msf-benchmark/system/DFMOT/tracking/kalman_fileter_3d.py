# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Author: wangxy
# The code refers to https://github.com/xinshuoweng/AB3DMOT

import numpy as np
from shared.kalman import initialize_box_filter


class KalmanBoxTracker(object):
    count = 0

    def __init__(self, bbox3_d):
        """
        Initialises a tracker using initial bounding box.
        """
        self.kf = initialize_box_filter(bbox3_d)
        self.history = []
        self.still_first = True

    def update(self, bbox3_d):
        """
        Updates the state vector with observed bbox.
        """
        self.history = []
        # ######################### orientation correction
        if self.kf.x[3] >= np.pi:
            self.kf.x[3] -= np.pi * 2  # make the theta still in the range
        if self.kf.x[3] < -np.pi:
            self.kf.x[3] += np.pi * 2

        new_theta = bbox3_d[3]
        if new_theta >= np.pi:
            new_theta -= np.pi * 2  # make the theta still in the range
        if new_theta < -np.pi:
            new_theta += np.pi * 2
        bbox3_d[3] = new_theta

        predicted_theta = self.kf.x[3]
        if (
            abs(new_theta - predicted_theta) > np.pi / 2.0
            and abs(new_theta - predicted_theta) < np.pi * 3 / 2.0
        ):  # if the angle of two theta is not acute angle
            self.kf.x[3] += np.pi
            if self.kf.x[3] > np.pi:
                self.kf.x[3] -= np.pi * 2  # make the theta still in the range
            if self.kf.x[3] < -np.pi:
                self.kf.x[3] += np.pi * 2

        # now the angle is acute: < 90 or > 270, convert the case of > 270 to < 90
        if abs(new_theta - self.kf.x[3]) >= np.pi * 3 / 2.0:
            if new_theta > 0:
                self.kf.x[3] += np.pi * 2
            else:
                self.kf.x[3] -= np.pi * 2

        # # flip

        self.kf.update(bbox3_d)

        if self.kf.x[3] >= np.pi:
            self.kf.x[3] -= np.pi * 2  # make the theta still in the rage
        if self.kf.x[3] < -np.pi:
            self.kf.x[3] += np.pi * 2

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        self.kf.predict()
        if self.kf.x[3] >= np.pi:
            self.kf.x[3] -= np.pi * 2
        if self.kf.x[3] < -np.pi:
            self.kf.x[3] += np.pi * 2
        self.history.append(self.kf.x)
        pose = self.history[-1].tolist()
        pose = np.concatenate(pose[:7], axis=0)
        return pose

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return self.kf.x[:7].reshape((7,))
