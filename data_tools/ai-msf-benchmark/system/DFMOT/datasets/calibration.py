# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Author: wangxy

import numpy as np


def inverse_rigid_trans(tr):
    """Inverse a rigid body transform matrix (3x4 as [R|t])
    [R'|-R't; 0|1]
    """
    inv_tr = np.zeros_like(tr)  # 3x4
    inv_tr[0:3, 0:3] = np.transpose(tr[0:3, 0:3])
    inv_tr[0:3, 3] = np.dot(-np.transpose(tr[0:3, 0:3]), tr[0:3, 3])
    return inv_tr


class Calibration(object):
    """Calibration matrices and utils
    3d XYZ in <label>measure are in rect camera coord.
    2d box xy are in image2 coord
    Points in <lidar>.bin are in Velodyne coord.

    y_image2 = P^2_rect * x_rect
    y_image2 = P^2_rect * R0_rect * Tr_velo_to_cam * x_velo
    x_ref = Tr_velo_to_cam * x_velo
    x_rect = R0_rect * x_ref

    P^2_rect = [f^2_u,  0,      c^2_u,  -f^2_u b^2_x;
                0,      f^2_v,  c^2_v,  -f^2_v b^2_y;
                0,      0,      1,      0]
             = K * [1|t]

    image2 coord:
     ----> x-axis (u)
    |
    |
    v y-axis (v)

    velodyne coord:
    front x, left y, up z

    rect/ref camera coord:
    right x, down y, front z

    Ref (KITTI paper): http://www.cvlibs.net/publications/Geiger2013IJRR.pdf

    """

    def __init__(self, calib_filepath):
        with open(calib_filepath) as f:
            self.p0 = np.fromstring(f.readline().split(":")[1], sep=" ").reshape((3, 4))
            self.p1 = np.fromstring(f.readline().split(":")[1], sep=" ").reshape((3, 4))

            # Projection matrix from rectified camera coord to image2/3 coord
            self.p2 = np.fromstring(f.readline().split(":")[1], sep=" ").reshape((3, 4))
            self.p3 = np.fromstring(f.readline().split(":")[1], sep=" ").reshape((3, 4))

            # Rotation from reference camera coord to rectified camera coord
            line = f.readline()
            self.r0_rect = np.fromstring(line[line.index(" "):], sep=" ").reshape(
                (3, 3)
            )

            # Rigid transform from lidar coord to reference camera coord
            line = f.readline()
            self.tr_lidar_to_cam = np.fromstring(
                line[line.index(" "):], sep=" "
            ).reshape(
                (3, 4)
            )  # lidar_to_cam
            self.tr_cam_to_lidar = inverse_rigid_trans(
                self.tr_lidar_to_cam
            )  # cam_to_lidar

            line = f.readline()
            self.tr_imu_to_lidar = np.fromstring(
                line[line.index(" "):], sep=" "
            ).reshape(
                (3, 4)
            )  # imu_to_lidar
            self.tr_lidar_to_imu = inverse_rigid_trans(
                self.tr_imu_to_lidar
            )  # lidar_to_imu

    @staticmethod
    def inverse_rigid_trans(tr):
        """Inverse a rigid body transform matrix (3x4 as [R|t])
        [R'|-R't; 0|1]
        """
        inv_tr = np.zeros_like(tr)  # 3x4
        inv_tr[0:3, 0:3] = np.transpose(tr[0:3, 0:3])
        inv_tr[0:3, 3] = np.dot(-np.transpose(tr[0:3, 0:3]), tr[0:3, 3])
        return inv_tr

    def cart2hom(self, pts_3d):
        """Input: nx3 points in Cartesian
        Oupput: nx4 points in Homogeneous by pending 1
        """
        n = pts_3d.shape[0]
        pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
        return pts_3d_hom
