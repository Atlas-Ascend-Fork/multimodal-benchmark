# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import argparse
import logging
import os

from lib.config import cfg
from lib.datasets.kitti_dataset import KittiDataset
from lib.datasets.kitti_rcnn_dataset import interpolate_img_by_xy
import _init_path  # noqa: F401
import numpy as np
from lib.datasets.generation_utils import DatasetGeneratorMixin


# Created by silver at 2019/9/16 15:21
# Email: xiwuchencn[at]gmail[dot]com


parser = argparse.ArgumentParser()
parser.add_argument("--save_dir", type=str, default="./gt_database")
parser.add_argument("--class_name", type=str, default="Car")
parser.add_argument("--split", type=str, default="train")
args = parser.parse_args()


LOGGER = logging.getLogger(__name__)


class GTDatabaseGenerator(DatasetGeneratorMixin, KittiDataset):

    def __init__(self, root_dir, split="train", classes=args.class_name):
        super().__init__(root_dir, split=split)
        self.configure_classes(classes)
        self.configure_point_filter(cfg.PC_REDUCE_BY_RANGE, cfg.PC_AREA_SCOPE)
        self.velodyne_rgb_dir = os.path.join(
            root_dir, "KITTI/object/training/velodyne_rgb"
        )
        os.makedirs(self.velodyne_rgb_dir, exist_ok=True)

    def vis_img(self):
        _ = []
        for _, sample_id in enumerate(self.image_idx_list):
            sample_id = int(sample_id)
            LOGGER.info("%s", "process gt sample (id=%06d)" % sample_id)

            pts_lidar = self.get_lidar(sample_id)
            calib = self.get_calib(sample_id)
            pts_rect = calib.lidar_to_rect(pts_lidar[:, 0:3])
            _ = pts_lidar[:, 3]

            img = self.get_image_rgb_with_normal(sample_id)
            img = ((img * self.std + self.mean) * 255)[:, :, ::-1]
            img = np.ascontiguousarray(img)
            pts_img, pts_depth = calib.rect_to_img(pts_rect)
            img_shape = self.get_image_shape(sample_id)
            pts_valid_flag = self.get_valid_flag(
                pts_rect, pts_img, pts_depth, img_shape
            )
            pts_img = pts_img[pts_valid_flag][:, :]
            shape = self.get_image_shape_with_padding(sample_id)
            shape = np.array([shape[1], shape[0]])

            cur_pts_rgb = interpolate_img_by_xy(img, pts_img, shape)

            img = img.astype(np.uint8)

            #
            img_mask = np.zeros_like(img).astype(np.uint8)

            cur_pts_rgb = ((cur_pts_rgb)).astype(np.uint8)
            pts_img = np.around(pts_img).astype(np.int)
            for i in range(pts_img.shape[0]):
                LOGGER.info(
                    "%s %s", img[pts_img[i][1], pts_img[i][0], :], cur_pts_rgb[i, :]
                )
                img_mask[pts_img[i][1], pts_img[i][0], :] = cur_pts_rgb[i, :]

            input()

            #
            #
            #
            # # save


if __name__ == "__main__":
    #
    # # args.split=
    dataset = GTDatabaseGenerator(root_dir="../../data/", split="train")
    dataset.vis_img()
