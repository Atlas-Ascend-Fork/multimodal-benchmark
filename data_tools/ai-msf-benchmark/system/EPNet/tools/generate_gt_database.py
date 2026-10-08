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
import pickle

from lib.datasets.kitti_dataset import KittiDataset
from lib.datasets.kitti_rcnn_dataset import interpolate_img_by_xy
import _init_path  # noqa: F401
import lib.utils.roipool3d.roipool3d_utils as roipool3d_utils
import numpy as np
import torch
from lib.datasets.generation_utils import DatasetGeneratorMixin

LOGGER = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--save_dir", type=str, default="./gt_database")
parser.add_argument("--class_name", type=str, default="Car")
parser.add_argument("--split", type=str, default="train")
args = parser.parse_args()


class GTDatabaseGenerator(DatasetGeneratorMixin, KittiDataset):
    def __init__(self, root_dir, split="train", classes=args.class_name):
        super().__init__(root_dir, split=split)
        self.configure_classes(classes)

    @staticmethod
    def _objects_to_boxes(objects):
        boxes = np.zeros((len(objects), 7), dtype=np.float32)
        for index, obj in enumerate(objects):
            boxes[index] = (*obj.pos, obj.h, obj.w, obj.length_value, obj.ry)
        return boxes

    def generate_gt_database(self):
        gt_database = []
        for sample_id in self.image_idx_list:
            sample_id = int(sample_id)
            LOGGER.info("%s", "process gt sample (id=%06d)" % sample_id)
            gt_database.extend(self._sample_database_entries(sample_id))

        save_file_name = os.path.join(
            args.save_dir,
            "%s_gt_database_3level_%s.pkl" % (args.split, self.classes[-1]),
        )
        with open(save_file_name, "wb") as f:
            pickle.dump(gt_database, f)

        self.gt_database = gt_database
        LOGGER.info(
            "%s", "Save refine training sample info file to %s" % save_file_name
        )

    def _sample_database_entries(self, sample_id):
        lidar = self.get_lidar(sample_id)
        calibration = self.get_calib(sample_id)
        points = calibration.lidar_to_rect(lidar[:, :3])
        image_points, _ = calibration.rect_to_img(points)
        image = self.get_image_rgb_with_normal(sample_id)
        objects = self.filtrate_objects(self.get_label(sample_id))
        boxes = self._objects_to_boxes(objects)
        if boxes.shape[0] == 0:
            LOGGER.info("%s", "No gt object")
            return []
        masks = roipool3d_utils.pts_in_boxes3d_cpu(
            torch.from_numpy(points), torch.from_numpy(boxes)
        )
        shape = self.image_hw_with_padding_np.reshape((1, 1, 2))
        entries = []
        for index, mask in enumerate(masks):
            selected = mask.numpy() == 1
            entries.append(
                {
                    "sample_id": sample_id,
                    "cls_type": objects[index].cls_type,
                    "gt_box3d": boxes[index],
                    "points": points[selected].astype(np.float32),
                    "rgb": interpolate_img_by_xy(
                        image, image_points[selected].astype(np.float32), shape
                    ),
                    "intensity": lidar[:, 3][selected].astype(np.float32),
                    "obj": objects[index],
                }
            )
        return entries


if __name__ == "__main__":
    dataset = GTDatabaseGenerator(root_dir="../../data/", split=args.split)
    os.makedirs(args.save_dir, exist_ok=True)

    dataset.generate_gt_database()
