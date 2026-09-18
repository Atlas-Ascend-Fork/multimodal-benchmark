# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import os

import cv2
import kitti_utils
import numpy as np
import torch.utils.data as torch_data
from lib.datasets.generation_utils import (
    filter_lidar_sample, image_shape, lidar_points, load_lidar_sample, populate_point_sample,
    sample_point_indices,
)

_HELPER_CONTINUE = object()

USE_INTENSITY = False


def _kittidataset_getitem_45_part_1(_state):
    _state["sample_id"] = int(_state["self"].sample_id_list[_state["index"]])
    (_state["calib"], _state["img_shape"], _state["pts_rect"],
     _state["pts_intensity"]) = load_lidar_sample(_state["self"], _state["sample_id"])
    filtered = filter_lidar_sample(
        _state["self"], _state["calib"], _state["img_shape"],
        _state["pts_rect"], _state["pts_intensity"],
    )
    _state["pts_rect"], _state["pts_intensity"] = filtered[:2]
    _state["choice"] = sample_point_indices(_state["pts_rect"], _state["self"].npoints)
    _state["ret_pts_rect"] = _state["pts_rect"][_state["choice"], :]
    _state["ret_pts_intensity"] = _state["pts_intensity"][_state["choice"]] - 0.5


def _kittidataset_getitem_45_part_2(_state):
    _state["pts_features"] = [_state["ret_pts_intensity"].reshape(-1, 1)]
    _state["ret_pts_features"] = (
        np.concatenate(_state["pts_features"], axis=1)
        if _state["pts_features"].__len__() > 1
        else _state["pts_features"][0]
    )
    _state["sample_info"] = {"sample_id": _state["sample_id"]}
    if _state["self"].mode == "TEST":
        return populate_point_sample(
            _state["sample_info"], _state["ret_pts_rect"],
            _state["ret_pts_features"], USE_INTENSITY,
        )
    _state["gt_obj_list"] = _state["self"].filtrate_objects(
        _state["self"].get_label(_state["sample_id"])
    )
    _state["gt_boxes3d"] = kitti_utils.objs_to_boxes3d(_state["gt_obj_list"])
    if USE_INTENSITY:
        _state["pts_input"] = np.concatenate(
            (_state["ret_pts_rect"], _state["ret_pts_features"]), axis=1
        )
    else:
        _state["pts_input"] = _state["ret_pts_rect"]
    _state["cls_labels"] = _state["self"].generate_training_labels(
        _state["ret_pts_rect"], _state["gt_boxes3d"]
    )
    _state["sample_info"]["pts_input"] = _state["pts_input"]
    _state["sample_info"]["pts_rect"] = _state["ret_pts_rect"]
    _state["sample_info"]["cls_labels"] = _state["cls_labels"]
    return _HELPER_CONTINUE


class KittiDataset(torch_data.Dataset):
    def __init__(self, root_dir, split="train", mode="TRAIN"):
        self.split = split
        self.mode = mode
        self.classes = ["Car"]
        is_test = self.split == "test"
        self.imageset_dir = os.path.join(
            root_dir, "KITTI", "object", "testing" if is_test else "training"
        )

        split_dir = os.path.join(root_dir, "KITTI", "ImageSets", split + ".txt")
        with open(split_dir, encoding="utf-8") as split_file:
            self.image_idx_list = [line.strip() for line in split_file]
        self.sample_id_list = [int(sample_id) for sample_id in self.image_idx_list]
        self.num_sample = self.image_idx_list.__len__()

        self.npoints = 16384

        self.image_dir = os.path.join(self.imageset_dir, "image_2")
        self.lidar_dir = os.path.join(self.imageset_dir, "velodyne")
        self.calib_dir = os.path.join(self.imageset_dir, "calib")
        self.label_dir = os.path.join(self.imageset_dir, "label_2")
        self.plane_dir = os.path.join(self.imageset_dir, "planes")

    def __len__(self):
        return len(self.sample_id_list)

    def __getitem__(self, index):
        _state = locals()
        _kittidataset_getitem_45_part_1(_state)
        _helper_result = _kittidataset_getitem_45_part_2(_state)
        if _helper_result is not _HELPER_CONTINUE:
            return _helper_result
        return _state["sample_info"]

    @staticmethod
    def get_valid_flag(pts_rect, pts_img, pts_rect_depth, img_shape):
        val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
        val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
        pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)
        return pts_valid_flag

    @staticmethod
    def generate_training_labels(pts_rect, gt_boxes3d):
        cls_label = np.zeros((pts_rect.shape[0]), dtype=np.int32)
        gt_corners = kitti_utils.boxes3d_to_corners3d(gt_boxes3d, rotate=True)
        extend_gt_boxes3d = kitti_utils.enlarge_box3d(gt_boxes3d, extra_width=0.2)
        extend_gt_corners = kitti_utils.boxes3d_to_corners3d(
            extend_gt_boxes3d, rotate=True
        )
        for k in range(gt_boxes3d.shape[0]):
            box_corners = gt_corners[k]
            fg_pt_flag = kitti_utils.in_hull(pts_rect, box_corners)
            cls_label[fg_pt_flag] = 1

            # enlarge the bbox3d, ignore nearby points
            extend_box_corners = extend_gt_corners[k]
            fg_enlarge_flag = kitti_utils.in_hull(pts_rect, extend_box_corners)
            ignore_flag = np.logical_xor(fg_pt_flag, fg_enlarge_flag)
            cls_label[ignore_flag] = -1

        return cls_label

    def get_image(self, idx):
        img_file = os.path.join(self.image_dir, "%06d.png" % idx)
        if not (os.path.exists(img_file)):
            raise AssertionError
        return cv2.imread(img_file)  # (H, W, 3) BGR mode

    def get_image_shape(self, idx):
        return image_shape(self.image_dir, idx)

    def get_lidar(self, idx):
        return lidar_points(self.lidar_dir, idx)

    def get_calib(self, idx):
        calib_file = os.path.join(self.calib_dir, "%06d.txt" % idx)
        if not (os.path.exists(calib_file)):
            raise AssertionError
        return kitti_utils.Calibration(calib_file)

    def get_label(self, idx):
        label_file = os.path.join(self.label_dir, "%06d.txt" % idx)
        if not (os.path.exists(label_file)):
            raise AssertionError
        return kitti_utils.get_objects_from_label(label_file)

    def filtrate_objects(self, obj_list):
        type_whitelist = self.classes
        if self.mode == "TRAIN":
            type_whitelist = list(self.classes)
            if "Car" in self.classes:
                type_whitelist.append("Van")

        valid_obj_list = []
        for obj in obj_list:
            if obj.cls_type not in type_whitelist:
                continue

            valid_obj_list.append(obj)
        return valid_obj_list

    def collate_batch(self, batch):
        batch_size = batch.__len__()
        ans_dict = {}

        for key in batch[0].keys():
            if isinstance(batch[0][key], np.ndarray):
                ans_dict[key] = np.concatenate(
                    [batch[k][key][np.newaxis, ...] for k in range(batch_size)], axis=0
                )

            else:
                ans_dict[key] = [batch[k][key] for k in range(batch_size)]
                if isinstance(batch[0][key], int):
                    ans_dict[key] = np.array(ans_dict[key], dtype=np.int32)
                elif isinstance(batch[0][key], float):
                    ans_dict[key] = np.array(ans_dict[key], dtype=np.float32)

        return ans_dict
