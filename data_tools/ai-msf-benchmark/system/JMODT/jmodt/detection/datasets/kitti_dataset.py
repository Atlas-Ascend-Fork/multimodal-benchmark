# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import NamedTuple
import logging
import os

from PIL import Image
from torch.utils.data import Dataset
import numpy as np

from jmodt.config import cfg
from jmodt.utils import calibration, kitti_utils
from shared.dataset_geometry import ConfiguredGeometryMixin, filter_objects


from shared.parameters import ParameterSchema as _ParameterSchema, bind_parameters as _bind_parameters


class KittidatasetInitParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    root_dir: object
    npoints: object
    split: object
    classes: object
    mode: object
    logger: object
    challenge: object
    fixed_img_size: object


LOGGER = logging.getLogger(__name__)


_KITTIDATASET_INIT_PARAMETERS_SCHEMA = _ParameterSchema(
    KittidatasetInitParameters,
    (
        "root_dir",
        "npoints",
        "split",
        "classes",
        "mode",
        "logger",
        "challenge",
        "fixed_img_size",
    ),
    {
        "npoints": 16384,
        "split": "train",
        "classes": "Car",
        "mode": "TRAIN",
        "logger": None,
        "challenge": "detection",
        "fixed_img_size": (384, 1280),
    },
    frozenset(()),
    8,
    None,
    None,
)


class KittiDataset(ConfiguredGeometryMixin, Dataset):
    geometry_config = cfg
    geometry_box_utils = kitti_utils

    def __init__(self, *call_args, **call_kwargs):
        bound_parameters = _bind_parameters(
            _KITTIDATASET_INIT_PARAMETERS_SCHEMA, call_args, call_kwargs
        )
        root_dir, npoints, split, classes, mode, logger, challenge, fixed_img_size = (
            bound_parameters
        )
        self.split = split
        is_test = self.split == "test"
        self.challenge = challenge
        self.fixed_img_size = fixed_img_size
        if mode not in ["TRAIN", "EVAL", "TEST"]:
            raise ValueError("Invalid mode: %s" % mode)
        self.mode = mode
        self._set_data_paths(root_dir, split, is_test)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.npoints = npoints
        self.logger = logger
        self._set_classes(classes)
        self._initialize_samples()

    def __len__(self):
        if self.challenge == "detection" or self.mode != "TRAIN":
            return len(self.sample_id_list)
        else:
            return len(self.sample_pair_id_list)

    def __getitem__(self, index):
        if self.challenge == "detection" or self.mode != "TRAIN":
            sample_id = self.sample_id_list[index]
            return self.get_sample_dict(sample_id)
        else:
            prev_sample_id, next_sample_id = self.sample_pair_id_list[index]
            return self.get_sample_dict(prev_sample_id), self.get_sample_dict(
                next_sample_id
            )

    @staticmethod
    def data_augmentation(aug_pts_rect, aug_gt_boxes3d, gt_alpha):
        """
        :param aug_pts_rect: (N, 3)
        :param aug_gt_boxes3d: (N, 7)
        :param gt_alpha: (N)
        :return:
        """
        aug_list = cfg.AUG_METHOD_LIST
        aug_enable = 1 - np.random.rand(3)
        if "rotation" in aug_list and aug_enable[0] < cfg.AUG_METHOD_PROB[0]:
            angle = np.random.uniform(
                -np.pi / cfg.AUG_ROT_RANGE, np.pi / cfg.AUG_ROT_RANGE
            )
            aug_pts_rect = kitti_utils.rotate_pc_along_y(aug_pts_rect, rot_angle=angle)
            # xyz change, hwl unchanged
            aug_gt_boxes3d = kitti_utils.rotate_pc_along_y(
                aug_gt_boxes3d, rot_angle=angle
            )

            # calculate the ry after rotation
            x, z = aug_gt_boxes3d[:, 0], aug_gt_boxes3d[:, 2]
            beta = np.arctan2(z, x)
            new_ry = np.sign(beta) * np.pi / 2 + gt_alpha - beta
            aug_gt_boxes3d[:, 6] = new_ry

        if "scaling" in aug_list and aug_enable[1] < cfg.AUG_METHOD_PROB[1]:
            scale = np.random.uniform(0.95, 1.05)
            aug_pts_rect = aug_pts_rect * scale
            aug_gt_boxes3d[:, 0:6] = aug_gt_boxes3d[:, 0:6] * scale

        if "flip" in aug_list and aug_enable[2] < cfg.AUG_METHOD_PROB[2]:
            # flip horizontal
            aug_pts_rect[:, 0] = -aug_pts_rect[:, 0]
            aug_gt_boxes3d[:, 0] = -aug_gt_boxes3d[:, 0]
            # flip orientation: ry > 0: pi - ry, ry < 0: -pi - ry
            aug_gt_boxes3d[:, 6] = (
                np.sign(aug_gt_boxes3d[:, 6]) * np.pi - aug_gt_boxes3d[:, 6]
            )

        return aug_pts_rect, aug_gt_boxes3d

    @staticmethod
    def _ground_truth_arrays(objects):
        boxes = np.zeros((len(objects), 7), dtype=np.float32)
        alpha = np.zeros(len(objects), dtype=np.float32)
        tids = np.zeros(len(objects), dtype=np.float32)
        for index, obj in enumerate(objects):
            boxes[index] = (
                obj.pos[0], obj.pos[1], obj.pos[2], obj.h,
                obj.w, obj.length_value, obj.ry,
            )
            alpha[index] = obj.alpha
            tids[index] = obj.score
        return boxes, alpha, tids

    @staticmethod
    def _point_input(points, features):
        if cfg.RPN.USE_INTENSITY:
            return np.concatenate((points, features), axis=1)
        return points

    def get_normalized_image(self, idx):
        """
        :return: fixed_size_img (H,W,3)
        """
        img_file = os.path.join(self.image_dir, "%06d.png" % idx)
        if not (os.path.exists(img_file)):
            raise AssertionError
        with Image.open(img_file) as image:
            img = np.array(image.convert("RGB")).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std
        fixed_size_img = np.zeros(
            [self.fixed_img_size[0], self.fixed_img_size[1], 3], dtype=np.float32
        )
        fixed_size_img[: img.shape[0], : img.shape[1], :] = img

        return fixed_size_img

    def get_image_shape(self, idx):
        img_file = os.path.join(self.image_dir, "%06d.png" % idx)
        if not (os.path.exists(img_file)):
            raise AssertionError(f"Path {img_file} does not exist")
        with Image.open(img_file) as image:
            width, height = image.size
        return height, width, 3

    def get_lidar(self, idx):
        lidar_file = os.path.join(self.lidar_dir, "%06d.bin" % idx)
        if not (os.path.exists(lidar_file)):
            raise AssertionError(f"Path {lidar_file} does not exist")
        points = np.fromfile(lidar_file, dtype=np.float32)
        points = points.reshape(-1, 4)
        return points

    def get_calib(self, idx):
        calib_file = os.path.join(self.calib_dir, "%06d.txt" % idx)
        if not (os.path.exists(calib_file)):
            raise AssertionError(f"Path {calib_file} does not exist")
        return calibration.Calibration(calib_file)

    def get_label(self, idx):
        label_file = os.path.join(self.label_dir, "%06d.txt" % idx)
        if not (os.path.exists(label_file)):
            raise AssertionError(f"Path {label_file} does not exist")
        return kitti_utils.get_objects_from_label(label_file)

    def filtrate_objects(self, obj_list):
        return filter_objects(obj_list, self.classes, self.mode, cfg, self.check_pc_range)

    def get_sample_dict(self, sample_id):
        """Load and prepare one point-cloud/image sample."""
        calib = self.get_calib(sample_id)
        img = self.get_normalized_image(sample_id)
        img_shape = self.get_image_shape(sample_id)
        pts_lidar = self.get_lidar(sample_id)
        pts_rect = calib.lidar_to_rect(pts_lidar[:, 0:3])
        pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
        pts_valid_flag = self.get_valid_flag(
            pts_rect, pts_img, pts_rect_depth, img_shape
        )
        pts_rect = pts_rect[pts_valid_flag]
        pts_intensity = pts_lidar[pts_valid_flag, 3]
        pts_xy = pts_img[pts_valid_flag]
        choice = self._sample_point_indices(pts_rect)
        ret_pts_rect = pts_rect[choice, :]
        ret_pts_intensity = pts_intensity[choice] - 0.5
        ret_pts_xy = pts_xy[choice, :]
        ret_pts_xy[:, 0] = ret_pts_xy[:, 0] / (self.fixed_img_size[1] - 1.0) * 2.0 - 1.0
        ret_pts_xy[:, 1] = ret_pts_xy[:, 1] / (self.fixed_img_size[0] - 1.0) * 2.0 - 1.0
        ret_pts_features = ret_pts_intensity.reshape(-1, 1)
        sample_info = {"sample_id": sample_id, "img": img, "pts_xy": ret_pts_xy}
        if self.mode == "TEST":
            sample_info["pts_input"] = self._point_input(
                ret_pts_rect, ret_pts_features
            )
        else:
            sample_info.update(self._training_sample(
                sample_id, ret_pts_rect, ret_pts_features
            ))
        return sample_info

    def collate_batch(self, batch):
        if self.mode != "TRAIN" and cfg.RCNN.ENABLED and not cfg.RPN.ENABLED:
            if len(batch) != 1:
                raise ValueError("RCNN evaluation requires a batch size of one")
            return batch[0]
        if self.challenge == "detection" or self.mode != "TRAIN":
            return self._collate_detection_batch(batch)
        return self._collate_tracking_batch(batch)

    def _set_data_paths(self, root_dir, split, is_test):
        challenge_dir = "object" if self.challenge == "detection" else "tracking_object"
        data_dir = os.path.join(
            root_dir, challenge_dir, "testing" if is_test else "training"
        )
        split_file = os.path.join(root_dir, challenge_dir, "ImageSets", split + ".txt")
        with open(split_file, encoding="utf-8") as file_handle:
            if self.challenge != "detection" and self.mode == "TRAIN":
                self.sample_pair_list = [line.split() for line in file_handle]
            else:
                self.image_idx_list = [line.strip() for line in file_handle]
        self.image_dir = os.path.join(data_dir, "image_2")
        self.lidar_dir = os.path.join(data_dir, "velodyne")
        self.calib_dir = os.path.join(data_dir, "calib")
        self.label_dir = os.path.join(data_dir, "label_2")

    def _set_classes(self, classes):
        class_groups = {
            "Car": ("Background", "Car"),
            "People": ("Background", "Pedestrian", "Cyclist"),
            "Pedestrian": ("Background", "Pedestrian"),
            "Cyclist": ("Background", "Cyclist"),
        }
        if classes not in class_groups:
            raise ValueError("Invalid classes: %s" % classes)
        self.classes = class_groups[classes]
        self.num_class = len(self.classes)

    def _initialize_detection_samples(self):
        if self.mode != "TRAIN":
            self.sample_id_list = [int(sample_id) for sample_id in self.image_idx_list]
            return
        self.logger.info("Loading %s samples from %s ..." % (self.mode, self.label_dir))
        self.sample_id_list = []
        for sample in self.image_idx_list:
            sample_id = int(sample)
            if self.filtrate_objects(self.get_label(sample_id)):
                self.sample_id_list.append(sample_id)
        self.logger.info(
            f"Done filtering: {len(self.sample_id_list)} / {len(self.image_idx_list)}\n"
        )

    def _initialize_tracking_samples(self):
        if self.mode != "TRAIN":
            self.sample_id_list = [int(sample_id) for sample_id in self.image_idx_list]
            return
        self.logger.info("Loading %s samples from %s ..." % (self.mode, self.label_dir))
        self.sample_pair_id_list = []
        for previous, following in self.sample_pair_list:
            previous_id, following_id = int(previous), int(following)
            previous_objects = self.filtrate_objects(self.get_label(previous_id))
            following_objects = self.filtrate_objects(self.get_label(following_id))
            if previous_objects and following_objects:
                self.sample_pair_id_list.append((previous_id, following_id))
        self.logger.info(
            f"Done filtering: {len(self.sample_pair_id_list)} / {len(self.sample_pair_list)}\n"
        )

    def _initialize_samples(self):
        if cfg.RPN.ENABLED:
            initializer = (
                self._initialize_detection_samples
                if self.challenge == "detection"
                else self._initialize_tracking_samples
            )
            initializer()
            return
        if cfg.RCNN.ENABLED:
            if self.challenge == "detection" or self.mode != "TRAIN":
                self.sample_id_list = [
                    int(sample_id) for sample_id in self.image_idx_list
                ]
            else:
                self.sample_pair_id_list = [
                    (int(first), int(second)) for first, second in self.sample_pair_list
                ]

    def _sample_point_indices(self, pts_rect):
        if self.npoints < len(pts_rect):
            near_mask = pts_rect[:, 2] < 40.0
            far_indices = np.where(near_mask == 0)[0]
            near_indices = np.where(near_mask == 1)[0]
            near_choice = np.random.choice(
                near_indices, self.npoints - len(far_indices), replace=False
            )
            choice = (
                np.concatenate((near_choice, far_indices), axis=0)
                if len(far_indices) > 0 else near_choice
            )
        else:
            choice = np.arange(len(pts_rect), dtype=np.int32)
            missing = self.npoints - len(pts_rect)
            if missing > 0 and missing < len(choice):
                choice = np.concatenate(
                    (choice, np.random.choice(choice, missing, replace=False)), axis=0
                )
            elif missing >= len(choice) and missing > 0:
                LOGGER.info(
                    "%s %s %s %s", "self.npoints - len(pts_rect)",
                    missing, "choice", choice,
                )
        np.random.shuffle(choice)
        return choice

    def _training_sample(self, sample_id, points, features):
        objects = self.filtrate_objects(self.get_label(sample_id))
        boxes, alpha, tids = self._ground_truth_arrays(objects)
        augmented_points, augmented_boxes = points.copy(), boxes.copy()
        if cfg.AUG_DATA and self.mode == "TRAIN":
            augmented_points, augmented_boxes = self.data_augmentation(
                augmented_points, augmented_boxes, alpha
            )
        result = {
            "pts_input": self._point_input(augmented_points, features),
            "gt_boxes3d": augmented_boxes,
            "gt_tids": tids,
        }
        if not cfg.RPN.FIXED:
            labels = self.generate_rpn_training_labels(
                augmented_points, augmented_boxes
            )
            result["rpn_cls_label"], result["rpn_reg_label"] = labels
        return result

    def _collate_detection_batch(self, batch):
        batch_size = len(batch)
        ans_dict = {}
        for key in batch[0].keys():
            pad_boxes = cfg.RPN.ENABLED and key == "gt_boxes3d"
            pad_boxes = pad_boxes or (
                cfg.RCNN.ENABLED
                and cfg.RCNN.ROI_SAMPLE_JIT
                and key in {"gt_boxes3d", "roi_boxes3d"}
            )
            if pad_boxes:
                max_gt = max(item[key].shape[0] for item in batch)
                values = np.zeros((batch_size, max_gt, 7), dtype=np.float32)
                for index, item in enumerate(batch):
                    values[index, : item[key].shape[0], :] = item[key]
                ans_dict[key] = values
                continue
            if isinstance(batch[0][key], np.ndarray):
                values = [item[key][np.newaxis, ...] for item in batch]
                ans_dict[key] = (
                    values[0] if batch_size == 1 else np.concatenate(values, axis=0)
                )
                continue
            values = [item[key] for item in batch]
            if isinstance(values[0], int):
                values = np.array(values, dtype=np.int32)
            if isinstance(values[0], float):
                values = np.array(values, dtype=np.float32)
            ans_dict[key] = values
        return ans_dict

    def _collate_tracking_batch(self, batch):
        batch_size = len(batch)
        ans_dict = {}
        for key in batch[0][0].keys():
            pad_values = cfg.RPN.ENABLED and key == "gt_boxes3d"
            pad_values = pad_values or (
                cfg.RCNN.ENABLED
                and cfg.RCNN.ROI_SAMPLE_JIT
                and key in {"gt_boxes3d", "roi_boxes3d", "gt_tids"}
            )
            if pad_values:
                max_gt = max(
                    max(pair[0][key].shape[0], pair[1][key].shape[0]) for pair in batch
                )
                shape = (
                    (batch_size * 2, max_gt)
                    if key == "gt_tids"
                    else (batch_size * 2, max_gt, 7)
                )
                dtype = np.int32 if key == "gt_tids" else np.float32
                values = np.zeros(shape, dtype=dtype)
                for index, pair in enumerate(batch):
                    values[2 * index, : pair[0][key].shape[0], ...] = pair[0][key]
                    values[2 * index + 1, : pair[1][key].shape[0], ...] = pair[1][key]
                ans_dict[key] = values
                continue
            if isinstance(batch[0][0][key], np.ndarray):
                values = [item[key][np.newaxis, ...] for pair in batch for item in pair]
                ans_dict[key] = np.concatenate(values)
                continue
            values = [item[key] for pair in batch for item in pair]
            if isinstance(values[0], int):
                ans_dict[key] = np.array(values, dtype=np.int32)
                continue
            if isinstance(values[0], float):
                ans_dict[key] = np.array(values, dtype=np.float32)
                continue
            ans_dict[key] = values
        return ans_dict
