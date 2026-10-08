# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import NamedTuple
import argparse
import logging
import os
import pickle
import shutil

from lib.datasets.kitti_dataset import KittiDataset
import _init_path  # noqa: F401
import lib.utils.iou3d.iou3d_utils as iou3d_utils
import lib.utils.kitti_utils as kitti_utils
import lib.utils.roipool3d.roipool3d_utils as roipool3d_utils
import numpy as np
import torch
from lib.datasets.generation_utils import DatasetGeneratorMixin


from shared.parameters import ParameterSchema as _ParameterSchema, bind_parameters as _bind_parameters
from shared.kitti_detection import format_detection_line, observation_angle


class AugscenegeneratorAugOneEpochSceneParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    base_id: object
    data_save_dir: object
    label_save_dir: object
    split_list: object
    log_fp: object


LOGGER = logging.getLogger(__name__)

np.random.seed(1024)

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="generator")
parser.add_argument("--class_name", type=str, default="Car")
parser.add_argument(
    "--save_dir", type=str, default="../../data/KITTI/aug_scene/training"
)
parser.add_argument("--split", type=str, default="train")
parser.add_argument(
    "--gt_database_dir",
    type=str,
    default="gt_database/train_gt_database_3level_Car.pkl",
)
parser.add_argument("--include_similar", action="store_true", default=False)
parser.add_argument("--aug_times", type=int, default=4)
args = parser.parse_args()

PC_REDUCE_BY_RANGE = True
if args.class_name == "Car":
    PC_AREA_SCOPE = np.array(
        [[-40, 40], [-1, 3], [0, 70.4]]
    )  # x, y, z scope in rect camera coords
else:
    PC_AREA_SCOPE = np.array([[-30, 30], [-1, 3], [0, 50]])


def log_print(info, fp=None):
    LOGGER.info("%s", info)
    if fp is not None:
        fp.write(str(info) + "\n")


def save_kitti_format(calib, bbox3d, obj_list, img_shape, save_fp):
    corners3d = kitti_utils.boxes3d_to_corners3d(bbox3d)
    img_boxes, _ = calib.corners3d_to_img_boxes(corners3d)

    img_boxes[:, 0] = np.clip(img_boxes[:, 0], 0, img_shape[1] - 1)
    img_boxes[:, 1] = np.clip(img_boxes[:, 1], 0, img_shape[0] - 1)
    img_boxes[:, 2] = np.clip(img_boxes[:, 2], 0, img_shape[1] - 1)
    img_boxes[:, 3] = np.clip(img_boxes[:, 3], 0, img_shape[0] - 1)

    # Discard boxes that are larger than 80% of the image width OR height
    img_boxes_w = img_boxes[:, 2] - img_boxes[:, 0]
    img_boxes_h = img_boxes[:, 3] - img_boxes[:, 1]
    box_valid_mask = np.logical_and(
        img_boxes_w < img_shape[1] * 0.8, img_boxes_h < img_shape[0] * 0.8
    )

    for k in range(bbox3d.shape[0]):
        if box_valid_mask[k] == 0:
            continue
        save_fp.write(format_detection_line(
            args.class_name, observation_angle(bbox3d[k]), img_boxes[k], bbox3d[k],
            {"truncation": obj_list[k].trucation,
             "occlusion": int(obj_list[k].occlusion)},
        ) + "\n")


_AUGSCENEGENERATOR_AUG_ONE_EPOCH_SCENE_PARAMETERS_SCHEMA = _ParameterSchema(
    AugscenegeneratorAugOneEpochSceneParameters,
    ("base_id", "data_save_dir", "label_save_dir", "split_list", "log_fp"),
    {"log_fp": None},
    frozenset(()),
    5,
    None,
    None,
)


class AugSceneGenerator(DatasetGeneratorMixin, KittiDataset):
    def __init__(
        self, root_dir, gt_database=None, split="train", classes=args.class_name
    ):
        super().__init__(root_dir, split=split)
        self.configure_classes(classes)
        self.configure_point_filter(PC_REDUCE_BY_RANGE, PC_AREA_SCOPE)
        self.gt_database = gt_database

    @staticmethod
    def check_pc_range(xyz):
        """
        :param xyz: [x, y, z]
        :return:
        """
        x_range, y_range, z_range = PC_AREA_SCOPE
        if (
            (x_range[0] <= xyz[0] <= x_range[1])
            and (y_range[0] <= xyz[1] <= y_range[1])
            and (z_range[0] <= xyz[2] <= z_range[1])
        ):
            return True
        return False

    @staticmethod
    def _insert_database_object(candidate, pts_rect, src_pts_flag, accumulators,
                                cur_gt_boxes3d):
        new_box, new_points, new_intensity, new_obj = candidate
        enlarged_box = new_box.copy()
        enlarged_box[3] += 2
        point_masks = roipool3d_utils.pts_in_boxes3d_cpu(
            torch.from_numpy(pts_rect), torch.from_numpy(enlarged_box.reshape(1, 7))
        )
        src_pts_flag[point_masks[0].numpy() == 1] = 0
        new_pts_list, new_intensity_list, box_list, object_list = accumulators
        new_pts_list.append(new_points)
        new_intensity_list.append(new_intensity)
        enlarged_box = new_box.copy()
        enlarged_box[4:6] += 0.5
        cur_gt_boxes3d = np.concatenate(
            (cur_gt_boxes3d, enlarged_box.reshape(1, 7)), axis=0
        )
        box_list.append(new_box.reshape(1, 7))
        object_list.append(new_obj)
        return cur_gt_boxes3d

    @staticmethod
    def _merge_augmented_points(pts_rect, pts_intensity, src_pts_flag,
                                new_pts_list, new_intensity_list):
        pts_rect = np.concatenate(
            (pts_rect[src_pts_flag == 1], np.concatenate(new_pts_list, axis=0)),
            axis=0,
        )
        pts_intensity = np.concatenate(
            (pts_intensity[src_pts_flag == 1], np.concatenate(new_intensity_list, axis=0)),
            axis=0,
        )
        return pts_rect, pts_intensity

    @staticmethod
    def _augmentation_state(all_gt_boxes3d, point_count):
        current_boxes = all_gt_boxes3d.copy()
        current_boxes[:, 4:6] += 0.5
        accumulators = ([], [], [], [])
        source_point_mask = np.ones(point_count, dtype=np.int32)
        return current_boxes, accumulators, source_point_mask

    @staticmethod
    def _save_augmented_scene(save_args, scene_data):
        sample_id, base_id, data_save_dir, label_save_dir, log_fp = save_args
        calib, pts_rect, pts_intensity, img_shape, obj_list = scene_data[:5]
        aug_flag, extra_gt_boxes3d, extra_gt_obj_list = scene_data[5:]
        pts_info = np.concatenate((pts_rect, pts_intensity.reshape(-1, 1)), axis=1)
        bin_file = os.path.join(data_save_dir, "%06d.bin" % (base_id + sample_id))
        pts_info.astype(np.float32).tofile(bin_file)
        label_file = os.path.join(label_save_dir, "%06d.txt" % (base_id + sample_id))
        with open(label_file, "w") as file_handle:
            for obj in obj_list:
                file_handle.write(str(obj.to_kitti_format()) + "\n")
            if aug_flag:
                save_kitti_format(
                    calib, extra_gt_boxes3d, extra_gt_obj_list,
                    img_shape=img_shape, save_fp=file_handle,
                )
            else:
                extra_gt_boxes3d = np.zeros((0, 7), dtype=np.float32)
        log_print(
            "Save to file (new_obj: %s): %s" % (len(extra_gt_boxes3d), label_file),
            fp=log_fp,
        )

    def filtrate_dc_objects(self, obj_list):
        valid_obj_list = []
        for obj in obj_list:
            if obj.cls_type in ["DontCare"]:
                continue
            valid_obj_list.append(obj)

        return valid_obj_list

    def filtrate_objects(self, obj_list):
        valid_obj_list = []
        type_whitelist = self.classes
        if args.include_similar:
            type_whitelist = list(self.classes)
            if "Car" in self.classes:
                type_whitelist.append("Van")
            if "Pedestrian" in self.classes or "Cyclist" in self.classes:
                type_whitelist.append("Person_sitting")

        for obj in obj_list:
            if obj.cls_type in type_whitelist:
                valid_obj_list.append(obj)
        return valid_obj_list

    def aug_one_scene(self, sample_id, pts_rect, pts_intensity, all_gt_boxes3d):
        """Augment one point-cloud scene with sampled ground-truth objects."""
        if not (self.gt_database is not None):
            raise AssertionError
        extra_gt_num, try_times, cnt = np.random.randint(10, 15), 50, 0
        cur_gt_boxes3d, accumulators, src_pts_flag = self._augmentation_state(
            all_gt_boxes3d, pts_rect.shape[0]
        )
        new_pts_list, new_pts_intensity_list = accumulators[:2]
        extra_gt_boxes3d_list, extra_gt_obj_list = accumulators[2:]

        road_plane = self.get_road_plane(sample_id)

        while try_times > 0:
            try_times -= 1

            rand_idx = np.random.randint(0, self.gt_database.__len__() - 1)
            new_gt_dict = self.gt_database[rand_idx]
            if PC_REDUCE_BY_RANGE and not self.check_pc_range(
                new_gt_dict["gt_box3d"][:3]
            ):
                continue
            if cnt > extra_gt_num:
                break
            candidate = self._prepare_database_object(
                new_gt_dict, road_plane
            )
            if candidate is None:
                continue
            cnt += 1
            new_gt_box3d = candidate[0]
            iou3d = (
                iou3d_utils.boxes_iou3d_gpu(
                    torch.from_numpy(new_gt_box3d.reshape(1, 7)).cuda(),
                    torch.from_numpy(cur_gt_boxes3d).cuda(),
                )
                .cpu()
                .numpy()
            )

            if iou3d.max() >= 1e-8:
                continue
            cur_gt_boxes3d = self._insert_database_object(
                candidate, pts_rect, src_pts_flag, accumulators, cur_gt_boxes3d
            )

        if new_pts_list.__len__() == 0:
            return False, pts_rect, pts_intensity, None, None

        extra_gt_boxes3d = np.concatenate(extra_gt_boxes3d_list, axis=0)
        pts_rect, pts_intensity = self._merge_augmented_points(
            pts_rect, pts_intensity, src_pts_flag,
            new_pts_list, new_pts_intensity_list,
        )

        return True, pts_rect, pts_intensity, extra_gt_boxes3d, extra_gt_obj_list

    def aug_one_epoch_scene(self, *call_args, **call_kwargs):
        bound_parameters = _bind_parameters(
            _AUGSCENEGENERATOR_AUG_ONE_EPOCH_SCENE_PARAMETERS_SCHEMA,
            call_args,
            call_kwargs,
        )
        base_id, data_save_dir, label_save_dir, split_list, log_fp = bound_parameters
        for _, sample_id in enumerate(self.image_idx_list):
            sample_id = int(sample_id)
            LOGGER.info(
                "%s", "process gt sample (%s, id=%06d)" % (args.split, sample_id)
            )

            calib, pts_rect, pts_intensity, img_shape = self._load_scene_points(sample_id)
            all_gt_boxes3d, obj_list = self._scene_labels(sample_id)
            if args.class_name != "Car" and obj_list.__len__() == 0:
                continue
            aug_flag, pts_rect, pts_intensity, extra_gt_boxes3d, extra_gt_obj_list = (
                self.aug_one_scene(sample_id, pts_rect, pts_intensity, all_gt_boxes3d)
            )
            save_args = (sample_id, base_id, data_save_dir, label_save_dir, log_fp)
            scene_data = (
                calib, pts_rect, pts_intensity, img_shape, obj_list,
                aug_flag, extra_gt_boxes3d, extra_gt_obj_list,
            )
            self._save_augmented_scene(save_args, scene_data)
            split_list.append("%06d" % (base_id + sample_id))

    def generate_aug_scene(self, aug_times, log_fp=None):
        data_save_dir = os.path.join(args.save_dir, "rectified_data")
        label_save_dir = os.path.join(args.save_dir, "aug_label")
        os.makedirs(data_save_dir, exist_ok=True)
        os.makedirs(label_save_dir, exist_ok=True)

        split_file = os.path.join(args.save_dir, "%s_aug.txt" % args.split)
        split_list = self.image_idx_list.copy()
        for epoch in range(aug_times):
            base_id = (epoch + 1) * 10000
            self.aug_one_epoch_scene(
                base_id, data_save_dir, label_save_dir, split_list, log_fp=log_fp
            )

        with open(split_file, "w") as f:
            for idx, sample_id in enumerate(split_list):
                f.write(str(sample_id) + "")
                if idx != len(split_list) - 1:
                    f.write(str("") + "\n")
        log_print("Save split file to %s" % split_file, fp=log_fp)
        target_dir = "../../data/KITTI/ImageSets/"
        shutil.copy2(split_file, target_dir)
        log_print("Copy split file from %s to %s" % (split_file, target_dir), fp=log_fp)

    def _prepare_database_object(self, new_gt_dict, road_plane):
        box = new_gt_dict["gt_box3d"].copy()
        points = new_gt_dict["points"].copy()
        if len(points) < 5:
            return None
        a, b, c, d = road_plane
        road_height = (-d - a * box[0] - c * box[2]) / b
        move_height = box[1] - road_height
        box[1] -= move_height
        points[:, 1] -= move_height
        return box, points, new_gt_dict["intensity"].copy(), new_gt_dict["obj"]

    def _load_scene_points(self, sample_id):
        pts_lidar = self.get_lidar(sample_id)
        calib = self.get_calib(sample_id)
        pts_rect = calib.lidar_to_rect(pts_lidar[:, 0:3])
        pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
        img_shape = self.get_image_shape(sample_id)
        valid = self.get_valid_flag(pts_rect, pts_img, pts_rect_depth, img_shape)
        return calib, pts_rect[valid, :3], pts_lidar[valid, 3], img_shape

    def _scene_labels(self, sample_id):
        all_obj_list = self.filtrate_dc_objects(self.get_label(sample_id))
        all_gt_boxes3d = np.zeros((len(all_obj_list), 7), dtype=np.float32)
        for index, obj in enumerate(all_obj_list):
            all_gt_boxes3d[index] = (
                *obj.pos, obj.h, obj.w, obj.length_value, obj.ry
            )
        return all_gt_boxes3d, self.filtrate_objects(self.get_label(sample_id))


if __name__ == "__main__":
    os.makedirs(args.save_dir, exist_ok=True)
    info_file = os.path.join(args.save_dir, "log_info.txt")

    if args.mode == "generator":
        with open(info_file, "w") as log_fp:
            with open(args.gt_database_dir, "rb") as database_file:
                gt_database = pickle.load(database_file)
            log_print(
                "Loading gt_database(%d) from %s"
                % (gt_database.__len__(), args.gt_database_dir),
                fp=log_fp,
            )

            dataset = AugSceneGenerator(
                root_dir="../../data", gt_database=gt_database, split=args.split
            )
            dataset.generate_aug_scene(aug_times=args.aug_times, log_fp=log_fp)

    else:
        pass
