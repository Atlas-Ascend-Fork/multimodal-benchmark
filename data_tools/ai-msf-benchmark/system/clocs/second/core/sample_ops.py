# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import NamedTuple
from functools import reduce
import copy
import logging
import pathlib

import numpy as np

from second.core import box_np_ops
from second.core import preprocess as prep
from second.utils.check import shape_mergeable
from shared.parameters import ParameterSchema as _ParameterSchema, bind_parameters as _bind_parameters


class Databasesamplerv2InitParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    db_infos: object
    groups: object
    db_prepor: object
    rate: object
    global_rot_range: object


class Databasesamplerv2SampleAllParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    root_path: object
    gt_boxes: object
    gt_names: object
    num_point_features: object
    random_crop: object
    gt_group_ids: object
    rect: object
    trv2c: object
    p2: object


LOGGER = logging.getLogger(__name__)


_DATABASESAMPLERV2_INIT_PARAMETERS_SCHEMA = _ParameterSchema(
    Databasesamplerv2InitParameters,
    ("db_infos", "groups", "db_prepor", "rate", "global_rot_range"),
    {"db_prepor": None, "rate": 1.0, "global_rot_range": None},
    frozenset(()),
    5,
    None,
    None,
)
_DATABASESAMPLERV2_SAMPLE_ALL_PARAMETERS_SCHEMA = _ParameterSchema(
    Databasesamplerv2SampleAllParameters,
    (
        "root_path",
        "gt_boxes",
        "gt_names",
        "num_point_features",
        "random_crop",
        "gt_group_ids",
        "rect",
        "trv2c",
        "p2",
    ),
    {
        "random_crop": False,
        "gt_group_ids": None,
        "rect": None,
        "trv2c": None,
        "p2": None,
    },
    frozenset(()),
    9,
    None,
    None,
)


class DataBaseSamplerV2:
    def __init__(self, *call_args, **call_kwargs):
        bound_parameters = _bind_parameters(
            _DATABASESAMPLERV2_INIT_PARAMETERS_SCHEMA, call_args, call_kwargs
        )
        db_infos, groups, db_prepor, rate, global_rot_range = bound_parameters
        self._log_database_sizes(db_infos)
        if db_prepor is not None:
            db_infos = db_prepor(db_infos)
            self._log_database_sizes(db_infos, "After filter database:")
        self.db_infos = db_infos
        self._rate = rate
        self._groups = groups
        self._group_db_infos = {}
        self._group_name_to_names = []
        self._sample_classes = []
        self._sample_max_nums = []
        self._set_sampling_groups(groups)
        self._sampler_dict = {
            name: prep.BatchSampler(items, name)
            for name, items in self._group_db_infos.items()
        }
        self._global_rot_range, self._enable_global_rot = (
            self._normalize_rotation_range(global_rot_range)
        )

    @property
    def use_group_sampling(self):
        return self._use_group_sampling

    @staticmethod
    def _log_database_sizes(db_infos, prefix=None):
        if prefix is not None:
            LOGGER.info("%s", prefix)
        for name, items in db_infos.items():
            LOGGER.info("load %d %s database infos", len(items), name)

    @staticmethod
    def _group_composition(group_info, grouped_items):
        if len(group_info) <= 1:
            return {}
        counts = {}
        for group in grouped_items:
            name = ", ".join(sorted(item["name"] for item in group))
            counts[name] = counts.get(name, 0) + 1
        return counts

    @staticmethod
    def _normalize_rotation_range(global_rot_range):
        if global_rot_range is None:
            return None, False
        if not isinstance(global_rot_range, (list, tuple, np.ndarray)):
            global_rot_range = [-global_rot_range, global_rot_range]
        elif not shape_mergeable(global_rot_range, [2]):
            raise ValueError("global rotation range must contain two values")
        enabled = np.abs(global_rot_range[0] - global_rot_range[1]) >= 1e-3
        return global_rot_range, enabled

    @staticmethod
    def _stack_sample_boxes(samples):
        if len(samples) == 1:
            return samples[0]["box3d_lidar"][np.newaxis, ...]
        return np.stack([sample["box3d_lidar"] for sample in samples], axis=0)

    @staticmethod
    def _append_group_ids(total_group_ids, samples):
        if len(samples) == 1:
            sampled_ids = np.array(samples[0]["group_id"])[np.newaxis, ...]
        else:
            sampled_ids = np.stack([sample["group_id"] for sample in samples])
        return np.concatenate([total_group_ids, sampled_ids], axis=0)

    @staticmethod
    def _load_sample_points(root_path, samples, num_features):
        point_sets = []
        for info in samples:
            points = np.fromfile(
                str(pathlib.Path(root_path) / info["path"]), dtype=np.float32
            ).reshape([-1, num_features])
            if "rot_transform" in info:
                points[:, :3] = box_np_ops.rotation_points_single_angle(
                    points[:, :3], info["rot_transform"], axis=2
                )
            points[:, :3] += info["box3d_lidar"][:3]
            point_sets.append(points)
        return point_sets

    @staticmethod
    def _random_crop_points(point_sets, gt_boxes, rect, trv2c, p2):
        bboxes = box_np_ops.box3d_to_bbox(gt_boxes, rect, trv2c, p2)
        frustums = prep.random_crop_frustum(bboxes, rect, trv2c, p2)
        for index, frustum in enumerate(frustums):
            points = point_sets[index]
            mask = prep.mask_points_in_corners(
                points, frustum[np.newaxis, ...]
            ).reshape(-1)
            remaining = points.shape[0] - np.sum(mask)
            if np.any(mask) and remaining > 15:
                point_sets[index] = points[np.logical_not(mask)]
        return point_sets

    @staticmethod
    def _remap_sample_group_ids(sampled, gt_group_ids):
        gid_map = {}
        sampled_gid = np.max(gt_group_ids) + 1
        for sample in sampled:
            gid = sample["group_id"]
            if gid in gid_map:
                sample["group_id"] = gid_map[gid]
            else:
                gid_map[gid] = sampled_gid
                sample["group_id"] = sampled_gid
                sampled_gid += 1

    def sample_all(self, *call_args, **call_kwargs):
        bound_parameters = _bind_parameters(
            _DATABASESAMPLERV2_SAMPLE_ALL_PARAMETERS_SCHEMA, call_args, call_kwargs
        )
        (
            root_path,
            gt_boxes,
            gt_names,
            num_point_features,
            random_crop,
            gt_group_ids,
            rect,
            trv2c,
            p2,
        ) = bound_parameters
        counts = self._sample_counts(gt_names)
        names, sample_counts, group_ids = self._sampling_plan(counts, gt_group_ids)
        sampled, sampled_boxes = self._sample_plan(
            names, sample_counts, gt_boxes, group_ids
        )
        if not sampled:
            return None
        sampled_boxes = np.concatenate(sampled_boxes, axis=0)
        point_sets = self._load_sample_points(root_path, sampled, num_point_features)
        if random_crop:
            point_sets = self._random_crop_points(
                point_sets, sampled_boxes, rect, trv2c, p2
            )
        return self._sample_result(
            sampled, sampled_boxes, point_sets, gt_boxes.shape[0]
        )

    def sample(self, name, num):
        if self._use_group_sampling:
            group_name = name
            ret = self._sampler_dict[group_name].sample(num)
            groups_num = [len(length_value) for length_value in ret]
            return reduce(lambda x, y: x + y, ret), groups_num
        else:
            ret = self._sampler_dict[name].sample(num)
            return ret, np.ones((len(ret),), dtype=np.int64)

    def sample_v1(self, name, num):
        if isinstance(name, (list, tuple)):
            group_name = ", ".join(name)
            ret = self._sampler_dict[group_name].sample(num)
            groups_num = [len(length_value) for length_value in ret]
            return reduce(lambda x, y: x + y, ret), groups_num
        else:
            ret = self._sampler_dict[name].sample(num)
            return ret, np.ones((len(ret),), dtype=np.int64)

    def sample_class_v2(self, name, num, gt_boxes):
        sampled = self._sampler_dict[name].sample(num)
        sampled = copy.deepcopy(sampled)
        num_gt = gt_boxes.shape[0]
        num_sampled = len(sampled)
        gt_boxes_bv = box_np_ops.center_to_corner_box2d(
            gt_boxes[:, 0:2], gt_boxes[:, 3:5], gt_boxes[:, 6]
        )

        sp_boxes = np.stack([i["box3d_lidar"] for i in sampled], axis=0)

        valid_mask = np.zeros([gt_boxes.shape[0]], dtype=np.bool_)
        valid_mask = np.concatenate(
            [valid_mask, np.ones([sp_boxes.shape[0]], dtype=np.bool_)], axis=0
        )
        boxes = np.concatenate([gt_boxes, sp_boxes], axis=0).copy()
        if self._enable_global_rot:
            # place samples to any place in a circle.
            prep.noise_per_object_v3_(
                boxes, None, valid_mask, 0, 0, self._global_rot_range, num_try=100
            )
        sp_boxes_new = boxes[gt_boxes.shape[0]:]
        sp_boxes_bv = box_np_ops.center_to_corner_box2d(
            sp_boxes_new[:, 0:2], sp_boxes_new[:, 3:5], sp_boxes_new[:, 6]
        )

        total_bv = np.concatenate([gt_boxes_bv, sp_boxes_bv], axis=0)
        coll_mat = prep.box_collision_test(total_bv, total_bv)
        diag = np.arange(total_bv.shape[0])
        coll_mat[diag, diag] = False

        valid_samples = []
        for i in range(num_gt, num_gt + num_sampled):
            if coll_mat[i].any():
                coll_mat[i] = False
                coll_mat[:, i] = False
            else:
                if self._enable_global_rot:
                    sampled[i - num_gt]["box3d_lidar"][:2] = boxes[i, :2]
                    sampled[i - num_gt]["box3d_lidar"][-1] = boxes[i, -1]
                    sampled[i - num_gt]["rot_transform"] = (
                        boxes[i, -1] - sp_boxes[i - num_gt, -1]
                    )
                valid_samples.append(sampled[i - num_gt])
        return valid_samples

    def sample_group(self, name, num, gt_boxes, gt_group_ids):
        sampled, group_num = self.sample(name, num)
        sampled = copy.deepcopy(sampled)
        self._remap_sample_group_ids(sampled, gt_group_ids)

        num_gt = gt_boxes.shape[0]
        gt_boxes_bv = box_np_ops.center_to_corner_box2d(
            gt_boxes[:, 0:2], gt_boxes[:, 3:5], gt_boxes[:, 6]
        )

        sp_boxes = np.stack([i["box3d_lidar"] for i in sampled], axis=0)
        sp_group_ids = np.stack([i["group_id"] for i in sampled], axis=0)
        valid_mask = np.zeros([gt_boxes.shape[0]], dtype=np.bool_)
        valid_mask = np.concatenate(
            [valid_mask, np.ones([sp_boxes.shape[0]], dtype=np.bool_)], axis=0
        )
        boxes = np.concatenate([gt_boxes, sp_boxes], axis=0).copy()
        group_ids = np.concatenate([gt_group_ids, sp_group_ids], axis=0)
        if self._enable_global_rot:
            # place samples to any place in a circle.
            prep.noise_per_object_v3_(
                boxes,
                None,
                valid_mask,
                0,
                0,
                self._global_rot_range,
                group_ids=group_ids,
                num_try=100,
            )
        sp_boxes_new = boxes[gt_boxes.shape[0]:]
        sp_boxes_bv = box_np_ops.center_to_corner_box2d(
            sp_boxes_new[:, 0:2], sp_boxes_new[:, 3:5], sp_boxes_new[:, 6]
        )
        total_bv = np.concatenate([gt_boxes_bv, sp_boxes_bv], axis=0)
        coll_mat = prep.box_collision_test(total_bv, total_bv)
        diag = np.arange(total_bv.shape[0])
        coll_mat[diag, diag] = False
        return self._valid_sample_groups(
            sampled, group_num, (boxes, sp_boxes), coll_mat, num_gt
        )

    def _set_sampling_groups(self, groups):
        self._use_group_sampling = any(len(group) > 1 for group in groups)
        if self._use_group_sampling:
            self._set_combined_groups(groups)
            return
        self._group_db_infos = self.db_infos
        for group_info in groups:
            self._sample_classes.extend(group_info)
            self._sample_max_nums.extend(group_info.values())

    def _set_combined_groups(self, groups):
        for group_info in groups:
            names = list(group_info)
            group_name = ", ".join(names)
            self._sample_classes.extend(names)
            self._sample_max_nums.extend(group_info.values())
            self._group_name_to_names.append((group_name, names))
            grouped_items = self._group_items_by_id(names)
            if group_name in self._group_db_infos:
                raise ValueError("group must be unique")
            self._group_db_infos[group_name] = grouped_items
            LOGGER.info("%s", self._group_composition(group_info, grouped_items))

    def _group_items_by_id(self, names):
        grouped = {}
        for name in names:
            for item in self.db_infos[name]:
                grouped.setdefault(item["group_id"], []).append(item)
        return list(grouped.values())

    def _sample_counts(self, gt_names):
        counts = {}
        for name, maximum in zip(self._sample_classes, self._sample_max_nums):
            missing = maximum - np.sum([gt_name == name for gt_name in gt_names])
            counts[name] = np.round(self._rate * int(missing)).astype(np.int64)
        return counts

    def _sampling_plan(self, counts, gt_group_ids):
        if not self._use_group_sampling:
            return self._sample_classes, list(counts.values()), gt_group_ids
        if gt_group_ids is None:
            raise ValueError("gt_group_ids is required for group sampling")
        names = []
        sample_counts = []
        for group_name, class_names in self._group_name_to_names:
            names.append(group_name)
            sample_counts.append(np.max([counts[name] for name in class_names]))
        return names, sample_counts, gt_group_ids

    def _sample_plan(self, names, counts, gt_boxes, group_ids):
        sampled = []
        sampled_boxes = []
        collision_boxes = gt_boxes
        for name, count in zip(names, counts):
            if count <= 0:
                continue
            if self._use_group_sampling:
                items = self.sample_group(name, count, collision_boxes, group_ids)
            else:
                items = self.sample_class_v2(name, count, collision_boxes)
            sampled.extend(items)
            if not items:
                continue
            boxes = self._stack_sample_boxes(items)
            sampled_boxes.append(boxes)
            collision_boxes = np.concatenate([collision_boxes, boxes], axis=0)
            if self._use_group_sampling:
                group_ids = self._append_group_ids(group_ids, items)
        return sampled, sampled_boxes

    def _sample_result(self, samples, boxes, points, original_count):
        result = {
            "gt_names": np.array([sample["name"] for sample in samples]),
            "difficulty": np.array([sample["difficulty"] for sample in samples]),
            "gt_boxes": boxes,
            "points": np.concatenate(points, axis=0),
            "gt_masks": np.ones((len(samples),), dtype=np.bool_),
        }
        if self._use_group_sampling:
            result["group_ids"] = np.array([sample["group_id"] for sample in samples])
        else:
            result["group_ids"] = np.arange(
                original_count, original_count + len(samples)
            )
        return result

    def _valid_sample_groups(self, sampled, group_num, box_data, coll_mat, num_gt):
        boxes, sp_boxes = box_data
        valid_samples = []
        idx = num_gt
        for group_size in group_num:
            if coll_mat[idx: idx + group_size].any():
                coll_mat[idx: idx + group_size] = False
                coll_mat[:, idx: idx + group_size] = False
                idx += group_size
                continue
            for offset in range(group_size):
                sample_idx = idx - num_gt + offset
                if self._enable_global_rot:
                    sampled[sample_idx]["box3d_lidar"][:2] = boxes[idx + offset, :2]
                    sampled[sample_idx]["box3d_lidar"][-1] = boxes[idx + offset, -1]
                    sampled[sample_idx]["rot_transform"] = (
                        boxes[idx + offset, -1] - sp_boxes[sample_idx, -1]
                    )
                valid_samples.append(sampled[sample_idx])
            idx += group_size
        return valid_samples
