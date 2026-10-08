# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

""" Prepare KITTI data for 3D object detection.

Author: Charles R. Qi
Date: September 2017

Modified by Zhixin Wang
"""
from collections import namedtuple
from typing import NamedTuple
import argparse
import logging
import os
import pickle
import sys

from draw_util import get_lidar_in_image_fov
from KittiObject import KittiObject
from ops.pybind11.rbbox_iou import bbox_overlaps_2d
from tqdm import tqdm
import kitti_util as utils
import numpy as np
from utils.frustum_data import extract_pc_in_box2d, extract_pc_in_box3d
from utils.frustum_data import (
    add_generation_arguments,
    collect_box_dimensions,
    in_hull,
    make_2d_detection_writer,
    prepare_generation_output,
    read_detection_file,
    read_detection_pickle,
    run_default_kitti_splits,
)
from utils.frustum_data import random_shift_box2d as _random_shift_box2d


from shared.parameters import ParameterSchema as _ParameterSchema, bind_parameters as _bind_parameters


KittiScene = namedtuple(
    "KittiScene", "calibration points image_coordinates image_mask height width"
)


class ExtractFrustumDetDataParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    idx_filename: object
    split: object
    output_filename: object
    det_filename: object
    perturb_box2d: object
    augment_x: object
    type_whitelist: object


class ExtractFrustumDataParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    idx_filename: object
    split: object
    output_filename: object
    perturb_box2d: object
    augment_x: object
    type_whitelist: object


class ExtractFrustumDataRgbDetectionParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    det_filename: object
    split: object
    output_filename: object
    type_whitelist: object
    img_height_threshold: object
    lidar_point_threshold: object


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)


LOGGER = logging.getLogger(__name__)


def random_shift_box2d(box2d, img_height, img_width, shift_ratio=0.1):
    return _random_shift_box2d(
        box2d, img_height, img_width, shift_ratio, require_clipped_valid=True
    )


def extract_boxes(objects, type_whitelist):
    boxes_2d = []
    boxes_3d = []

    filter_objects = []

    for obj_idx, _ in enumerate(objects):
        obj = objects[obj_idx]
        if obj.type not in type_whitelist:
            continue

        boxes_2d += [obj.box2d]
        boxes_3d += [
            np.array(
                [obj.t[0], obj.t[1], obj.t[2], obj.length_value, obj.w, obj.h, obj.ry]
            )
        ]
        filter_objects += [obj]

    if len(boxes_3d) != 0:
        boxes_3d = np.stack(boxes_3d, 0)
        boxes_2d = np.stack(boxes_2d, 0)

    return filter_objects, boxes_2d, boxes_3d


_EXTRACT_FRUSTUM_DET_DATA_PARAMETERS_SCHEMA = _ParameterSchema(
    ExtractFrustumDetDataParameters,
    (
        "idx_filename",
        "split",
        "output_filename",
        "det_filename",
        "perturb_box2d",
        "augment_x",
        "type_whitelist",
    ),
    {"perturb_box2d": False, "augment_x": 1, "type_whitelist": None},
    frozenset(()),
    7,
    None,
    None,
)
_EXTRACT_FRUSTUM_DATA_PARAMETERS_SCHEMA = _ParameterSchema(
    ExtractFrustumDataParameters,
    (
        "idx_filename",
        "split",
        "output_filename",
        "perturb_box2d",
        "augment_x",
        "type_whitelist",
    ),
    {"perturb_box2d": False, "augment_x": 1, "type_whitelist": None},
    frozenset(()),
    6,
    None,
    None,
)
_EXTRACT_FRUSTUM_DATA_RGB_DETECTION_PARAMETERS_SCHEMA = _ParameterSchema(
    ExtractFrustumDataRgbDetectionParameters,
    (
        "det_filename",
        "split",
        "output_filename",
        "type_whitelist",
        "img_height_threshold",
        "lidar_point_threshold",
    ),
    {"type_whitelist": None, "img_height_threshold": 5, "lidar_point_threshold": 1},
    frozenset(()),
    6,
    None,
    None,
)


def _load_kitti_scene(dataset, data_idx):
    calib = dataset.get_calibration(data_idx)
    pc_velo = dataset.get_lidar(data_idx)
    pc_rect = np.zeros_like(pc_velo)
    pc_rect[:, :3] = calib.project_velo_to_rect(pc_velo[:, :3])
    pc_rect[:, 3] = pc_velo[:, 3]
    img_height, img_width = dataset.get_image(data_idx).shape[:2]
    _, image_coords, image_mask = get_lidar_in_image_fov(
        pc_velo[:, :3], calib, 0, 0, img_width, img_height, True
    )
    return KittiScene(
        calib, pc_rect, image_coords, image_mask, img_height, img_width
    )


def _crop_scene_frustum(box2d, scene):
    _, pc_rect, image_coords, image_mask, _, _ = scene
    xmin, ymin, xmax, ymax = box2d
    box_mask = (
        (image_coords[:, 0] < xmax) & (image_coords[:, 0] >= xmin)
        & (image_coords[:, 1] < ymax) & (image_coords[:, 1] >= ymin)
        & image_mask
    )
    return pc_rect[box_mask, :], (xmin, ymin, xmax, ymax)


def _frustum_angle(calib, bounds):
    xmin, ymin, xmax, ymax = bounds
    uvdepth = np.array([[(xmin + xmax) / 2.0, (ymin + ymax) / 2.0, 20]])
    center = calib.project_image_to_rect(uvdepth)[0]
    return -np.arctan2(center[2], center[0])


def _dump_pickle_values(output_filename, values):
    with open(output_filename, "wb") as file_handle:
        for value in values:
            pickle.dump(value, file_handle, -1)


def _append_labeled_frustum(outputs, data_idx, obj, source):
    source_box2d, bounds, scene = source
    pc_in_box_fov, bounds = _crop_scene_frustum(bounds, scene)
    calib = scene[0]
    _, box3d = utils.compute_box_3d(obj, calib.P)
    _, indices = extract_pc_in_box3d(pc_in_box_fov, box3d)
    label = np.zeros(pc_in_box_fov.shape[0])
    label[indices] = 1
    if source_box2d[3] - source_box2d[1] < 25 or np.sum(label) == 0:
        return None
    (id_list, box2d_list, box3d_list, input_list, label_list, type_list,
     heading_list, size_list, angle_list, gt_box2d_list, calib_list) = outputs
    id_list.append(data_idx)
    box2d_list.append(np.array(bounds))
    box3d_list.append(box3d)
    input_list.append(pc_in_box_fov.astype(np.float32, copy=False))
    label_list.append(label)
    type_list.append(obj.type)
    heading_list.append(obj.ry)
    size_list.append(np.array([obj.length_value, obj.w, obj.h]))
    angle_list.append(_frustum_angle(calib, bounds))
    gt_box2d_list.append(source_box2d)
    calib_list.append(calib.calib_dict)
    return np.sum(label), pc_in_box_fov.shape[0]


def _group_detections(det_ids, det_types, det_boxes, det_probs):
    grouped = {}
    for det_id, det_type, box2d, probability in zip(
        det_ids, det_types, det_boxes, det_probs
    ):
        grouped.setdefault(det_id, []).append(
            {"type": det_type, "box2d": box2d, "prob": probability}
        )
    return grouped


def _matching_gt_object(detection, gt_objects, gt_boxes_2d, threshold):
    overlaps = bbox_overlaps_2d(detection["box2d"].reshape(-1, 4), gt_boxes_2d)[0]
    max_index = overlaps.argmax(0)
    return gt_objects[max_index] if overlaps[max_index] >= threshold else None


def _read_grouped_detections(det_filename):
    detections = read_det_file(det_filename)
    return _group_detections(*detections)


def _finish_training_output(output_filename, outputs, pos_count, point_count):
    LOGGER.info("%s", "total_objects %d" % len(outputs[0]))
    LOGGER.info("%s", "Average pos ratio: %f" % (pos_count / float(point_count)))
    LOGGER.info("%s", "Average npoints: %f" % (float(point_count) / len(outputs[0])))
    _dump_pickle_values(output_filename, outputs)
    LOGGER.info("%s", "save in {}".format(output_filename))


def _append_detection_frustum(outputs, detection, scene, thresholds):
    data_idx, det_type, box2d, probability = detection
    min_height, min_points = thresholds
    box2d = box2d.copy()
    img_height, img_width = scene[4:]
    box2d[[0, 2]] = np.clip(box2d[[0, 2]], 0, img_width - 1)
    box2d[[1, 3]] = np.clip(box2d[[1, 3]], 0, img_height - 1)
    points, bounds = _crop_scene_frustum(box2d, scene)
    xmin, ymin, xmax, ymax = bounds
    if ymax - ymin < min_height or xmax - xmin < 1 or len(points) < min_points:
        return
    id_list, box_list, input_list, type_list, angle_list, prob_list, calib_list = outputs
    id_list.append(data_idx)
    box_list.append(np.array(bounds))
    input_list.append(points.astype(np.float32, copy=False))
    type_list.append(det_type)
    angle_list.append(_frustum_angle(scene[0], bounds))
    prob_list.append(probability)
    calib_list.append(scene[0].calib_dict)


def _collect_labeled_augmentations(context, obj, box2d):
    positive_count = total_count = 0
    for _ in range(context["augment_x"]):
        if context["perturb_box2d"]:
            bounds = random_shift_box2d(
                box2d, context["img_height"], context["img_width"], 0.1
            )
        else:
            bounds = box2d
        stats = _append_labeled_frustum(
            context["outputs"], context["data_idx"], obj,
            (obj.box2d, tuple(bounds), context["scene"]),
        )
        if stats is not None:
            positive_count += stats[0]
            total_count += stats[1]
    return positive_count, total_count


def _make_augmentation_config(ctx_tuple, data_idx):
    """Build the config dict for _collect_labeled_augmentations."""
    augment_x, perturb_box2d, img_height, img_width, outputs, scene = ctx_tuple
    return {
        "augment_x": augment_x,
        "perturb_box2d": perturb_box2d,
        "img_height": img_height,
        "img_width": img_width,
        "outputs": outputs,
        "data_idx": data_idx,
        "scene": scene,
    }


def extract_frustum_det_data(*call_args, **call_kwargs):
    bound_parameters = _bind_parameters(
        _EXTRACT_FRUSTUM_DET_DATA_PARAMETERS_SCHEMA, call_args, call_kwargs
    )
    (idx_filename, split, output_filename, det_filename, perturb_box2d,
     augment_x, type_whitelist) = bound_parameters
    if type_whitelist is None:
        type_whitelist = ["Car"]
    dataset = KittiObject(os.path.join(ROOT_DIR, "data/kitti"), split)
    data_idx_list = [int(line.rstrip()) for line in open(idx_filename)]

    all_boxes_2d = _read_grouped_detections(det_filename)
    outputs = tuple([] for _ in range(11))
    pos_cnt = all_cnt = 0
    thresh = 0.5 if "Car" in type_whitelist else 0.25
    for data_idx in tqdm(data_idx_list):
        scene = _load_kitti_scene(dataset, data_idx)
        gt_objects = dataset.get_label_objects(data_idx)

        gt_objects, gt_boxes_2d, gt_boxes_3d = extract_boxes(gt_objects, type_whitelist)

        if len(gt_objects) == 0:
            continue

        img_height, img_width = scene[4:]

        det_objects = all_boxes_2d.get(data_idx)
        if det_objects is None:
            continue

        ctx = (augment_x, perturb_box2d, img_height, img_width, outputs, scene)
        for cur_obj in det_objects:
            if cur_obj["type"] not in type_whitelist:
                continue
            assign_obj = _matching_gt_object(
                cur_obj, gt_objects, gt_boxes_2d, thresh
            )
            if assign_obj is None:
                continue
            cfg = _make_augmentation_config(ctx, data_idx)
            stats = _collect_labeled_augmentations(
                cfg, assign_obj, cur_obj["box2d"],
            )
            pos_cnt += stats[0]
            all_cnt += stats[1]

    _finish_training_output(output_filename, outputs, pos_cnt, all_cnt)


def extract_frustum_data(*call_args, **call_kwargs):
    """Extract labeled point clouds from ground-truth 2D frustums."""
    bound_parameters = _bind_parameters(
        _EXTRACT_FRUSTUM_DATA_PARAMETERS_SCHEMA, call_args, call_kwargs
    )
    idx_filename, split, output_filename, perturb_box2d, augment_x, type_whitelist = (
        bound_parameters
    )
    if type_whitelist is None:
        type_whitelist = ["Car"]
    dataset = KittiObject(os.path.join(ROOT_DIR, "data/kitti"), split)
    data_idx_list = [int(line.rstrip()) for line in open(idx_filename)]

    outputs = tuple([] for _ in range(11))
    pos_cnt = 0
    all_cnt = 0
    for data_idx in tqdm(data_idx_list):
        scene = _load_kitti_scene(dataset, data_idx)
        objects = dataset.get_label_objects(data_idx)
        img_height, img_width = scene[4:]

        ctx = (augment_x, perturb_box2d, img_height, img_width, outputs, scene)
        for obj_idx, _ in enumerate(objects):
            if objects[obj_idx].type not in type_whitelist:
                continue

            obj = objects[obj_idx]
            aug_cfg = _make_augmentation_config(ctx, data_idx)
            stats = _collect_labeled_augmentations(
                aug_cfg, obj, obj.box2d,
            )
            pos_cnt += stats[0]
            all_cnt += stats[1]

    _finish_training_output(output_filename, outputs, pos_cnt, all_cnt)


def get_box3d_dim_statistics(idx_filename):
    """Collect 3D bounding box statistics"""
    dataset = KittiObject(os.path.join(ROOT_DIR, "data/kitti"))
    dimension_list, type_list, _ = collect_box_dimensions(
        dataset, idx_filename, tqdm
    )

    LOGGER.info("%s", "number of objects: {} ".format(len(type_list)))
    LOGGER.info("%s %s", "categories:", set(type_list))

    # Get average box size for different categories
    for class_type in sorted(set(type_list)):
        box3d_list = []
        for i, _ in enumerate(dimension_list):
            if type_list[i] == class_type:
                box3d_list.append(dimension_list[i])

        m_box3d = np.mean(box3d_list, 0)
        LOGGER.info(
            "%s",
            "'%s': np.array([%f,%f,%f]),"
            % (class_type, m_box3d[0], m_box3d[1], m_box3d[2]),
        )


def read_det_file(det_filename):
    """Parse lines in 2D detection output files"""
    return read_detection_file(det_filename, allow_named_types=True)


read_det_pkl_file = read_detection_pickle


def extract_frustum_data_rgb_detection(*call_args, **call_kwargs):
    """Extract point clouds from 2D detection frustums."""
    bound_parameters = _bind_parameters(
        _EXTRACT_FRUSTUM_DATA_RGB_DETECTION_PARAMETERS_SCHEMA, call_args, call_kwargs
    )
    (det_filename, split, output_filename, type_whitelist,
     img_height_threshold, lidar_point_threshold) = bound_parameters
    if type_whitelist is None:
        type_whitelist = ["Car"]
    dataset = KittiObject(os.path.join(ROOT_DIR, "data/kitti"), split=split)
    if det_filename.split(".")[-1] == "pkl":
        det_id_list, det_type_list, det_box2d_list, det_prob_list = read_det_pkl_file(
            det_filename
        )
    else:
        det_id_list, det_type_list, det_box2d_list, det_prob_list = read_det_file(
            det_filename
        )
    cache_id = -1
    cache = None

    outputs = tuple([] for _ in range(7))

    for det_idx, data_idx in enumerate(tqdm(det_id_list)):
        if cache_id != data_idx:
            cache = _load_kitti_scene(dataset, data_idx)
            cache_id = data_idx

        if det_type_list[det_idx] not in type_whitelist:
            continue
        detection = (
            data_idx, det_type_list[det_idx],
            det_box2d_list[det_idx], det_prob_list[det_idx],
        )
        thresholds = (img_height_threshold, lidar_point_threshold)
        _append_detection_frustum(outputs, detection, cache, thresholds)

    _dump_pickle_values(output_filename, outputs)
    LOGGER.info("%s", "total_objects %d" % len(outputs[0]))
    LOGGER.info("%s", "save in {}".format(output_filename))


write_2d_rgb_detection = make_2d_detection_writer(read_det_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_generation_arguments(parser)

    parser.add_argument(
        "--gen_trainval",
        action="store_true",
        help="Generate trainval split frustum data with perturbed GT 2D boxes",
    )

    parser.add_argument(
        "--gen_test_rgb_detection",
        action="store_true",
        help="Generate test split frustum data with RGB detection 2D boxes",
    )

    parser.add_argument("--car_only", action="store_true", help="Only generate cars")
    parser.add_argument(
        "--people_only", action="store_true", help="Only generate peds and cycs"
    )
    parser.add_argument(
        "--save_dir", default=None, type=str, help="data directory to save data"
    )

    parser.add_argument(
        "--gen_avg_dim", action="store_true", help="get average dimension of each class"
    )

    args = parser.parse_args()

    if args.gen_avg_dim:
        get_box3d_dim_statistics(
            os.path.join(BASE_DIR, "image_sets/train.txt"),
        )

    save_dir, type_whitelist, output_prefix = prepare_generation_output(
        args, "kitti/data/pickle_data"
    )
    generation_paths = BASE_DIR, save_dir, output_prefix

    run_default_kitti_splits(
        extract_frustum_data, generation_paths, type_whitelist, args
    )

    if args.gen_val_rgb_detection:
        extract_frustum_data_rgb_detection(
            os.path.join(BASE_DIR, "rgb_detections/rgb_detection_val.txt"),
            "training",
            os.path.join(save_dir, output_prefix + "val_rgb_detection.pickle"),
            type_whitelist=type_whitelist,
        )
