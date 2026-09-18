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
import sys

from draw_util import get_lidar_in_image_fov
from KittiObject import KittiObject
from ops.pybind11.rbbox_iou import rbbox_iou_3d
import kitti_util as utils
import numpy as np
from tqdm import tqdm
from utils.frustum_data import extract_pc_in_box2d, extract_pc_in_box3d
from utils.frustum_data import (
    read_detection_pickle,
    read_detection_file,
    prepare_generation_output,
    make_2d_detection_writer,
    in_hull,
    collect_box_dimensions,
    add_generation_arguments,
    run_default_kitti_splits,
)
from utils.frustum_data import random_shift_box2d as _random_shift_box2d


from shared.parameters import ParameterSchema as _ParameterSchema, bind_parameters as _bind_parameters


class ComputeBox3dObjParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    cx: object
    cy: object
    cz: object
    length_value: object
    w: object
    h: object
    ry: object


class ExtractFrustumDataParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    idx_filename: object
    split: object
    output_filename: object
    perturb_box2d: object
    augment_x: object
    type_whitelist: object
    remove_diff: object


class ExtractFrustumDetDataParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    idx_filename: object
    split: object
    output_filename: object
    res_label_dir: object
    perturb_box2d: object
    augment_x: object
    type_whitelist: object
    remove_diff: object


class ExtractFrustumDataRgbDetectionParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    idx_filename: object
    split: object
    output_filename: object
    res_label_dir: object
    type_whitelist: object
    img_height_threshold: object
    lidar_point_threshold: object



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)


LOGGER = logging.getLogger(__name__)


def extract_boxes(objects, type_whitelist, remove_diff=False):
    boxes_2d = []
    boxes_3d = []

    filter_objects = []

    for obj_idx, _ in enumerate(objects):
        obj = objects[obj_idx]
        if obj.type not in type_whitelist:
            continue

        if remove_diff:
            if obj.occlusion > 2 or obj.truncation > 0.5 or obj.ymax - obj.ymin < 25:
                continue

        boxes_2d += [obj.box2d]

        length_value, w, h = obj.length_value, obj.w, obj.h
        cx, cy, cz = obj.t
        ry = obj.ry
        cy = cy - h / 2
        boxes_3d += [np.array([cx, cy, cz, length_value, w, h, ry])]
        filter_objects += [obj]

    if len(boxes_3d) != 0:
        boxes_3d = np.stack(boxes_3d, 0)
        boxes_2d = np.stack(boxes_2d, 0)

    return filter_objects, boxes_2d, boxes_3d


def compute_box_3d_obj_array(obj_array):
    """
    cx, cy, cz, l, w, h, ry
    """

    center, dimensions, angle = obj_array[:3], obj_array[3:6], obj_array[6]
    return utils.compute_box_3d(center, dimensions, angle)


_COMPUTE_BOX3D_OBJ_PARAMETERS_SCHEMA = _ParameterSchema(
    ComputeBox3dObjParameters,
    ("cx", "cy", "cz", "length_value", "w", "h", "ry"),
    {},
    frozenset(()),
    7,
    None,
    None,
)
_FRUSTUM_DATA_PARAMETER_NAMES = (
    "idx_filename",
    "split",
    "output_filename",
    "perturb_box2d",
    "augment_x",
    "type_whitelist",
    "remove_diff",
)


def _frustum_data_schema(parameter_type, with_result_directory):
    names = list(_FRUSTUM_DATA_PARAMETER_NAMES)
    if with_result_directory:
        names.insert(3, "res_label_dir")
    return _ParameterSchema(
        parameter_type,
        tuple(names),
        {
        "perturb_box2d": False,
        "augment_x": 1,
        "type_whitelist": None,
        "remove_diff": False,
        },
        frozenset(()),
        len(names),
        None,
        None,
    )


_EXTRACT_FRUSTUM_DATA_PARAMETERS_SCHEMA = _frustum_data_schema(
    ExtractFrustumDataParameters, False
)
_EXTRACT_FRUSTUM_DET_DATA_PARAMETERS_SCHEMA = _frustum_data_schema(
    ExtractFrustumDetDataParameters, True
)
_EXTRACT_FRUSTUM_DATA_RGB_DETECTION_PARAMETERS_SCHEMA = _ParameterSchema(
    ExtractFrustumDataRgbDetectionParameters,
    (
        "idx_filename",
        "split",
        "output_filename",
        "res_label_dir",
        "type_whitelist",
        "img_height_threshold",
        "lidar_point_threshold",
    ),
    {"type_whitelist": None, "img_height_threshold": 5, "lidar_point_threshold": 1},
    frozenset(()),
    7,
    None,
    None,
)


def compute_box_3d_obj(*call_args, **call_kwargs):
    """Takes an object and a projection matrix (P) and projects the 3d
    bounding box into the image plane.
    Returns:
        corners_2d: (8,2) array in left image coord.
        corners_3d: (8,3) array in in rect camera coord.
    """
    bound_parameters = _bind_parameters(
        _COMPUTE_BOX3D_OBJ_PARAMETERS_SCHEMA, call_args, call_kwargs
    )
    return compute_box_3d_obj_array(bound_parameters)


def single_overlap(box1, box2):
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    x_w = min(box1[2], box2[2]) - max(box1[0], box2[0])
    x_h = min(box1[3], box2[3]) - max(box1[1], box2[1])

    condition_124 = x_w <= 0 or x_h <= 0 or area1 <= 0 or area2 <= 0
    if condition_124:
        return 0

    return (x_w * x_h) / (area1 + area2 - (x_w * x_h))


def random_shift_box2d(box2d, img_height, img_width, shift_ratio=0.1):
    return _random_shift_box2d(
        box2d, img_height, img_width, shift_ratio, require_clipped_valid=False
    )


def random_shift_box3d(box3d, shift_ratio=0.1):
    """Randomly shift box center, randomly scale width and height"""
    r = shift_ratio
    xmin, ymin, zmin, xmax, ymax, zmax = box3d

    length_value = xmax - xmin
    h = ymax - ymin
    w = zmax - zmin

    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    cz = (zmin + zmax) / 2.0

    if not (xmin < xmax and ymin < ymax and (zmin < zmax)):
        raise AssertionError

    while True:
        cx2 = cx + length_value * r * (np.random.random() * 2 - 1)
        cy2 = cy + h * r * (np.random.random() * 2 - 1)
        cz2 = cz + w * r * (np.random.random() * 2 - 1)

        l2 = length_value * (1 + np.random.random() * 2 * r - r)  # 0.9 to 1.1
        h2 = h * (1 + np.random.random() * 2 * r - r)  # 0.9 to 1.1
        w2 = w * (1 + np.random.random() * 2 * r - r)  # 0.9 to 1.1

        new_box3d = np.array(
            [
                cx2 - l2 / 2.0,
                cy2 - h2 / 2.0,
                cz2 - w2 / 2.0,
                cx2 + l2 / 2.0,
                cy2 + h2 / 2.0,
                cz2 + w2 / 2.0,
            ]
        )

        o = single_overlap(box3d[[0, 2, 3, 5]], new_box3d[[0, 2, 3, 5]])
        condition_210 = l2 > 0 and h2 > 0 and w2 > 0 and o <= 0.8 and o >= 0.5
        if condition_210:
            return new_box3d


def random_shift_rotate_box3d(obj_array, shift_ratio=0.1):
    """Randomly shift box center, randomly scale width and height"""
    r = shift_ratio

    cx, cy, cz, length_value, w, h, angle = obj_array
    # [-pi, pi] -> [0, 2pi]
    angle = angle + np.pi

    if not (length_value > 0 and w > 0 and (h > 0)):
        raise AssertionError

    while True:
        l1 = length_value + length_value * r * (np.random.random() * 2 - 1)
        h1 = h + h * r * (np.random.random() * 2 - 1)
        w1 = w + w * r * (np.random.random() * 2 - 1)

        cx1 = cx + length_value * r * (np.random.random() * 2 - 1)
        cy1 = cy + h * r * (np.random.random() * 2 - 1)
        cz1 = cz + w * r * (np.random.random() * 2 - 1)

        angle1 = angle + r * (np.random.random() * 2 - 1) * np.pi

        angle1 = angle1 % (2 * np.pi)

        if l1 > 0 and h1 > 0 and w1 > 0:
            angle1 = angle1 - np.pi
            if not (angle1 > -np.pi - 0.001 and angle1 < np.pi + 0.001):
                raise AssertionError
            new_box3d = np.array([cx1, cy1, cz1, l1, w1, h1, angle1])

            return new_box3d


def _load_kitti_scene(dataset, data_index):
    calibration = dataset.get_calibration(data_index)
    lidar = dataset.get_lidar(data_index)
    points = np.zeros_like(lidar)
    points[:, :3] = calibration.project_velo_to_rect(lidar[:, :3])
    points[:, 3] = lidar[:, 3]
    height, width = dataset.get_image(data_index).shape[:2]
    _, image_coordinates, selected = get_lidar_in_image_fov(
        lidar[:, :3], calibration, 0, 0, width, height, True
    )
    return calibration, points[selected], image_coordinates[selected]


def _object_box_array(obj):
    center_x, center_y, center_z = obj.t
    return np.array([
        center_x, center_y - obj.h / 2, center_z,
        obj.length_value, obj.w, obj.h, obj.ry,
    ])


def _new_refine_output():
    keys = (
        "id", "box3d", "input", "label", "type", "heading", "size", "angle",
        "box2d", "calib", "enlarged_box", "enlarged_size", "enlarged_angle",
    )
    return {key: [] for key in keys}


def _training_frustum_sample(points, gt_corners, gt_obj, enlarged_array, perturb):
    if perturb:
        enlarged_array = random_shift_rotate_box3d(enlarged_array, 0.05)
    enlarged_corners = compute_box_3d_obj_array(enlarged_array)
    _, selected = extract_pc_in_box3d(points, enlarged_corners)
    frustum_points = points[selected]
    _, foreground = extract_pc_in_box3d(frustum_points, gt_corners)
    labels = np.zeros(len(frustum_points))
    labels[foreground] = 1
    if np.sum(labels) == 0:
        return enlarged_array, None
    center = enlarged_array[:3]
    sample = {
        "box3d": gt_corners, "input": frustum_points, "label": labels,
        "heading": gt_obj.ry,
        "size": np.array([gt_obj.length_value, gt_obj.w, gt_obj.h]),
        "angle": -np.arctan2(center[2], center[0]),
        "enlarged_box": enlarged_corners, "enlarged_size": enlarged_array[3:6],
        "enlarged_angle": enlarged_array[-1],
    }
    return enlarged_array, sample


def _append_refine_sample(output, metadata, sample):
    data_index, obj_type, box2d, calibration = metadata
    common = {
        "id": data_index, "type": obj_type, "box2d": box2d,
        "calib": calibration.calib_dict,
    }
    for key, value in common.items():
        output[key].append(value)
    for key, value in sample.items():
        output[key].append(value)


def _save_refine_training_output(output, filename):
    order = (
        "id", "box3d", "input", "label", "type", "heading", "size", "angle",
        "box2d", "calib", "enlarged_box", "enlarged_size", "enlarged_angle",
    )
    with open(filename, "wb") as file_handle:
        for key in order:
            pickle.dump(output[key], file_handle, -1)
    LOGGER.info("%s", "save in {}".format(filename))


def _valid_refine_gt_object(obj, whitelist, remove_difficult):
    if obj.type not in whitelist:
        return False
    if not remove_difficult:
        return True
    xmin, ymin, xmax, ymax = obj.box2d
    return obj.occlusion <= 2 and obj.truncation <= 0.5 and ymax - ymin >= 25


def _matched_refine_ground_truth(detection, gt_objects, gt_boxes, threshold):
    detection_array = _object_box_array(detection)
    overlaps = rbbox_iou_3d(detection_array.reshape(-1, 7), gt_boxes)[0]
    best_index = overlaps.argmax(0)
    if overlaps.max(0) < threshold:
        return None
    enlarged_array = detection_array.copy()
    enlarged_array[3:6] *= 1.2
    return gt_objects[best_index], enlarged_array


def _collect_refine_augmentations(context, obj, enlarged_array, gt_corners):
    positive_count = point_count = 0
    for _ in range(context["augment_x"]):
        enlarged_array, sample = _training_frustum_sample(
            context["points"], gt_corners, obj, enlarged_array,
            context["perturb_box2d"],
        )
        if sample is None:
            continue
        _append_refine_sample(context["output"], context["metadata"], sample)
        positive_count += np.sum(sample["label"])
        point_count += len(sample["input"])
    return positive_count, point_count


def extract_frustum_data(*call_args, **call_kwargs):
    parameters = _bind_parameters(
        _EXTRACT_FRUSTUM_DATA_PARAMETERS_SCHEMA, call_args, call_kwargs
    )
    whitelist = parameters.type_whitelist or ["Car"]
    dataset = KittiObject(os.path.join(ROOT_DIR, "data/kitti"), parameters.split)
    data_indices = [int(line.rstrip()) for line in open(parameters.idx_filename)]
    output = _new_refine_output()
    positive_count, point_count = 0, 0
    for data_index in tqdm(data_indices):
        calibration, points, _ = _load_kitti_scene(dataset, data_index)
        for obj in dataset.get_label_objects(data_index):
            if not _valid_refine_gt_object(obj, whitelist, parameters.remove_diff):
                continue
            object_array = _object_box_array(obj)
            gt_corners = compute_box_3d_obj_array(object_array)
            enlarged_array = object_array.copy()
            enlarged_array[3:6] *= 1.2
            counts = _collect_refine_augmentations(
                {
                    "augment_x": parameters.augment_x,
                    "points": points,
                    "perturb_box2d": parameters.perturb_box2d,
                    "output": output,
                    "metadata": (data_index, obj.type, obj.box2d, calibration),
                },
                obj,
                enlarged_array,
                gt_corners,
            )
            positive_count += counts[0]
            point_count += counts[1]
    LOGGER.info("%s", "total_objects %d" % len(output["id"]))
    LOGGER.info("%s", "Average pos ratio: %f" % (positive_count / float(point_count)))
    LOGGER.info("%s", "Average npoints: %f" % (float(point_count) / len(output["id"])))
    _save_refine_training_output(output, parameters.output_filename)


def extract_frustum_det_data(*call_args, **call_kwargs):
    parameters = _bind_parameters(
        _EXTRACT_FRUSTUM_DET_DATA_PARAMETERS_SCHEMA, call_args, call_kwargs
    )
    whitelist = parameters.type_whitelist or ["Car"]
    threshold = 0.5 if "Car" in whitelist else 0.25
    dataset = KittiObject(os.path.join(ROOT_DIR, "data/kitti"), parameters.split)
    data_indices = [int(line.rstrip()) for line in open(parameters.idx_filename)]
    output = _new_refine_output()
    positive_count, point_count = 0, 0
    for data_index in tqdm(data_indices):
        calibration, points, _ = _load_kitti_scene(dataset, data_index)
        gt_objects, _, gt_boxes = extract_boxes(
            dataset.get_label_objects(data_index), whitelist, parameters.remove_diff
        )
        if not gt_objects:
            continue
        filename = os.path.join(parameters.res_label_dir, "%06d.txt" % data_index)
        for detection in utils.read_label(filename):
            if detection.type not in whitelist:
                continue
            matched = _matched_refine_ground_truth(
                detection, gt_objects, gt_boxes, threshold
            )
            if matched is None:
                continue
            gt_obj, enlarged_array = matched
            gt_corners = compute_box_3d_obj_array(_object_box_array(gt_obj))
            counts = _collect_refine_augmentations(
                {
                    "augment_x": parameters.augment_x,
                    "points": points,
                    "perturb_box2d": parameters.perturb_box2d,
                    "output": output,
                    "metadata": (
                        data_index, detection.type, gt_obj.box2d, calibration
                    ),
                },
                gt_obj,
                enlarged_array,
                gt_corners,
            )
            positive_count += counts[0]
            point_count += counts[1]
    LOGGER.info("%s", "total_objects %d" % len(output["id"]))
    LOGGER.info("%s", "Average pos ratio: %f" % (positive_count / float(point_count)))
    LOGGER.info("%s", "Average npoints: %f" % (float(point_count) / len(output["id"])))
    _save_refine_training_output(output, parameters.output_filename)


def get_box3d_dim_statistics(idx_filename):
    """Collect and dump 3D bounding box statistics"""
    dataset = KittiObject(os.path.join(ROOT_DIR, "data/kitti"))
    dimension_list, type_list, ry_list = collect_box_dimensions(
        dataset, idx_filename, tqdm
    )

    with open("box3d_dimensions.pickle", "wb") as fp:
        pickle.dump(type_list, fp)
        pickle.dump(dimension_list, fp)
        pickle.dump(ry_list, fp)


def read_det_file(det_filename):
    """Parse lines in 2D detection output files"""
    return read_detection_file(det_filename)


read_det_pkl_file = read_detection_pickle


def _rgb_refine_sample(obj, points, height_threshold, point_threshold):
    xmin, ymin, xmax, ymax = obj.box2d
    object_array = _object_box_array(obj)
    predicted_corners = compute_box_3d_obj_array(object_array)
    enlarged_array = object_array.copy()
    enlarged_array[3:6] *= 1.2
    enlarged_corners = compute_box_3d_obj_array(enlarged_array)
    _, selected = extract_pc_in_box3d(points, enlarged_corners)
    frustum_points = points[selected]
    if ymax - ymin < height_threshold or xmax - xmin < 1 or len(frustum_points) < point_threshold:
        return None
    center = enlarged_array[:3]
    return {
        "input": frustum_points.astype(np.float32, copy=False),
        "angle": -np.arctan2(center[2], center[0]), "probability": obj.score,
        "predicted_box": predicted_corners, "enlarged_box": enlarged_corners,
        "enlarged_size": enlarged_array[3:6], "enlarged_angle": enlarged_array[-1],
    }


def _save_refine_detection_output(output, filename):
    order = (
        "id", "box2d", "input", "type", "angle", "probability", "calib",
        "enlarged_box", "enlarged_size", "enlarged_angle",
    )
    with open(filename, "wb") as file_handle:
        for key in order:
            pickle.dump(output[key], file_handle, -1)
    LOGGER.info("%s", len(output["id"]))
    LOGGER.info("%s", "save in {}".format(filename))


def extract_frustum_data_rgb_detection(*call_args, **call_kwargs):
    parameters = _bind_parameters(
        _EXTRACT_FRUSTUM_DATA_RGB_DETECTION_PARAMETERS_SCHEMA, call_args, call_kwargs
    )
    whitelist = parameters.type_whitelist or ["Car"]
    dataset = KittiObject(os.path.join(ROOT_DIR, "data/kitti"), parameters.split)
    data_indices = [int(line.rstrip()) for line in open(parameters.idx_filename)]
    keys = (
        "id", "box2d", "input", "type", "angle", "probability", "calib",
        "enlarged_box", "enlarged_size", "enlarged_angle",
    )
    output = {key: [] for key in keys}
    for data_index in tqdm(data_indices):
        calibration, points, _ = _load_kitti_scene(dataset, data_index)
        filename = os.path.join(parameters.res_label_dir, "%06d.txt" % data_index)
        for obj in utils.read_label(filename):
            if obj.type not in whitelist:
                continue
            sample = _rgb_refine_sample(
                obj, points, parameters.img_height_threshold,
                parameters.lidar_point_threshold,
            )
            if sample is None:
                continue
            values = {
                "id": data_index, "box2d": obj.box2d, "type": obj.type,
                "calib": calibration.calib_dict,
                "input": sample["input"], "angle": sample["angle"],
                "probability": sample["probability"],
                "enlarged_box": sample["enlarged_box"],
                "enlarged_size": sample["enlarged_size"],
                "enlarged_angle": sample["enlarged_angle"],
            }
            for key in keys:
                output.get(key).append(values.get(key))
    _save_refine_detection_output(output, parameters.output_filename)


write_2d_rgb_detection = make_2d_detection_writer(read_det_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_generation_arguments(parser)

    parser.add_argument(
        "--gen_val_det",
        action="store_true",
        help="Generate val split frustum data with DET boxes",
    )

    parser.add_argument("--car_only", action="store_true", help="Only generate cars")
    parser.add_argument(
        "--people_only", action="store_true", help="Only generate person"
    )

    parser.add_argument(
        "--save_dir", default=None, type=str, help="data directory to save data"
    )

    parser.add_argument(
        "--gen_from_folder",
        default=None,
        type=str,
        help="Generate frustum data from folder",
    )

    args = parser.parse_args()

    save_dir, type_whitelist, output_prefix = prepare_generation_output(
        args, "kitti/data/pickle_data_refine"
    )
    generation_paths = BASE_DIR, save_dir, output_prefix

    run_default_kitti_splits(
        extract_frustum_data, generation_paths, type_whitelist, args,
        val_options={"remove_diff": True},
    )

    if args.gen_val_det:

        if args.people_only:
            res_label_dir = "./output/people_train/val_nms/result/data"
        elif args.car_only:
            res_label_dir = "./output/car_train/val_nms/result/data"
        else:
            if not (False):
                raise AssertionError

        extract_frustum_det_data(
            os.path.join(BASE_DIR, "image_sets/val.txt"),
            "training",
            os.path.join(save_dir, output_prefix + "val_det.pickle"),
            res_label_dir,
            perturb_box2d=False,
            augmentX=1,
            type_whitelist=type_whitelist,
            remove_diff=True,
        )

    if args.gen_val_rgb_detection:
        if args.people_only:
            res_label_dir = "./output/people_train/val_nms/result/data"
        elif args.car_only:
            res_label_dir = "./output/car_train/val_nms/result/data"
        else:
            if not (False):
                raise AssertionError

        extract_frustum_data_rgb_detection(
            os.path.join(BASE_DIR, "image_sets/val.txt"),
            "training",
            os.path.join(save_dir, output_prefix + "val_rgb_detection_refine.pickle"),
            res_label_dir,
            type_whitelist=type_whitelist,
        )

    if args.gen_from_folder:

        res_label_dir = args.gen_from_folder
        postfix = "val_rgb_detection_refine.pickle"
        save_dir = os.path.join(res_label_dir, "..")

        extract_frustum_data_rgb_detection(
            os.path.join(BASE_DIR, "image_sets/val.txt"),
            "training",
            os.path.join(save_dir, output_prefix + postfix),
            res_label_dir,
            type_whitelist=type_whitelist,
        )
