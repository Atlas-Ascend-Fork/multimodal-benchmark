# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import NamedTuple
from collections import defaultdict
import pathlib
import sys

import numpy as np

from second.core import box_np_ops
from second.core import preprocess as prep
from second.core.geometry import points_in_convex_polygon_3d_jit
from second.data import kitti_common as kitti


from shared.parameters import ParameterSchema as _ParameterSchema, bind_parameters as _bind_parameters


def finalize_example(prep_func, input_dict, image_idx):
    example = prep_func(input_dict=input_dict)
    example["image_idx"] = image_idx
    example["image_shape"] = input_dict["image_shape"]
    if "anchors_mask" in example:
        example["anchors_mask"] = example["anchors_mask"].astype(np.uint8)
    return example


class PrepPointcloudParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    input_dict: object
    root_path: object
    voxel_generator: object
    target_assigner: object
    db_sampler: object
    max_voxels: object
    class_names: object
    remove_outside_points: object
    training: object
    create_targets: object
    shuffle_points: object
    reduce_valid_area: object
    remove_unknown: object
    gt_rotation_noise: object
    gt_loc_noise_std: object
    global_rotation_noise: object
    global_scaling_noise: object
    global_random_rot_range: object
    generate_bev: object
    without_reflectivity: object
    num_point_features: object
    anchor_area_threshold: object
    gt_points_drop: object
    gt_drop_max_keep: object
    remove_points_after_sample: object
    anchor_cache: object
    remove_environment: object
    random_crop: object
    reference_detections: object
    add_rgb_to_points: object
    lidar_input: object
    unlabeled_db_sampler: object
    out_size_factor: object
    min_gt_point_dict: object
    bev_only: object
    use_group_id: object
    out_dtype: object


if "/opt/ros/kinetic/lib/python2.7/dist-packages" in sys.path:
    sys.path.remove("/opt/ros/kinetic/lib/python2.7/dist-packages")


def _merge_batch_field(key, elements):
    if key in {
        "voxels",
        "num_points",
        "num_gt",
        "gt_boxes",
        "voxel_labels",
        "match_indices",
        "match_indices_num",
    }:
        return np.concatenate(elements, axis=0)
    if key == "images":
        return elements[0]
    if key == "coordinates":
        coordinates = []
        for index, coordinate in enumerate(elements):
            coordinates.append(
                np.pad(
                    coordinate, ((0, 0), (1, 0)), mode="constant", constant_values=index
                )
            )
        return np.concatenate(coordinates, axis=0)
    return np.stack(elements, axis=0)


def merge_second_batch(batch_list, _unused=False):
    example_merged = defaultdict(list)
    count = 0
    for example in batch_list:
        count = count + 1
        for k, v in example.items():
            example_merged[k].append(v)
    ret = {}
    example_merged.pop("num_voxels")
    for key, elems in example_merged.items():
        ret[key] = _merge_batch_field(key, elems)
    return ret


_PREP_POINTCLOUD_PARAMETERS_SCHEMA = _ParameterSchema(
    PrepPointcloudParameters,
    (
        "input_dict",
        "root_path",
        "voxel_generator",
        "target_assigner",
        "db_sampler",
        "max_voxels",
        "class_names",
        "remove_outside_points",
        "training",
        "create_targets",
        "shuffle_points",
        "reduce_valid_area",
        "remove_unknown",
        "gt_rotation_noise",
        "gt_loc_noise_std",
        "global_rotation_noise",
        "global_scaling_noise",
        "global_random_rot_range",
        "generate_bev",
        "without_reflectivity",
        "num_point_features",
        "anchor_area_threshold",
        "gt_points_drop",
        "gt_drop_max_keep",
        "remove_points_after_sample",
        "anchor_cache",
        "remove_environment",
        "random_crop",
        "reference_detections",
        "add_rgb_to_points",
        "lidar_input",
        "unlabeled_db_sampler",
        "out_size_factor",
        "min_gt_point_dict",
        "bev_only",
        "use_group_id",
        "out_dtype",
    ),
    {
        "db_sampler": None,
        "max_voxels": 20000,
        "class_names": None,
        "remove_outside_points": False,
        "training": True,
        "create_targets": True,
        "shuffle_points": False,
        "reduce_valid_area": False,
        "remove_unknown": False,
        "gt_rotation_noise": None,
        "gt_loc_noise_std": None,
        "global_rotation_noise": None,
        "global_scaling_noise": None,
        "global_random_rot_range": None,
        "generate_bev": False,
        "without_reflectivity": False,
        "num_point_features": 4,
        "anchor_area_threshold": 1,
        "gt_points_drop": 0.0,
        "gt_drop_max_keep": 10,
        "remove_points_after_sample": True,
        "anchor_cache": None,
        "remove_environment": False,
        "random_crop": False,
        "reference_detections": None,
        "add_rgb_to_points": False,
        "lidar_input": False,
        "unlabeled_db_sampler": None,
        "out_size_factor": 2,
        "min_gt_point_dict": None,
        "bev_only": False,
        "use_group_id": False,
        "out_dtype": np.float32,
    },
    frozenset(()),
    37,
    None,
    None,
)


def _select_training_annotations(state, selected):
    for key in ("gt_boxes", "gt_2d_boxes", "gt_names", "difficulty"):
        state[key] = state[key][selected]
    if state["group_ids"] is not None:
        state["group_ids"] = state["group_ids"][selected]


def _reference_filtered_points(points, params, rect, trv2c, projection):
    if params.reference_detections is None:
        return points
    camera, rotation, translation = box_np_ops.projection_matrix_to_crt_kitti(
        projection
    )
    frustums = box_np_ops.get_frustum_v2(params.reference_detections, camera)
    frustums -= translation
    frustums = np.einsum("ij, akj->aki", np.linalg.inv(rotation), frustums)
    frustums = box_np_ops.camera_to_lidar(frustums, rect, trv2c)
    surfaces = box_np_ops.corner_to_surfaces_3d_jit(frustums)
    masks = points_in_convex_polygon_3d_jit(points, surfaces)
    return points[masks.any(-1)]


def _training_annotations(params):
    source = params.input_dict
    return {
        "gt_boxes": source["gt_boxes"],
        "gt_2d_boxes": source["gt_2d_bbox"],
        "gt_names": source["gt_names"],
        "difficulty": source["difficulty"],
        "group_ids": (
            source.get("group_ids")
            if params.use_group_id and "group_ids" in source
            else None
        ),
    }


def _prepare_training_annotations(state, params, points, rect, trv2c):
    class_names = params.class_names or ["Car"]
    if params.remove_environment:
        selected = kitti.keep_arrays_by_name(state["gt_names"], class_names)
        _select_training_annotations(state, selected)
        points = prep.remove_points_outside_boxes(points, state["gt_boxes"])
    selected = kitti.drop_arrays_by_name(state["gt_names"], ["DontCare"])
    _select_training_annotations(state, selected)
    state["gt_boxes"] = box_np_ops.box_camera_to_lidar(
        state["gt_boxes"], rect, trv2c
    )
    if params.remove_unknown:
        _select_training_annotations(state, state["difficulty"] != -1)
    class_mask = np.array(
        [name in class_names for name in state["gt_names"]], dtype=np.bool_
    )
    if params.without_reflectivity:
        axes = list(range(params.num_point_features))
        axes.pop(3)
        points = points[:, axes]
    point_cloud_range = params.voxel_generator.point_cloud_range
    if params.bev_only:
        state["gt_boxes"][:, 2] = point_cloud_range[2]
        state["gt_boxes"][:, 5] = point_cloud_range[5] - point_cloud_range[2]
    _select_training_annotations(state, class_mask)
    state["gt_classes"] = np.array(
        [class_names.index(name) + 1 for name in state["gt_names"]],
        dtype=np.int32,
    )
    bev_range = point_cloud_range[[0, 1, 3, 4]]
    range_mask = prep.filter_gt_box_outside_range(state["gt_boxes"], bev_range)
    _select_training_annotations(state, range_mask)
    state["gt_classes"] = state["gt_classes"][range_mask]
    return points


def _voxelized_example(params, points, calibration, training_state):
    voxels, coordinates, num_points = params.voxel_generator.generate(
        points, params.max_voxels
    )
    example = {
        "voxels": voxels,
        "num_points": num_points,
        "coordinates": coordinates,
        "num_voxels": np.array([voxels.shape[0]], dtype=np.int64),
    }
    if training_state is not None:
        example["d3_gt_boxes"] = training_state["gt_boxes"]
    example.update(calibration)
    return example, coordinates


def _anchor_data(params):
    grid_size = params.voxel_generator.grid_size
    feature_map_size = [*(grid_size[:2] // params.out_size_factor), 1][::-1]
    if params.anchor_cache is not None:
        return (
            params.anchor_cache["anchors"],
            params.anchor_cache["anchors_bv"],
            params.anchor_cache["anchors_dict"],
        )
    generated = params.target_assigner.generate_anchors(feature_map_size)
    anchors = generated["anchors"].reshape([-1, 7])
    anchors_dict = params.target_assigner.generate_anchors_dict(feature_map_size)
    anchors_bv = box_np_ops.rbbox2d_to_near_bbox(anchors[:, [0, 1, 3, 4, 6]])
    return anchors, anchors_bv, anchors_dict


def _add_anchor_mask(example, params, coordinates, anchors_bv):
    if params.anchor_area_threshold < 0:
        return None
    grid_size = params.voxel_generator.grid_size
    dense_map = box_np_ops.sparse_sum_for_anchors_mask(
        coordinates, tuple(grid_size[::-1][1:])
    )
    dense_map = dense_map.cumsum(0).cumsum(1)
    anchor_area = box_np_ops.fused_get_anchors_area(
        dense_map,
        anchors_bv,
        params.voxel_generator.voxel_size,
        params.voxel_generator.point_cloud_range,
        grid_size,
    )
    mask = anchor_area > params.anchor_area_threshold
    example["anchors_mask"] = mask
    return mask


def prep_pointcloud(*call_args, **call_kwargs):
    """Convert a point cloud to voxels and optionally create targets."""
    params = _bind_parameters(
        _PREP_POINTCLOUD_PARAMETERS_SCHEMA, call_args, call_kwargs
    )
    source = params.input_dict
    rect, trv2c, projection = source["rect"], source["Trv2c"], source["P2"]
    points = _reference_filtered_points(
        source["points"], params, rect, trv2c, projection
    )
    if params.remove_outside_points and not params.lidar_input:
        points = box_np_ops.remove_outside_points(
            points, rect, trv2c, projection, source["image_shape"]
        )
    training_state = _training_annotations(params) if params.training else None
    if training_state is not None:
        points = _prepare_training_annotations(
            training_state, params, points, rect, trv2c
        )
    if params.shuffle_points:
        np.random.shuffle(points)
    calibration = {"rect": rect, "Trv2c": trv2c, "P2": projection}
    example, coordinates = _voxelized_example(
        params, points, calibration, training_state
    )
    anchors, anchors_bv, anchors_dict = _anchor_data(params)
    example.update({"anchors": anchors, "images": source["images"]})
    anchors_mask = _add_anchor_mask(example, params, coordinates, anchors_bv)
    if training_state is not None and params.create_targets:
        targets = params.target_assigner.assign_v2(
            anchors_dict,
            training_state["gt_boxes"],
            anchors_mask,
            gt_classes=training_state["gt_classes"],
            gt_names=training_state["gt_names"],
        )
        example.update(
            {
                "labels": targets["labels"],
                "reg_targets": targets["bbox_targets"],
                "reg_weights": targets["bbox_outside_weights"],
                "gt_2d_boxes": training_state["gt_2d_boxes"],
            }
        )
    return example


def _read_and_prep_v9(info, root_path, num_point_features, prep_func):
    """read data from KITTI-format infos, then call prep function."""
    v_path = pathlib.Path(root_path) / info["velodyne_path"]
    v_path = v_path.parent.parent / (v_path.parent.stem + "_reduced") / v_path.name
    points = np.fromfile(str(v_path), dtype=np.float32, count=-1).reshape(
        [-1, num_point_features]
    )
    image_idx = info["image_idx"]
    input_dict = {
        "points": points,
        "rect": info["calib/R0_rect"].astype(np.float32),
        "Trv2c": info["calib/Tr_velo_to_cam"].astype(np.float32),
        "P2": info["calib/P2"].astype(np.float32),
        "image_shape": np.array(info["img_shape"], dtype=np.int32),
        "image_idx": image_idx,
        "image_path": info["img_path"],
        "images": None,
    }

    if "annos" in info:
        annos = info["annos"]
        # we need other objects to avoid collision when sample
        annos = kitti.remove_dontcare(annos)
        gt_boxes = np.concatenate(
            [annos["location"], annos["dimensions"], annos["rotation_y"][..., None]],
            axis=1,
        ).astype(np.float32)
        input_dict.update(
            {
                "gt_boxes": gt_boxes,
                "gt_names": annos["name"],
                "difficulty": annos["difficulty"],
                "gt_2d_bbox": annos["bbox"],
            }
        )
        if "group_ids" in annos:
            input_dict["group_ids"] = annos["group_ids"]
    return finalize_example(prep_func, input_dict, image_idx)
