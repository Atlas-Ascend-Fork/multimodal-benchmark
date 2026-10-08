# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import NamedTuple
from math import sin, cos, atan2, sqrt
import logging
import pathlib
import pickle
import random
import shutil

import fire
import numpy as np

from second.core import box_np_ops
from second.data import kitti_common as kitti
from second.utils.progress_bar import list_bar as prog_bar


from shared.parameters import ParameterSchema as _ParameterSchema, bind_parameters as _bind_parameters


class CreateNoisedPointCloudParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    data_path: object
    train_info_path: object
    val_info_path: object
    test_info_path: object
    save_path: object
    with_back: object


CreateReducedPointCloudParameters = CreateNoisedPointCloudParameters
CreateReducedPointCloudOnlyTestParameters = CreateNoisedPointCloudParameters


class CreateGroundtruthDatabaseParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    data_path: object
    info_path: object
    used_classes: object
    database_save_path: object
    db_info_save_path: object
    relative_path: object
    lidar_only: object
    bev_only: object
    coors_range: object


LOGGER = logging.getLogger(__name__)


def _read_imageset_file(path):
    with open(path, "r") as f:
        lines = f.readlines()
    return [int(line) for line in lines]


def _calculate_num_points_in_gt(
    data_path, infos, relative_path, remove_outside=True, num_features=4
):
    for info in infos:
        if relative_path:
            v_path = str(pathlib.Path(data_path) / info["velodyne_path"])
        else:
            v_path = info["velodyne_path"]
        LOGGER.info("%s", v_path)
        points_v = np.fromfile(v_path, dtype=np.float32, count=-1).reshape(
            [-1, num_features]
        )
        rect = info["calib/R0_rect"]
        trv2c = info["calib/Tr_velo_to_cam"]
        p2 = info["calib/P2"]
        if remove_outside:
            points_v = box_np_ops.remove_outside_points(
                points_v, rect, trv2c, p2, info["img_shape"]
            )

        annos = info["annos"]
        num_obj = len([n for n in annos["name"] if n != "DontCare"])
        dims = annos["dimensions"][:num_obj]
        loc = annos["location"][:num_obj]
        rots = annos["rotation_y"][:num_obj]
        gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)
        gt_boxes_lidar = box_np_ops.box_camera_to_lidar(gt_boxes_camera, rect, trv2c)
        indices = box_np_ops.points_in_rbbox(points_v[:, :3], gt_boxes_lidar)
        num_points_in_gt = indices.sum(0)
        num_ignored = len(annos["dimensions"]) - num_obj
        num_points_in_gt = np.concatenate([num_points_in_gt, -np.ones([num_ignored])])
        annos["num_points_in_gt"] = num_points_in_gt.astype(np.int32)


def _save_kitti_infos(save_path, name, infos):
    filename = save_path / "kitti_infos_{}.pkl".format(name)
    LOGGER.info("%s", "Kitti info {} file is saved to {}".format(name, filename))
    with open(filename, "wb") as file_handle:
        pickle.dump(infos, file_handle)


def _build_kitti_infos(data_path, image_ids, training, relative_path):
    return kitti.get_kitti_image_info(
        data_path,
        training=training,
        label_info=training,
        velodyne=True,
        calib=True,
        image_ids=image_ids,
        relative_path=relative_path,
    )


def create_kitti_info_file(
    data_path,
    save_path=None,
    imageset_path=None,
    create_trainval=False,
    relative_path=True,
):
    if imageset_path is not None:
        train_img_ids = _read_imageset_file(f"{imageset_path}/train.txt")
        val_img_ids = _read_imageset_file(f"{imageset_path}/val.txt")
        test_img_ids = _read_imageset_file(f"{imageset_path}/test.txt")
    else:
        train_img_ids = _read_imageset_file("./clocs/second/data/ImageSets/train.txt")
        val_img_ids = _read_imageset_file("./clocs/second/data/ImageSets/val.txt")
        test_img_ids = _read_imageset_file("./clocs/second/data/ImageSets/test.txt")

    LOGGER.info("%s", "Generate info. this may take several minutes.")
    if save_path is None:
        save_path = pathlib.Path(data_path)
    else:
        save_path = pathlib.Path(save_path)
    kitti_infos_train = _build_kitti_infos(
        data_path, train_img_ids, True, relative_path
    )
    _calculate_num_points_in_gt(data_path, kitti_infos_train, relative_path)
    _save_kitti_infos(save_path, "train", kitti_infos_train)
    kitti_infos_val = _build_kitti_infos(data_path, val_img_ids, True, relative_path)
    _calculate_num_points_in_gt(data_path, kitti_infos_val, relative_path)
    _save_kitti_infos(save_path, "val", kitti_infos_val)
    _save_kitti_infos(save_path, "trainval", kitti_infos_train + kitti_infos_val)
    kitti_infos_test = _build_kitti_infos(
        data_path, test_img_ids, False, relative_path
    )
    _save_kitti_infos(save_path, "test", kitti_infos_test)


def create_kitti_info_file_only_test(
    data_path, save_path=None, create_trainval=False, relative_path=True
):
    test_img_ids = _read_imageset_file("./data/ImageSets/test.txt")

    LOGGER.info("%s", "Generate info. this may take several minutes.")
    if save_path is None:
        save_path = pathlib.Path(data_path)
    else:
        save_path = pathlib.Path(save_path)
    kitti_infos_test = kitti.get_kitti_image_info(
        data_path,
        training=False,
        label_info=False,
        velodyne=True,
        calib=True,
        image_ids=test_img_ids,
        relative_path=relative_path,
    )
    filename = save_path / "kitti_infos_test.pkl"
    LOGGER.info("%s", f"Kitti info test file is saved to {filename}")
    with open(filename, "wb") as f:
        pickle.dump(kitti_infos_test, f)


def _create_reduced_point_cloud(data_path, info_path, save_path=None, back=False):
    with open(info_path, "rb") as f:
        kitti_infos = pickle.load(f)
    for info in prog_bar(kitti_infos):
        v_path = info["velodyne_path"]
        v_path = pathlib.Path(data_path) / v_path
        points_v = np.fromfile(str(v_path), dtype=np.float32, count=-1).reshape([-1, 4])
        rect = info["calib/R0_rect"]
        p2 = info["calib/P2"]
        trv2c = info["calib/Tr_velo_to_cam"]
        # first remove z < 0 points
        # then remove outside.
        if back:
            points_v[:, 0] = -points_v[:, 0]
        points_v = box_np_ops.remove_outside_points(
            points_v, rect, trv2c, p2, info["img_shape"]
        )

        if save_path is None:
            save_filename = (
                v_path.parent.parent / (v_path.parent.stem + "_reduced") / v_path.name
            )
            if back:
                save_filename += "_back"
        else:
            save_filename = str(pathlib.Path(save_path) / v_path.name)
            if back:
                save_filename += "_back"
        with open(save_filename, "w") as f:
            points_v.tofile(f)


def _create_noised_point_cloud(data_path, info_path, save_path=None, back=False):
    with open(info_path, "rb") as f:
        kitti_infos = pickle.load(f)
    for info in prog_bar(kitti_infos):
        v_path = info["velodyne_path"]
        v_path = pathlib.Path(data_path) / v_path
        points_v = np.fromfile(str(v_path), dtype=np.float32, count=-1).reshape([-1, 4])
        # first remove z < 0 points
        # then remove outside.
        if back:
            points_v[:, 0] = -points_v[:, 0]
        # info["img_shape"])
        for i, p in enumerate(points_v[:, :3]):
            noise = random.uniform(-0.05, 0.05)
            alpha = atan2(p[2], sqrt(p[0] ** 2 + p[1] ** 2))
            beta = atan2(p[1], p[0])
            p[2] = p[2] + noise * sin(alpha)
            p[1] = p[1] + noise * cos(alpha) * sin(beta)
            p[0] = p[0] + noise * cos(alpha) * cos(beta)
            points_v[i, :3] = p[:3]
        if save_path is None:
            save_filename = (
                v_path.parent.parent / (v_path.parent.stem + "_noised") / v_path.name
            )
            if back:
                save_filename += "_back"
        else:
            save_filename = str(pathlib.Path(save_path) / v_path.name)
            if back:
                save_filename += "_back"
        with open(save_filename, "w") as f:
            points_v.tofile(f)


_CREATE_NOISED_POINT_CLOUD_PARAMETERS_SCHEMA = _ParameterSchema(
    CreateNoisedPointCloudParameters,
    (
        "data_path",
        "train_info_path",
        "val_info_path",
        "test_info_path",
        "save_path",
        "with_back",
    ),
    {
        "train_info_path": None,
        "val_info_path": None,
        "test_info_path": None,
        "save_path": None,
        "with_back": False,
    },
    frozenset(()),
    6,
    None,
    None,
)
_CREATE_REDUCED_POINT_CLOUD_PARAMETERS_SCHEMA = (
    _CREATE_NOISED_POINT_CLOUD_PARAMETERS_SCHEMA
)
_CREATE_REDUCED_POINT_CLOUD_ONLY_TEST_PARAMETERS_SCHEMA = (
    _CREATE_NOISED_POINT_CLOUD_PARAMETERS_SCHEMA
)
_CREATE_GROUNDTRUTH_DATABASE_PARAMETERS_SCHEMA = _ParameterSchema(
    CreateGroundtruthDatabaseParameters,
    (
        "data_path",
        "info_path",
        "used_classes",
        "database_save_path",
        "db_info_save_path",
        "relative_path",
        "lidar_only",
        "bev_only",
        "coors_range",
    ),
    {
        "info_path": None,
        "used_classes": None,
        "database_save_path": None,
        "db_info_save_path": None,
        "relative_path": True,
        "lidar_only": False,
        "bev_only": False,
        "coors_range": None,
    },
    frozenset(()),
    9,
    None,
    None,
)


def _point_cloud_paths(data_path, train_info_path, val_info_path, test_info_path):
    defaults = (
        "kitti_infos_train.pkl",
        "kitti_infos_val.pkl",
        "kitti_infos_test.pkl",
    )
    paths = [train_info_path, val_info_path, test_info_path]
    return tuple(
        pathlib.Path(data_path) / default if path is None else path
        for path, default in zip(paths, defaults)
    )


def create_noised_point_cloud(*call_args, **call_kwargs):
    bound_parameters = _bind_parameters(
        _CREATE_NOISED_POINT_CLOUD_PARAMETERS_SCHEMA, call_args, call_kwargs
    )
    data_path, train_info_path, val_info_path, test_info_path, save_path, with_back = (
        bound_parameters
    )
    train_info_path, val_info_path, test_info_path = _point_cloud_paths(
        data_path, train_info_path, val_info_path, test_info_path
    )

    _create_noised_point_cloud(data_path, train_info_path, save_path)
    _create_noised_point_cloud(data_path, val_info_path, save_path)

    if with_back:
        _create_noised_point_cloud(data_path, train_info_path, save_path, back=True)
        _create_noised_point_cloud(data_path, val_info_path, save_path, back=True)


def create_reduced_point_cloud(*call_args, **call_kwargs):
    bound_parameters = _bind_parameters(
        _CREATE_REDUCED_POINT_CLOUD_PARAMETERS_SCHEMA, call_args, call_kwargs
    )
    data_path, train_info_path, val_info_path, test_info_path, save_path, with_back = (
        bound_parameters
    )
    train_info_path, val_info_path, test_info_path = _point_cloud_paths(
        data_path, train_info_path, val_info_path, test_info_path
    )

    _create_reduced_point_cloud(data_path, train_info_path, save_path)
    _create_reduced_point_cloud(data_path, val_info_path, save_path)
    _create_reduced_point_cloud(data_path, test_info_path, save_path)
    if with_back:
        _create_reduced_point_cloud(data_path, train_info_path, save_path, back=True)
        _create_reduced_point_cloud(data_path, val_info_path, save_path, back=True)
        _create_reduced_point_cloud(data_path, test_info_path, save_path, back=True)


def create_reduced_point_cloud_only_test(*call_args, **call_kwargs):
    bound_parameters = _bind_parameters(
        _CREATE_REDUCED_POINT_CLOUD_ONLY_TEST_PARAMETERS_SCHEMA, call_args, call_kwargs
    )
    data_path, train_info_path, val_info_path, test_info_path, save_path, with_back = (
        bound_parameters
    )
    test_info_path = pathlib.Path(data_path) / "kitti_infos_test.pkl"
    _create_reduced_point_cloud(data_path, test_info_path, save_path)


def _groundtruth_database_paths(root_path, info_path, database_path, db_info_path):
    return (
        pathlib.Path(info_path or root_path / "kitti_infos_train.pkl"),
        pathlib.Path(database_path or root_path / "gt_database"),
        pathlib.Path(db_info_path or root_path / "kitti_dbinfos_train.pkl"),
    )


def _database_sample_data(info, config):
    velodyne_path = info["velodyne_path"]
    if config["relative_path"]:
        velodyne_path = str(config["root_path"] / velodyne_path)
    feature_count = info.get("pointcloud_num_features", 4)
    points = np.fromfile(velodyne_path, dtype=np.float32, count=-1).reshape(
        [-1, feature_count]
    )
    rect, p2, trv2c = (
        info["calib/R0_rect"],
        info["calib/P2"],
        info["calib/Tr_velo_to_cam"],
    )
    if not config["lidar_only"]:
        points = box_np_ops.remove_outside_points(
            points, rect, trv2c, p2, info["img_shape"]
        )
    annos = info["annos"]
    num_obj = np.sum(annos["index"] >= 0)
    boxes_camera = kitti.anno_to_rbboxes(annos)[:num_obj]
    boxes_lidar = box_np_ops.box_camera_to_lidar(boxes_camera, rect, trv2c)
    if config["bev_only"]:
        if config["coors_range"] is None:
            raise AssertionError
        boxes_lidar[:, 2] = config["coors_range"][2]
        boxes_lidar[:, 5] = config["coors_range"][5] - config["coors_range"][2]
    return points, annos, boxes_lidar, num_obj


def _database_object_info(sample, object_index, config, group_state):
    info, points, annos, boxes_lidar, point_indices = sample
    names, indices = annos["name"], annos["index"]
    filename = f"{info['image_idx']}_{names[object_index]}_{indices[object_index]}.bin"
    filepath = config["database_path"] / filename
    object_points = points[point_indices[:, object_index]]
    object_points[:, :3] -= boxes_lidar[object_index, :3]
    with open(filepath, "w") as point_file:
        object_points.tofile(point_file)
    if names[object_index] not in config["used_classes"]:
        return None
    path = (
        str(config["database_path"].stem + "/" + filename)
        if config["relative_path"]
        else str(filepath)
    )
    database_info = {
        "name": names[object_index],
        "path": path,
        "image_idx": info["image_idx"],
        "gt_idx": indices[object_index],
        "box3d_lidar": boxes_lidar[object_index],
        "num_points_in_gt": object_points.shape[0],
        "difficulty": annos["difficulty"][object_index],
    }
    local_group_id = group_state["ids"][object_index]
    if local_group_id not in group_state["mapped"]:
        group_state["mapped"][local_group_id] = group_state["counter"]
        group_state["counter"] += 1
    database_info["group_id"] = group_state["mapped"][local_group_id]
    if "score" in annos:
        database_info["score"] = annos["score"][object_index]
    return database_info


def _collect_database_sample(info, config, all_database_info, group_counter):
    points, annos, boxes_lidar, num_obj = _database_sample_data(info, config)
    group_state = {
        "ids": annos.get("group_ids", np.arange(len(annos["bbox"]), dtype=np.int64)),
        "mapped": {},
        "counter": group_counter,
    }
    point_indices = box_np_ops.points_in_rbbox(points, boxes_lidar)
    sample = info, points, annos, boxes_lidar, point_indices
    for object_index in range(num_obj):
        database_info = _database_object_info(
            sample, object_index, config, group_state
        )
        if database_info is not None:
            all_database_info[database_info["name"]].append(database_info)
    return group_state["counter"]


def create_groundtruth_database(*call_args, **call_kwargs):
    bound_parameters = _bind_parameters(
        _CREATE_GROUNDTRUTH_DATABASE_PARAMETERS_SCHEMA, call_args, call_kwargs
    )
    (
        data_path,
        info_path,
        used_classes,
        database_save_path,
        db_info_save_path,
        relative_path,
        lidar_only,
        bev_only,
        coors_range,
    ) = bound_parameters
    root_path = pathlib.Path(data_path)
    info_path, database_save_path, db_info_save_path = _groundtruth_database_paths(
        root_path, info_path, database_save_path, db_info_save_path
    )
    if database_save_path.exists():
        shutil.rmtree(database_save_path)
    database_save_path.mkdir(parents=True, exist_ok=True)
    with open(info_path, "rb") as f:
        kitti_infos = pickle.load(f)
    if used_classes is None:
        used_classes = list(kitti.get_classes())
        used_classes.pop(used_classes.index("DontCare"))
    all_db_infos = {name: [] for name in used_classes}
    config = {
        "root_path": root_path,
        "database_path": database_save_path,
        "used_classes": used_classes,
        "relative_path": relative_path,
        "lidar_only": lidar_only,
        "bev_only": bev_only,
        "coors_range": coors_range,
    }
    group_counter = 0
    for info in prog_bar(kitti_infos):
        group_counter = _collect_database_sample(
            info, config, all_db_infos, group_counter
        )
    for k, v in all_db_infos.items():
        LOGGER.info("%s", f"load {len(v)} {k} database infos")

    with open(db_info_save_path, "wb") as f:
        pickle.dump(all_db_infos, f)


if __name__ == "__main__":
    fire.Fire()
