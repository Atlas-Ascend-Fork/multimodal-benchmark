# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

""" Helper class and functions for loading KITTI objects

Author: Charles R. Qi
Date: September 2017
"""
from __future__ import print_function
from collections import namedtuple
from typing import NamedTuple

import argparse
import logging
import os
import sys

from tqdm import tqdm
import cv2
import kitti_util as utils
import numpy as np
from shared.kitti_labels import get_label_annotation
from shared.kitti_detection import (
    KittiVideoDataset,
    load_indexed_calibration,
    project_detection_boxes,
)
from shared.parameters import (
    ParameterSchema as _ParameterSchema,
    bind_parameters as _bind_parameters,
    named_parameter_schema,
)


VisualizationSample = namedtuple(
    "VisualizationSample", "point_cloud objects predicted calibration image depth"
)


class KittiobjectInitParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    root_dir: object
    lidardir: object
    preddir: object
    split: object
    args: object


class ShowImageWithBoxes3typeParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    img: object
    objects: object
    calib: object
    objects2d: object
    name: object
    objects_pred: object


class GetLidarInImageFovParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    pc_velo: object
    calib: object
    xmin: object
    ymin: object
    xmax: object
    ymax: object
    return_more: object
    clip_distance: object


GetLidarIndexInImageFovParameters = GetLidarInImageFovParameters


class ShowLidarWithDepthParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    pc_velo: object
    objects: object
    calib: object
    fig: object
    img_fov: object
    img_width: object
    img_height: object
    objects_pred: object
    depth: object
    cam_img: object
    constraint_box: object
    pc_label: object
    save: object


class SaveDepth0Parameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    data_idx: object
    pc_velo: object
    calib: object
    img_fov: object
    img_width: object
    img_height: object
    depth: object
    constraint_box: object


SaveDepthParameters = SaveDepth0Parameters


class ShowLidarWithBoxesParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    pc_velo: object
    objects: object
    calib: object
    img_fov: object
    img_width: object
    img_height: object
    objects_pred: object
    depth: object
    cam_img: object


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, "mayavi"))

LOGGER = logging.getLogger(__name__)

try:
    raw_input  # Python 2
except NameError:
    raw_input = input  # Python 3

cbox = np.array([[0, 70.4], [-40, 40], [-3, 1]])


_SHOW_IMAGE_WITH_BOXES3TYPE_PARAMETERS_SCHEMA = _ParameterSchema(
    ShowImageWithBoxes3typeParameters,
    ("img", "objects", "calib", "objects2d", "name", "objects_pred"),
    {},
    frozenset(()),
    6,
    None,
    None,
)
_GET_LIDAR_IN_IMAGE_FOV_PARAMETERS_SCHEMA = named_parameter_schema(
    ("pc_velo", "calib", "xmin", "ymin", "xmax", "ymax", "return_more", "clip_distance"),
    {"return_more": False, "clip_distance": 2.0},
)
_GET_LIDAR_INDEX_IN_IMAGE_FOV_PARAMETERS_SCHEMA = (
    _GET_LIDAR_IN_IMAGE_FOV_PARAMETERS_SCHEMA
)
_SHOW_LIDAR_WITH_DEPTH_PARAMETERS_SCHEMA = _ParameterSchema(
    ShowLidarWithDepthParameters,
    (
        "pc_velo",
        "objects",
        "calib",
        "fig",
        "img_fov",
        "img_width",
        "img_height",
        "objects_pred",
        "depth",
        "cam_img",
        "constraint_box",
        "pc_label",
        "save",
    ),
    {
        "img_fov": False,
        "img_width": None,
        "img_height": None,
        "objects_pred": None,
        "depth": None,
        "cam_img": None,
        "constraint_box": False,
        "pc_label": False,
        "save": False,
    },
    frozenset(()),
    13,
    None,
    None,
)
_SAVE_DEPTH0_PARAMETERS_SCHEMA = _ParameterSchema(
    SaveDepth0Parameters,
    (
        "data_idx",
        "pc_velo",
        "calib",
        "img_fov",
        "img_width",
        "img_height",
        "depth",
        "constraint_box",
    ),
    {"constraint_box": False},
    frozenset(()),
    8,
    None,
    None,
)
_SAVE_DEPTH_PARAMETERS_SCHEMA = _SAVE_DEPTH0_PARAMETERS_SCHEMA
_SHOW_LIDAR_WITH_BOXES_PARAMETERS_SCHEMA = _ParameterSchema(
    ShowLidarWithBoxesParameters,
    (
        "pc_velo",
        "objects",
        "calib",
        "img_fov",
        "img_width",
        "img_height",
        "objects_pred",
        "depth",
        "cam_img",
    ),
    {
        "img_fov": False,
        "img_width": None,
        "img_height": None,
        "objects_pred": None,
        "depth": None,
        "cam_img": None,
    },
    frozenset(()),
    9,
    None,
    None,
)
_KITTIOBJECT_INIT_PARAMETERS_SCHEMA = _ParameterSchema(
    KittiobjectInitParameters,
    ("root_dir", "lidardir", "preddir", "split", "args"),
    {"lidardir": None, "preddir": None, "split": "training", "args": None},
    frozenset(()),
    5,
    None,
    None,
)


class KittiObject(object):
    """Load and parse object data into a usable format."""

    def __init__(self, *call_args, **call_kwargs):
        """root_dir contains training and testing folders"""
        bound_parameters = _bind_parameters(
            _KITTIOBJECT_INIT_PARAMETERS_SCHEMA, call_args, call_kwargs
        )
        root_dir, lidardir, preddir, split, args = bound_parameters
        self.root_dir = root_dir
        self.split = split
        self.split_dir = os.path.join(root_dir, split)

        if split == "training":
            self.num_samples = 7481
        elif split == "testing":
            self.num_samples = 7518
        else:
            raise ValueError(f"Unknown split: {split}")

        lidar_dir = "velodyne"
        depth_dir = "depth"
        pred_dir = "pred"
        d2_pred_dir = "d2_detection_data"

        if args is not None:
            lidar_dir = args.lidar
            depth_dir = args.depthdir
            pred_dir = args.preddir
        if preddir is not None:
            pred_dir = preddir
        if lidardir is not None:
            lidar_dir = lidardir

        self.image_dir = os.path.join(self.split_dir, "image_2")
        self.label_dir = os.path.join(self.split_dir, "label_2")
        self.calib_dir = os.path.join(self.split_dir, "calib")

        self.depthpc_dir = os.path.join(self.split_dir, "depth_pc")
        self.lidar_dir = os.path.join(self.split_dir, lidar_dir)
        self.depth_dir = os.path.join(self.split_dir, depth_dir)
        self.pred_dir = os.path.join(self.split_dir, pred_dir)
        self.d2_pred_dir = os.path.join(self.split_dir, d2_pred_dir)

    def __len__(self):
        return self.num_samples

    def get_image(self, idx):
        if not (idx < self.num_samples):
            raise AssertionError
        img_filename = os.path.join(self.image_dir, "%06d.png" % (idx))
        return utils.load_image(img_filename)

    def get_lidar(self, idx, dtype=np.float32, n_vec=4):
        if not (idx < self.num_samples):
            raise AssertionError
        lidar_filename = os.path.join(self.lidar_dir, "%06d.bin" % (idx))
        return utils.load_velo_scan(lidar_filename, dtype, n_vec)

    def get_calibration(self, idx):
        return load_indexed_calibration(utils, self.calib_dir, idx, self.num_samples)

    def get_label_objects(self, idx):
        if not (idx < self.num_samples and self.split == "training"):
            raise AssertionError
        label_filename = os.path.join(self.label_dir, "%06d.txt" % (idx))
        return utils.read_label(label_filename)

    def get_pred_objects(self, idx):
        if not (idx < self.num_samples):
            raise AssertionError
        pred_filename = os.path.join(self.pred_dir, "%06d.txt" % (idx))
        is_exist = os.path.exists(pred_filename)
        if is_exist:
            return utils.read_label(pred_filename)
        else:
            raise ValueError("{} not exist".format(pred_filename))

    def get_d2_pred_objects(self, idx):
        if not (idx < self.num_samples):
            raise AssertionError
        pred_filename = os.path.join(self.d2_pred_dir, "%06d.txt" % (idx))
        is_exist = os.path.exists(pred_filename)
        if is_exist:
            return utils.read_label(pred_filename)
        else:
            return None

    def get_depth(self, idx):
        if not (idx < self.num_samples):
            raise AssertionError
        img_filename = os.path.join(self.depth_dir, "%06d.png" % (idx))
        return utils.load_depth(img_filename)

    def get_depth_image(self, idx):
        if not (idx < self.num_samples):
            raise AssertionError
        img_filename = os.path.join(self.depth_dir, "%06d.png" % (idx))
        return utils.load_depth(img_filename)

    def get_depth_pc(self, idx):
        if not (idx < self.num_samples):
            raise AssertionError
        lidar_filename = os.path.join(self.depthpc_dir, "%06d.bin" % (idx))
        is_exist = os.path.exists(lidar_filename)
        if is_exist:
            return utils.load_velo_scan(lidar_filename), is_exist
        else:
            return None, is_exist

    def get_top_down(self, idx):
        pass

    def isexist_pred_objects(self, idx):
        if not (idx < self.num_samples and self.split == "training"):
            raise AssertionError
        pred_filename = os.path.join(self.pred_dir, "%06d.txt" % (idx))
        return os.path.exists(pred_filename)

    def isexist_depth(self, idx):
        if not (idx < self.num_samples and self.split == "training"):
            raise AssertionError
        depth_filename = os.path.join(self.depth_dir, "%06d.txt" % (idx))
        return os.path.exists(depth_filename)

    def get_label_anno(self, idx, obj_interesting=None):
        label_filename = os.path.join(self.label_dir, "%06d.txt" % (idx))
        return self._get_label_anno(label_filename, obj_interesting)

    def get_pred_anno(self, idx, obj_interesting=None):
        label_filename = os.path.join(self.pred_dir, "%06d.txt" % (idx))
        return self._get_label_anno(label_filename, obj_interesting)

    def _get_label_anno(self, label_path, obj_interesting):
        return [get_label_annotation(label_path, obj_interesting)]


class KittiObjectVideo(KittiVideoDataset):
    """Load data for KITTI videos"""

    def __init__(self, img_dir, lidar_dir, calib_dir):
        super().__init__(img_dir, lidar_dir, calib_dir, utils, LOGGER)


def viz_kitti_video():
    from viz_util import draw_lidar

    video_path = os.path.join(ROOT_DIR, "dataset/2011_09_26/")
    dataset = KittiObjectVideo(
        os.path.join(video_path, "2011_09_26_drive_0023_sync/image_02/data"),
        os.path.join(video_path, "2011_09_26_drive_0023_sync/velodyne_points/data"),
        video_path,
    )
    LOGGER.info("%s", len(dataset))
    for _, _ in enumerate(dataset):
        img = dataset.get_image(0)
        pc = dataset.get_lidar(0)
        cv2.imshow("video", img)
        draw_lidar(pc)
        raw_input()
        pc[:, 0:3] = dataset.get_calibration().project_velo_to_rect(pc[:, 0:3])
        draw_lidar(pc)
        raw_input()
    return


def show_image_with_dt_dt_boxes(img, dt_objects, gt_objects, save_path=None, hull=None):
    """Show image with 2D bounding boxes"""
    import matplotlib.pyplot as plt

    img1 = np.copy(img)  # for 2d bbox
    h, w, c = img.shape
    plt.imshow(img1)  # the x in sitk is z in ITKSnap
    for objects, color in zip([gt_objects, dt_objects], ["r", "b"]):
        if objects is None:
            continue
        for obj in objects:
            if obj.type == "DontCare":
                continue
            if obj.type == "Car":
                color = (0, 1, 0)
            if obj.type == "Pedestrian":
                color = (1, 1, 1)
            if obj.type == "Cyclist":
                color = (0, 1, 1)
            if obj.conf is not None:
                plt.text(obj.xmin, obj.ymin, str(obj.conf), fontsize=5, color=color)
            plt.gca().add_patch(
                plt.Rectangle(
                    (obj.xmin, obj.ymin),
                    obj.xmax - obj.xmin,
                    obj.ymax - obj.ymin,
                    fill=False,
                    edgecolor=color,
                    linewidth=1,
                )
            )  # attention, plt.Rectangle's x is the y for image and y is image's x

    if hull is not None:
        LOGGER.info("%s", hull.shape)
        LOGGER.info("%s", np.array([hull[0]]).shape)
        hull = np.concatenate([np.array([hull[0]]), hull], axis=0)
        LOGGER.info("%s", hull.shape)
        plt.plot(hull[:, 0], hull[:, 1], "r--", lw=1)
    if save_path is not None:
        plt.xticks([])
        plt.yticks([])
        plt.savefig(save_path, bbox_inches="tight", pad_inches=-0.05)
    else:
        plt.show()
    plt.close()


def show_image_with_boxes(img, objects, calib, show3d=True, depth=None):
    """Show image with 2D bounding boxes"""
    img1 = np.copy(img)  # for 2d bbox
    img2 = np.copy(img)  # for 3d bbox
    for obj in objects:
        if obj.type == "DontCare":
            continue
        if obj.type == "Car":
            cv2.rectangle(
                img1,
                (int(obj.xmin), int(obj.ymin)),
                (int(obj.xmax), int(obj.ymax)),
                (0, 255, 0),
                2,
            )
        if obj.type == "Pedestrian":
            cv2.rectangle(
                img1,
                (int(obj.xmin), int(obj.ymin)),
                (int(obj.xmax), int(obj.ymax)),
                (255, 255, 0),
                2,
            )
        if obj.type == "Cyclist":
            cv2.rectangle(
                img1,
                (int(obj.xmin), int(obj.ymin)),
                (int(obj.xmax), int(obj.ymax)),
                (0, 255, 255),
                2,
            )
        box3d_pts_2d, _ = utils.compute_box_3d(obj, calib.P)
        if box3d_pts_2d is None:
            LOGGER.info("%s", "something wrong in the 3D box.")
            continue
        if obj.type == "Car":
            img2 = utils.draw_projected_box3d(img2, box3d_pts_2d, color=(0, 255, 0))
        elif obj.type == "Pedestrian":
            img2 = utils.draw_projected_box3d(img2, box3d_pts_2d, color=(255, 255, 0))
        elif obj.type == "Cyclist":
            img2 = utils.draw_projected_box3d(img2, box3d_pts_2d, color=(0, 255, 255))

        # project
    cv2.imshow("2dbox", img1)
    show3d = True
    if show3d:
        cv2.imshow("3dbox", img2)
    if depth is not None:
        cv2.imshow("depth", depth)

    return img1, img2


def _draw_image_entries(image, entries, style, title, start_x):
    color, width = style
    for box, _ in entries:
        cv2.rectangle(
            image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
            color, width,
        )
    labels = [title] + [label for _, label in entries]
    for index, label in enumerate(labels):
        cv2.putText(
            image, label, (start_x, 25 * (index + 1)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 0, cv2.LINE_AA,
        )


def show_image_with_boxes_3type(*call_args, **call_kwargs):
    """Show image with 2D bounding boxes"""
    bound_parameters = _bind_parameters(
        _SHOW_IMAGE_WITH_BOXES3TYPE_PARAMETERS_SCHEMA, call_args, call_kwargs
    )
    img, objects, calib, objects2d, name, objects_pred = bound_parameters
    img1 = np.copy(img)  # for 2d bbox
    type_list = ["Pedestrian", "Car", "Cyclist"]
    label_entries = [
        ((obj.xmin, obj.ymin, obj.xmax, obj.ymax), obj.type)
        for obj in objects if obj.type in type_list
    ]
    _draw_image_entries(img1, label_entries, ((0, 255, 0), 3), "Label:", 5)
    prediction_2d = [(obj.box2d, type_list[obj.typeid - 1]) for obj in objects2d]
    _draw_image_entries(img1, prediction_2d, ((0, 0, 255), 2), "2D Pred:", 85)
    if objects_pred is not None:
        prediction_3d = [
            ((obj.xmin, obj.ymin, obj.xmax, obj.ymax), obj.type)
            for obj in objects_pred if obj.type in type_list
        ]
        _draw_image_entries(
            img1, prediction_3d, ((255, 0, 0), 1), "3D Pred:", 165
        )

    cv2.imshow("with_bbox", img1)
    cv2.imwrite("imgs/" + str(name) + ".png", img1)


def _lidar_image_fov(pc_velo, calib, bounds, clip_distance):
    xmin, ymin, xmax, ymax = bounds
    pts_2d = calib.project_velo_to_image(pc_velo)
    fov_inds = (
        (pts_2d[:, 0] < xmax)
        & (pts_2d[:, 0] >= xmin)
        & (pts_2d[:, 1] < ymax)
        & (pts_2d[:, 1] >= ymin)
    )
    fov_inds = fov_inds & (pc_velo[:, 0] > clip_distance)
    return pts_2d, fov_inds


def get_lidar_in_image_fov(*call_args, **call_kwargs):
    """Filter lidar points, keep those in image FOV"""
    bound_parameters = _bind_parameters(
        _GET_LIDAR_IN_IMAGE_FOV_PARAMETERS_SCHEMA, call_args, call_kwargs
    )
    pc_velo, calib, xmin, ymin, xmax, ymax, return_more, clip_distance = (
        bound_parameters
    )
    bounds = xmin, ymin, xmax, ymax
    pts_2d, fov_inds = _lidar_image_fov(pc_velo, calib, bounds, clip_distance)
    imgfov_pc_velo = pc_velo[fov_inds, :]
    if return_more:
        return imgfov_pc_velo, pts_2d, fov_inds
    else:
        return imgfov_pc_velo


def get_lidar_index_in_image_fov(*call_args, **call_kwargs):
    """Filter lidar points, keep those in image FOV"""
    bound_parameters = _bind_parameters(
        _GET_LIDAR_INDEX_IN_IMAGE_FOV_PARAMETERS_SCHEMA, call_args, call_kwargs
    )
    pc_velo, calib, xmin, ymin, xmax, ymax, return_more, clip_distance = (
        bound_parameters
    )
    bounds = xmin, ymin, xmax, ymax
    _, fov_inds = _lidar_image_fov(pc_velo, calib, bounds, clip_distance)
    return fov_inds


def depth_region_pt3d(depth, obj):
    b = obj.box2d
    pt3d = []
    for i in range(int(b[0]), int(b[2])):
        for j in range(int(b[1]), int(b[3])):
            pt3d.append([j, i, depth[j, i]])
    return np.array(pt3d)


def get_depth_pt3d(depth):
    pt3d = []
    for i in range(depth.shape[0]):
        for j in range(depth.shape[1]):
            pt3d.append([i, j, depth[i, j]])
    return np.array(pt3d)


def _draw_depth_cloud(depth, calib, constraint_box, fig, save):
    from viz_util import draw_lidar

    depth_velo = calib.project_depth_to_velo(depth, constraint_box)
    intensity = np.ones((depth_velo.shape[0], 1)) * 0.5
    depth_velo = np.hstack((depth_velo, intensity))
    LOGGER.info("%s %s", "depth_pc_velo:", depth_velo.shape)
    LOGGER.info("%s %s", "depth_pc_velo:", type(depth_velo))
    LOGGER.info("%s", depth_velo[:5])
    draw_lidar(depth_velo, fig=fig, pts_color=(1, 1, 1))
    if save:
        save_filename = os.path.join("data/object/training/depth_pc", "%06d.bin" % 0)
        LOGGER.info("%s", save_filename)
        depth_velo.astype(np.float32).tofile(save_filename)


def _draw_colored_lidar_boxes(objects, calib, fig):
    from viz_util import draw_gt_boxes3d

    colors = {"Car": (0, 1, 0), "Pedestrian": (0, 1, 1), "Cyclist": (1, 1, 0)}
    for obj in objects:
        if obj.type not in colors:
            continue
        _, box3d = utils.compute_box_3d(obj, calib.P)
        box3d_velo = calib.project_rect_to_velo(box3d)
        LOGGER.info("%s", "box3d_pts_3d_velo:")
        LOGGER.info("%s", box3d_velo)
        draw_gt_boxes3d([box3d_velo], fig=fig, color=colors[obj.type], label=obj.type)


def show_lidar_with_depth(*call_args, **call_kwargs):
    """Show LiDAR points and 3D boxes in the velocity coordinate system."""
    bound_parameters = _bind_parameters(
        _SHOW_LIDAR_WITH_DEPTH_PARAMETERS_SCHEMA, call_args, call_kwargs
    )
    (
        pc_velo,
        objects,
        calib,
        fig,
        img_fov,
        img_width,
        img_height,
        objects_pred,
        depth,
        cam_img,
        constraint_box,
        pc_label,
        save,
    ) = bound_parameters
    import mayavi.mlab as mlab
    from viz_util import draw_lidar

    LOGGER.info("%s", ("All point num: ", pc_velo.shape[0]))
    if img_fov:
        pc_velo_index = get_lidar_index_in_image_fov(
            pc_velo[:, :3], calib, 0, 0, img_width, img_height
        )
        pc_velo = pc_velo[pc_velo_index, :]
        LOGGER.info("%s", ("FOV point num: ", pc_velo.shape))
    LOGGER.info("%s %s", "pc_velo", pc_velo.shape)
    draw_lidar(pc_velo, fig=fig, pc_label=pc_label)

    if depth is not None:
        _draw_depth_cloud(depth, calib, constraint_box, fig, save)

    _draw_colored_lidar_boxes(objects, calib, fig)

    if objects_pred is not None:
        _draw_lidar_objects(objects_pred, calib, (fig, mlab), (1, 0, 0))
    mlab.show(1)


def save_depth0(*call_args, **call_kwargs):
    bound_parameters = _bind_parameters(
        _SAVE_DEPTH0_PARAMETERS_SCHEMA, call_args, call_kwargs
    )
    data_idx, pc_velo, calib, img_fov, img_width, img_height, depth, constraint_box = (
        bound_parameters
    )
    if img_fov:
        pc_velo_index = get_lidar_index_in_image_fov(
            pc_velo[:, :3], calib, 0, 0, img_width, img_height
        )
        pc_velo = pc_velo[pc_velo_index, :]
        entry_type = np.zeros((pc_velo.shape[0], 1))
        pc_velo = np.hstack((pc_velo, entry_type))
        LOGGER.info("%s", ("FOV point num: ", pc_velo.shape))
    # Draw depth
    if depth is not None:
        depth_pc_velo = calib.project_depth_to_velo(depth, constraint_box)

        indensity = np.ones((depth_pc_velo.shape[0], 1)) * 0.5
        depth_pc_velo = np.hstack((depth_pc_velo, indensity))

        entry_type = np.ones((depth_pc_velo.shape[0], 1))
        depth_pc_velo = np.hstack((depth_pc_velo, entry_type))
        LOGGER.info("%s %s", "depth_pc_velo:", depth_pc_velo.shape)

        depth_pc = np.concatenate((pc_velo, depth_pc_velo), axis=0)
        LOGGER.info("%s %s", "depth_pc:", depth_pc.shape)

    vely_dir = "data/object/training/depth_pc"
    save_filename = os.path.join(vely_dir, "%06d.bin" % (data_idx))

    depth_pc = depth_pc.astype(np.float32)
    depth_pc.tofile(save_filename)


def save_depth(*call_args, **call_kwargs):
    bound_parameters = _bind_parameters(
        _SAVE_DEPTH_PARAMETERS_SCHEMA, call_args, call_kwargs
    )
    data_idx, pc_velo, calib, img_fov, img_width, img_height, depth, constraint_box = (
        bound_parameters
    )
    if depth is not None:
        depth_pc_velo = calib.project_depth_to_velo(depth, constraint_box)

        indensity = np.ones((depth_pc_velo.shape[0], 1)) * 0.5
        depth_pc = np.hstack((depth_pc_velo, indensity))

        LOGGER.info("%s %s", "depth_pc:", depth_pc.shape)

    vely_dir = "data/object/training/depth_pc"
    save_filename = os.path.join(vely_dir, "%06d.bin" % (data_idx))

    depth_pc = depth_pc.astype(np.float32)
    depth_pc.tofile(save_filename)


def _draw_lidar_objects(objects, calib, rendering, color, depth=None):
    fig, mlab = rendering
    from viz_util import draw_lidar, draw_gt_boxes3d

    for obj in objects:
        if obj.type == "DontCare":
            continue
        _, box3d = utils.compute_box_3d(obj, calib.P)
        box3d_velo = calib.project_rect_to_velo(box3d)
        LOGGER.info("%s", "box3d_pts_3d_velo:")
        LOGGER.info("%s", box3d_velo)
        draw_gt_boxes3d([box3d_velo], fig=fig, color=color)
        if depth is not None:
            depth_pt3d = depth_region_pt3d(depth, obj)
            depth_uv = np.zeros_like(depth_pt3d)
            depth_uv[:, 0], depth_uv[:, 1] = depth_pt3d[:, 1], depth_pt3d[:, 0]
            depth_uv[:, 2] = depth_pt3d[:, 2]
            depth_velo = calib.project_image_to_velo(depth_uv)
            LOGGER.info("%s %s", "dep_pc_velo:", depth_velo)
            draw_lidar(depth_velo, fig=fig, pts_color=(1, 1, 1))
        _, orientation = utils.compute_orientation_3d(obj, calib.P)
        orientation = calib.project_rect_to_velo(orientation)
        start, end = orientation[0, :], orientation[1, :]
        mlab.plot3d(
            [start[0], end[0]], [start[1], end[1]], [start[2], end[2]],
            color=color, tube_radius=None, line_width=1, figure=fig,
        )


def show_lidar_with_boxes(*call_args, **call_kwargs):
    """Show LiDAR points and 3D boxes in the velocity coordinate system."""
    bound_parameters = _bind_parameters(
        _SHOW_LIDAR_WITH_BOXES_PARAMETERS_SCHEMA, call_args, call_kwargs
    )
    (
        pc_velo,
        objects,
        calib,
        img_fov,
        img_width,
        img_height,
        objects_pred,
        depth,
        cam_img,
    ) = bound_parameters
    import mayavi.mlab as mlab
    from viz_util import draw_lidar

    LOGGER.info("%s", ("All point num: ", pc_velo.shape[0]))
    fig = mlab.figure(
        figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1000, 500)
    )
    if img_fov:
        pc_velo = get_lidar_in_image_fov(
            pc_velo[:, 0:3], calib, 0, 0, img_width, img_height
        )
        LOGGER.info("%s", ("FOV point num: ", pc_velo.shape[0]))
    LOGGER.info("%s %s", "pc_velo", pc_velo.shape)
    draw_lidar(pc_velo, fig=fig)

    _draw_lidar_objects(objects, calib, (fig, mlab), (0, 1, 0), depth)
    if objects_pred is not None:
        _draw_lidar_objects(objects_pred, calib, (fig, mlab), (1, 0, 0))
    mlab.show()


def box_min_max(box3d):
    box_min = np.min(box3d, axis=0)
    box_max = np.max(box3d, axis=0)
    return box_min, box_max


def get_velo_whl(box3d, pc):
    bmin, bmax = box_min_max(box3d)
    ind = np.where(
        (pc[:, 0] >= bmin[0])
        & (pc[:, 0] <= bmax[0])
        & (pc[:, 1] >= bmin[1])
        & (pc[:, 1] <= bmax[1])
        & (pc[:, 2] >= bmin[2])
        & (pc[:, 2] <= bmax[2])
    )[0]
    if len(ind) > 0:
        vmin, vmax = box_min_max(pc[ind, :])
        return vmax - vmin
    else:
        return 0, 0, 0, 0


def stat_lidar_with_boxes(pc_velo, objects, calib):
    """Log LiDAR statistics for 3D boxes in the velocity coordinate system."""

    for obj in objects:
        if obj.type == "DontCare":
            continue
        _, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
        box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
        v_l, v_w, v_h, _ = get_velo_whl(box3d_pts_3d_velo, pc_velo)
        LOGGER.info("%s", "%.4f %.4f %.4f %s" % (v_w, v_h, v_l, obj.type))


def get_lidar_on_image(pc_velo, calib, img_width, img_height):
    """Project LiDAR points to image"""
    pc_velo = pc_velo[:, 0:3]
    imgfov_pc_velo, pts_2d, fov_inds = get_lidar_in_image_fov(
        pc_velo, calib, 0, 0, img_width, img_height, True
    )
    imgfov_pts_2d = pts_2d[fov_inds, :]
    imgfov_pc_rect = calib.project_velo_to_rect(imgfov_pc_velo)

    return imgfov_pc_rect, imgfov_pts_2d


def _lidar_projection_context(pc_velo, img, calib, img_width, img_height):
    img = np.copy(img)
    imgfov_pc_velo, pts_2d, fov_inds = get_lidar_in_image_fov(
        pc_velo, calib, 0, 0, img_width, img_height, True
    )
    imgfov_pts_2d = pts_2d[fov_inds, :]
    imgfov_pc_rect = calib.project_velo_to_rect(imgfov_pc_velo)

    import matplotlib.pyplot as plt

    cmap = plt.cm.get_cmap("hsv", 256)
    cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255
    return img, imgfov_pts_2d, imgfov_pc_rect, cmap


def show_lidar_on_image(pc_velo, img, calib, img_width, img_height):
    """Project LiDAR points to image"""
    img, imgfov_pts_2d, imgfov_pc_rect, cmap = _lidar_projection_context(
        pc_velo, img, calib, img_width, img_height
    )

    for i in range(imgfov_pts_2d.shape[0]):
        depth = imgfov_pc_rect[i, 2]
        ix = int(640.0 / depth)
        if ix >= 256:
            ix = 255
        color = cmap[ix, :]
        cv2.circle(
            img,
            (int(np.round(imgfov_pts_2d[i, 0])), int(np.round(imgfov_pts_2d[i, 1]))),
            2,
            color=tuple(color),
            thickness=-1,
        )
    cv2.imshow("projection", img)
    return img


def my_show_lidar_on_image(pc_velo, img, calib, img_width, img_height):
    """Project LiDAR points to image"""
    img, imgfov_pts_2d, imgfov_pc_rect, cmap = _lidar_projection_context(
        pc_velo, img, calib, img_width, img_height
    )

    for i in tqdm(range(imgfov_pts_2d.shape[0])):
        depth = imgfov_pc_rect[i, 2]
        ix = int(640.0 / depth)
        if ix >= 256:
            ix = 255
        _ = cmap[ix, :]
        y_index = int(np.round(imgfov_pts_2d[i, 0]))
        x_index = int(np.round(imgfov_pts_2d[i, 1]))
        if y_index >= img_width:
            y_index = img_width - 1
        if x_index >= img_height:
            x_index = img_height - 1
        img[x_index, y_index] = (255, 255, 0)

    return img


def _draw_topview_boxes(top_image, objects, calib, is_ground_truth):
    def bbox3d(obj):
        _, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
        return calib.project_rect_to_velo(box3d_pts_3d)

    selected = [obj for obj in objects if obj.type != "DontCare"]
    return utils.draw_box3d_on_top(
        top_image,
        np.array([bbox3d(obj) for obj in selected]),
        text_lables=[obj.type for obj in selected],
        scores=None,
        thickness=1,
        is_gt=is_ground_truth,
    )


def _populate_topview(top_image, objects, calib, objects_pred):
    top_image = _draw_topview_boxes(top_image, objects, calib, True)
    if objects_pred is not None:
        top_image = _draw_topview_boxes(top_image, objects_pred, calib, False)
    return top_image


def show_lidar_topview_with_boxes(pc_velo, objects, calib, objects_pred=None):
    """Show LiDAR boxes in a top-view OpenCV window."""
    top_image = utils.draw_top_image(utils.lidar_to_top(pc_velo))
    LOGGER.info("%s %s", "top_image:", top_image.shape)
    top_image = _populate_topview(top_image, objects, calib, objects_pred)

    cv2.imshow("top_image", top_image)
    return top_image


def my_show_lidar_topview_with_boxes(pc_velo, objects, calib, objects_pred=None):
    """Render LiDAR boxes in a Matplotlib top view."""
    import matplotlib.pyplot as plt

    top_image = utils.draw_top_image(utils.lidar_to_top(pc_velo))
    plt.imshow(top_image)
    plt.show()
    LOGGER.info("%s %s", "top_image:", top_image.shape)
    return _populate_topview(top_image, objects, calib, objects_pred)


def _load_visualization_sample(dataset, data_index, args):
    objects = dataset.get_label_objects(data_index) if args.split == "training" else []
    predicted = dataset.get_pred_objects(data_index) if args.pred else None
    if args.pred and predicted is None:
        return None
    if predicted is None:
        LOGGER.info("%s", "no pred file")
    dimensions = 5 if args.pc_label else 4
    dtype = np.float64 if args.dtype64 else np.float32
    point_cloud = dataset.get_lidar(data_index, dtype, dimensions)[:, :dimensions]
    calibration = dataset.get_calibration(data_index)
    image = dataset.get_image(data_index)
    LOGGER.info("%s image shape: %s", data_index, image.shape)
    LOGGER.info("%s velo shape: %s", data_index, point_cloud.shape)
    depth = None
    if args.depth:
        depth, _ = dataset.get_depth(data_index)
        LOGGER.info("%s depth shape: %s", data_index, depth.shape)
    return VisualizationSample(
        point_cloud, objects, predicted, calibration, image, depth
    )


def _log_ground_truth_objects(objects):
    LOGGER.info("%s", "======== Objects in Ground Truth ========")
    object_index = 0
    for object_3d in objects:
        if object_3d.type == "DontCare":
            continue
        object_index += 1
        LOGGER.info("=== %d object ===", object_index)
        object_3d.print_object()


def _show_visualization(sample, args, figure):
    point_cloud, objects, predicted, calibration, image, depth = sample
    image_height, image_width, _ = image.shape
    if args.show_lidar_topview_with_boxes:
        show_lidar_topview_with_boxes(point_cloud, objects, calibration, predicted)
    if args.show_image_with_boxes:
        show_image_with_boxes(image, objects, calibration, True, depth)
    if args.show_lidar_with_depth:
        show_lidar_with_depth(
            point_cloud,
            objects,
            calibration,
            figure,
            args.img_fov,
            image_width,
            image_height,
            predicted,
            depth,
            image,
            constraint_box=args.const_box,
            save=args.save_depth,
            pc_label=args.pc_label,
        )
    if args.show_lidar_on_image:
        show_lidar_on_image(
            point_cloud[:, :3], image, calibration, image_width, image_height
        )


def dataset_viz(root_dir, args):
    dataset = KittiObject(root_dir, split=args.split, args=args)
    figure = None
    if args.show_lidar_with_depth:
        import mayavi.mlab as mlab

        figure = mlab.figure(
            figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1000, 500)
        )
    for data_idx, _ in enumerate(dataset):
        if args.ind > 0:
            data_idx = args.ind
        sample = _load_visualization_sample(dataset, data_idx, args)
        if sample is None:
            continue
        point_cloud, objects, _, calibration, _, _ = sample
        if args.stat:
            stat_lidar_with_boxes(point_cloud, objects, calibration)
            continue
        _log_ground_truth_objects(objects)
        _show_visualization(sample, args, figure)
        input_str = raw_input()
        if args.show_lidar_with_depth:
            mlab.clf()
        if input_str == "killall":
            break


def depth_to_lidar_format(root_dir, args):
    dataset = KittiObject(root_dir, split=args.split, args=args)
    for data_idx, _ in enumerate(dataset):
        # Load data from dataset

        pc_velo = dataset.get_lidar(data_idx)[:, 0:4]
        calib = dataset.get_calibration(data_idx)
        depth, _ = dataset.get_depth(data_idx)
        img = dataset.get_image(data_idx)
        img_height, img_width, _ = img.shape
        LOGGER.info("%s %s %s", data_idx, "image shape: ", img.shape)
        LOGGER.info("%s %s %s", data_idx, "velo  shape: ", pc_velo.shape)
        LOGGER.info("%s %s %s", data_idx, "depth shape: ", depth.shape)

        save_depth(
            data_idx,
            pc_velo,
            calib,
            args.img_fov,
            img_width,
            img_height,
            depth,
            constraint_box=args.const_box,
        )


def read_det_file(det_filename):
    """Parse lines in 2D detection output files"""
    objects = {}
    with open(det_filename, "r") as f:
        for line in f.readlines():
            obj = utils.Object2d(line.rstrip())
            if obj.img_name not in objects.keys():
                objects[obj.img_name] = []
            objects[obj.img_name].append(obj)

    return objects


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KIITI Object Visualization")
    parser.add_argument(
        "-d",
        "--dir",
        type=str,
        default="_adv_tempdir/cloc_workplace",
        metavar="N",
        help="input  (default: data/object)",
    )
    parser.add_argument(
        "-i",
        "--ind",
        type=int,
        default=0,
        metavar="N",
        help="input  (default: data/object)",
    )
    parser.add_argument(
        "-p", "--pred", action="store_true", help="show predict results"
    )
    parser.add_argument(
        "-s",
        "--stat",
        action="store_true",
        help=" stat the w/h/l of point cloud in gt bbox",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="training",
        help="use training split or testing split (default: training)",
    )
    parser.add_argument(
        "-l",
        "--lidar",
        type=str,
        default="velodyne",
        metavar="N",
        help="velodyne dir  (default: velodyne)",
    )
    parser.add_argument(
        "-e",
        "--depthdir",
        type=str,
        default="depth",
        metavar="N",
        help="depth dir  (default: depth)",
    )
    parser.add_argument(
        "-r",
        "--preddir",
        type=str,
        default="pred",
        metavar="N",
        help="predicted boxes  (default: pred)",
    )
    parser.add_argument("--gen_depth", action="store_true", help="generate depth")
    parser.add_argument("--vis", action="store_true", help="show images")
    parser.add_argument("--depth", action="store_true", help="load depth")
    parser.add_argument("--img_fov", action="store_true", help="front view mapping")
    parser.add_argument("--const_box", action="store_true", help="constraint box")
    parser.add_argument(
        "--save_depth", action="store_true", help="save depth into file"
    )
    parser.add_argument(
        "--pc_label", action="store_true", help="5-verctor lidar, pc with label"
    )
    parser.add_argument(
        "--dtype64", action="store_true", help="for float64 datatype, default float64"
    )

    parser.add_argument(
        "--show_lidar_on_image", action="store_true", help="project lidar on image"
    )
    parser.add_argument(
        "--show_lidar_with_depth",
        action="store_true",
        help="--show_lidar, depth is supported",
    )
    parser.add_argument(
        "--show_image_with_boxes", action="store_true", help="show lidar"
    )
    parser.add_argument(
        "--show_lidar_topview_with_boxes",
        action="store_true",
        help="show lidar topview",
    )
    args = parser.parse_args()
    if args.pred:
        LOGGER.info("%s", args.dir + "/" + args.split + "/pred")
        if not (os.path.exists(args.dir + "/" + args.split + "/pred")):
            raise AssertionError

    if args.vis:
        dataset_viz(args.dir, args)
    if args.gen_depth:
        depth_to_lidar_format(args.dir, args)
