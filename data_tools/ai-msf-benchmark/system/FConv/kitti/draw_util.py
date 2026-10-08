# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import NamedTuple
import logging

from kitti.KittiObject import KittiObject
from PIL import Image
import cv2
import kitti.kitti_util as utils
import matplotlib.pyplot as plt
import numpy as np
from shared.box_geometry import draw_projected_box


from shared.parameters import ParameterSchema as _ParameterSchema, bind_parameters as _bind_parameters


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


class ShowImageWithBoxesParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    img: object
    objects: object
    calib: object
    show3d: object
    color: object
    scores: object
    colors: object
    show: object


class ShowLidarWithBoxesParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    pc_velo: object
    objects: object
    calib: object
    img_fov: object
    img_width: object
    img_height: object


LOGGER = logging.getLogger(__name__)


_GET_LIDAR_IN_IMAGE_FOV_PARAMETERS_SCHEMA = _ParameterSchema(
    GetLidarInImageFovParameters,
    (
        "pc_velo",
        "calib",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
        "return_more",
        "clip_distance",
    ),
    {"return_more": False, "clip_distance": 2.0},
    frozenset(()),
    8,
    None,
    None,
)
_SHOW_IMAGE_WITH_BOXES_PARAMETERS_SCHEMA = _ParameterSchema(
    ShowImageWithBoxesParameters,
    ("img", "objects", "calib", "show3d", "color", "scores", "colors", "show"),
    {
        "show3d": True,
        "color": (0, 255, 0),
        "scores": None,
        "colors": None,
        "show": True,
    },
    frozenset(()),
    8,
    None,
    None,
)
_SHOW_LIDAR_WITH_BOXES_PARAMETERS_SCHEMA = _ParameterSchema(
    ShowLidarWithBoxesParameters,
    ("pc_velo", "objects", "calib", "img_fov", "img_width", "img_height"),
    {"img_fov": False, "img_width": None, "img_height": None},
    frozenset(()),
    6,
    None,
    None,
)


def get_lidar_in_image_fov(*call_args, **call_kwargs):
    """Filter lidar points, keep those in image FOV"""
    bound_parameters = _bind_parameters(
        _GET_LIDAR_IN_IMAGE_FOV_PARAMETERS_SCHEMA, call_args, call_kwargs
    )
    pc_velo, calib, xmin, ymin, xmax, ymax, return_more, clip_distance = (
        bound_parameters
    )
    pts_2d = calib.project_velo_to_image(pc_velo[:, :3])
    fov_inds = (
        (pts_2d[:, 0] < xmax)
        & (pts_2d[:, 0] >= xmin)
        & (pts_2d[:, 1] < ymax)
        & (pts_2d[:, 1] >= ymin)
    )
    fov_inds = fov_inds & (pc_velo[:, 0] > clip_distance)
    imgfov_pc_velo = pc_velo[fov_inds, :]
    if return_more:
        return imgfov_pc_velo, pts_2d, fov_inds
    else:
        return imgfov_pc_velo


def _draw_box_score(image, score, bounds, color):
    x_min, y_min, x_max, y_max = bounds
    if (x_max - x_min) * (y_max - y_min) <= 25:
        return
    cv2.putText(
        image,
        "%.2f" % score,
        (int((x_min + x_max) / 2), max(int(y_min) - 3, 0)),
        cv2.FONT_HERSHEY_PLAIN,
        fontScale=0.8,
        color=color,
        thickness=1,
        lineType=cv2.LINE_AA,
    )


def show_image_with_boxes(*call_args, **call_kwargs):
    """Show image with 2D bounding boxes"""
    bound_parameters = _bind_parameters(
        _SHOW_IMAGE_WITH_BOXES_PARAMETERS_SCHEMA, call_args, call_kwargs
    )
    img, objects, calib, show3d, color, scores, colors, show = bound_parameters
    if scores is None:
        scores = []
    if colors is None:
        colors = []
    img1 = np.copy(img)  # for 2d bbox
    img2 = np.copy(img)  # for 3d bbox

    if len(colors) != 0:
        if not (len(colors) == len(objects)):
            raise AssertionError

    for i, obj in enumerate(objects):
        if obj.type == "DontCare":
            continue
        if len(colors) != 0:
            color = colors[i]
        cv2.rectangle(
            img1,
            (int(obj.xmin), int(obj.ymin)),
            (int(obj.xmax), int(obj.ymax)),
            color,
            2,
        )

        if len(scores) != 0:
            _draw_box_score(
                img1, scores[i], (obj.xmin, obj.ymin, obj.xmax, obj.ymax), color
            )

        box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
        if box3d_pts_2d is not None:
            img2 = draw_projected_box3d(img2, box3d_pts_2d, color)

            if len(scores) != 0:
                xx1, yy1 = np.min(box3d_pts_2d, 0)
                xx2, yy2 = np.max(box3d_pts_2d, 0)
                _draw_box_score(img2, scores[i], (xx1, yy1, xx2, yy2), color)

    if show:
        Image.fromarray(img1).show()
        if show3d:
            Image.fromarray(img2).show()

    return img1, img2


def show_lidar_with_boxes(*call_args, **call_kwargs):
    """Show LiDAR points and 3D boxes in the velocity coordinate system."""
    bound_parameters = _bind_parameters(
        _SHOW_LIDAR_WITH_BOXES_PARAMETERS_SCHEMA, call_args, call_kwargs
    )
    pc_velo, objects, calib, img_fov, img_width, img_height = bound_parameters
    import mayavi.mlab as mlab
    from utils.mayavi_viz_util import draw_gt_boxes3d, draw_lidar

    fig = mlab.figure(
        figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1000, 500)
    )
    if img_fov:
        pc_velo = get_lidar_in_image_fov(pc_velo, calib, 0, 0, img_width, img_height)
        LOGGER.info("%s %s", "FOV point num: ", pc_velo.shape[0])
    draw_lidar(pc_velo, fig=fig)

    for obj in objects:
        if obj.type == "DontCare":
            continue
        # Draw 3d bounding box
        box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
        box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
        # Draw heading arrow
        ori3d_pts_2d, ori3d_pts_3d = utils.compute_orientation_3d(obj, calib.P)
        ori3d_pts_3d_velo = calib.project_rect_to_velo(ori3d_pts_3d)
        x1, y1, z1 = ori3d_pts_3d_velo[0, :]
        x2, y2, z2 = ori3d_pts_3d_velo[1, :]
        draw_gt_boxes3d([box3d_pts_3d_velo], fig=fig)
        mlab.plot3d(
            [x1, x2],
            [y1, y2],
            [z1, z2],
            color=(0.5, 0.5, 0.5),
            tube_radius=None,
            line_width=1,
            figure=fig,
        )
    mlab.show(1)


def show_lidar_on_image(pc_velo, img, calib, img_width, img_height):
    """Project LiDAR points to image"""
    imgfov_pc_velo, pts_2d, fov_inds = get_lidar_in_image_fov(
        pc_velo, calib, 0, 0, img_width, img_height, True
    )
    imgfov_pts_2d = pts_2d[fov_inds, :]
    imgfov_pc_rect = calib.project_velo_to_rect(imgfov_pc_velo)

    cmap = plt.cm.get_cmap("hsv", 256)
    cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255

    for i in range(imgfov_pts_2d.shape[0]):
        depth = imgfov_pc_rect[i, 2]
        color = cmap[int(640.0 / depth), :]
        cv2.circle(
            img,
            (int(np.round(imgfov_pts_2d[i, 0])), int(np.round(imgfov_pts_2d[i, 1]))),
            2,
            color=tuple(color),
            thickness=-1,
        )
    Image.fromarray(img).show()
    return img


def draw_projected_box3d(image, qs, color=(255, 255, 255), thickness=2):
    """Draw 3d bounding box in image
    qs: (8,3) array of vertices for the 3d box in following order:
        1 -------- 0
       /|         /|
      2 -------- 3 .
      | |        | |
      . 5 -------- 4
      |/         |/
      6 -------- 7
    """
    return draw_projected_box(image, qs, cv2, color, thickness)


if __name__ == "__main__":
    dataset = KittiObject("data/kitti/training")

    for data_idx in range(len(dataset)):
        # Load data from dataset
        objects = dataset.get_label_objects(data_idx)
        objects[0].print_object()
        img = dataset.get_image(data_idx)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_height, img_width, img_channel = img.shape
        LOGGER.info("%s %s", "Image shape: ", img.shape)
        pc_velo = dataset.get_lidar(data_idx)[:, 0:3]
        calib = dataset.get_calibration(data_idx)

        show_image_with_boxes(img, objects, calib, False)
        input()
        show_lidar_with_boxes(pc_velo, objects, calib, True, img_width, img_height)
        input()
