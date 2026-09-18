# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import NamedTuple
import random

import cv2
import numpy as np

from datasets.coordinate_transformation import compute_box_3dto2d


from shared.parameters import ParameterSchema as _ParameterSchema, bind_parameters as _bind_parameters


class ShowImageWithBoxesParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    img2: object
    bbox3d_tmp: object
    image_path: object
    color: object
    img0_name: object
    label: object
    calib_file: object
    line_thickness: object


MAX_COLOR = 30
score_threshold = -10000
WIDTH = 1242
HEIGHT = 374


def draw_projected_box3d(image, qs, color=(255, 255, 255), thickness=2):
    """Draw 3d bounding box in image
    qs: (8,2) array of vertices for the 3d box in following order:
        1 -------- 0
       /|         /|
      2 -------- 3 .
      | |        | |
      . 5 -------- 4
      |/         |/
      6 -------- 7
    """
    if qs is not None:
        qs = qs.astype(np.int32)
        for k in range(0, 4):
            i, j = k, (k + 1) % 4
            image = cv2.line(
                image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness
            )  # use LINE_AA for opencv3

            i, j = k + 4, (k + 1) % 4 + 4
            image = cv2.line(
                image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness
            )

            i, j = k, k + 4
            image = cv2.line(
                image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness
            )
    return image


_SHOW_IMAGE_WITH_BOXES_PARAMETERS_SCHEMA = _ParameterSchema(
    ShowImageWithBoxesParameters,
    (
        "img2",
        "bbox3d_tmp",
        "image_path",
        "color",
        "img0_name",
        "label",
        "calib_file",
        "line_thickness",
    ),
    {},
    frozenset(()),
    8,
    None,
    None,
)


def show_image_with_boxes(*call_args, **call_kwargs):
    bound_parameters = _bind_parameters(
        _SHOW_IMAGE_WITH_BOXES_PARAMETERS_SCHEMA, call_args, call_kwargs
    )
    (
        img2,
        bbox3d_tmp,
        image_path,
        color,
        img0_name,
        label,
        calib_file,
        line_thickness,
    ) = bound_parameters
    box3d_pts_2d = compute_box_3dto2d(bbox3d_tmp, calib_file)
    tl = (
        line_thickness or round(0.002 * (img2.shape[0] + img2.shape[1]) / 2) + 1
    )  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    if box3d_pts_2d is not None:
        c1, c2 = (int(box3d_pts_2d[4, 0]), int(box3d_pts_2d[4, 1])), (
            int(box3d_pts_2d[3, 0]),
            int(box3d_pts_2d[3, 1]),
        )
    else:
        c1, c2 = (0, 0), (0, 0)
    color_tmp = color
    img2 = draw_projected_box3d(img2, box3d_pts_2d, color=color_tmp)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(str(label), 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img2, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img2,
            str(label),
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )
    cv2.imwrite(image_path + "/" + "{}.png".format(img0_name), img2)
