# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Author: wangxy

from typing import NamedTuple
from pathlib import Path
import glob
import random
import re
import sys

import cv2
import numpy as np
import torch


from shared.parameters import ParameterSchema as _ParameterSchema, bind_parameters as _bind_parameters


class PlotOneBoxParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    x: object
    im: object
    image_path: object
    color: object
    img0_name: object
    label: object
    line_thickness: object


class SaveOneBoxParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    xyxy: object
    im: object
    file: object
    gain: object
    pad: object
    square: object
    bgr: object


sys.path.append("..")


_PLOT_ONE_BOX_PARAMETERS_SCHEMA = _ParameterSchema(
    PlotOneBoxParameters,
    ("x", "im", "image_path", "color", "img0_name", "label", "line_thickness"),
    {"color": None, "img0_name": None, "label": None, "line_thickness": 3},
    frozenset(()),
    7,
    None,
    None,
)
_SAVE_ONE_BOX_PARAMETERS_SCHEMA = _ParameterSchema(
    SaveOneBoxParameters,
    ("xyxy", "im", "file", "gain", "pad", "square", "bgr"),
    {"file": "image.jpg", "gain": 1.02, "pad": 10, "square": False, "bgr": False},
    frozenset(()),
    7,
    None,
    None,
)


def plot_one_box(*call_args, **call_kwargs):
    bound_parameters = _bind_parameters(
        _PLOT_ONE_BOX_PARAMETERS_SCHEMA, call_args, call_kwargs
    )
    x, im, image_path, color, img0_name, label, line_thickness = bound_parameters
    if not (im.data.contiguous):
        raise AssertionError(
            "Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image."
        )
    tl = (
        line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1
    )  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[10]), int(x[11])), (int(x[12]), int(x[13]))
    cv2.rectangle(im, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)

    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(str(label), 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            im,
            str(label),
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )
    cv2.imwrite(image_path + "/" + "{}.png".format(img0_name), im)


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2


def save_one_box(*call_args, **call_kwargs):
    bound_parameters = _bind_parameters(
        _SAVE_ONE_BOX_PARAMETERS_SCHEMA, call_args, call_kwargs
    )
    xyxy, im, file, gain, pad, square, bgr = bound_parameters
    xyxy = torch.tensor(xyxy).view(-1, 4)
    b = xywh2xyxy(xyxy)  # boxes
    if square:
        b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # attempt rectangle to square
    b[:, 2:] = b[:, 2:] * gain + pad  # box wh * gain + pad
    xyxy = xywh2xyxy(b).long()
    clip_coords(xyxy, im.shape)
    crop = im[int(xyxy[0, 1]): int(xyxy[0, 3]), int(xyxy[0, 0]): int(xyxy[0, 2])]
    cv2.imwrite(
        str(increment_path(file, mkdir=True).with_suffix(".jpg")),
        crop if bgr else crop[..., ::-1],
    )


def increment_path(path, exist_ok=False, sep="", mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        suffix = path.suffix
        path = path.with_suffix("")
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        path = Path(f"{path}{sep}{n}{suffix}")  # update path
    directory = path if path.suffix == "" else path.parent  # directory
    if not directory.exists() and mkdir:
        directory.mkdir(parents=True, exist_ok=True)  # make directory
    return path
