# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import NamedTuple
from enum import Enum

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pyqtgraph.opengl.GLGraphicsItem import GLGraphicsItem
from pyqtgraph.Qt import QtCore, QtGui
import matplotlib.patches as patches
import numpy as np
import pyqtgraph.opengl as gl

from second.core.box_np_ops import minmax_to_corner_3d
from second.utils.qt_gl import GLLabelItem as _SharedGLLabelItem
from second.utils.qt_gl import pltcolor_to_qtcolor as _pltcolor_to_qtcolor
from shared.parameters import ParameterSchema as _ParameterSchema, bind_parameters as _bind_parameters


class GltextitemInitParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    x: object
    y: object
    z: object
    text: object
    color: object


class DrawBboxInAxParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    ax: object
    bboxes: object
    rotations: object
    fmt: object
    labels: object
    label_size: object
    edgecolors: object
    linestyle: object
    alpha: object


class Draw2dBboxInAxParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    ax: object
    bboxes: object
    colors: object
    alpha: object
    with_arrow: object
    behind_axes: object


class Draw3dBboxlinesInPyqtParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    widget: object
    bboxes: object
    colors: object
    width: object
    labels: object
    alpha: object
    label_color: object
    line_item: object
    label_item: object


DrawBboxlinesInPyqtParameters = Draw3dBboxlinesInPyqtParameters


class Draw3dBboxlinesInPyqtV1Parameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    widget: object
    bboxes: object
    colors: object
    width: object
    labels: object
    label_color: object


class FORMAT(Enum):
    """enum that indicate format of a bbox"""

    Center = "format_bbox_centor"
    Corner = "format_bbox_corner"
    Length = "format_bbox_length"


class GLColor(Enum):
    Red = (1.0, 0.0, 0.0)
    Lime = (0.0, 1.0, 0.0)
    Green = (0.0, 0.5, 0.0)
    Blue = (0.0, 0.0, 1.0)
    Gray = (0.5, 0.5, 0.5)
    Yellow = (1.0, 1.0, 0.0)
    Write = (1.0, 1.0, 1.0)
    Cyan = (0.0, 1.0, 1.0)
    Magenta = (1.0, 0.0, 1.0)
    Silver = (0.75, 0.75, 0.75)
    Maroon = (0.5, 0.0, 0.0)
    Olive = (0.5, 0.5, 0.0)
    Teal = (0.0, 0.5, 0.5)
    Navy = (0.0, 0.0, 0.5)
    Purple = (0.5, 0.0, 0.5)


def gl_color(color, alpha=1.0):
    return (*color.value, alpha)


def corner_to_length(bboxes):
    if len(bboxes.shape) == 1:
        bboxes = bboxes.reshape([1, -1])
    ndim = bboxes.shape[1] // 2
    return np.concatenate(
        [bboxes[:, :ndim], bboxes[:, ndim:] - bboxes[:, :ndim]], axis=1
    )


_DRAW_BBOX_IN_AX_PARAMETERS_SCHEMA = _ParameterSchema(
    DrawBboxInAxParameters,
    (
        "ax",
        "bboxes",
        "rotations",
        "fmt",
        "labels",
        "label_size",
        "edgecolors",
        "linestyle",
        "alpha",
    ),
    {
        "rotations": None,
        "fmt": FORMAT.Corner,
        "labels": None,
        "label_size": "small",
        "edgecolors": "r",
        "linestyle": "dashed",
        "alpha": 0.5,
    },
    frozenset(()),
    9,
    None,
    None,
)
_DRAW2D_BBOX_IN_AX_PARAMETERS_SCHEMA = _ParameterSchema(
    Draw2dBboxInAxParameters,
    ("ax", "bboxes", "colors", "alpha", "with_arrow", "behind_axes"),
    {"colors": "r", "alpha": 0.5, "with_arrow": True, "behind_axes": None},
    frozenset(()),
    6,
    None,
    None,
)
_DRAW3D_BBOXLINES_IN_PYQT_PARAMETERS_SCHEMA = _ParameterSchema(
    Draw3dBboxlinesInPyqtParameters,
    (
        "widget",
        "bboxes",
        "colors",
        "width",
        "labels",
        "alpha",
        "label_color",
        "line_item",
        "label_item",
    ),
    {
        "colors": GLColor.Green,
        "width": 1.0,
        "labels": None,
        "alpha": 1.0,
        "label_color": "r",
        "line_item": None,
        "label_item": None,
    },
    frozenset(()),
    9,
    None,
    None,
)
_DRAW_BBOXLINES_IN_PYQT_PARAMETERS_SCHEMA = (
    _DRAW3D_BBOXLINES_IN_PYQT_PARAMETERS_SCHEMA
)
_DRAW3D_BBOXLINES_IN_PYQT_V1_PARAMETERS_SCHEMA = _ParameterSchema(
    Draw3dBboxlinesInPyqtV1Parameters,
    ("widget", "bboxes", "colors", "width", "labels", "label_color"),
    {"colors": (0.0, 1.0, 0.0, 1.0), "width": 1.0, "labels": None, "label_color": "r"},
    frozenset(()),
    6,
    None,
    None,
)
_GLTEXTITEM_INIT_PARAMETERS_SCHEMA = _ParameterSchema(
    GltextitemInitParameters,
    ("x", "y", "z", "text", "color"),
    {"x": None, "y": None, "z": None, "text": None, "color": None},
    frozenset(()),
    5,
    None,
    None,
)


def draw_bbox_in_ax(*call_args, **call_kwargs):
    bound_parameters = _bind_parameters(
        _DRAW_BBOX_IN_AX_PARAMETERS_SCHEMA, call_args, call_kwargs
    )
    ax, bboxes, rotations, fmt, labels, label_size, edgecolors, linestyle, alpha = (
        bound_parameters
    )
    if rotations is None:
        rotations = np.zeros([bboxes.shape[0]])
    else:
        rotations = rotations / np.pi * 180
    if labels is None:
        labels = [None] * bboxes.shape[0]
    if not isinstance(edgecolors, list):
        edgecolors = [edgecolors for i in range(len(bboxes))]
    if fmt == FORMAT.Corner:
        bboxes = corner_to_length(bboxes)
    for bbox, rot, e_color, label in zip(bboxes, rotations, edgecolors, labels):
        rect_p = patches.Rectangle(
            bbox[:2],
            bbox[2],
            bbox[3],
            rot,
            fill=False,
            edgecolor=e_color,
            linestyle=linestyle,
            alpha=alpha,
        )
        ax.add_patch(rect_p)
        if label is not None:
            _ = ax.text(
                bbox[0],
                bbox[1],
                label,
                ha="left",
                va="bottom",
                color=e_color,
                size=label_size,
            )
    return ax


def draw_3d_bbox_in_ax(ax, bboxes, colors="r", alpha=0.5, image_shape=None):
    # assume bboxes has right format.
    if not isinstance(colors, list):
        colors = [colors for i in range(len(bboxes))]
    for box, color in zip(bboxes, colors):
        box_a, box_b = box[:4], box[4:]
        for pa, pb in zip(box_a, box_a[[1, 2, 3, 0]]):
            ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color=color, alpha=alpha)
        for pa, pb in zip(box_b, box_b[[1, 2, 3, 0]]):
            ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color=color, alpha=alpha)
        for pa, pb in zip(box_a, box_b):
            ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color=color, alpha=alpha)
    if image_shape is not None:
        patch = patches.Rectangle([0, 0], image_shape[1], image_shape[0])
        ax.set_clip_path(patch)
    return ax


def draw_2d_bbox_in_ax(*call_args, **call_kwargs):
    bound_parameters = _bind_parameters(
        _DRAW2D_BBOX_IN_AX_PARAMETERS_SCHEMA, call_args, call_kwargs
    )
    ax, bboxes, colors, alpha, with_arrow, behind_axes = bound_parameters
    if behind_axes is None:
        behind_axes = [0, 1]
    if not isinstance(colors, list):
        colors = [colors for i in range(len(bboxes))]
    for box, color in zip(bboxes, colors):
        for pa, pb in zip(box, box[[1, 2, 3, 0]]):
            ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color=color, alpha=alpha)
        if with_arrow:
            center = np.mean(box, axis=0)
            start = np.mean(
                np.concatenate([center[np.newaxis, ...], box[behind_axes]]), axis=0
            )
            front_axes = [i for i in range(4) if i not in behind_axes]
            end = np.mean(
                np.concatenate([center[np.newaxis, ...], box[front_axes]]), axis=0
            )
            ax.arrow(
                start[0],
                start[1],
                end[0] - start[0],
                end[1] - start[1],
                head_width=0.2,
                head_length=0.2,
                fc=color,
                ec=color,
            )
    return ax


def draw_3d_bbox_in_3dax(ax, bboxes, colors="r", alpha=0.25, facecolors=None):
    if not isinstance(colors, list):
        colors = [colors for i in range(len(bboxes))]
    if not isinstance(facecolors, list):
        facecolors = [facecolors for i in range(len(bboxes))]

    for box, color, facecolor in zip(bboxes, colors, facecolors):
        ax.scatter3D(box[:, 0], box[:, 1], box[:, 2], marker=".", color=color)
        verts = np.array(
            [
                [box[0], box[1], box[2], box[3]],
                [box[4], box[5], box[6], box[7]],
                [box[0], box[3], box[7], box[4]],
                [box[1], box[2], box[6], box[5]],
                [box[0], box[1], box[5], box[4]],
                [box[3], box[2], box[6], box[7]],
            ]
        )
        mp3dcoll = Poly3DCollection(
            verts, linewidths=1, edgecolors=color, alpha=alpha, facecolors=facecolor
        )
        mp3dcoll.set_facecolor(facecolor)
        mp3dcoll.set_edgecolor(color)
        mp3dcoll.set_alpha(alpha)
        ax.add_collection3d(mp3dcoll)
    return ax


class GLTextItem(GLGraphicsItem):
    def __init__(self, *call_args, **call_kwargs):
        bound_parameters = _bind_parameters(
            _GLTEXTITEM_INIT_PARAMETERS_SCHEMA, call_args, call_kwargs
        )
        x, y, z, text, color = bound_parameters
        super().__init__()
        self.color = color
        if color is None:
            self.color = QtCore.Qt.white
        self.text = text
        self.x = x
        self.y = y
        self.z = z

    def set_gl_view_widget(self, gl_view_widget):
        self.gl_view_widget = gl_view_widget

    def set_text(self, text):
        self.text = text
        self.update()

    def set_x(self, x):
        self.x = x
        self.update()

    def set_y(self, y):
        self.y = y
        self.update()

    def set_z(self, z):
        self.z = z
        self.update()

    def paint(self):
        self.gl_view_widget.qglColor(self.color)
        self.gl_view_widget.renderText(self.x, self.y, self.z, self.text)


class GLLabelItem(_SharedGLLabelItem):
    def __init__(self, pos=None, text=None, color=None, font=None):
        super().__init__(pos, text, color, font)
        self.font.setPointSizeF(20)


def draw_bounding_box(widget, box_minmax, color):
    bbox = minmax_to_corner_3d(box_minmax)
    return draw_3d_bboxlines_in_pyqt(widget, bbox, color)


def _box_edge_lines(box):
    indices = (0, 1, 1, 2, 2, 3, 3, 0, 1, 5, 5, 4, 4, 0, 2, 6, 6, 7, 7, 3, 5, 6, 4, 7)
    return np.array([box[index] for index in indices])


def _update_line_item(widget, line_item, lines, colors, width):
    line_options = {
        "pos": lines,
        "color": colors,
        "width": width,
        "antialias": True,
        "mode": "lines",
    }
    if line_item is None:
        line_item = gl.GLLinePlotItem(**line_options)
        widget.addItem(line_item)
    else:
        line_item.setData(**line_options)
    return line_item


def _update_label_item(widget, label_item, bboxes, labels, label_color):
    if labels is None:
        return label_item
    color = _pltcolor_to_qtcolor(label_color)
    if label_item is None:
        label_item = GLLabelItem(bboxes[:, 0, :], labels, color)
        label_item.set_gl_view_widget(widget)
        widget.addItem(label_item)
    else:
        label_item.set_data(pos=bboxes[:, 0, :], text=labels, color=color)
    return label_item


def draw_3d_bboxlines_in_pyqt(*call_args, **call_kwargs):
    bound_parameters = _bind_parameters(
        _DRAW3D_BBOXLINES_IN_PYQT_PARAMETERS_SCHEMA, call_args, call_kwargs
    )
    widget, bboxes, colors, width, labels, alpha, label_color, line_item, label_item = (
        bound_parameters
    )
    if bboxes.shape[0] == 0:
        bboxes = np.zeros([0, 8, 3])
    if not isinstance(colors, (list, np.ndarray)):
        if isinstance(colors, GLColor):
            colors = gl_color(colors, alpha)
        colors = [colors for i in range(len(bboxes))]
    if not isinstance(labels, (list, np.ndarray)):
        labels = [labels for i in range(len(bboxes))]
    total_lines = []
    total_colors = []
    for box, facecolor in zip(bboxes, colors):
        lines = _box_edge_lines(box)
        total_lines.append(lines)
        color = np.array([list(facecolor) for i in range(len(lines))])
        total_colors.append(color)
    if bboxes.shape[0] != 0:
        total_lines = np.concatenate(total_lines, axis=0)
        total_colors = np.concatenate(total_colors, axis=0)
    else:
        total_lines = None
        total_colors = None
    line_item = _update_line_item(widget, line_item, total_lines, total_colors, width)
    label_item = _update_label_item(widget, label_item, bboxes, labels, label_color)
    return line_item, label_item


def draw_bboxlines_in_pyqt(*call_args, **call_kwargs):
    bound_parameters = _bind_parameters(
        _DRAW_BBOXLINES_IN_PYQT_PARAMETERS_SCHEMA, call_args, call_kwargs
    )
    widget, bboxes, colors, width, labels, alpha, label_color, line_item, label_item = (
        bound_parameters
    )
    if bboxes.shape[0] == 0:
        return None, None
    if not isinstance(colors, list):
        if isinstance(colors, GLColor):
            colors = gl_color(colors, alpha)
        colors = [colors for i in range(len(bboxes))]
    if not isinstance(labels, list):
        labels = [labels for i in range(len(bboxes))]
    total_lines = []
    total_colors = []
    for box, facecolor in zip(bboxes, colors):
        lines = np.array(
            [box[0], box[1], box[1], box[2], box[2], box[3], box[3], box[0]]
        )
        total_lines.append(lines)
        color = np.array([list(facecolor) for i in range(len(lines))])
        total_colors.append(color)
    total_lines = np.concatenate(total_lines, axis=0)
    total_colors = np.concatenate(total_colors, axis=0)
    line_item = _update_line_item(widget, line_item, total_lines, total_colors, width)
    label_item = _update_label_item(widget, label_item, bboxes, labels, label_color)

    return line_item, label_item


def _3d_bbox_to_mesh(bboxes):
    bbox_faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 5, 6],
            [4, 6, 7],
            [0, 4, 7],
            [0, 7, 3],
            [1, 5, 6],
            [1, 6, 2],
            [3, 2, 6],
            [3, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
        ]
    )
    verts_list = []
    faces_list = []
    for i, bbox in enumerate(bboxes):
        verts_list.append(bbox)
        faces_list.append(bbox_faces + 8 * i)
    verts = np.concatenate(verts_list, axis=0)
    faces = np.concatenate(faces_list, axis=0)
    return verts, faces


def draw_3d_bbox_meshes_in_pyqt(
    widget, bboxes, colors=GLColor.Gray, alpha=1.0, edgecolors=None
):
    verts, faces = _3d_bbox_to_mesh(bboxes)
    if not isinstance(colors, list):
        if isinstance(colors, GLColor):
            colors = gl_color(colors, alpha)
        colors = np.array([colors for i in range(len(verts))])
    m1 = gl.GLMeshItem(vertexes=verts, faces=faces, faceColors=colors, smooth=False)
    m1.setGLOptions("additive")
    widget.addItem(m1)
    return widget


def draw_3d_bboxlines_in_pyqt_v1(*call_args, **call_kwargs):
    bound_parameters = _bind_parameters(
        _DRAW3D_BBOXLINES_IN_PYQT_V1_PARAMETERS_SCHEMA, call_args, call_kwargs
    )
    widget, bboxes, colors, width, labels, label_color = bound_parameters
    if not isinstance(colors, list):
        colors = [colors for i in range(len(bboxes))]
    if not isinstance(labels, list):
        labels = [labels for i in range(len(bboxes))]
    for box, facecolor, label in zip(bboxes, colors, labels):
        lines = _box_edge_lines(box)
        color = np.array([list(facecolor) for i in range(len(lines))])
        line_item = gl.GLLinePlotItem(
            pos=lines, color=color, width=width, antialias=True, mode="lines"
        )
        widget.addItem(line_item)
        if label is not None:
            label_color_qt = _pltcolor_to_qtcolor(label_color)
            t = GLTextItem(
                X=box[0, 0], Y=box[0, 1], Z=box[0, 2], text=label, color=label_color_qt
            )
            t.set_gl_view_widget(widget)
            widget.addItem(t)

    return widget
