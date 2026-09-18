# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import NamedTuple
import logging

from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import pyqtSignal
import numpy as np
import pyqtgraph.opengl as gl

from second.utils.bbox_plot import GLColor
from second.utils.qt_gl import GLLabelItem, pltcolor_to_qtcolor as _pltcolor_to_qtcolor
from shared.parameters import ParameterSchema as _ParameterSchema, bind_parameters as _bind_parameters


class KittiglviewwidgetBoxes3dParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    name: object
    boxes: object
    colors: object
    width: object
    alpha: object


class World2cameraParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    points: object
    elevation: object
    azimuth: object
    distance: object
    center: object
    fov: object
    w: object
    h: object


class KittiglviewwidgetScatterParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    name: object
    points: object
    colors: object
    alphas: object
    size: object
    translucent: object


class KittiglviewwidgetCirclesParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    name: object
    poses: object
    radiuses: object
    colors: object
    num_points: object
    alphas: object
    width: object
    antialias: object


class KittiglviewwidgetLinesParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    name: object
    lines: object
    colors: object
    alphas: object
    width: object
    antialias: object


class KittiglviewwidgetSurfaceParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    name: object
    x: object
    y: object
    z: object
    colors: object
    alphas: object


class KittiglviewwidgetLabelsParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    name: object
    pos: object
    labels: object
    color: object
    size: object
    alpha: object


LOGGER = logging.getLogger(__name__)


def gl_color(color, alpha=1.0):
    return (*color.value, alpha)


def _extend_color_if_necessary(colors, shape, alphas):
    if not isinstance(shape, (list, tuple)):
        shape = [shape]
    num = np.prod(shape)
    if isinstance(colors, (list, tuple, np.ndarray)):
        colors_np = np.array(colors)
        if colors_np.shape[0] == 4 and len(colors_np.shape) == 1:
            colors = (*colors,)
        elif colors_np.shape[0] == 3 and len(colors_np.shape) == 1:
            colors = (*colors, 1.0)
    if not isinstance(colors, (list, np.ndarray)):
        if isinstance(alphas, (list, tuple, np.ndarray)):
            colors = []
            for i in range(num):
                colors.append(gl_color(colors, alphas[i]))
        else:
            if isinstance(colors, GLColor):
                colors = gl_color(colors, alphas)
            colors = [colors for i in range(num)]
    return np.array(colors).reshape([num, 4])


def _glcolor_to_qtcolor(glcolor):
    color = np.array(gl_color(glcolor)) * 255
    return QtGui.QColor(*color.astype(np.int32))


def get_rotation_matrix_3d(angle, axis=0):
    # return counter-clockwise rotation matrix per axis
    rot_sin = np.sin(angle)
    rot_cos = np.cos(angle)
    if axis == 1:
        rot_mat_t = np.stack([[rot_cos, 0, -rot_sin], [0, 1, 0], [rot_sin, 0, rot_cos]])
    elif axis == 2 or axis == -1:
        rot_mat_t = np.stack([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]])
    elif axis == 0:
        rot_mat_t = np.stack([[0, rot_cos, -rot_sin], [0, rot_sin, rot_cos], [1, 0, 0]])
    else:
        raise ValueError("axis should in range")
    return rot_mat_t.T


def get_c(fov, w, h):
    f = w / np.tan(fov / 2) / 2
    c = np.array([[f, 0, w / 2], [0, -f, h / 2], [0, 0, 1]], dtype=np.float64)
    return c


def get_rt(elevation, azimuth, distance, center):
    ry = get_rotation_matrix_3d(-elevation, axis=1)
    rz = get_rotation_matrix_3d(-(-azimuth + np.pi), axis=2)

    t0 = np.array([distance, 0, 0], dtype=np.float64)
    r = rz @ ry
    t = t0 - r @ np.array(center, dtype=np.float64)
    rect = np.array([[0, -1, 0], [0, 0, 1], [1, 0, 0]])
    return rect @ r, rect @ t


def get_rt_(elevation, azimuth, distance, center):
    ry = get_rotation_matrix_3d(-elevation, axis=1)
    rz = get_rotation_matrix_3d(-(-azimuth + np.pi), axis=2)

    t0 = np.array([distance, 0, 0], dtype=np.float64)
    r = rz @ ry
    t = t0 - r @ np.array(center, dtype=np.float64)
    return r, t


_WORLD2CAMERA_PARAMETERS_SCHEMA = _ParameterSchema(
    World2cameraParameters,
    ("points", "elevation", "azimuth", "distance", "center", "fov", "w", "h"),
    {},
    frozenset(()),
    8,
    None,
    None,
)
_KITTIGLVIEWWIDGET_SCATTER_PARAMETERS_SCHEMA = _ParameterSchema(
    KittiglviewwidgetScatterParameters,
    ("name", "points", "colors", "alphas", "size", "translucent"),
    {"colors": GLColor.Write, "alphas": 0.5, "size": 0.1, "translucent": False},
    frozenset(()),
    6,
    None,
    None,
)
_KITTIGLVIEWWIDGET_CIRCLES_PARAMETERS_SCHEMA = _ParameterSchema(
    KittiglviewwidgetCirclesParameters,
    (
        "name",
        "poses",
        "radiuses",
        "colors",
        "num_points",
        "alphas",
        "width",
        "antialias",
    ),
    {
        "colors": GLColor.Red,
        "num_points": 100,
        "alphas": 1.0,
        "width": 1.0,
        "antialias": True,
    },
    frozenset(()),
    8,
    None,
    None,
)
_KITTIGLVIEWWIDGET_LINES_PARAMETERS_SCHEMA = _ParameterSchema(
    KittiglviewwidgetLinesParameters,
    ("name", "lines", "colors", "alphas", "width", "antialias"),
    {"alphas": 1.0, "width": 1.0, "antialias": True},
    frozenset(()),
    6,
    None,
    None,
)
_KITTIGLVIEWWIDGET_SURFACE_PARAMETERS_SCHEMA = _ParameterSchema(
    KittiglviewwidgetSurfaceParameters,
    ("name", "x", "y", "z", "colors", "alphas"),
    {"alphas": 1.0},
    frozenset(()),
    6,
    None,
    None,
)
_KITTIGLVIEWWIDGET_LABELS_PARAMETERS_SCHEMA = _ParameterSchema(
    KittiglviewwidgetLabelsParameters,
    ("name", "pos", "labels", "color", "size", "alpha"),
    {"size": None, "alpha": 1.0},
    frozenset(()),
    6,
    None,
    None,
)
_KITTIGLVIEWWIDGET_BOXES3D_PARAMETERS_SCHEMA = _ParameterSchema(
    KittiglviewwidgetBoxes3dParameters,
    ("name", "boxes", "colors", "width", "alpha"),
    {"width": 1.0, "alpha": 1.0},
    frozenset(()),
    5,
    None,
    None,
)


def world2camera(*call_args, **call_kwargs):
    bound_parameters = _bind_parameters(
        _WORLD2CAMERA_PARAMETERS_SCHEMA, call_args, call_kwargs
    )
    points, elevation, azimuth, distance, center, fov, w, h = bound_parameters
    r, t = get_rt(elevation, azimuth, distance, center)
    c = get_c(fov, w, h)
    return c @ (r @ points + t)


def sph_to_device_pos(elevation, azimuth, distance, center):
    ry = get_rotation_matrix_3d(elevation, axis=1)
    rz = get_rotation_matrix_3d(-(azimuth + np.pi), axis=2)
    t0 = np.array([distance, 0, 0], dtype=np.float64)
    return rz @ ry @ t0 + rz @ ry @ np.array(center, dtype=np.float64)


def device_pos_to_sph(pos, center):
    pos -= center
    distance = np.linalg.norm(pos)
    elevation = np.arctan2(np.linalg.norm(pos[:2]), pos[2])
    azimuth = np.arctan2(pos[1], pos[0])
    return (elevation - np.pi / 2), (azimuth - np.pi), distance


def get_lines_for_circle(radius, num_point=100):
    rads = np.linspace(0, 2 * np.pi, num=num_point)
    points = np.stack([radius * np.cos(rads), radius * np.sin(rads)], axis=1)
    lines = np.stack([points, points[[*np.arange(1, rads.shape[0]), 0]]], axis=1)
    return lines


class KittiGLViewWidget(gl.GLViewWidget):
    mousePressed = pyqtSignal(tuple, name="MousePressed")

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self._named_items = {}
        self.noRepeatKeys.append(QtCore.Qt.Key_W)
        self.noRepeatKeys.append(QtCore.Qt.Key_S)
        self.noRepeatKeys.append(QtCore.Qt.Key_A)
        self.noRepeatKeys.append(QtCore.Qt.Key_D)
        self.w_gl_axis = gl.GLAxisItem(
            size=None, antialias=True, glOptions="translucent"
        )
        self.addItem(self.w_gl_axis)

    @property
    def camera_center(self):
        c = self.opts["center"]
        center = [c.x(), c.y(), c.z()]
        return np.array(center)

    @property
    def camera_position(self):
        c = self.opts["center"]
        center = [c.x(), c.y(), c.z()]
        elevation = self.opts["elevation"] / 180 * np.pi
        azimuth = self.opts["azimuth"] / 180 * np.pi
        distance = self.opts["distance"]

        camera_pos = sph_to_device_pos(elevation, azimuth, distance, center)
        return camera_pos

    @property
    def camera_vector(self):
        vec = self.camera_position - self.camera_center
        return vec / np.linalg.norm(vec)

    def scatter(self, *call_args, **call_kwargs):
        bound_parameters = _bind_parameters(
            _KITTIGLVIEWWIDGET_SCATTER_PARAMETERS_SCHEMA, call_args, call_kwargs
        )
        name, points, colors, alphas, size, translucent = bound_parameters
        colors = _extend_color_if_necessary(colors, points.shape[0], alphas)
        if name not in self._named_items:
            w_gl_scatter = gl.GLScatterPlotItem(
                pos=points, size=size, color=colors, pxMode=False
            )
            if translucent:
                w_gl_scatter.setGLOptions("translucent")
            self._named_items[name] = w_gl_scatter
            self.addItem(w_gl_scatter)
        else:
            self._named_items[name].setData(
                pos=points, size=size, color=colors, pxMode=False
            )

    def circles(self, *call_args, **call_kwargs):
        bound_parameters = _bind_parameters(
            _KITTIGLVIEWWIDGET_CIRCLES_PARAMETERS_SCHEMA, call_args, call_kwargs
        )
        name, poses, radiuses, colors, num_points, alphas, width, antialias = (
            bound_parameters
        )
        if not isinstance(radiuses, (list, tuple, np.ndarray)):
            if not (isinstance(poses, (list, tuple, np.ndarray))):
                raise AssertionError
            radiuses = np.full([len(poses)], radiuses)
        lines_list = []
        for pos, radius in zip(poses, radiuses):
            lines = get_lines_for_circle(radius, num_points)

            shape = [lines.shape[0], 2, 1]
            lines_with_z = np.concatenate([lines, np.full(shape, pos[-1])], axis=-1)
            lines_list.append(lines_with_z)
        lines = np.concatenate(lines_list, axis=0)
        return self.lines(name, lines.reshape(-1, 3), colors, alphas, width, antialias)

    def lines(self, *call_args, **call_kwargs):
        bound_parameters = _bind_parameters(
            _KITTIGLVIEWWIDGET_LINES_PARAMETERS_SCHEMA, call_args, call_kwargs
        )
        name, lines, colors, alphas, width, antialias = bound_parameters
        if lines is None:
            return

        colors = _extend_color_if_necessary(colors, lines.shape[0], alphas)
        if name not in self._named_items:
            w_gl_item = gl.GLLinePlotItem(
                pos=lines, color=colors, width=width, antialias=antialias, mode="lines"
            )
            self._named_items[name] = w_gl_item
            self.addItem(w_gl_item)
        else:
            self._named_items[name].setData(
                pos=lines, color=colors, width=width, antialias=antialias, mode="lines"
            )

    def surface(self, *call_args, **call_kwargs):
        bound_parameters = _bind_parameters(
            _KITTIGLVIEWWIDGET_SURFACE_PARAMETERS_SCHEMA, call_args, call_kwargs
        )
        name, x, y, z, colors, alphas = bound_parameters
        colors = _extend_color_if_necessary(colors, list(z.shape), alphas)
        if name not in self._named_items:
            w_gl_item = gl.GLSurfacePlotItem(
                x=x, y=y, z=z, shader=None, colors=colors, glOptions="translucent"
            )
            self._named_items[name] = w_gl_item
            self.addItem(w_gl_item)
        else:
            self._named_items[name].setData(x=x, y=y, z=z, colors=colors)

    def labels(self, *call_args, **call_kwargs):
        bound_parameters = _bind_parameters(
            _KITTIGLVIEWWIDGET_LABELS_PARAMETERS_SCHEMA, call_args, call_kwargs
        )
        name, pos, labels, color, size, alpha = bound_parameters
        font = QtGui.QFont()
        if size is not None:
            font.setPointSizeF(size)
        if isinstance(color, GLColor):
            color = _glcolor_to_qtcolor(color)
        if isinstance(color, str):
            color = _pltcolor_to_qtcolor(color)

        if name not in self._named_items:
            w_gl_item = GLLabelItem(pos=pos, text=labels, color=color, font=font)
            self._named_items[name] = w_gl_item
            w_gl_item.set_gl_view_widget(self)
            self.addItem(w_gl_item)
        else:
            self._named_items[name].set_data(
                pos=pos, text=labels, color=color, font=font
            )

    def remove(self, name):
        if name in self._named_items:
            try:
                self.removeItem(self._named_items[name])
                return self._named_items.pop(name)
            except (KeyError, RuntimeError):
                LOGGER.info("%s", "remove failed.")
        return None

    def boxes3d(self, *call_args, **call_kwargs):
        bound_parameters = _bind_parameters(
            _KITTIGLVIEWWIDGET_BOXES3D_PARAMETERS_SCHEMA, call_args, call_kwargs
        )
        name, boxes, colors, width, alpha = bound_parameters
        if boxes.shape[0] == 0:
            self.remove(name)
            return
        colors = _extend_color_if_necessary(colors, boxes.shape[0], alpha)
        total_lines = []
        total_colors = []
        for box, facecolor in zip(boxes, colors):
            lines = np.array(
                [
                    box[0],
                    box[1],
                    box[1],
                    box[2],
                    box[2],
                    box[3],
                    box[3],
                    box[0],
                    box[1],
                    box[5],
                    box[5],
                    box[4],
                    box[4],
                    box[0],
                    box[2],
                    box[6],
                    box[6],
                    box[7],
                    box[7],
                    box[3],
                    box[5],
                    box[6],
                    box[4],
                    box[7],
                ]
            )
            total_lines.append(lines)
            color = np.array([list(facecolor) for i in range(len(lines))])
            total_colors.append(color)
        if boxes.shape[0] != 0:
            total_lines = np.concatenate(total_lines, axis=0)
            total_colors = np.concatenate(total_colors, axis=0)
        else:
            total_lines = None
            total_colors = None
        self.lines(name, total_lines, total_colors, alphas=alpha, width=width)

    def set_camera_position(
        self, center=None, distance=None, elevation=None, azimuth=None
    ):
        if center is not None:
            self.opts["center"].setX(center[0])
            self.opts["center"].setY(center[1])
            self.opts["center"].setZ(center[2])
        if distance is not None:
            self.opts["distance"] = distance
        if elevation is not None:
            self.opts["elevation"] = elevation
        if azimuth is not None:
            self.opts["azimuth"] = azimuth
        self.update()

    def reset_camera(self):
        self.set_camera_position(
            center=(5, 0, 0), distance=20, azimuth=-180, elevation=30
        )
        self.update()

    def eval_key_state(self):
        speed = 2.0
        world_speed = 0.32
        actions = {
            QtCore.Qt.Key_Right: (self.orbit, {"azim": -speed, "elev": 0}),
            QtCore.Qt.Key_Left: (self.orbit, {"azim": speed, "elev": 0}),
            QtCore.Qt.Key_Up: (self.orbit, {"azim": 0, "elev": -speed}),
            QtCore.Qt.Key_Down: (self.orbit, {"azim": 0, "elev": speed}),
            QtCore.Qt.Key_W: (self.camera_move, {"dx": world_speed}),
            QtCore.Qt.Key_S: (self.camera_move, {"dx": -world_speed}),
            QtCore.Qt.Key_A: (self.camera_move, {"dy": world_speed}),
            QtCore.Qt.Key_D: (self.camera_move, {"dy": -world_speed}),
            QtCore.Qt.Key_Backspace: (self.reset_camera, {}),
        }
        if not self.keysPressed:
            self.keyTimer.stop()
            return
        for key in self.keysPressed:
            action = actions.get(key)
            if action is not None:
                callback, keywords = action
                callback(**keywords)
            self.keyTimer.start(16)

    def set_camera_position_cartesian(self, camera_pos, center):
        elevation, azimuth, distance = device_pos_to_sph(camera_pos, center)
        elevation = elevation * 180 / np.pi
        azimuth = azimuth * 180 / np.pi
        self.setCameraPosition(center, distance, elevation, azimuth)

    def camera_move(self, dx=0, dy=0, dz=0):
        xp = [dx, 0, 0]
        yp = [0, dy, 0]
        zp = [0, 0, dz]
        c = self.opts["center"]
        center = [c.x(), c.y(), c.z()]
        elevation = self.opts["elevation"] / 180 * np.pi
        azimuth = self.opts["azimuth"] / 180 * np.pi
        distance = self.opts["distance"]
        r, t = get_rt_(elevation, azimuth, distance, center)
        dx_world = np.linalg.inv(r) @ (xp)
        dy_world = np.linalg.inv(r) @ (yp)
        dz_world = np.linalg.inv(r) @ (zp)
        return self.world_move(*(dx_world + dy_world + dz_world))

    def world_move(self, dx=0, dy=0, dz=0):
        c = self.opts["center"]
        center_new = [c.x() + dx, c.y() + dy, c.z() + dz]
        self.setCameraPosition(center=center_new)

    def mouse_press_event(self, ev):
        super().mousePressEvent(ev)
        self.mousePressed.emit((ev.x(), ev.y()))

    def mouse_release_event(self, ev):

        # Example item selection code:
        c = self.opts["center"]
        _ = (
            f"fov={self.opts['fov']:.2f}\n"
            f"center=[{c.x():.2f}, {c.y():.2f}, {c.z():.2f}]\n"
            f"distance={self.opts['distance']:.2f}\n"
            f"elevation={self.opts['elevation']:.2f}\n"
            f"azimuth={self.opts['azimuth']:.2f}"
        )
        w_norm = self.width()
        h_norm = self.height()
        elevation = self.opts["elevation"] / 180 * np.pi
        azimuth = self.opts["azimuth"] / 180 * np.pi
        distance = self.opts["distance"]
        center = [c.x(), c.y(), c.z()]
        fov = self.opts["fov"] / 180 * np.pi
        point = [0, 0, 0.0]
        _ = world2camera(
            np.array(point), elevation, azimuth, distance, center, fov, w_norm, h_norm
        )
        _ = sph_to_device_pos(elevation, azimuth, distance, center)
        # debugging code: draw the picking region


setattr(KittiGLViewWidget, "mousePressEvent", KittiGLViewWidget.mouse_press_event)
setattr(KittiGLViewWidget, "mouseReleaseEvent", KittiGLViewWidget.mouse_release_event)
