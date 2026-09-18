# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import NamedTuple
from functools import partial
from pathlib import Path
import datetime
import io as sysio
import json
import logging
import pickle
import sys
import time

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QTextEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from shapely.geometry import Polygon
from skimage import io
import numpy as np

from second.core.anchor_generator import AnchorGeneratorStride
from second.core.box_coders import GroundBox3dCoder
from second.core.point_cloud.point_cloud_ops import points_to_voxel
from second.core.region_similarity import NearestIouSimilarity
from second.core.sample_ops import DataBaseSamplerV2
from second.core.target_assigner import TargetAssigner
from second.data import kitti_common as kitti
from second.kittiviewer.glwidget import KittiGLViewWidget
from second.pytorch.inference import TorchInferenceContext
from second.utils import bbox_plot
from second.utils.bbox_plot import GLColor
from second.utils.eval import get_official_eval_result
from second.utils.progress_bar import list_bar
import second.core.box_np_ops as box_np_ops
import second.core.preprocess as prep
import second.kittiviewer.control_panel as panel


from shared.parameters import ParameterSchema as _ParameterSchema, bind_parameters as _bind_parameters


class KittipointcloudviewInitParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    config: object
    parent: object
    voxel_size: object
    coors_range: object
    max_voxels: object
    max_num_points: object


LOGGER = logging.getLogger(__name__)


def _kittidrawcontrol_initialize_part_1(_state):
    _state["self"].setWindowTitle(_state["title"])


def _kittidrawcontrol_initialize_part_2(_state):
    with _state["self"].tab(0, "common"):
        with _state["self"].column(0):
            _state["self"].add_listedit("UsedClass", str)
            _state["self"].add_fspinbox("PointSize", 0.01, 0.5, 0.01, 0.05)
            _state["self"].add_fspinbox("PointAlpha", 0.0, 1.0, 0.05, 0.5)
            _state["self"].add_colorbutton(
                "PointColor", bbox_plot.gl_color(GLColor.Gray)
            )
            _state["self"].add_fspinbox("GTPointSize", 0.01, 0.5, 0.01, 0.2)
            _state["self"].add_fspinbox("GTPointAlpha", 0.0, 1.0, 0.05, 0.5)
            _state["self"].add_colorbutton(
                "GTPointColor", bbox_plot.gl_color(GLColor.Purple)
            )
            _state["self"].add_checkbox("WithReflectivity")
            _state["self"].add_checkbox("DrawGTBoxes")
            _state["self"].add_checkbox("DrawGTLabels")
            _state["self"].add_colorbutton(
                "GTBoxColor", bbox_plot.gl_color(GLColor.Green)
            )
            _state["self"].add_fspinbox("GTBoxAlpha", 0.0, 1.0, 0.05, 0.5)
            _state["self"].add_checkbox("DrawDTBoxes")
            _state["self"].add_checkbox("DrawDTLabels")
            _state["self"].add_checkbox("DTScoreAsAlpha")
            _state["self"].add_fspinbox("DTScoreThreshold", 0.0, 1.0, 0.01, 0.3)
            _state["self"].add_colorbutton(
                "DTBoxColor", bbox_plot.gl_color(GLColor.Blue)
            )
            _state["self"].add_fspinbox("DTBoxAlpha", 0.0, 1.0, 0.05, 0.5)
            _state["self"].add_fspinbox("DTBoxLineWidth", 0.25, 10.0, 0.25, 1.0)
        with _state["self"].column(1):
            _state["self"].add_arrayedit(
                "CoorsRange", np.float64, [-40, -40, -2, 40, 40, 4], [6]
            )
            _state["self"].add_arrayedit("VoxelSize", np.float64, [0.2, 0.2, 0.4], [3])
            _state["self"].add_checkbox("DrawVoxels")
            _state["self"].add_colorbutton(
                "PosVoxelColor", bbox_plot.gl_color(GLColor.Yellow)
            )
            _state["self"].add_fspinbox("PosVoxelAlpha", 0.0, 1.0, 0.05, 0.5)
            _state["self"].add_colorbutton(
                "NegVoxelColor", bbox_plot.gl_color(GLColor.Purple)
            )
            _state["self"].add_fspinbox("NegVoxelAlpha", 0.0, 1.0, 0.05, 0.5)
            _state["self"].add_checkbox("DrawPositiveVoxelsOnly")
            _state["self"].add_checkbox("RemoveOutsidePoint")


def _kittidrawcontrol_initialize_part_3(_state):
    with _state["self"].tab(1, "inference"):
        with _state["self"].column(0):
            _state["self"].add_checkbox("TensorflowInference")
    with _state["self"].tab(2, "anchors"):
        with _state["self"].column(0):
            _state["self"].add_checkbox("DrawAnchors")
            _state["self"].add_arrayedit(
                "AnchorSize", np.float64, [1.6, 3.9, 1.56], [3]
            )
            _state["self"].add_arrayedit(
                "AnchorOffset", np.float64, [0, -39.8, -1.0], [3]
            )
            _state["self"].add_arrayedit(
                "AnchorStride", np.float64, [0.4, 0.4, 0.0], [3]
            )
            _state["self"].add_fspinbox("MatchThreshold", 0.0, 1.0, 0.1)
            _state["self"].add_fspinbox("UnMatchThreshold", 0.0, 1.0, 0.1)
            _state["self"].add_combobox("IoUMethod", ["RotateIoU", "NearestIoU"])
    with _state["self"].tab(3, "sample and augmentation"):
        with _state["self"].column(0):
            _state["self"].add_checkbox("EnableSample")
            _state["self"].add_jsonedit("SampleGroups")
            _state["self"].add_arrayedit(
                "SampleGlobleRotRange", np.float64, [0.78, 2.35], [2]
            )
        with _state["self"].column(1):
            _state["self"].add_checkbox("EnableAugmentation")
            _state["self"].add_checkbox("GroupNoisePerObject")


class KittiDrawControl(panel.ControlPanel):
    def __init__(self, title, parent=None):
        super().__init__(column_nums=[2, 1, 1, 2], tab_num=4, parent=parent)
        _state = locals()
        _kittidrawcontrol_initialize_part_1(_state)
        _kittidrawcontrol_initialize_part_2(_state)
        _kittidrawcontrol_initialize_part_3(_state)


class Settings:
    def __init__(self, cfg_path):
        self._cfg_path = cfg_path
        self._settings = {}
        self._setting_defaultvalue = {}
        if not Path(self._cfg_path).exists():
            with open(self._cfg_path, "w") as f:
                f.write(json.dumps(self._settings, indent=2, sort_keys=True))
        else:
            with open(self._cfg_path, "r") as f:
                self._settings = json.loads(f.read())

    def set(self, name, value):
        self._settings[name] = value
        with open(self._cfg_path, "w") as f:
            f.write(json.dumps(self._settings, indent=2, sort_keys=True))

    def get(self, name, default_value=None):
        if name in self._settings:
            return self._settings[name]
        if default_value is None:
            raise ValueError("name not exist")
        return default_value

    def save(self, path):
        with open(path, "w") as f:
            f.write(json.dumps(self._settings, indent=2, sort_keys=True))

    def load(self, path):
        with open(self._cfg_path, "r") as f:
            self._settings = json.loads(f.read())


def _riou3d_shapely(rbboxes1, rbboxes2):
    n, k = rbboxes1.shape[0], rbboxes2.shape[0]
    corners1 = box_np_ops.center_to_corner_box2d(
        rbboxes1[:, :2], rbboxes1[:, 3:5], rbboxes1[:, 6]
    )
    corners2 = box_np_ops.center_to_corner_box2d(
        rbboxes2[:, :2], rbboxes2[:, 3:5], rbboxes2[:, 6]
    )
    iou = np.zeros([n, k], dtype=np.float32)
    for i in range(n):
        for j in range(k):
            iw = min(
                rbboxes1[i, 2] + rbboxes1[i, 5], rbboxes2[j, 2] + rbboxes2[j, 5]
            ) - max(rbboxes1[i, 2], rbboxes2[j, 2])
            if iw <= 0:
                continue
            p1 = Polygon(corners1[i])
            p2 = Polygon(corners2[j])
            inc = p1.intersection(p2).area * iw
            if inc > 0:
                iou[i, j] = inc / (
                    p1.area * rbboxes1[i, 5] + p2.area * rbboxes2[j, 5] - inc
                )

    return iou


def kitti_anno_to_corners(info, annos=None):
    rect = info["calib/R0_rect"]
    _ = info["calib/P2"]
    tr_velo_to_cam = info["calib/Tr_velo_to_cam"]
    if annos is None:
        annos = info["annos"]
    dims = annos["dimensions"]
    loc = annos["location"]
    rots = annos["rotation_y"]
    scores = None
    if "score" in annos:
        scores = annos["score"]
    boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)
    boxes_lidar = box_np_ops.box_camera_to_lidar(boxes_camera, rect, tr_velo_to_cam)
    boxes_corners = box_np_ops.center_to_corner_box3d(
        boxes_lidar[:, :3],
        boxes_lidar[:, 3:6],
        boxes_lidar[:, 6],
        origin=[0.5, 0.5, 0],
        axis=2,
    )
    return boxes_corners, scores, boxes_lidar


class MatPlotLibView(FigureCanvas):
    def __init__(self, parent=None, rect=None, dpi=100):
        if rect is None:
            rect = [5, 4]
        self.fig = Figure(figsize=(rect[0], rect[1]), dpi=dpi)
        self.ax = self.fig.add_subplot(1, 1, 1)
        super().__init__(self.fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.draw()

    def reset_plot(self):
        self.fig.clf()
        self.ax = self.fig.add_subplot(1, 1, 1)


class MatPlotLibViewTab(QWidget):
    def __init__(self, num_rect=None, dpi=100, parent=None):
        if num_rect is None:
            num_rect = [5, 4]
        self.fig = Figure(figsize=(num_rect[0], num_rect[1]), dpi=dpi)
        self.ax = self.fig.add_subplot(1, 1, 1)
        super().__init__(self.fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.draw()

    def reset_plot(self):
        self.fig.clf()
        self.ax = self.fig.add_subplot(1, 1, 1)


class MatPlotLibWidget(QWidget):
    def __init__(self, parent=None, rect=None, dpi=100):
        if rect is None:
            rect = [5, 4]
        self.w_plot = MatPlotLibView(self, rect, dpi)
        self.w_plt_toolbar = NavigationToolbar(self.w_plot, self)
        plt_layout = QVBoxLayout()
        plt_layout.addWidget(self.w_plot)
        plt_layout.addWidget(self.w_plt_toolbar)

    @property
    def axis(self):
        return self.w_plot.ax

    def reset_plot(self):
        return self.w_plot.reset_plot()


_KITTIPOINTCLOUDVIEW_INIT_PARAMETERS_SCHEMA = _ParameterSchema(
    KittipointcloudviewInitParameters,
    ("config", "parent", "voxel_size", "coors_range", "max_voxels", "max_num_points"),
    {
        "parent": None,
        "voxel_size": None,
        "coors_range": None,
        "max_voxels": 50000,
        "max_num_points": 35,
    },
    frozenset(()),
    6,
    None,
    None,
)


class KittiPointCloudView(KittiGLViewWidget):
    def __init__(self, *call_args, **call_kwargs):
        bound_parameters = _bind_parameters(
            _KITTIPOINTCLOUDVIEW_INIT_PARAMETERS_SCHEMA, call_args, call_kwargs
        )
        config, parent, voxel_size, coors_range, max_voxels, max_num_points = (
            bound_parameters
        )
        super().__init__(parent=parent)
        if voxel_size is None:
            voxel_size = [0.2, 0.2, 0.4]
        if coors_range is None:
            coors_range = [0, -40, -3, 70.4, 40, 1]
        self.w_config = config
        self._voxel_size = voxel_size
        self._coors_range = coors_range
        self._max_voxels = max_voxels
        self._max_num_points = max_num_points
        bk_color = (0.8, 0.8, 0.8, 1.0)
        bk_color = list([int(v * 255) for v in bk_color])
        self.mousePressed.connect(self.on_mouse_pressed)
        self.setCameraPosition(distance=20, azimuth=-180, elevation=30)

    @staticmethod
    def _anchor_target_assigner():
        anchor_generator = AnchorGeneratorStride(
            sizes=[1.6, 3.9, 1.56],
            anchor_strides=[0.4, 0.4, 0.0],
            anchor_offsets=[0.2, -39.8, -1.55442884],
            rotations=[0, 1.5707963267948966],
            match_threshold=0.6,
            unmatch_threshold=0.45,
        )
        return TargetAssigner(
            GroundBox3dCoder(), [anchor_generator], NearestIouSimilarity()
        )

    def on_mouse_pressed(self, pos):
        pass

    def reset_camera(self):
        self.set_camera_position(
            center=(5, 0, 0), distance=20, azimuth=-180, elevation=30
        )
        self.update()

    def draw_frustum(self, bboxes, rect, trv2c, p2):
        frustums = []
        c, r, t = box_np_ops.projection_matrix_to_crt_kitti(p2)
        frustums = box_np_ops.get_frustum_v2(bboxes, c)
        frustums -= t
        frustums = np.einsum("ij, akj->aki", np.linalg.inv(r), frustums)
        frustums = box_np_ops.camera_to_lidar(frustums, rect, trv2c)
        self.boxes3d("frustums", frustums, colors=GLColor.Write, alpha=0.5)

    def draw_cropped_frustum(self, bboxes, rect, trv2c, p2):
        self.boxes3d(
            "cropped_frustums",
            prep.random_crop_frustum(bboxes, rect, trv2c, p2),
            colors=GLColor.Write,
            alpha=0.5,
        )

    def draw_anchors(self, gt_boxes_lidar, points=None, image_idx=0, gt_names=None):
        voxel_size = np.array(self._voxel_size, dtype=np.float32)
        coors_range = np.array(self._coors_range, dtype=np.float32)
        grid_size = (coors_range[3:] - coors_range[:3]) / voxel_size
        grid_size = np.round(grid_size).astype(np.int64)
        _ = coors_range[[0, 1, 3, 4]]
        target_assigner = self._anchor_target_assigner()
        feature_map_size = grid_size[:2] // 2
        feature_map_size = [*feature_map_size, 1][::-1]
        LOGGER.info("%s", feature_map_size)
        ret = target_assigner.generate_anchors(feature_map_size)
        anchors = ret["anchors"]
        anchors = anchors.reshape([-1, 7])
        anchors_bv = box_np_ops.rbbox2d_to_near_bbox(anchors[:, [0, 1, 3, 4, 6]])
        matched_thresholds = ret["matched_thresholds"]
        unmatched_thresholds = ret["unmatched_thresholds"]
        LOGGER.info("%s", f"num_anchors_ {len(anchors)}")
        if points is None:
            raise ValueError("points are required to draw anchors")
        geometry = (voxel_size, coors_range, grid_size)
        anchors_mask = self._points_anchor_mask(points, anchors_bv, geometry)
        LOGGER.info("%s %s", np.sum(anchors_mask), anchors_mask.shape)
        class_names = [
            "Car",
            "Pedestrian",
            "Cyclist",
            "Van",
            "Truck",
            "Tram",
            "Misc",
            "Person_sitting",
        ]
        gt_classes = np.array(
            [class_names.index(n) + 1 for n in gt_names], dtype=np.int32
        )
        t = time.time()
        target_dict = target_assigner.assign(
            anchors,
            gt_boxes_lidar,
            anchors_mask,
            gt_classes=gt_classes,
            matched_thresholds=matched_thresholds,
            unmatched_thresholds=unmatched_thresholds,
        )
        LOGGER.info("%s %s", "target time", time.time() - t)
        self._draw_assigned_anchors(anchors, anchors_mask, target_dict)

    def draw_bounding_box(self):
        bbox = box_np_ops.minmax_to_corner_3d(
            np.array([self.w_config.get("CoorsRange")])
        )
        self.boxes3d("bound", bbox, GLColor.Green)

    def draw_voxels(self, points, gt_boxes=None):
        pos_color = self.w_config.get("PosVoxelColor")[:3]
        pos_color = (*pos_color, self.w_config.get("PosVoxelAlpha"))
        neg_color = self.w_config.get("NegVoxelColor")[:3]
        neg_color = (*neg_color, self.w_config.get("NegVoxelAlpha"))

        voxel_size = np.array(self.w_config.get("VoxelSize"), dtype=np.float32)
        coors_range = np.array(self.w_config.get("CoorsRange"), dtype=np.float32)
        voxels, coors, num_points = points_to_voxel(
            points,
            voxel_size,
            coors_range,
            self._max_num_points,
            reverse_index=True,
            max_voxels=self._max_voxels,
        )
        grid_size = (coors_range[3:] - coors_range[:3]) / voxel_size
        grid_size = np.round(grid_size).astype(np.int64)

        shift = coors_range[:3]
        voxel_origins = coors[:, ::-1] * voxel_size + shift
        voxel_maxs = voxel_origins + voxel_size
        voxel_boxes = np.concatenate([voxel_origins, voxel_maxs], axis=1)
        voxel_box_corners = box_np_ops.minmax_to_corner_3d(voxel_boxes)
        pos_only = self.w_config.get("DrawPositiveVoxelsOnly")
        if gt_boxes is not None:
            labels = box_np_ops.assign_label_to_voxel(
                gt_boxes, coors, voxel_size, coors_range
            ).astype(np.bool)
            if pos_only:
                voxel_box_corners = voxel_box_corners[labels]
            colors = np.zeros([voxel_box_corners.shape[0], 4])
            if pos_only:
                colors[:] = pos_color
            else:
                colors[np.logical_not(labels)] = neg_color
                colors[labels] = pos_color
        else:
            if not pos_only:
                colors = np.zeros([voxel_box_corners.shape[0], 4])
                colors[:] = neg_color
            else:
                voxel_box_corners = np.zeros((0, 8, 3))
                colors = np.zeros((0, 4))
        self.boxes3d("voxels", voxel_box_corners, colors)

    def _points_anchor_mask(self, points, anchors_bv, geometry):
        voxel_size, coors_range, grid_size = geometry
        voxels, coors, num_points = points_to_voxel(
            points,
            self._voxel_size,
            coors_range,
            self._max_num_points,
            reverse_index=True,
            max_voxels=self._max_voxels,
        )
        dense_voxel_map = box_np_ops.sparse_sum_for_anchors_mask(
            coors, tuple(grid_size[::-1][1:])
        )
        dense_voxel_map = dense_voxel_map.cumsum(0).cumsum(1)
        return box_np_ops.fused_get_anchors_area(
            dense_voxel_map, anchors_bv, voxel_size, coors_range, grid_size
        ) > 1

    def _draw_assigned_anchors(self, anchors, anchors_mask, target_dict):
        labels = target_dict.get("labels")
        LOGGER.info("%s", f"num_pos={np.sum(labels > 0)}")
        colors = np.zeros([anchors.shape[0], 4])
        colors[labels == -1] = bbox_plot.gl_color(GLColor.Gray, 0.5)
        colors[labels > 0] = bbox_plot.gl_color(GLColor.Cyan, 0.5)
        cared_mask = np.logical_and(labels != 0, anchors_mask)
        corners = box_np_ops.rbbox3d_to_corners(anchors)[cared_mask]
        self.boxes3d("anchors", corners, colors=colors[cared_mask])


def _kittiviewer_init_ui_665_part_1(_state):
    _state["self"].setWindowTitle(_state["self"].title)
    _state["self"].setGeometry(*_state["self"].bbox_window)
    _state["control_panel_layout"] = QVBoxLayout()
    _state["root_path"] = _state["self"].json_setting.get("kitti_root_path", "")
    _state["self"].w_root_path = QLineEdit(_state["root_path"])
    _state["iamge_idx"] = _state["self"].json_setting.get("image_idx", "0")
    _state["self"].w_imgidx = QLineEdit(_state["iamge_idx"])
    _state["info_path"] = _state["self"].json_setting.get("latest_info_path", "")
    _state["self"].w_info_path = QLineEdit(_state["info_path"])
    _state["det_path"] = _state["self"].json_setting.get("latest_det_path", "")
    _state["self"].w_det_path = QLineEdit(_state["det_path"])
    _state["self"].w_load = QPushButton("load info")
    _state["self"].w_load.clicked.connect(_state["self"].on_load_button_pressed)
    _state["self"].w_load_det = QPushButton("load detection")
    _state["self"].w_load_det.clicked.connect(_state["self"].on_load_det_pressed)
    _state["self"].w_config = KittiDrawControl("ctrl")
    _state["config"] = _state["self"].json_setting.get("config", "")
    if _state["config"] != "":
        _state["self"].w_config.loads(_state["config"])
    _state["self"].w_config.configChanged.connect(_state["self"].on_configchanged)
    _state["self"].w_plot = QPushButton("plot")
    _state["self"].w_plot.clicked.connect(_state["self"].on_plot_button_pressed)
    _state["self"].w_show_panel = QPushButton("control panel")
    _state["self"].w_show_panel.clicked.connect(_state["self"].on_panel_clicked)
    _state["center_widget"] = QWidget(_state["self"])
    _state["self"].w_output = QTextEdit()
    _state["self"].w_config_gbox = QGroupBox("Read Config")
    _state["layout"] = QFormLayout()
    _state["layout"].addRow(QLabel("root path:"), _state["self"].w_root_path)
    _state["layout"].addRow(QLabel("info path:"), _state["self"].w_info_path)


def _kittiviewer_init_ui_665_part_2(_state):
    _state["layout"].addRow(QLabel("image idx:"), _state["self"].w_imgidx)
    _state["layout"].addRow(QLabel("det path:"), _state["self"].w_det_path)
    _state["self"].w_config_gbox.setLayout(_state["layout"])
    _state["self"].w_plt = MatPlotLibView()
    _state["self"].w_plt_toolbar = NavigationToolbar(
        _state["self"].w_plt, _state["center_widget"]
    )
    _state["plt_layout"] = QVBoxLayout()
    _state["plt_layout"].addWidget(_state["self"].w_plt)
    _state["plt_layout"].addWidget(_state["self"].w_plt_toolbar)
    _state["control_panel_layout"].addWidget(_state["self"].w_config_gbox)
    _state["h_layout"] = QHBoxLayout()
    _state["h_layout"].addWidget(_state["self"].w_load)
    _state["h_layout"].addWidget(_state["self"].w_load_det)
    _state["control_panel_layout"].addLayout(_state["h_layout"])
    _state["h_layout"] = QHBoxLayout()
    _state["h_layout"].addWidget(_state["self"].w_plot)
    _state["control_panel_layout"].addLayout(_state["h_layout"])
    _state["control_panel_layout"].addWidget(_state["self"].w_show_panel)
    _state["vcfg_path"] = _state["self"].json_setting.get("latest_vxnet_cfg_path", "")
    _state["self"].w_vconfig_path = QLineEdit(_state["vcfg_path"])
    _state["vckpt_path"] = _state["self"].json_setting.get("latest_vxnet_ckpt_path", "")
    _state["self"].w_vckpt_path = QLineEdit(_state["vckpt_path"])
    _state["layout"] = QFormLayout()
    _state["layout"].addRow(QLabel("config path:"), _state["self"].w_vconfig_path)
    _state["layout"].addRow(QLabel("ckpt path:"), _state["self"].w_vckpt_path)
    _state["control_panel_layout"].addLayout(_state["layout"])
    _state["self"].w_build_net = QPushButton("Build Network")
    _state["self"].w_build_net.clicked.connect(_state["self"].on_build_vx_net_pressed)
    _state["self"].w_load_ckpt = QPushButton("load Network checkpoint")
    _state["self"].w_load_ckpt.clicked.connect(
        _state["self"].on_load_vx_net_ckpt_pressed
    )
    _state["h_layout"] = QHBoxLayout()


def _kittiviewer_init_ui_665_part_3(_state):
    _state["h_layout"].addWidget(_state["self"].w_build_net)
    _state["h_layout"].addWidget(_state["self"].w_load_ckpt)
    _state["control_panel_layout"].addLayout(_state["h_layout"])
    _state["self"].w_inference = QPushButton("Inference Network")
    _state["self"].w_inference.clicked.connect(
        _state["self"].on_inference_vx_net_pressed
    )
    _state["control_panel_layout"].addWidget(_state["self"].w_inference)
    _state["self"].w_load_infer = QPushButton("Load and Inference Network")
    _state["self"].w_load_infer.clicked.connect(
        _state["self"].on_load_inference_vx_net_pressed
    )
    _state["control_panel_layout"].addWidget(_state["self"].w_load_infer)
    _state["layout"] = QFormLayout()
    _state["self"].w_cb_gt_curcls = QCheckBox("Indexed by GroundTruth Class")
    _state["self"].w_cb_gt_curcls.setChecked(True)
    _state["self"].w_cb_gt_curcls.stateChanged.connect(
        _state["self"].on_gt_checkbox_statechanged
    )
    _state["self"].gt_combobox = QComboBox()
    _state["self"].gt_combobox.addItem("All")
    for _state["cls_name"] in kitti.get_classes():
        _state["self"].gt_combobox.addItem(_state["cls_name"])
    _state["self"].current_gt_class_ids = None
    _state["self"].current_gt_class_index = 0
    _state["self"].gt_combobox.currentTextChanged.connect(
        _state["self"].on_gt_combobox_changed
    )
    _state["layout"].addRow(_state["self"].w_cb_gt_curcls, _state["self"].gt_combobox)
    _state["self"].w_cb_dt_curcls = QCheckBox("Indexed by Detection Class")
    _state["self"].w_cb_dt_curcls.setChecked(False)
    _state["self"].w_cb_dt_curcls.stateChanged.connect(
        _state["self"].on_dt_checkbox_statechanged
    )
    _state["self"].dt_combobox = QComboBox()
    _state["self"].dt_combobox.addItem("All")
    _state["self"].current_detection_class_ids = None
    _state["self"].current_detection_class_index = 0
    _state["self"].dt_combobox.currentTextChanged.connect(
        _state["self"].on_dt_combobox_changed
    )
    _state["layout"].addRow(_state["self"].w_cb_dt_curcls, _state["self"].dt_combobox)


def _kittiviewer_init_ui_665_part_4(_state):
    _state["control_panel_layout"].addLayout(_state["layout"])
    _state["self"].w_next = QPushButton("next")
    _state["self"].w_next.clicked.connect(
        partial(_state["self"].on_next_or_prev_pressed, prev=False)
    )
    _state["self"].w_prev = QPushButton("prev")
    _state["self"].w_prev.clicked.connect(
        partial(_state["self"].on_next_or_prev_pressed, prev=True)
    )
    _state["layout"] = QHBoxLayout()
    _state["layout"].addWidget(_state["self"].w_prev)
    _state["layout"].addWidget(_state["self"].w_next)
    _state["control_panel_layout"].addLayout(_state["layout"])
    _state["self"].w_next = QPushButton("next current class")
    _state["self"].w_next.clicked.connect(
        partial(_state["self"].on_next_or_prev_cur_cls_pressed, prev=False)
    )
    _state["self"].w_prev = QPushButton("prev current class")
    _state["self"].w_prev.clicked.connect(
        partial(_state["self"].on_next_or_prev_cur_cls_pressed, prev=True)
    )
    _state["layout"] = QHBoxLayout()
    _state["layout"].addWidget(_state["self"].w_prev)
    _state["layout"].addWidget(_state["self"].w_next)
    _state["control_panel_layout"].addLayout(_state["layout"])
    _state["control_panel_layout"].addLayout(_state["plt_layout"])
    _state["save_image_path"] = _state["self"].json_setting.get("save_image_path", "")
    _state["self"].w_image_save_path = QLineEdit(_state["save_image_path"])
    _state["self"].w_save_image = QPushButton("save image")
    _state["self"].w_save_image.clicked.connect(_state["self"].on_saveimg_clicked)
    _state["control_panel_layout"].addWidget(_state["self"].w_image_save_path)
    _state["control_panel_layout"].addWidget(_state["self"].w_save_image)
    _state["control_panel_layout"].addWidget(_state["self"].w_output)
    _state["self"].center_layout = QHBoxLayout()


def _kittiviewer_init_ui_665_part_5(_state):
    _state["self"].w_pc_viewer = KittiPointCloudView(
        _state["self"].w_config, coors_range=_state["self"].w_config.get("CoorsRange")
    )
    _state["self"].center_layout.addWidget(_state["self"].w_pc_viewer)
    _state["self"].center_layout.addLayout(_state["control_panel_layout"])
    _state["self"].center_layout.setStretch(0, 2)
    _state["self"].center_layout.setStretch(1, 1)
    _state["center_widget"].setLayout(_state["self"].center_layout)
    _state["self"].setCentralWidget(_state["center_widget"])
    _state["self"].show()


class KittiViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.title = "KittiViewer"
        self.bbox_window = [10, 10, 1600, 900]
        self.sstream = sysio.StringIO()
        self.json_setting = Settings(str(Path.home() / ".kittiviewerrc"))
        self.kitti_infos = None
        self.detection_annos = None
        self.image_idxes = None
        self.root_path = None
        self.current_idx = 0
        self.dt_image_idxes = None
        self.current_image = None
        self.init_ui()
        self.kitti_info = None
        self.points = None
        self.gt_boxes = None
        self.gt_names = None
        self.difficulty = None
        self.group_ids = None
        self.inference_ctx = None

    @staticmethod
    def get_simpify_labels(labels):
        label_map = {
            "Car": "V",
            "Pedestrian": "P",
            "Cyclist": "C",
            "car": "C",
            "tractor": "T1",
            "trailer": "T2",
        }
        label_count = {
            "Car": 0,
            "Pedestrian": 0,
            "Cyclist": 0,
            "car": 0,
            "tractor": 0,
            "trailer": 0,
        }
        ret = []
        for _, name in enumerate(labels):
            count = 0
            if name in label_count:
                count = label_count[name]
                label_count[name] += 1
            else:
                label_count[name] = 0
            ret.append(f"{label_map.get(name)}{count}")
        return ret

    @staticmethod
    def get_false_pos_neg(gt_boxes, dt_boxes, labels, fp_thresh=0.1):
        iou = _riou3d_shapely(gt_boxes, dt_boxes)
        ret = np.full([len(gt_boxes)], 2, dtype=np.int64)
        assigned_dt = np.zeros([len(dt_boxes)], dtype=np.bool_)
        label_thresh_map = {
            "Car": 0.7,
            "Pedestrian": 0.5,
            "Cyclist": 0.5,
            "car": 0.7,
            "tractor": 0.7,
            "trailer": 0.7,
        }
        tp_thresh = np.array([label_thresh_map.get(n) for n in labels])
        if len(gt_boxes) != 0 and len(dt_boxes) != 0:
            iou_max_dt_for_gt = iou.max(1)
            dt_iou_max_dt_for_gt = iou.argmax(1)
            ret[iou_max_dt_for_gt >= tp_thresh] = 0
            ret[
                np.logical_and(
                    iou_max_dt_for_gt < tp_thresh, iou_max_dt_for_gt > fp_thresh
                )
            ] = 1  # FP
            assigned_dt_inds = dt_iou_max_dt_for_gt
            assigned_dt_inds = assigned_dt_inds[iou_max_dt_for_gt >= fp_thresh]
            assigned_dt[assigned_dt_inds] = True
        return ret, assigned_dt

    def init_ui(self):

        _state = locals()
        _kittiviewer_init_ui_665_part_1(_state)
        _kittiviewer_init_ui_665_part_2(_state)
        _kittiviewer_init_ui_665_part_3(_state)
        _kittiviewer_init_ui_665_part_4(_state)
        _kittiviewer_init_ui_665_part_5(_state)

    def on_panel_clicked(self):
        if self.w_config.isHidden():
            self.w_config.show()
        else:
            self.w_config.hide()

    def on_saveimg_clicked(self):
        self.save_image(self.current_image)

    def on_gt_checkbox_statechanged(self):
        self.w_cb_gt_curcls.setChecked(True)
        self.w_cb_dt_curcls.setChecked(False)

    def on_dt_checkbox_statechanged(self):
        self.w_cb_gt_curcls.setChecked(False)
        self.w_cb_dt_curcls.setChecked(True)

    def on_gt_combobox_changed(self):
        self.current_gt_class_index = 0
        self.on_load_button_pressed()

    def on_dt_combobox_changed(self):
        self.current_detection_class_index = 0
        annos = kitti.filter_empty_annos(self.detection_annos)
        if self.dt_image_idxes is not None and annos is not None:
            current_class = self.dt_combobox.currentText()
            if current_class == "All":
                self.current_detection_class_ids = self.dt_image_idxes
            else:
                self.current_detection_class_ids = [
                    anno["image_idx"][0]
                    for anno in annos
                    if current_class in anno["name"]
                ]

    def message(self, value, *arg, color="Black"):
        color_html = f'<font color="{color}">'
        end_html = "</font><br>"
        msg = self.print_str(value, *arg)
        self.w_output.insertHtml(color_html + msg + end_html)
        self.w_output.verticalScrollBar().setValue(
            self.w_output.verticalScrollBar().maximum()
        )

    def error(self, value, *arg):
        time_str = datetime.datetime.now(datetime.timezone.utc).strftime("[%H:%M:%S]")
        return self.message(time_str, value, *arg, color="Red")

    def info(self, value, *arg):
        time_str = datetime.datetime.now(datetime.timezone.utc).strftime("[%H:%M:%S]")
        return self.message(time_str, value, *arg, color="Black")

    def warning(self, value, *arg):
        time_str = datetime.datetime.now(datetime.timezone.utc).strftime("[%H:%M:%S]")
        return self.message(time_str, value, *arg, color="Yellow")

    def save_image(self, image):

        img_path = self.w_image_save_path.text()
        self.json_setting.set("save_image_path", img_path)
        if self.current_image is not None:
            io.imsave(img_path, image)
        p = self.w_pc_viewer.grabFrameBuffer()

        pc_img_path = str(
            Path(img_path).parent / (str(Path(img_path).stem) + "_pc.jpg")
        )
        p.save(pc_img_path, "jpg")
        self.info("image saved to", img_path)

    def print_str(self, value, *arg):
        self.sstream.truncate(0)
        self.sstream.seek(0)
        self.sstream.write(" ".join(map(str, (value, *arg))) + "\n")
        return self.sstream.getvalue()

    def on_next_or_prev_pressed(self, prev):
        if prev:
            self.current_idx = max(self.current_idx - 1, 0)
        else:
            info_len = len(self.image_idxes)
            self.current_idx = min(self.current_idx + 1, info_len - 1)
        image_idx = self.image_idxes[self.current_idx]
        self.w_imgidx.setText(str(image_idx))
        self.plot_all(image_idx)

    def on_next_or_prev_cur_cls_pressed(self, prev):
        if self.w_cb_dt_curcls.isChecked():
            if prev:
                self.current_detection_class_index = max(self.current_detection_class_index - 1, 0)
            else:
                info_len = len(self.current_detection_class_ids)
                self.current_detection_class_index = min(
                    self.current_detection_class_index + 1, info_len - 1
                )
            image_idx = self.current_detection_class_ids[self.current_detection_class_index]
            self.info("current dt image idx:", image_idx)
        elif self.w_cb_gt_curcls.isChecked():
            if prev:
                self.current_gt_class_index = max(self.current_gt_class_index - 1, 0)
            else:
                info_len = len(self.current_gt_class_ids)
                self.current_gt_class_index = min(
                    self.current_gt_class_index + 1, info_len - 1
                )
            image_idx = self.current_gt_class_ids[self.current_gt_class_index]
            self.info("current gt image idx:", image_idx)
        self.plot_all(image_idx)

    def on_cmd_return_pressed(self):
        cmd = self.print_str(self.cmd.text())
        self.output.insertPlainText(cmd)

    def on_load_button_pressed(self):
        self.root_path = Path(self.w_root_path.text())
        if not (self.root_path / "training").exists():
            self.error("ERROR: your root path is incorrect.")
            return
        self.json_setting.set("kitti_root_path", str(self.root_path))
        info_path = self.w_info_path.text()
        if info_path == "":
            info_path = self.root_path / "kitti_infos_val.pkl"
        else:
            info_path = Path(info_path)
        if not info_path.exists():
            self.error("ERROR: info file not exist")
            return
        self.json_setting.set("latest_info_path", str(info_path))
        with open(info_path, "rb") as f:
            self.kitti_infos = pickle.load(f)
        db_infos_path = Path(self.root_path) / "kitti_dbinfos_train.pkl"
        if db_infos_path.exists():
            with open(db_infos_path, "rb") as f:
                self.db_infos = pickle.load(f)
                global_rot_range = self.w_config.get("SampleGlobleRotRange")
                groups = self.w_config.get("SampleGroups")
                self.info("init database sampler with group:")
                self.info(groups)
                self.db_sampler = DataBaseSamplerV2(
                    self.db_infos, groups, global_rot_range=global_rot_range
                )
        self.info("load db_infos.")
        self.image_idxes = [info["image_idx"] for info in self.kitti_infos]
        self.info("load", len(self.kitti_infos), "infos.")
        current_class = self.gt_combobox.currentText()
        if current_class == "All":
            self.current_gt_class_ids = self.image_idxes
        else:
            self.current_gt_class_ids = [
                info["image_idx"]
                for info in self.kitti_infos
                if current_class in info["annos"]["name"]
            ]
        self.current_gt_class_index = 0

    def on_load_det_pressed(self):
        det_path = self.w_det_path.text()
        if Path(det_path).is_file():
            with open(det_path, "rb") as f:
                dt_annos = pickle.load(f)
        else:
            dt_annos = kitti.get_label_annos(det_path)
        if len(dt_annos) == 0:
            self.warning("detection path contain nothing.")
            return
        self.detection_annos = dt_annos
        self.info(f"load {len(dt_annos)} detections.")
        self.json_setting.set("latest_det_path", det_path)
        annos = kitti.filter_empty_annos(self.detection_annos)
        self.dt_image_idxes = [anno["image_idx"][0] for anno in annos]
        # get class in dt
        available_cls = []
        for anno in self.detection_annos:
            for name in anno["name"]:
                if name not in available_cls:
                    available_cls.append(name)

        self.dt_combobox.clear()
        self.dt_combobox.addItem("All")
        for cls_name in available_cls:
            self.dt_combobox.addItem(cls_name)

        current_class = self.dt_combobox.currentText()
        if current_class == "All":
            self.current_detection_class_ids = self.dt_image_idxes
        else:
            self.current_detection_class_ids = [
                anno["image_idx"][0] for anno in annos if anno["name"] == current_class
            ]
        self.current_detection_class_index = 0

    def sample_to_current_data(self):
        if self.kitti_info is None:
            self.error("you must load infos and choose a existing image idx first.")
            return

        rect = self.kitti_info["calib/R0_rect"]
        p2 = self.kitti_info["calib/P2"]
        trv2c = self.kitti_info["calib/Tr_velo_to_cam"]
        num_features = 4
        if "pointcloud_num_features" in self.kitti_info:
            num_features = self.kitti_info["pointcloud_num_features"]

        if self.db_sampler is not None:
            # [n in class_names for n in self.gt_names], dtype=np.bool_)
            gt_boxes_mask = np.ones((self.gt_names.shape[0],), np.bool_)
            sampled_dict = self.db_sampler.sample_all(
                self.root_path,
                self.gt_boxes,
                self.gt_names,
                num_features,
                False,
                gt_group_ids=self.group_ids,
                rect=rect,
                Trv2c=trv2c,
                P2=p2,
            )
            if sampled_dict is not None:
                sampled_gt_names = sampled_dict["gt_names"]
                sampled_gt_boxes = sampled_dict["gt_boxes"]
                sampled_points = sampled_dict["points"]
                sampled_gt_masks = sampled_dict["gt_masks"]
                sampled_difficulty = sampled_dict["difficulty"]
                self.gt_names = np.concatenate(
                    [self.gt_names, sampled_gt_names], axis=0
                )
                self.gt_boxes = np.concatenate([self.gt_boxes, sampled_gt_boxes])
                gt_boxes_mask = np.concatenate(
                    [gt_boxes_mask, sampled_gt_masks], axis=0
                )
                self.difficulty = np.concatenate(
                    [self.difficulty, sampled_difficulty], axis=0
                )
                self.points = np.concatenate([sampled_points, self.points], axis=0)
                sampled_group_ids = sampled_dict["group_ids"]
                if self.group_ids is not None:
                    self.group_ids = np.concatenate([self.group_ids, sampled_group_ids])

            # should remove unrelated objects after noise per object
            self.gt_boxes = self.gt_boxes[gt_boxes_mask]
            self.gt_names = self.gt_names[gt_boxes_mask]
            self.difficulty = self.difficulty[gt_boxes_mask]
            if self.group_ids is not None:
                self.group_ids = self.group_ids[gt_boxes_mask]
        else:
            self.error("you enable sample but not provide a database")

    def data_augmentation(self):
        if self.kitti_info is None:
            self.error("you must load infos and choose a existing image idx first.")
            return

        seed = np.random.randint(5000000)
        np.random.seed(seed)
        self.info(f"prep random seed: {seed}")
        t = time.time()
        group_ids = None
        if self.w_config.get("GroupNoisePerObject"):
            group_ids = self.group_ids
        prep.noise_per_object_v3_(
            self.gt_boxes,
            self.points,
            global_random_rot_range=[np.pi / 4, np.pi / 4 * 3],
            group_ids=group_ids,
            num_try=100,
        )
        self.info("noise time", time.time() - t)
        # self.gt_boxes, self.points)
        # self.gt_boxes, self.points)
        # self.gt_boxes, self.points)
        # self.gt_boxes, [0, -40, 70.4, 40])

    def draw_gt_in_image(self):
        if self.kitti_info is None:
            self.error("you must load infos and choose a existing image idx first.")
            return
        if self.gt_boxes is None:
            return
        rect = self.kitti_info["calib/R0_rect"]
        p2 = self.kitti_info["calib/P2"]
        trv2c = self.kitti_info["calib/Tr_velo_to_cam"]
        gt_boxes_camera = box_np_ops.box_lidar_to_camera(self.gt_boxes, rect, trv2c)
        boxes_3d = box_np_ops.center_to_corner_box3d(
            gt_boxes_camera[:, :3], gt_boxes_camera[:, 3:6], gt_boxes_camera[:, 6]
        )
        boxes_3d = boxes_3d.reshape((-1, 3))
        boxes_3d_p2 = box_np_ops.project_to_image(boxes_3d, p2)
        boxes_3d_p2 = boxes_3d_p2.reshape([-1, 8, 2])
        if self.current_image is not None:
            bbox_plot.draw_3d_bbox_in_ax(self.w_plt.ax, boxes_3d_p2, colors="b")

    def draw_detection(self, detection_anno, label_color=GLColor.Blue):
        if self.kitti_info is None:
            self.error("you must load infos and choose a existing image idx first.")
            return

        dt_box_color = self.w_config.get("DTBoxColor")[:3]
        dt_box_color = (*dt_box_color, self.w_config.get("DTBoxAlpha"))

        rect = self.kitti_info["calib/R0_rect"]
        p2 = self.kitti_info["calib/P2"]
        trv2c = self.kitti_info["calib/Tr_velo_to_cam"]
        detection_anno = kitti.remove_low_score(
            detection_anno, self.w_config.get("DTScoreThreshold")
        )

        _ = detection_anno["bbox"]

        dt_boxes_corners, scores, dt_box_lidar = kitti_anno_to_corners(
            self.kitti_info, detection_anno
        )
        num_dt = dt_box_lidar.shape[0]
        dt_boxes_corners_cam = box_np_ops.lidar_to_camera(dt_boxes_corners, rect, trv2c)
        dt_boxes_corners_cam = dt_boxes_corners_cam.reshape((-1, 3))
        dt_boxes_corners_cam_p2 = box_np_ops.project_to_image(dt_boxes_corners_cam, p2)
        dt_boxes_corners_cam_p2 = dt_boxes_corners_cam_p2.reshape([-1, 8, 2])
        _ = detection_anno["name"]

        self._draw_detection_score_labels(
            scores, dt_boxes_corners, dt_box_lidar, label_color
        )
        dt_box_color = np.tile(np.array(dt_box_color)[np.newaxis, ...], [num_dt, 1])
        if self.w_config.get("DTScoreAsAlpha") and scores is not None:
            dt_box_color = np.concatenate(
                [dt_box_color[:, :3], scores[..., np.newaxis]], axis=1
            )
        self.w_pc_viewer.boxes3d(
            "dt_boxes",
            dt_boxes_corners,
            dt_box_color,
            self.w_config.get("DTBoxLineWidth"),
            1.0,
        )

    def plot_gt_boxes_in_pointcloud(self):
        if self.kitti_info is None:
            self.error("you must load infos and choose a existing image idx first.")
            return
        if "annos" in self.kitti_info:
            gt_box_color = self.w_config.get("GTBoxColor")[:3]
            gt_box_color = (*gt_box_color, self.w_config.get("GTBoxAlpha"))
            diff = self.difficulty.tolist()
            diff_to_name = {-1: "unk", 0: "easy", 1: "moderate", 2: "hard"}
            diff_names = [diff_to_name.get(d) for d in diff]
            label_idx = list(range(self.gt_names.shape[0]))
            labels_ = [
                f"{i}:{length_value}, {d}"
                for i, length_value, d in zip(label_idx, self.gt_names, diff_names)
            ]
            boxes_corners = box_np_ops.center_to_corner_box3d(
                self.gt_boxes[:, :3],
                self.gt_boxes[:, 3:6],
                self.gt_boxes[:, 6],
                origin=[0.5, 0.5, 0],
                axis=2,
            )
            self.w_pc_viewer.boxes3d("gt_boxes", boxes_corners, gt_box_color, 3.0, 1.0)
            if self.w_config.get("DrawGTLabels"):
                self.w_pc_viewer.labels(
                    "gt_boxes/labels",
                    boxes_corners[:, 0, :],
                    labels_,
                    GLColor.Green,
                    15,
                )

    def plot_pointcloud(self):
        if self.kitti_info is None:
            self.error("you must load infos and choose a existing image idx first.")
            return

        point_color = self.w_config.get("PointColor")[:3]
        point_color = (*point_color, self.w_config.get("PointAlpha"))
        point_color = np.tile(np.array(point_color), [self.points.shape[0], 1])

        point_size = np.full(
            [self.points.shape[0]], self.w_config.get("PointSize"), dtype=np.float32
        )
        self.w_pc_viewer.draw_bounding_box()
        idx = self.image_idxes.index(self.kitti_info["image_idx"])
        if "annos" in self.kitti_info:
            # "circles", poses, np.linalg.norm(
            gt_point_mask = box_np_ops.points_in_rbbox(self.points, self.gt_boxes).any(
                1
            )
            point_size[gt_point_mask] = self.w_config.get("GTPointSize")
            gt_point_color = self.w_config.get("GTPointColor")
            gt_point_color = (*gt_point_color[:3], self.w_config.get("GTPointAlpha"))
            point_color[gt_point_mask] = gt_point_color
        self.w_pc_viewer.remove("dt_boxes/labels")
        self.w_pc_viewer.remove("dt_boxes")
        if self.detection_annos is not None and self.w_config.get("DrawDTBoxes"):
            detection_anno = self.detection_annos[idx]
            self.draw_detection(detection_anno)
        if self.w_config.get("WithReflectivity"):
            if self.points.shape[1] < 4:
                self.error("Your pointcloud don't contain reflectivity.")
            else:
                point_color = np.concatenate(
                    [point_color[:, :3], self.points[:, 3:4] * 0.8 + 0.2], axis=1
                )

        self.w_pc_viewer.scatter(
            "pointcloud", self.points[:, :3], point_color, size=point_size
        )

    def load_info(self, image_idx):
        if self.kitti_infos is None:
            self.error("you must load infos first.")
            return False

        if image_idx not in self.image_idxes:
            self.error(f"index{image_idx} not exist.")
            return False
        self.json_setting.set("image_idx", str(image_idx))
        idx = self.image_idxes.index(image_idx)
        self.kitti_info = self.kitti_infos[idx]
        if "timestamp" in self.kitti_info:
            self.message("timestamp", self.kitti_info["timestamp"])
        self._load_info_image()
        v_path = str(self.root_path / self.kitti_info["velodyne_path"])
        num_features = 4
        if "pointcloud_num_features" in self.kitti_info:
            num_features = self.kitti_info["pointcloud_num_features"]

        points = np.fromfile(v_path, dtype=np.float32, count=-1).reshape(
            [-1, num_features]
        )
        self.points = points
        rect = self.kitti_info["calib/R0_rect"]
        p2 = self.kitti_info["calib/P2"]
        trv2c = self.kitti_info["calib/Tr_velo_to_cam"]
        image_shape = None
        if "img_shape" in self.kitti_info:
            image_shape = self.kitti_info["img_shape"]
            if self.w_config.get("RemoveOutsidePoint"):
                self.points = box_np_ops.remove_outside_points(
                    self.points, rect, trv2c, p2, image_shape
                )
        img_path = self.w_image_save_path.text()
        img_path = str(Path(img_path).parent / f"{image_idx}.jpg")
        self.w_image_save_path.setText(img_path)
        self.json_setting.set("save_image_path", img_path)

        self._load_info_annotations(rect, trv2c)
        if self.w_config.get("EnableSample"):
            self.sample_to_current_data()
        if self.w_config.get("EnableAugmentation"):
            self.data_augmentation()
        return True

    def plot_image(self):
        if self.kitti_info is None:
            self.error("you need to load the info first before plot image")
            return False
        if self.current_image is not None:
            self.w_plt.ax.imshow(self.current_image)
            if "annos" in self.kitti_info:
                annos = self.kitti_info["annos"]
                annos = kitti.filter_kitti_anno(annos, self.w_config.get("UsedClass"))
                labels = annos["name"]
                num_obj = len([n for n in annos["name"] if n != "DontCare"])
                bbox_plot.draw_bbox_in_ax(
                    self.w_plt.ax,
                    annos["bbox"],
                    edgecolors=["g"] * num_obj + ["b"] * num_obj,
                    labels=[f"{i}: {labels[i]}" for i in range(len(labels))],
                )
        return True

    def plot_all(self, image_idx):
        self.w_plt.reset_plot()
        self.load_info(image_idx)
        self.plot_image()
        self.draw_gt_in_image()
        self.w_plt.draw()  # this isn't supported in ubuntu.
        self.plot_pointcloud()
        if self.w_config.get("DrawGTBoxes"):
            self.plot_gt_boxes_in_pointcloud()
        if self.w_config.get("DrawVoxels"):
            self.w_pc_viewer.draw_voxels(self.points, self.gt_boxes)

        return True

    def on_plot_button_pressed(self):
        if self.kitti_infos is None:
            self.error("you must load Kitti Infos first.")
            return
        image_idx = int(self.w_imgidx.text())
        if self.plot_all(image_idx):
            self.current_idx = self.image_idxes.index(image_idx)

    def close_event(self, event):
        config_str = self.w_config.dumps()
        self.json_setting.set("config", config_str)
        return super().closeEvent(event)

    def on_configchanged(self, msg):
        # save config to file
        idx = self.image_idxes.index(self.kitti_info["image_idx"])
        config_str = self.w_config.dumps()
        self.json_setting.set("config", config_str)
        pc_redraw_msgs = ["PointSize", "PointAlpha", "GTPointSize"]
        pc_redraw_msgs += ["GTPointAlpha", "WithReflectivity"]
        pc_redraw_msgs += ["PointColor", "GTPointColor"]
        box_redraw = ["GTBoxColor", "GTBoxAlpha"]
        dt_redraw = [
            "DTBoxColor",
            "DTBoxAlpha",
            "DrawDTLabels",
            "DTScoreAsAlpha",
            "DTScoreThreshold",
            "DTBoxLineWidth",
        ]

        vx_redraw_msgs = ["DrawPositiveVoxelsOnly", "DrawVoxels"]
        vx_redraw_msgs += ["PosVoxelColor", "PosVoxelAlpha"]
        vx_redraw_msgs += ["NegVoxelColor", "NegVoxelAlpha"]
        all_redraw_msgs = ["RemoveOutsidePoint"]
        if msg.name in vx_redraw_msgs:
            if self.w_config.get("DrawVoxels"):
                self.w_pc_viewer.draw_voxels(self.points, self.gt_boxes)
            else:
                self.w_pc_viewer.remove("voxels")
            return
        redraw_actions = (
            (pc_redraw_msgs, self.plot_pointcloud),
            (all_redraw_msgs, self.on_plot_button_pressed),
            (box_redraw, self.plot_gt_boxes_in_pointcloud),
        )
        for message_names, action in redraw_actions:
            if msg.name in message_names:
                action()
                return
        if (
            msg.name in dt_redraw
            and self.detection_annos is not None
            and self.w_config.get("DrawDTBoxes")
        ):
            self.draw_detection(self.detection_annos[idx])

    def on_load_vx_net_ckpt_pressed(self):
        ckpt_path = Path(self.w_vckpt_path.text())
        self.json_setting.set("latest_vxnet_ckpt_path", self.w_vckpt_path.text())
        self.inference_ctx.restore(ckpt_path)
        self.info("load VoxelNet ckpt succeed.")

    def on_build_vx_net_pressed(self):
        self.inference_ctx = TorchInferenceContext()
        vconfig_path = Path(self.w_vconfig_path.text())
        self.inference_ctx.build(vconfig_path)
        self.json_setting.set("latest_vxnet_cfg_path", str(vconfig_path))
        self.info("Build VoxelNet ckpt succeed.")

    def on_inference_vx_net_pressed(self):
        t = time.time()
        inputs = self.inference_ctx.get_inference_input_dict(
            self.kitti_info, self.points
        )
        self.info("input preparation time:", time.time() - t)
        t = time.time()
        with self.inference_ctx.ctx():
            det_annos = self.inference_ctx.inference(inputs)
        self.info("detection time:", time.time() - t)
        self.draw_detection(det_annos[0])

    def on_load_inference_vx_net_pressed(self):
        self.on_build_vx_net_pressed()
        self.on_load_vx_net_ckpt_pressed()
        self.on_inference_vx_net_pressed()

    def on_eval_vx_net_pressed(self):
        if "annos" not in self.kitti_infos[0]:
            self.error("ERROR: infos don't contain gt label.")
        t = time.time()
        det_annos = []
        input_cfg = self.inference_ctx.config.eval_input_reader
        model_cfg = self.inference_ctx.config.model.second

        class_names = list(input_cfg.class_names)
        num_features = model_cfg.num_point_features
        with self.inference_ctx.ctx():
            for info in list_bar(self.kitti_infos):
                v_path = self.root_path / info["velodyne_path"]
                # v_path.parent.stem + "_reduced") / v_path.name
                points = np.fromfile(str(v_path), dtype=np.float32, count=-1).reshape(
                    [-1, num_features]
                )
                rect = info["calib/R0_rect"]
                p2 = info["calib/P2"]
                trv2c = info["calib/Tr_velo_to_cam"]
                image_shape = info["img_shape"]
                if self.w_config.get("RemoveOutsidePoint"):
                    points = box_np_ops.remove_outside_points(
                        points, rect, trv2c, p2, image_shape
                    )
                inputs = self.inference_ctx.get_inference_input_dict(info, points)
                det_annos += self.inference_ctx.inference(inputs)
        self.info("total detection time:", time.time() - t)
        gt_annos = [i["annos"] for i in self.kitti_infos]
        self.info(get_official_eval_result(gt_annos, det_annos, class_names))

    def _draw_detection_score_labels(self, scores, corners, boxes_lidar, label_color):
        if scores is None or not self.w_config.get("DrawDTLabels"):
            return
        if self.gt_boxes is not None:
            iou = _riou3d_shapely(self.gt_boxes, boxes_lidar)
            best_iou = iou.max(0) if iou.shape[0] != 0 else np.zeros([0, 0])
            score_text = [
                f"score={score:.2f}, iou={overlap:.2f}"
                for score, overlap in zip(scores, best_iou)
            ]
        else:
            score_text = [
                f"score={score:.2f}, z={z_value:.2f}"
                for score, z_value in zip(scores, boxes_lidar[:, 2])
            ]
        self.w_pc_viewer.labels(
            "dt_boxes/labels", corners[:, 1, :], score_text, label_color, 15
        )

    def _load_info_annotations(self, rect, trv2c):
        if "annos" not in self.kitti_info:
            self.gt_boxes = self.gt_names = self.difficulty = self.group_ids = None
            return
        annos = self.kitti_info["annos"]
        num_obj = len([name for name in annos["name"] if name != "DontCare"])
        dims = annos["dimensions"][:num_obj]
        loc = annos["location"][:num_obj]
        rots = annos["rotation_y"][:num_obj]
        self.difficulty = annos["difficulty"][:num_obj]
        self.gt_names = annos["name"][:num_obj]
        gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)
        self.gt_boxes = box_np_ops.box_camera_to_lidar(gt_boxes_camera, rect, trv2c)
        if "group_ids" in annos:
            self.group_ids = annos["group_ids"][:num_obj]

    def _load_info_image(self):
        if "img_path" not in self.kitti_info or not self.kitti_info["img_path"]:
            self.current_image = None
            return
        image_path = self.root_path / self.kitti_info["img_path"]
        self.current_image = io.imread(str(image_path))


setattr(KittiViewer, "closeEvent", KittiViewer.close_event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = KittiViewer()
    sys.exit(app.exec_())
