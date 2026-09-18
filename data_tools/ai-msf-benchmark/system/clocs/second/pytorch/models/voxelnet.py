# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import NamedTuple
from enum import Enum
import pathlib
import sys
import time

from torch import nn
from torch.nn import functional as F
import numpy as np
import torch

from second.pytorch.core import box_torch_ops
from second.pytorch.core.losses import WeightedSoftmaxClassificationLoss
from second.pytorch.models import rpn, middle, voxel_encoder
from torchplus import metrics
import second.data.kitti_common as kitti
import second.utils.eval as se
import torchplus
from shared.parameters import ParameterSchema as _ParameterSchema, bind_parameters as _bind_parameters


class VoxelnetInitParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    output_shape: object
    num_class: object
    num_input_features: object
    vfe_class_name: object
    vfe_num_filters: object
    with_distance: object
    middle_class_name: object
    middle_num_input_features: object
    middle_num_filters_d1: object
    middle_num_filters_d2: object
    rpn_class_name: object
    rpn_num_input_features: object
    rpn_layer_nums: object
    rpn_layer_strides: object
    rpn_num_filters: object
    rpn_upsample_strides: object
    rpn_num_upsample_filters: object
    use_norm: object
    use_groupnorm: object
    num_groups: object
    use_sparse_rpn: object
    use_voxel_classifier: object
    use_direction_classifier: object
    use_sigmoid_score: object
    encode_background_as_zeros: object
    use_rotate_nms: object
    multiclass_nms: object
    nms_score_threshold: object
    nms_pre_max_size: object
    nms_post_max_size: object
    nms_iou_threshold: object
    target_assigner: object
    use_bev: object
    use_rc_net: object
    lidar_only: object
    cls_loss_weight: object
    loc_loss_weight: object
    pos_cls_weight: object
    neg_cls_weight: object
    direction_loss_weight: object
    loss_norm_type: object
    encode_rad_error_by_sin: object
    loc_loss_ftor: object
    cls_loss_ftor: object
    measure_time: object
    name: object


class VoxelnetUpdateMetricsParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    cls_loss: object
    loc_loss: object
    cls_preds: object
    labels: object
    sampled: object
    vox_preds: object
    vox_labels: object
    vox_weights: object


class CreateLossParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    loc_loss_ftor: object
    cls_loss_ftor: object
    box_preds: object
    cls_preds: object
    cls_targets: object
    cls_weights: object
    reg_targets: object
    reg_weights: object
    num_class: object
    encode_background_as_zeros: object
    encode_rad_error_by_sin: object
    box_code_size: object


if "/opt/ros/kinetic/lib/python2.7/dist-packages" in sys.path:
    sys.path.remove("/opt/ros/kinetic/lib/python2.7/dist-packages")


def _get_pos_neg_loss(cls_loss, labels):
    batch_size = cls_loss.shape[0]
    if cls_loss.shape[-1] == 1 or len(cls_loss.shape) == 2:
        # this part is acutually running
        cls_pos_loss = (labels > 0).type_as(cls_loss) * cls_loss.view(batch_size, -1)
        cls_neg_loss = (labels == 0).type_as(cls_loss) * cls_loss.view(batch_size, -1)
        cls_pos_loss = cls_pos_loss.sum() / batch_size
        cls_neg_loss = cls_neg_loss.sum() / batch_size
    else:
        cls_pos_loss = cls_loss[..., 1:].sum() / batch_size
        cls_neg_loss = cls_loss[..., 0].sum() / batch_size
    return cls_pos_loss, cls_neg_loss


class LossNormType(Enum):
    NormByNumPositives = "norm_by_num_positives"
    NormByNumExamples = "norm_by_num_examples"
    NormByNumPosNeg = "norm_by_num_pos_neg"


_CREATE_LOSS_PARAMETERS_SCHEMA = _ParameterSchema(
    CreateLossParameters,
    (
        "loc_loss_ftor",
        "cls_loss_ftor",
        "box_preds",
        "cls_preds",
        "cls_targets",
        "cls_weights",
        "reg_targets",
        "reg_weights",
        "num_class",
        "encode_background_as_zeros",
        "encode_rad_error_by_sin",
        "box_code_size",
    ),
    {
        "encode_background_as_zeros": True,
        "encode_rad_error_by_sin": True,
        "box_code_size": 7,
    },
    frozenset(()),
    12,
    None,
    None,
)
_VOXELNET_INIT_PARAMETERS_SCHEMA = _ParameterSchema(
    VoxelnetInitParameters,
    (
        "output_shape",
        "num_class",
        "num_input_features",
        "vfe_class_name",
        "vfe_num_filters",
        "with_distance",
        "middle_class_name",
        "middle_num_input_features",
        "middle_num_filters_d1",
        "middle_num_filters_d2",
        "rpn_class_name",
        "rpn_num_input_features",
        "rpn_layer_nums",
        "rpn_layer_strides",
        "rpn_num_filters",
        "rpn_upsample_strides",
        "rpn_num_upsample_filters",
        "use_norm",
        "use_groupnorm",
        "num_groups",
        "use_sparse_rpn",
        "use_voxel_classifier",
        "use_direction_classifier",
        "use_sigmoid_score",
        "encode_background_as_zeros",
        "use_rotate_nms",
        "multiclass_nms",
        "nms_score_threshold",
        "nms_pre_max_size",
        "nms_post_max_size",
        "nms_iou_threshold",
        "target_assigner",
        "use_bev",
        "use_rc_net",
        "lidar_only",
        "cls_loss_weight",
        "loc_loss_weight",
        "pos_cls_weight",
        "neg_cls_weight",
        "direction_loss_weight",
        "loss_norm_type",
        "encode_rad_error_by_sin",
        "loc_loss_ftor",
        "cls_loss_ftor",
        "measure_time",
        "name",
    ),
    {
        "num_class": 2,
        "num_input_features": 4,
        "vfe_class_name": "VoxelFeatureExtractor",
        "vfe_num_filters": None,
        "with_distance": False,
        "middle_class_name": "SparseMiddleExtractor",
        "middle_num_input_features": -1,
        "middle_num_filters_d1": None,
        "middle_num_filters_d2": None,
        "rpn_class_name": "RPN",
        "rpn_num_input_features": -1,
        "rpn_layer_nums": None,
        "rpn_layer_strides": None,
        "rpn_num_filters": None,
        "rpn_upsample_strides": None,
        "rpn_num_upsample_filters": None,
        "use_norm": True,
        "use_groupnorm": False,
        "num_groups": 32,
        "use_sparse_rpn": False,
        "use_voxel_classifier": False,
        "use_direction_classifier": True,
        "use_sigmoid_score": False,
        "encode_background_as_zeros": True,
        "use_rotate_nms": True,
        "multiclass_nms": False,
        "nms_score_threshold": 0.5,
        "nms_pre_max_size": 1000,
        "nms_post_max_size": 20,
        "nms_iou_threshold": 0.1,
        "target_assigner": None,
        "use_bev": False,
        "use_rc_net": False,
        "lidar_only": False,
        "cls_loss_weight": 1.0,
        "loc_loss_weight": 1.0,
        "pos_cls_weight": 1.0,
        "neg_cls_weight": 1.0,
        "direction_loss_weight": 1.0,
        "loss_norm_type": LossNormType.NormByNumPositives,
        "encode_rad_error_by_sin": False,
        "loc_loss_ftor": None,
        "cls_loss_ftor": None,
        "measure_time": False,
        "name": "voxelnet",
    },
    frozenset(()),
    46,
    None,
    None,
)
_VOXELNET_UPDATE_METRICS_PARAMETERS_SCHEMA = _ParameterSchema(
    VoxelnetUpdateMetricsParameters,
    (
        "cls_loss",
        "loc_loss",
        "cls_preds",
        "labels",
        "sampled",
        "vox_preds",
        "vox_labels",
        "vox_weights",
    ),
    {"vox_preds": None, "vox_labels": None, "vox_weights": None},
    frozenset(()),
    8,
    None,
    None,
)


def _voxelnet_defaults(parameters):
    defaults = {
        "vfe_num_filters": [32, 128],
        "middle_num_filters_d1": [64],
        "middle_num_filters_d2": [64, 64],
        "rpn_layer_nums": [3, 5, 5],
        "rpn_layer_strides": [2, 2, 2],
        "rpn_num_filters": [128, 128, 256],
        "rpn_upsample_strides": [1, 2, 4],
        "rpn_num_upsample_filters": [256, 256, 256],
    }
    replacements = {
        name: value for name, value in defaults.items()
        if getattr(parameters, name) is None
    }
    schema = _VOXELNET_INIT_PARAMETERS_SCHEMA
    return schema.parameter_type(
        *(replacements.get(name, getattr(parameters, name)) for name in schema.names)
    )


def _configure_voxelnet_fields(module, parameters):
    aliases = {
        "name": "name",
        "_num_class": "num_class",
        "_use_rotate_nms": "use_rotate_nms",
        "_multiclass_nms": "multiclass_nms",
        "_nms_score_threshold": "nms_score_threshold",
        "_nms_pre_max_size": "nms_pre_max_size",
        "_nms_post_max_size": "nms_post_max_size",
        "_nms_iou_threshold": "nms_iou_threshold",
        "_use_sigmoid_score": "use_sigmoid_score",
        "_encode_background_as_zeros": "encode_background_as_zeros",
        "_use_sparse_rpn": "use_sparse_rpn",
        "_use_direction_classifier": "use_direction_classifier",
        "_use_bev": "use_bev",
        "_num_input_features": "num_input_features",
        "_lidar_only": "lidar_only",
        "target_assigner": "target_assigner",
        "_pos_cls_weight": "pos_cls_weight",
        "_neg_cls_weight": "neg_cls_weight",
        "_encode_rad_error_by_sin": "encode_rad_error_by_sin",
        "_loss_norm_type": "loss_norm_type",
        "_loc_loss_ftor": "loc_loss_ftor",
        "_cls_loss_ftor": "cls_loss_ftor",
        "_direction_loss_weight": "direction_loss_weight",
        "_cls_loss_weight": "cls_loss_weight",
        "_loc_loss_weight": "loc_loss_weight",
        "measure_time": "measure_time",
    }
    for attribute, parameter in aliases.items():
        setattr(module, attribute, getattr(parameters, parameter))
    box_coder = parameters.target_assigner.box_coder
    setattr(module, "_box_coder", box_coder)
    module.box_coder = box_coder
    module.num_class = parameters.num_class
    for name in (
        "encode_background_as_zeros", "use_direction_classifier", "use_sigmoid_score",
        "use_rotate_nms", "multiclass_nms", "nms_pre_max_size", "nms_post_max_size",
        "nms_iou_threshold", "nms_score_threshold",
    ):
        setattr(module, name, getattr(module, "_" + name))
    setattr(module, "_dir_loss_ftor", WeightedSoftmaxClassificationLoss())


def _build_voxel_feature_extractor(module, parameters):
    classes = {
        "VoxelFeatureExtractor": voxel_encoder.VoxelFeatureExtractor,
        "VoxelFeatureExtractorV2": voxel_encoder.VoxelFeatureExtractorV2,
        "VoxelFeatureExtractorV3": voxel_encoder.VoxelFeatureExtractorV3,
        "SimpleVoxel": voxel_encoder.SimpleVoxel,
    }
    extractor_class = classes.get(parameters.vfe_class_name)
    if extractor_class is None:
        raise KeyError("Unknown voxel feature extractor: %s" % parameters.vfe_class_name)
    module.voxel_feature_extractor = extractor_class(
        parameters.num_input_features,
        parameters.use_norm,
        num_filters=parameters.vfe_num_filters,
        with_distance=parameters.with_distance,
    )


def _rpn_input_filter_count(parameters):
    if parameters.middle_num_filters_d2:
        return parameters.middle_num_filters_d2[-1]
    if parameters.middle_num_filters_d1:
        return parameters.middle_num_filters_d1[-1]
    return parameters.vfe_num_filters[-1]


def _rpn_arguments(parameters):
    return {
        "use_norm": True,
        "num_class": parameters.num_class,
        "layer_nums": parameters.rpn_layer_nums,
        "layer_strides": parameters.rpn_layer_strides,
        "num_filters": parameters.rpn_num_filters,
        "upsample_strides": parameters.rpn_upsample_strides,
        "num_upsample_filters": parameters.rpn_num_upsample_filters,
        "num_anchor_per_loc": parameters.target_assigner.num_anchors_per_location,
        "encode_background_as_zeros": parameters.encode_background_as_zeros,
        "use_direction_classifier": parameters.use_direction_classifier,
        "use_bev": parameters.use_bev,
        "use_groupnorm": parameters.use_groupnorm,
        "num_groups": parameters.num_groups,
        "box_code_size": parameters.target_assigner.box_coder.code_size,
    }


def _build_voxelnet_rpn(module, parameters):
    arguments = _rpn_arguments(parameters)
    if parameters.use_sparse_rpn:
        arguments["num_input_features"] = _rpn_input_filter_count(parameters) * 2
        module.sparse_rpn = rpn.SparseRPN(
            parameters.output_shape,
            num_filters_down1=parameters.middle_num_filters_d1,
            num_filters_down2=parameters.middle_num_filters_d2,
            **arguments
        )
        return
    middle_classes = {
        "SparseMiddleExtractor": middle.SparseMiddleExtractor,
        "SpMiddleD4HD": middle.SpMiddleD4HD, "SpMiddleD8HD": middle.SpMiddleD8HD,
        "SpMiddleFHD": middle.SpMiddleFHD, "SpMiddleFHDV2": middle.SpMiddleFHDV2,
        "SpMiddleFHDLarge": middle.SpMiddleFHDLarge,
        "SpMiddleResNetFHD": middle.SpMiddleResNetFHD,
        "SpMiddleD4HDLite": middle.SpMiddleD4HDLite,
        "SpMiddleFHDLite": middle.SpMiddleFHDLite, "SpMiddle2K": middle.SpMiddle2K,
        "MiddleExtractor": middle.MiddleExtractor, "SpMiddleVision": middle.SpMiddleVision,
    }
    middle_class = middle_classes.get(parameters.middle_class_name)
    if middle_class is None:
        raise KeyError("Unknown middle extractor: %s" % parameters.middle_class_name)
    module.middle_feature_extractor = middle_class(
        parameters.output_shape, parameters.use_norm,
        num_input_features=parameters.middle_num_input_features,
        num_filters_down1=parameters.middle_num_filters_d1,
        num_filters_down2=parameters.middle_num_filters_d2,
    )
    arguments["num_input_features"] = parameters.rpn_num_input_features
    rpn_class = {"RPN": rpn.RPN, "RPNV2": rpn.RPNV2}.get(parameters.rpn_class_name)
    if rpn_class is None:
        raise KeyError("Unknown RPN: %s" % parameters.rpn_class_name)
    module.rpn = rpn_class(**arguments)


def _build_voxelnet_metrics(module, parameters):
    module.rpn_acc = metrics.Accuracy(
        dim=-1, encode_background_as_zeros=parameters.encode_background_as_zeros
    )
    module.rpn_precision = metrics.Precision(dim=-1)
    module.rpn_recall = metrics.Recall(dim=-1)
    module.rpn_metrics = metrics.PrecisionRecall(
        dim=-1, thresholds=[0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95],
        use_sigmoid_score=parameters.use_sigmoid_score,
        encode_background_as_zeros=parameters.encode_background_as_zeros,
    )
    module.rpn_cls_loss = metrics.Scalar()
    module.rpn_loc_loss = metrics.Scalar()
    module.rpn_total_loss = metrics.Scalar()
    module.register_buffer("global_step", torch.LongTensor(1).zero_())
    setattr(module, "_time_dict", {})
    setattr(module, "_time_total_dict", {})
    setattr(module, "_time_count_dict", {})


class VoxelNet(nn.Module):
    def __init__(self, *call_args, **call_kwargs):
        parameters = _bind_parameters(
            _VOXELNET_INIT_PARAMETERS_SCHEMA, call_args, call_kwargs
        )
        parameters = _voxelnet_defaults(parameters)
        super().__init__()
        _configure_voxelnet_fields(self, parameters)
        _build_voxel_feature_extractor(self, parameters)
        _build_voxelnet_rpn(self, parameters)
        _build_voxelnet_metrics(self, parameters)

    @staticmethod
    def convert_norm_to_float(net):
        """
        BatchNorm layers to have parameters in single precision.
        Find all layers and convert them back to float. This can't
        be done with built in .apply as that function will apply
        fn to all modules, parameters, and buffers. Thus we wouldn't
        be able to guard the float conversion based on the module type.
        """
        if isinstance(
            net, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)
        ):
            net.float()
        for child in net.children():
            VoxelNet.convert_norm_to_float(child)
        return net

    @staticmethod
    def _load_2d_predictions(example, detection_2d_path):
        image_index = example["image_idx"][0]
        filename = pathlib.Path(detection_2d_path) / (
            kitti.get_image_index_str(image_index) + ".txt"
        )
        with open(filename, "r") as file_handle:
            content = [line.strip().split(" ") for line in file_handle]
        predicted_class = np.array([row[0] for row in content], dtype="object")
        boxes = np.array([[float(value) for value in row[4:8]] for row in content]).reshape(-1, 4)
        scores = np.array([float(row[15]) for row in content]).reshape(-1, 1)
        detections = np.append(boxes, scores, axis=1)
        car_detections = detections[np.where(predicted_class == "Car"), :].reshape(-1, 5)
        return car_detections[np.where(car_detections[:, 4] >= -100)]

    @staticmethod
    def _clamp_image_boxes(boxes, image_shape):
        height, width = image_shape[0, 0], image_shape[0, 1]
        boxes[:, 0] = torch.clamp(boxes[:, 0], min=0, max=width)
        boxes[:, 1] = torch.clamp(boxes[:, 1], min=0, max=height)
        boxes[:, 2] = torch.clamp(boxes[:, 2], min=0, max=width)
        boxes[:, 3] = torch.clamp(boxes[:, 3], min=0, max=height)

    @staticmethod
    def _stage_two_tensors(prediction, top_predictions):
        boxes = prediction["bbox"]
        distance = torch.norm(prediction["box3d_lidar"][:, :2], p=2, dim=1, keepdim=True) / 82.0
        detector_boxes = top_predictions[:, :4]
        detector_scores = top_predictions[:, 4].reshape(-1, 1)
        overlaps = np.full((900000, 4), -1, dtype=boxes.detach().cpu().numpy().dtype)
        indices = np.full((900000, 2), -1, dtype=overlaps.dtype)
        iou_values, tensor_indices, max_num = se.build_stage2_training(
            (boxes.detach().cpu().numpy(), detector_boxes),
            (prediction["scores"].detach().cpu().numpy(), detector_scores,
             distance.detach().cpu().numpy()),
            (overlaps, indices), -1,
        )
        if max_num == 0:
            return torch.full((1, 4, 1, 2), -1.0), torch.full((2, 2), -1.0)
        iou_tensor = torch.FloatTensor(iou_values).permute(1, 0).reshape(1, 4, 1, 900000)
        index_tensor = torch.LongTensor(tensor_indices).reshape(-1, 2)
        return iou_tensor[:, :, :, :max_num], index_tensor[:max_num, :]

    def start_timer(self, *names):
        if not self.measure_time:
            return
        for name in names:
            self._time_dict[name] = time.time()
        torch.cuda.synchronize()

    def end_timer(self, name):
        if not self.measure_time:
            return
        torch.cuda.synchronize()
        time_elapsed = time.time() - self._time_dict[name]
        if name not in self._time_count_dict:
            self._time_count_dict[name] = 1
            self._time_total_dict[name] = time_elapsed
        else:
            self._time_count_dict[name] += 1
            self._time_total_dict[name] += time_elapsed
        self._time_dict[name] = 0

    def clear_timer(self):
        self._time_count_dict.clear()
        self._time_dict.clear()
        self._time_total_dict.clear()

    def get_avg_time_dict(self):
        ret = {}
        for name, val in self._time_total_dict.items():
            count = self._time_count_dict[name]
            ret[name] = val / max(1, count)
        return ret

    def set_global_step(self, step_value):
        self.global_step = step_value

    def update_global_step(self):
        self.global_step += 1

    def get_global_step(self):
        return int(self.global_step.cpu().numpy()[0])

    def forward(self, example, detection_2d_path):
        """module's forward should always accept dict and return loss."""
        voxels, num_points = example["voxels"], example["num_points"]
        coordinates = example["coordinates"]
        batch_anchors = example["anchors"]
        batch_size_dev = batch_anchors.shape[0]
        self.start_timer("voxel_feature_extractor")
        voxel_features = self.voxel_feature_extractor(voxels, num_points)
        self.end_timer("voxel_feature_extractor")
        preds_dict = self._network_predictions(
            example, voxel_features, coordinates, batch_size_dev
        )
        if self.training:
            return self._training_result(example, preds_dict, voxels, batch_size_dev)
        self.start_timer("predict")
        top_predictions = self._load_2d_predictions(example, detection_2d_path)
        res, iou_test, tensor_index = self.train_stage_2(
            example, preds_dict, top_predictions
        )
        self.end_timer("predict")
        return res, preds_dict, top_predictions, iou_test, tensor_index

    def train_stage_2(self, example, preds_dict, top_predictions):
        box_batches, cls_batches, direction_batches, masks = (
            self._decoded_prediction_batches(example, preds_dict)
        )
        predictions_dicts = []
        for box_preds, cls_preds, _, rect, trv2c, p2, image_index, mask in zip(
            box_batches, cls_batches, direction_batches, example["rect"],
            example["Trv2c"], example["P2"], example["image_idx"], masks,
        ):
            prediction = self._frame_prediction(
                box_preds, cls_preds, (rect, trv2c, p2), image_index, mask
            )
            self._clamp_image_boxes(prediction["bbox"], example["image_shape"])
            predictions_dicts.append(prediction)
            iou_tensor, index_tensor = self._stage_two_tensors(
                prediction, top_predictions
            )
        return predictions_dicts, iou_tensor, index_tensor

    def metrics_to_float(self):
        self.rpn_acc.float()
        self.rpn_metrics.float()
        self.rpn_cls_loss.float()
        self.rpn_loc_loss.float()
        self.rpn_total_loss.float()

    def update_metrics(self, *call_args, **call_kwargs):
        bound_parameters = _bind_parameters(
            _VOXELNET_UPDATE_METRICS_PARAMETERS_SCHEMA, call_args, call_kwargs
        )
        (
            cls_loss,
            loc_loss,
            cls_preds,
            labels,
            sampled,
            vox_preds,
            vox_labels,
            vox_weights,
        ) = bound_parameters
        batch_size = cls_preds.shape[0]
        num_class = self._num_class
        if not self._encode_background_as_zeros:
            num_class += 1
        cls_preds = cls_preds.view(batch_size, -1, num_class)
        rpn_acc = self.rpn_acc(labels, cls_preds, sampled).numpy()[0]
        prec, recall = self.rpn_metrics(labels, cls_preds, sampled)
        prec = prec.numpy()
        recall = recall.numpy()
        rpn_cls_loss = self.rpn_cls_loss(cls_loss).numpy()[0]
        rpn_loc_loss = self.rpn_loc_loss(loc_loss).numpy()[0]
        ret = {
            "cls_loss": float(rpn_cls_loss),
            "cls_loss_rt": float(cls_loss.data.cpu().numpy()),
            "loc_loss": float(rpn_loc_loss),
            "loc_loss_rt": float(loc_loss.data.cpu().numpy()),
            "rpn_acc": float(rpn_acc),
        }
        for i, thresh in enumerate(self.rpn_metrics.thresholds):
            ret[f"prec@{int(thresh*100)}"] = float(prec[i])
            ret[f"rec@{int(thresh*100)}"] = float(recall[i])
        return ret

    def clear_metrics(self):
        self.rpn_acc.clear()
        self.rpn_metrics.clear()
        self.rpn_cls_loss.clear()
        self.rpn_loc_loss.clear()
        self.rpn_total_loss.clear()

    def _network_predictions(self, example, voxel_features, coordinates, batch_size):
        if self._use_sparse_rpn:
            return self.sparse_rpn(voxel_features, coordinates, batch_size)
        self.start_timer("middle forward")
        spatial_features = self.middle_feature_extractor(
            voxel_features, coordinates, batch_size
        )
        self.end_timer("middle forward")
        self.start_timer("rpn forward")
        predictions = (
            self.rpn(spatial_features, example["bev_map"])
            if self._use_bev else self.rpn(spatial_features)
        )
        self.end_timer("rpn forward")
        return predictions

    def _training_result(self, example, predictions, voxels, batch_size):
        labels, reg_targets = example["labels"], example["reg_targets"]
        cls_weights, reg_weights, cared = prepare_loss_weights(
            labels, pos_cls_weight=self._pos_cls_weight,
            neg_cls_weight=self._neg_cls_weight, loss_norm_type=self._loss_norm_type,
            dtype=voxels.dtype,
        )
        cls_targets = (labels * cared.type_as(labels)).unsqueeze(-1)
        loc_loss, cls_loss = create_loss(
            self._loc_loss_ftor, self._cls_loss_ftor,
            box_preds=predictions["box_preds"], cls_preds=predictions["cls_preds"],
            cls_targets=cls_targets, cls_weights=cls_weights,
            reg_targets=reg_targets, reg_weights=reg_weights,
            num_class=self._num_class,
            encode_rad_error_by_sin=self._encode_rad_error_by_sin,
            encode_background_as_zeros=self._encode_background_as_zeros,
            box_code_size=self._box_coder.code_size,
        )
        loc_reduced = loc_loss.sum() / batch_size * self._loc_loss_weight
        cls_pos, cls_neg = _get_pos_neg_loss(cls_loss, labels)
        cls_pos /= self._pos_cls_weight
        cls_neg /= self._neg_cls_weight
        cls_reduced = cls_loss.sum() / batch_size * self._cls_loss_weight
        loss = loc_reduced + cls_reduced
        dir_loss = loss * 0
        if self._use_direction_classifier:
            targets = get_direction_target(example["anchors"], reg_targets)
            logits = predictions["dir_cls_preds"].view(batch_size, -1, 2)
            weights = (labels > 0).type_as(logits)
            weights /= torch.clamp(weights.sum(-1, keepdim=True), min=1.0)
            dir_loss = self._dir_loss_ftor(logits, targets, weights=weights).sum() / batch_size
            loss += dir_loss * self._direction_loss_weight
        return {
            "loss": loss, "cls_loss": cls_loss, "loc_loss": loc_loss,
            "cls_pos_loss": cls_pos, "cls_neg_loss": cls_neg,
            "cls_preds": predictions["cls_preds"], "dir_loss_reduced": dir_loss,
            "cls_loss_reduced": cls_reduced, "loc_loss_reduced": loc_reduced,
            "cared": cared,
        }

    def _decoded_prediction_batches(self, example, predictions):
        batch_size = example["anchors"].shape[0]
        anchors = example["anchors"].view(batch_size, -1, 7)
        box_predictions = predictions["box_preds"].view(
            batch_size, -1, self._box_coder.code_size
        )
        box_predictions = self._box_coder.decode_torch(box_predictions, anchors)
        class_count = self._num_class + int(not self._encode_background_as_zeros)
        class_predictions = predictions["cls_preds"].view(batch_size, -1, class_count)
        direction_predictions = (
            predictions["dir_cls_preds"].view(batch_size, -1, 2)
            if self._use_direction_classifier else [None] * batch_size
        )
        anchor_masks = (
            example["anchors_mask"].view(batch_size, -1)
            if "anchors_mask" in example else [None] * batch_size
        )
        return box_predictions, class_predictions, direction_predictions, anchor_masks

    def _frame_prediction(self, box_preds, cls_preds, calibration, image_index, mask):
        rect, trv2c, p2 = calibration
        if mask is not None:
            box_preds, cls_preds = box_preds[mask], cls_preds[mask]
        box_preds, cls_preds = box_preds.float(), cls_preds.float()
        if self._encode_background_as_zeros:
            if not self._use_sigmoid_score:
                raise AssertionError
            scores = torch.sigmoid(cls_preds)
        elif self._use_sigmoid_score:
            scores = torch.sigmoid(cls_preds)[..., 1:]
        else:
            scores = F.softmax(cls_preds, dim=-1)[..., 1:]
        camera_boxes = box_torch_ops.box_lidar_to_camera(
            box_preds, rect.float(), trv2c.float()
        )
        corners = box_torch_ops.center_to_corner_box3d(
            camera_boxes[:, :3], camera_boxes[:, 3:6], camera_boxes[:, 6],
            [0.5, 1.0, 0.5], axis=1,
        )
        projected = box_torch_ops.project_to_image(corners, p2.float())
        return {
            "bbox": torch.cat((torch.min(projected, dim=1)[0], torch.max(projected, dim=1)[0]), dim=1),
            "box3d_camera": camera_boxes,
            "box3d_lidar": box_preds,
            "scores": scores,
            "image_idx": image_index,
        }


# attention! sin(a-b) = sin(a)cos(b) - cos(a)sin(b), this is the logic behind this
def add_sin_difference(boxes1, boxes2):
    rad_pred_encoding = torch.sin(boxes1[..., -1:]) * torch.cos(boxes2[..., -1:])
    rad_tg_encoding = torch.cos(boxes1[..., -1:]) * torch.sin(boxes2[..., -1:])
    boxes1 = torch.cat([boxes1[..., :-1], rad_pred_encoding], dim=-1)
    boxes2 = torch.cat([boxes2[..., :-1], rad_tg_encoding], dim=-1)
    return boxes1, boxes2


def create_loss(*call_args, **call_kwargs):
    bound_parameters = _bind_parameters(
        _CREATE_LOSS_PARAMETERS_SCHEMA, call_args, call_kwargs
    )
    (
        loc_loss_ftor,
        cls_loss_ftor,
        box_preds,
        cls_preds,
        cls_targets,
        cls_weights,
        reg_targets,
        reg_weights,
        num_class,
        encode_background_as_zeros,
        encode_rad_error_by_sin,
        box_code_size,
    ) = bound_parameters
    batch_size = int(box_preds.shape[0])
    box_preds = box_preds.view(batch_size, -1, box_code_size)
    # cls_preds before here is in shape of [batch,200,176,2]
    if encode_background_as_zeros:
        # we do encode background as zeros
        cls_preds = cls_preds.view(batch_size, -1, num_class)
    else:
        cls_preds = cls_preds.view(batch_size, -1, num_class + 1)
    # cls_preds here is in shape of [batch,70400,1]
    cls_targets = cls_targets.squeeze(-1)
    # the shape of cls_targets after squeeze is [batch,70400]
    one_hot_targets = torchplus.nn.one_hot(
        cls_targets, depth=num_class + 1, dtype=box_preds.dtype
    )
    # shape of one_hot_targets here is [batch, 70400, 2]
    if encode_background_as_zeros:
        one_hot_targets = one_hot_targets[..., 1:]
    if encode_rad_error_by_sin:
        box_preds, reg_targets = add_sin_difference(box_preds, reg_targets)
    loc_losses = loc_loss_ftor(box_preds, reg_targets, weights=reg_weights)  # [N, M]
    cls_losses = cls_loss_ftor(
        cls_preds, one_hot_targets, weights=cls_weights
    )  # [N, M]
    # noted that for this focal loss function, alpha = 0.25, gamma = 2.0
    return loc_losses, cls_losses


def prepare_loss_weights(
    labels,
    pos_cls_weight=1.0,
    neg_cls_weight=1.0,
    loss_norm_type=LossNormType.NormByNumPositives,
    dtype=torch.float32,
):
    """get cls_weights and reg_weights from labels."""
    cared = labels >= 0  # becaues labels=-1 means dont't care
    positives = labels > 0  # because labels=1 means positive
    negatives = labels == 0  # because labels=0 means negative
    negative_cls_weights = negatives.type(dtype) * neg_cls_weight
    cls_weights = negative_cls_weights + pos_cls_weight * positives.type(dtype)
    reg_weights = positives.type(dtype)
    # because only positive regressions are included in the localization loss
    # for classification loss, both positive and negative are included
    if loss_norm_type == LossNormType.NormByNumExamples:
        num_examples = cared.type(dtype).sum(1, keepdim=True)
        num_examples = torch.clamp(num_examples, min=1.0)
        cls_weights /= num_examples
        bbox_normalizer = positives.sum(1, keepdim=True).type(dtype)
        reg_weights /= torch.clamp(bbox_normalizer, min=1.0)
    elif loss_norm_type == LossNormType.NormByNumPositives:  # for focal loss
        pos_normalizer = positives.sum(1, keepdim=True).type(dtype)
        # pos_normalizer shape: [batch,1]
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
    elif loss_norm_type == LossNormType.NormByNumPosNeg:
        pos_neg = torch.stack([positives, negatives], dim=-1).type(dtype)
        normalizer = pos_neg.sum(1, keepdim=True)  # [N, 1, 2]
        cls_normalizer = (pos_neg * normalizer).sum(-1)  # [N, M]
        cls_normalizer = torch.clamp(cls_normalizer, min=1.0)
        # cls_normalizer will be pos_or_neg_weight/num_pos_or_neg
        normalizer = torch.clamp(normalizer, min=1.0)
        reg_weights /= normalizer[:, 0:1, 0]
        cls_weights /= cls_normalizer
    else:
        raise ValueError(f"unknown loss norm type. available: {list(LossNormType)}")
    return cls_weights, reg_weights, cared


def assign_weight_to_each_class(
    labels, weight_per_class, norm_by_num=True, dtype=torch.float32
):
    weights = torch.zeros(labels.shape, dtype=dtype, device=labels.device)
    for label, weight in weight_per_class:
        positives = (labels == label).type(dtype)
        weight_class = weight * positives
        if norm_by_num:
            normalizer = positives.sum()
            normalizer = torch.clamp(normalizer, min=1.0)
            weight_class /= normalizer
        weights += weight_class
    return weights


def get_direction_target(anchors, reg_targets, one_hot=True):
    batch_size = reg_targets.shape[0]
    anchors = anchors.view(batch_size, -1, 7)
    # reg_targets[...,-1] is theta_g -theta_a, anchors[...,-1] is theta_a, so plus them together will give you theta_g
    rot_gt = reg_targets[..., -1] + anchors[..., -1]
    dir_cls_targets = (rot_gt > 0).long()
    if one_hot:
        dir_cls_targets = torchplus.nn.one_hot(dir_cls_targets, 2, dtype=anchors.dtype)
    return dir_cls_targets
