# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""VoxelNet builder.
"""

from second.protos import second_pb2
from second.pytorch.builder import losses_builder
from second.pytorch.models.voxelnet import LossNormType, VoxelNet


def _feature_options(model_cfg, num_input_features):
    extractor = model_cfg.voxel_feature_extractor
    middle = model_cfg.middle_feature_extractor
    return {
        "vfe_class_name": extractor.module_class_name,
        "vfe_num_filters": list(extractor.num_filters),
        "middle_class_name": middle.module_class_name,
        "middle_num_input_features": middle.num_input_features,
        "middle_num_filters_d1": list(middle.num_filters_down1),
        "middle_num_filters_d2": list(middle.num_filters_down2),
        "use_norm": True,
        "use_voxel_classifier": model_cfg.use_aux_classifier,
        "use_bev": model_cfg.use_bev,
        "num_input_features": num_input_features,
        "with_distance": extractor.with_distance,
    }


def _rpn_options(model_cfg):
    rpn = model_cfg.rpn
    return {
        "rpn_class_name": rpn.module_class_name,
        "rpn_num_input_features": rpn.num_input_features,
        "rpn_layer_nums": list(rpn.layer_nums),
        "rpn_layer_strides": list(rpn.layer_strides),
        "rpn_num_filters": list(rpn.num_filters),
        "rpn_upsample_strides": list(rpn.upsample_strides),
        "rpn_num_upsample_filters": list(rpn.num_upsample_filters),
        "num_groups": rpn.num_groups,
        "use_groupnorm": rpn.use_groupnorm,
        "use_rotate_nms": model_cfg.use_rotate_nms,
        "multiclass_nms": model_cfg.use_multi_class_nms,
        "nms_score_threshold": model_cfg.nms_score_threshold,
        "nms_pre_max_size": model_cfg.nms_pre_max_size,
        "nms_post_max_size": model_cfg.nms_post_max_size,
        "nms_iou_threshold": model_cfg.nms_iou_threshold,
        "use_sparse_rpn": False,
    }


def _loss_options(model_cfg, losses, loss_norm_type):
    cls_loss_ftor, loc_loss_ftor, cls_weight, loc_weight, _ = losses
    return {
        "cls_loss_weight": cls_weight,
        "loc_loss_weight": loc_weight,
        "pos_cls_weight": model_cfg.pos_class_weight,
        "neg_cls_weight": model_cfg.neg_class_weight,
        "direction_loss_weight": model_cfg.direction_loss_weight,
        "loss_norm_type": loss_norm_type,
        "encode_rad_error_by_sin": model_cfg.encode_rad_error_by_sin,
        "loc_loss_ftor": loc_loss_ftor,
        "cls_loss_ftor": cls_loss_ftor,
    }


def build(
    model_cfg: second_pb2.VoxelNet, voxel_generator, target_assigner, measure_time=False
) -> VoxelNet:
    """build second pytorch instance."""
    if not isinstance(model_cfg, second_pb2.VoxelNet):
        raise ValueError("model_cfg not of type " "second_pb2.VoxelNet.")
    vfe_num_filters = list(model_cfg.voxel_feature_extractor.num_filters)
    grid_size = voxel_generator.grid_size
    dense_shape = [1] + grid_size[::-1].tolist() + [vfe_num_filters[-1]]
    num_class = len(target_assigner.classes)

    num_input_features = model_cfg.num_point_features
    if model_cfg.without_reflectivity:
        num_input_features = 3
    loss_norm_type_dict = {
        0: LossNormType.NormByNumExamples,
        1: LossNormType.NormByNumPositives,
        2: LossNormType.NormByNumPosNeg,
    }
    loss_norm_type = loss_norm_type_dict.get(model_cfg.loss_norm_type)

    options = _feature_options(model_cfg, num_input_features)
    options.update(_rpn_options(model_cfg))
    options.update(
        _loss_options(model_cfg, losses_builder.build(model_cfg.loss), loss_norm_type)
    )
    options.update({
        "use_sigmoid_score": model_cfg.use_sigmoid_score,
        "encode_background_as_zeros": model_cfg.encode_background_as_zeros,
        "use_direction_classifier": model_cfg.use_direction_classifier,
    })
    return VoxelNet(
        dense_shape,
        num_class=num_class,
        target_assigner=target_assigner,
        measure_time=measure_time,
        **options,
    )
