# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from functools import partial


from second.builder import dbsampler_builder
from second.data.dataset import KittiDataset
from second.data.preprocess import prep_pointcloud
from second.protos import input_reader_pb2
from shared.parameters import bind_named as _bind_named


def _optional_sampler(sampler_config):
    if len(sampler_config.sample_groups) > 0:
        return dbsampler_builder.build(sampler_config)
    return None


def _build_prep_func(*call_args, **call_kwargs):
    (
        cfg,
        model_config,
        training,
        voxel_generator,
        target_assigner,
        out_size_factor,
    ) = _bind_named(
        call_args,
        call_kwargs,
        ('cfg', 'model_config', 'training', 'voxel_generator', 'target_assigner', 'out_size_factor'),
    )
    return partial(
        prep_pointcloud,
        root_path=cfg.kitti_root_path,
        class_names=target_assigner.classes,
        voxel_generator=voxel_generator,
        target_assigner=target_assigner,
        training=training,
        max_voxels=cfg.max_number_of_voxels,
        remove_outside_points=False,
        remove_unknown=cfg.remove_unknown_examples,
        create_targets=training,
        shuffle_points=cfg.shuffle_points,
        gt_rotation_noise=list(cfg.groundtruth_rotation_uniform_noise),
        gt_loc_noise_std=list(cfg.groundtruth_localization_noise_std),
        global_rotation_noise=list(cfg.global_rotation_uniform_noise),
        global_scaling_noise=list(cfg.global_scaling_uniform_noise),
        global_random_rot_range=list(cfg.global_random_rotation_range_per_object),
        db_sampler=_optional_sampler(cfg.database_sampler),
        unlabeled_db_sampler=_optional_sampler(cfg.unlabeled_database_sampler),
        generate_bev=model_config.use_bev,
        without_reflectivity=model_config.without_reflectivity,
        num_point_features=model_config.num_point_features,
        anchor_area_threshold=cfg.anchor_area_threshold,
        gt_points_drop=cfg.groundtruth_points_drop_percentage,
        gt_drop_max_keep=cfg.groundtruth_drop_max_keep_points,
        remove_points_after_sample=cfg.remove_points_after_sample,
        remove_environment=cfg.remove_environment,
        use_group_id=cfg.use_group_id,
        out_size_factor=out_size_factor,
    )


def build(
    input_reader_config, model_config, training, voxel_generator, target_assigner=None
):
    """Build a KITTI dataset from an InputReader configuration."""
    if not isinstance(input_reader_config, input_reader_pb2.InputReader):
        raise ValueError("input_reader_config not of type input_reader_pb2.InputReader.")
    stride = model_config.rpn.layer_strides[0]
    upsample_stride = model_config.rpn.upsample_strides[0]
    out_size_factor = int(
        stride / upsample_stride * model_config.middle_feature_extractor.downsample_factor
    )
    if not (out_size_factor > 0):
        raise AssertionError
    cfg = input_reader_config
    feature_map_size = [*(voxel_generator.grid_size[:2] // out_size_factor), 1][::-1]
    if not all(name != "" for name in target_assigner.classes):
        raise AssertionError("you must specify class_name in anchor_generators.")
    prep_func = _build_prep_func(
        cfg, model_config, training, voxel_generator, target_assigner, out_size_factor
    )
    dataset = KittiDataset(
        info_path=cfg.kitti_info_path,
        root_path=cfg.kitti_root_path,
        num_point_features=model_config.num_point_features,
        target_assigner=target_assigner,
        feature_map_size=feature_map_size,
        prep_func=prep_func,
    )

    return dataset
