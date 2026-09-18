# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Network assembly helpers for FConv detector variants."""

import time

import torch
from shared.parameters import bind_named as _bind_named


def run_detection_model(model, inputs, time_meter):
    torch.cuda.synchronize()
    start_time = time.time()
    with torch.no_grad():
        outputs = model(inputs)
    torch.cuda.synchronize()
    time_meter.update(time.time() - start_time)
    return tuple(value.data.cpu().numpy() for value in outputs)


def pooled_point_features(module, point_cloud, samples, feature, one_hot):
    pooled = []
    for index, sample in enumerate(samples, 1):
        pointnet = getattr(module, "pointnet{}".format(index))
        values, _ = torch.max(pointnet(point_cloud, feature, sample), -1)
        pooled.append(values)
    if one_hot is not None:
        if module.num_vec != one_hot.shape[1]:
            raise AssertionError
        pooled = [
            torch.cat(
                [value, one_hot.unsqueeze(-1).expand(-1, -1, value.shape[-1])],
                1,
            )
            for value in pooled
        ]
    return tuple(pooled)


def initialize_conv_feature_weights(module, nn_module):
    for layer in module.modules():
        if isinstance(layer, (nn_module.conv1d, nn_module.ConvTranspose1d)):
            nn_module.init.kaiming_normal_(layer.weight.data, mode="fan_in")
            if layer.bias is not None:
                layer.bias.data.zero_()
        if isinstance(layer, nn_module.BatchNorm1d):
            layer.weight.data.fill_(1)
            layer.bias.data.zero_()


def forward_conv_features(module, inputs):
    value = module.block1_conv1(inputs[0])
    stage_values = []
    for stage, skip in enumerate(inputs[1:], 2):
        value = getattr(module, "block{}_conv1".format(stage))(value)
        value = getattr(module, "block{}_conv2".format(stage))(value)
        value = torch.cat([value, skip], 1)
        value = getattr(module, "block{}_merge".format(stage))(value)
        stage_values.append(value)
    upsampled = [
        getattr(module, "block{}_deconv".format(stage))(value)
        for stage, value in enumerate(stage_values, 2)
    ]
    target_length = upsampled[0].shape[-1]
    return torch.cat(
        [upsampled[0]] + [value[:, :, :target_length] for value in upsampled[1:]],
        1,
    )


def initialize_pointnet_detector(*call_args, **call_kwargs):
    (
        module,
        config,
        dataset_info,
        feature_class,
        conv_class,
        input_channel,
        num_vec,
        num_classes,
        head_channels,
        nn_module,
    ) = _bind_named(
        call_args,
        call_kwargs,
        (
            'module', 'config', 'dataset_info', 'feature_class', 'conv_class',
            'input_channel', 'num_vec', 'num_classes', 'head_channels', 'nn_module',
        ),
    )
    dataset_name = config.DATA.DATASET_NAME
    if dataset_name not in dataset_info:
        raise AssertionError
    module.category_info = dataset_info[dataset_name]
    module.num_size_cluster = len(module.category_info.CLASSES)
    module.mean_size_array = module.category_info.MEAN_SIZE_ARRAY
    module.feat_net = feature_class(input_channel, num_vec)
    module.conv_net = conv_class(128, num_vec)
    module.num_classes = num_classes
    module.num_bins = config.DATA.NUM_HEADING_BIN
    output_size = 3 + module.num_bins * 2 + module.num_size_cluster * 4
    module.reg_out = nn_module.conv1d(head_channels, output_size, 1)
    module.cls_out = nn_module.conv1d(head_channels, 2, 1)
    module.relu = nn_module.ReLU(True)


def initialize_detector_forward(state, reference_count, flatten_class_scores=False):
    data = state["data_dicts"]
    for state_name, data_name in (
        ("point_cloud", "point_cloud"),
        ("one_hot_vec", "one_hot"),
        ("cls_label", "cls_label"),
        ("size_class_label", "size_class"),
        ("center_label", "box3d_center"),
        ("heading_label", "box3d_heading"),
        ("size_label", "box3d_size"),
    ):
        state[state_name] = data.get(data_name)
    references = []
    for index in range(1, reference_count + 1):
        name = "center_ref{}".format(index)
        state[name] = data.get(name)
        references.append(state[name])
    point_cloud = state["point_cloud"]
    state["batch_size"] = point_cloud.shape[0]
    state["object_point_cloud_xyz"] = point_cloud[:, :3, :].contiguous()
    state["object_point_cloud_i"] = (
        point_cloud[:, [3], :].contiguous() if point_cloud.shape[1] > 3 else None
    )
    state["mean_size_array"] = torch.from_numpy(
        state["self"].mean_size_array
    ).type_as(point_cloud)
    features = state["self"].feat_net(
        state["object_point_cloud_xyz"],
        references,
        state["object_point_cloud_i"],
        state["one_hot_vec"],
    )
    for index, feature in enumerate(features, 1):
        state["feat{}".format(index)] = feature
    state["x"] = state["self"].conv_net(*features)
    state["cls_scores"] = state["self"].cls_out(state["x"])
    state["outputs"] = state["self"].reg_out(state["x"])
    state["num_out"] = state["outputs"].shape[2]
    state["output_size"] = state["outputs"].shape[1]
    if flatten_class_scores:
        state["cls_scores"] = (
            state["cls_scores"].permute(0, 2, 1).contiguous().view(-1, 2)
        )
