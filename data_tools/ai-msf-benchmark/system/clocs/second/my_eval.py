# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import os
import pathlib
import pickle
import warnings

from google.protobuf import text_format
from numba.core.errors import (
    NumbaDeprecationWarning,
    NumbaPendingDeprecationWarning,
    NumbaPerformanceWarning,
    NumbaWarning,
)
from tqdm import tqdm
import numpy as np
from shared.kitti_labels import get_label_annotation
from shared.detection_merge import (
    MERGE_DP_PARAMETERS_SCHEMA,
    detection_paths,
    merge_all_detections,
    merge_without_overlap,
    read_detection_lines,
    write_detection_lines,
)

from second.builder import anchor_generator_builder
from second.protos import pipeline_pb2
from second.utils.eval import get_official_eval_result, calculate_iou_partly
import second.data.kitti_common as kitti


from shared.parameters import bind_parameters as _bind_parameters


warnings.simplefilter("ignore", category=NumbaDeprecationWarning)
warnings.simplefilter("ignore", category=NumbaPendingDeprecationWarning)
warnings.simplefilter("ignore", category=NumbaPerformanceWarning)
warnings.simplefilter("ignore", category=NumbaWarning)
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")


def get_label_anno(label_path, obj_interesting):
    return [get_label_annotation(label_path, obj_interesting)]


def _merge_car_detections(paths, output_dir, merge_config):
    paths1, paths2 = paths
    iou_threshold, score_threshold, metric = merge_config
    for path1, path2 in zip(paths1, paths2):
        lines1 = read_detection_lines(path1)
        lines2 = read_detection_lines(path2)
        cars1 = [line for line in lines1 if line.startswith("Car")]
        cars2 = [line for line in lines2 if line.startswith("Car")]
        merged = merge_without_overlap(lines1, lines2, score_threshold)
        if merged is None:
            fusion_results = get_label_anno(path1, "Car")
            signal_results = get_label_anno(path2, "Car")
            overlaps = calculate_iou_partly(
                signal_results, fusion_results, metric=metric, num_parts=1
            )[0][0]
            maximum_overlaps = np.max(overlaps, axis=1)
            selected = np.where(maximum_overlaps < iou_threshold)[0]
            additions = [
                cars2[index]
                for index in selected
                if float(cars2[index].strip().split(" ")[-1]) >= score_threshold
            ]
            merged = cars1 + additions
        write_detection_lines(path1, output_dir, merged)


def merge_dp(*call_args, **call_kwargs):
    bound_parameters = _bind_parameters(
        MERGE_DP_PARAMETERS_SCHEMA, call_args, call_kwargs
    )
    dir1, dir2, output_dir, iou_th, score, metric = bound_parameters
    dts1, dts2 = detection_paths(dir1, dir2, metric)
    if iou_th is None:
        merge_all_detections(dts1, dts2, output_dir)
    else:
        _merge_car_detections((dts1, dts2), output_dir, (iou_th, score, metric))


def evaluate(dt_path):
    data_path = "/home/niangao/disk1/kitti4cloc"
    root_path = pathlib.Path(data_path)
    info_path = root_path / "kitti_infos_val.pkl"
    dt_annos = kitti.get_label_annos(dt_path)

    with open(info_path, "rb") as f:
        kitti_infos = pickle.load(f)
    gt_annos = [info["annos"] for info in kitti_infos]
    config_path = (
        "/home/niangao/PycharmProjects/fusion/CLOCs/second/configs/car.fhd.config"
    )
    config = pipeline_pb2.TrainEvalPipelineConfig()
    with open(config_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, config)
    model_cfg = config.model.second
    target_assigner_cfg = model_cfg.target_assigner
    anchor_cfg = target_assigner_cfg.anchor_generators
    anchor_generators = []
    for a_cfg in anchor_cfg:
        anchor_generator = anchor_generator_builder.build(a_cfg)
        anchor_generators.append(anchor_generator)

    class_names = [a.class_name for a in anchor_generators]

    result = get_official_eval_result(gt_annos, dt_annos, class_names)
    result_path = dt_path + ".txt"
    with open(result_path, "w") as f:
        f.write(result)


def my_merge_and_eval(noise):
    img_op = noise["img"]
    lidar_op = noise["lidar"]
    if img_op is None:
        img_op = "clean"
    if lidar_op is None:
        lidar_op = "clean"
    dt_dir_clocs = "/home/niangao/PycharmProjects/fusion/result/CLOCs/img_{}_lidar_{}_calib_clean".format(
        img_op, lidar_op
    )
    dt_dir_second = "/home/niangao/PycharmProjects/fusion/result/second/img_{}_lidar_{}_calib_clean/data".format(
        img_op, lidar_op
    )
    output_dir = "/home/niangao/PycharmProjects/fusion/CLOCs/second/merge_dt/img_{}_lidar_{}_calib_clean".format(
        img_op, lidar_op
    )
    os.makedirs(output_dir, exist_ok=True)

    merge_dp(
        dt_dir_clocs, dt_dir_second, output_dir, iou_th=0.7, score=0.8, metric=2
    )  #
    evaluate(output_dir)


# CLOCS-Rb
if __name__ == "__main__":
    noise_arr = [
        # # light
        # # # image loss
        # # # lidar loss
        # # lidar noise
        # weather
        {"img": "fog_low", "lidar": "fog_low", "calib": None},
        {"img": "fog_moderate", "lidar": "fog_moderate", "calib": None},
        {"img": "fog_strong", "lidar": "fog_strong", "calib": None},
        {
            "img": "rain10",
            "lidar": "rain10",
            "calib": None,
        },
        {
            "img": "rain25",
            "lidar": "rain25",
            "calib": None,
        },
        {
            "img": "rain50",
            "lidar": "rain50",
            "calib": None,
        },
    ]
    for noise in tqdm(noise_arr):
        my_merge_and_eval(noise)
