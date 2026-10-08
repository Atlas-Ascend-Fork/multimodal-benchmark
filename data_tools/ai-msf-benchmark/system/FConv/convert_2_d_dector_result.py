# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


import fire


def read_id_list(p1):
    with open(p1, "r") as f:
        data = f.read().splitlines()
    return data


# Image ID, Category ID, Score, Box (x1, y1, x2, y2)
str2det_id = {"Pedestrian": 1, "Car": 2, "Cyclist": 3}


def _converted_detection_lines(input_path, sample_id, line_format):
    converted = []
    with open(input_path, "r") as stream:
        for input_line in stream.read().splitlines():
            values = input_line.split(" ")
            category_id = str2det_id[values[0]]
            score = values[-1]
            coordinates = [int(float(values[index])) for index in range(4, 8)]
            converted.append(line_format.format(
                sample_id, category_id, score, *coordinates
            ))
    return converted


def conver_2_d_result(from_dir, to_dir, id_list, id_type):
    lines_format = "{} {} {} {} {} {} {}\n"
    output_path = "{}/rgb_detection_{}.txt".format(to_dir, id_type)
    with open(output_path, "w") as fo:
        lines_arr = []
        for sample_id in id_list:
            input_path = "{}/{}.txt".format(from_dir, sample_id)
            lines_arr.extend(
                _converted_detection_lines(input_path, sample_id, lines_format)
            )
        fo.writelines(lines_arr)


def convert(from_dir, multi_insert=False):
    if multi_insert:  # mark adaptive msf insert project
        convert_msf_insert(from_dir)
    else:
        convert_msf_benchmark(from_dir)


def convert_msf_benchmark(from_dir):
    train_id_path = "./kitti/image_sets/train.txt"
    train_id_list = read_id_list(train_id_path)
    test_id_path = "./kitti/image_sets/val.txt"
    test_id_list = read_id_list(test_id_path)

    to_dir = "./kitti/rgb_detections"

    id_type_arr = ["train", "val"]
    id_list_arr = [train_id_list, test_id_list]
    for id_type, id_list in zip(id_type_arr, id_list_arr):
        conver_2_d_result(from_dir, to_dir, id_list, id_type)


def convert_msf_insert(from_dir):
    train_id_list = []
    test_id_path = "./kitti/image_sets/val.txt"
    test_id_list = read_id_list(test_id_path)
    to_dir = "./kitti/rgb_detections"
    id_type_arr = ["train", "val"]
    id_list_arr = [train_id_list, test_id_list]
    for id_type, id_list in zip(id_type_arr, id_list_arr):
        conver_2_d_result(from_dir, to_dir, id_list, id_type)


if __name__ == "__main__":
    fire.Fire(convert)
