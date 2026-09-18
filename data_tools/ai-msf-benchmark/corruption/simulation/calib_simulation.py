# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""'
@Project: fusion
@Description: Calib error (Spatial misalignment)
@Time:2022/1/13 17:26
@Author:NianGao

"""
import logging
import os
import shutil

from natsort import natsorted
from tqdm import tqdm
import numpy as np

from corruption.operator.calib_operator import CalibOperator

LOGGER = logging.getLogger(__name__)


class CalibSimulation(object):
    """
    :Author:  NianGao
    :Create:  2022/1/13
    """

    # calib format (object & tracking)
    # |-000001.txt
    # |-000002.txt
    @staticmethod
    def simulation(input_dir, output_dir, dop, sev, fm="%06d.txt"):
        if not (sev in range(1, 4)):
            raise AssertionError
        LOGGER.info("%s %s", "input_dir", input_dir)
        LOGGER.info("%s %s", "output_dir", output_dir)
        LOGGER.info("%s %s", "corruption", dop)
        LOGGER.info("%s %s", "severity", sev)
        fn_arr = os.listdir(input_dir)
        index_arr = [int(fn.split(".")[0]) for fn in fn_arr]
        index_arr.sort()
        for data_idx in tqdm(index_arr):
            input_fn = os.path.join(input_dir, fm % (data_idx))
            output_fn = os.path.join(output_dir, fm % (data_idx))
            CalibSimulation.simulation_one(input_fn, output_fn, dop, sev)

    @staticmethod
    def simulation_one(input_fn, output_fn, dop, sev):
        op = CalibOperator.operator_map()[dop]
        line_arr = []
        with open(input_fn, "r") as f:
            flag = False
            for _line in f.readlines():
                if len(_line.rstrip()) == 0:
                    continue
                key = CalibSimulation._calibration_key(_line)
                if key is not None:
                    flag = True
                    value = _line.replace(key, "")
                    v2_c = np.array([float(x) for x in value.split()])
                    v2_c = np.reshape(v2_c, [3, 4])
                    v2_c_noise = op(v2_c, sev)
                    v2_c_noise = numpy2str(v2_c_noise)
                    _line = key + v2_c_noise + "\n"
                line_arr.append(_line)
            if not (flag):
                raise AssertionError
        with open(output_fn, "w") as f:
            for line in line_arr:
                f.write(line)

    @staticmethod
    def simulation4depth(input_dir, output_dir, dop, sev):
        op = CalibOperator.operator_map()[dop]
        calib_dirs = natsorted(os.listdir(input_dir))
        for sub_dir in calib_dirs:
            calib_input_dir = os.path.join(input_dir, sub_dir)
            calib_output_dir = os.path.join(output_dir, sub_dir)
            os.makedirs(calib_output_dir, exist_ok=True)
            for fn in natsorted(os.listdir(calib_input_dir)):
                calib_input_txt = os.path.join(calib_input_dir, fn)
                calib_output_txt = os.path.join(calib_output_dir, fn)
                if fn == "calib_velo_to_cam.txt":
                    CalibSimulation._transform_depth_calibration(
                        op, calib_input_txt, calib_output_txt, sev
                    )
                else:
                    shutil.copyfile(calib_input_txt, calib_output_txt)

    @staticmethod
    def _transform_depth_calibration(op, input_path, output_path, severity):
        with open(input_path, "r") as stream:
            lines = stream.readlines()
        r_key, r_value = lines[1].strip().split(":")
        t_key, t_value = lines[2].strip().split(":")
        rotation = np.array(list(map(float, r_value.strip().split(" "))))
        translation = np.array(list(map(float, t_value.strip().split(" "))))
        velo2cam = np.hstack((rotation.reshape(3, 3), translation[..., np.newaxis]))
        if velo2cam.shape != (3, 4):
            raise AssertionError
        noisy = op(velo2cam, severity)
        if len(noisy) != 12:
            raise AssertionError
        rotation_noise = np.concatenate([noisy[:3], noisy[4:7], noisy[8:11]])
        translation_noise = np.array([noisy[3], noisy[7], noisy[11]])
        lines[1] = r_key + ": " + numpy2str(rotation_noise) + "\n"
        lines[2] = t_key + ": " + numpy2str(translation_noise) + "\n"
        with open(output_path, "w") as stream:
            stream.writelines(lines)

    # calib format (Depth complete)
    # 2011_0928
    # |-format calib_cam_to_cam.txt
    # |-calib_velo_to_cam.txt
    @staticmethod
    def _calibration_key(line):
        if "Tr_velo_to_cam: " in line:
            return "Tr_velo_to_cam: "
        if "Tr_velo_cam " in line:
            return "Tr_velo_cam "
        return None


def _parse_calibration_value(value):
    try:
        return np.array(list(map(float, value.split(" "))))
    except ValueError:
        return value


def read_calib_file(path):
    """Read KITTI calibration file
    (from https://github.com/hunse/kitti)
    """
    float_chars = set("0123456789.e+- ")
    data = {}
    with open(path, "r") as f:
        for line in f.readlines():
            key, value = line.split(":", 1)
            value = value.strip()
            data[key] = value
            if float_chars.issuperset(value):
                data[key] = _parse_calibration_value(value)

    return data


def numpy2str(arr):
    arr = np.array2string(arr, separator=" ")[1:-1].strip()
    arr = arr.replace(",", "")
    arr = arr.replace("\n", "")
    while True:
        arr = arr.replace("  ", " ")
        if "  " not in arr:
            break
    return arr
