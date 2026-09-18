# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""'
@Project: AI-MSF-benchmark
@Description: Please add Description
@Time:2022/9/6 16:10
@Author:NianGao

"""

from typing import NamedTuple
import os

from natsort import natsorted
from tqdm import tqdm

from corruption.operator.weather_operator import WeatherOperator
from utils.utils import symlink


from shared.parameters import ParameterSchema as _ParameterSchema, bind_parameters as _bind_parameters


class WeathersimulationSimulationParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    inputdir_c: object
    outputdir_c: object
    inputdir_l: object
    outputdir_l: object
    calibdir: object
    depthdir: object
    dop: object
    sev: object
    conflict_mode: object


class WeathersimulationSimulation4trackingParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    tracking2object_dir: object
    outputdir_c: object
    inputbasedir_l: object
    outputbasedir_l: object
    depthdir: object
    dop: object
    sev: object
    conflict_mode: object


class WeathersimulationSimulation4depthParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    depth2object_dir: object
    output_dir_c: object
    outputdir_l: object
    depthdir: object
    dop: object
    sev: object
    conflict_mode: object


_WEATHERSIMULATION_SIMULATION_PARAMETERS_SCHEMA = _ParameterSchema(
    WeathersimulationSimulationParameters,
    (
        "inputdir_c",
        "outputdir_c",
        "inputdir_l",
        "outputdir_l",
        "calibdir",
        "depthdir",
        "dop",
        "sev",
        "conflict_mode",
    ),
    {"conflict_mode": 0},
    frozenset(()),
    9,
    None,
    None,
)
_WEATHERSIMULATION_SIMULATION4TRACKING_PARAMETERS_SCHEMA = _ParameterSchema(
    WeathersimulationSimulation4trackingParameters,
    (
        "tracking2object_dir",
        "outputdir_c",
        "inputbasedir_l",
        "outputbasedir_l",
        "depthdir",
        "dop",
        "sev",
        "conflict_mode",
    ),
    {"conflict_mode": 0},
    frozenset(()),
    8,
    None,
    None,
)
_WEATHERSIMULATION_SIMULATION4DEPTH_PARAMETERS_SCHEMA = _ParameterSchema(
    WeathersimulationSimulation4depthParameters,
    (
        "depth2object_dir",
        "output_dir_c",
        "outputdir_l",
        "depthdir",
        "dop",
        "sev",
        "conflict_mode",
    ),
    {"conflict_mode": 0},
    frozenset(()),
    7,
    None,
    None,
)


class WeatherSimulation(object):

    @staticmethod
    def image_format(inputdir_c, calibdir, depthdir):
        dir_c = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "3rd_parts",
            "camera",
            "data",
            "source",
            "kitti",
            "data_object",
            "training",
        )
        os.makedirs(dir_c, exist_ok=True)
        symlink(inputdir_c, os.path.join(dir_c, "image_2"))
        symlink(calibdir, os.path.join(dir_c, "calib"))
        symlink(depthdir, os.path.join(dir_c, "image_2", "depth"))

    @staticmethod
    def simulation(*call_args, **call_kwargs):
        bound_parameters = _bind_parameters(
            _WEATHERSIMULATION_SIMULATION_PARAMETERS_SCHEMA, call_args, call_kwargs
        )
        (
            inputdir_c,
            outputdir_c,
            inputdir_l,
            outputdir_l,
            calibdir,
            depthdir,
            dop,
            sev,
            conflict_mode,
        ) = bound_parameters
        if not (sev in range(1, 6)):
            raise AssertionError
        WeatherSimulation.image_format(inputdir_c, calibdir, depthdir)
        op = WeatherOperator.operator_map()[dop]
        op(
            inputdir_c,
            outputdir_c,
            inputdir_l,
            outputdir_l,
            sev,
            conflict_mode=conflict_mode,
        )

    @staticmethod
    def simulation4tracking(*call_args, **call_kwargs):
        bound_parameters = _bind_parameters(
            _WEATHERSIMULATION_SIMULATION4TRACKING_PARAMETERS_SCHEMA,
            call_args,
            call_kwargs,
        )
        (
            tracking2object_dir,
            outputdir_c,
            inputbasedir_l,
            outputbasedir_l,
            depthdir,
            dop,
            sev,
            conflict_mode,
        ) = bound_parameters
        inputdir_c = os.path.join(tracking2object_dir, "image_2")
        calibdir = os.path.join(tracking2object_dir, "calib")
        WeatherSimulation.image_format(inputdir_c, calibdir, depthdir)
        l_inputbasedir_fns = natsorted(os.listdir(inputbasedir_l))
        op = WeatherOperator.operator_map()[dop]
        for l_subinputdir in tqdm(l_inputbasedir_fns):
            l_inputdir = os.path.join(inputbasedir_l, l_subinputdir)
            l_outputdir = os.path.join(outputbasedir_l, l_subinputdir)
            op(None, None, l_inputdir, l_outputdir, sev, tp="l")  #
        # covert tracking 2 object
        op(
            None, outputdir_c, None, None, sev, tp="c", conflict_mode=conflict_mode
        )  # iamge fix format

    @staticmethod
    def simulation4depth(*call_args, **call_kwargs):
        bound_parameters = _bind_parameters(
            _WEATHERSIMULATION_SIMULATION4DEPTH_PARAMETERS_SCHEMA,
            call_args,
            call_kwargs,
        )
        (
            depth2object_dir,
            output_dir_c,
            outputdir_l,
            depthdir,
            dop,
            sev,
            conflict_mode,
        ) = bound_parameters
        op = WeatherOperator.operator_map()[dop]
        inputdir_c = os.path.join(depth2object_dir, "image_2")
        inputdir_l = os.path.join(depth2object_dir, "velodyne")
        calibdir = os.path.join(depth2object_dir, "calib")
        WeatherSimulation.image_format(inputdir_c, calibdir, depthdir)
        op(
            inputdir_c,
            output_dir_c,
            inputdir_l,
            outputdir_l,
            sev,
            conflict_mode=conflict_mode,
        )
