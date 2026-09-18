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
@Time:2022/9/6 16:00
@Author:NianGao

"""

from typing import NamedTuple
import os

import fire


from shared.parameters import ParameterSchema as _ParameterSchema, bind_parameters as _bind_parameters


class SimulateCalibParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    input_dir: object
    output_dir: object
    corruption: object
    sev: object
    fm: object
    for_depth: object


class SimulateWeatherParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    input_dir_c: object
    output_dir_c: object
    input_dir_l: object
    output_dir_l: object
    calibdir: object
    depthdir: object
    corruption: object
    sev: object
    for_depth: object
    for_tracking: object
    x2object_dir: object


def simulate_image(input_dir, output_dir, corruption, sev, conflict_mode=0):
    from corruption.simulation.image_simulation import ImageSimulation

    os.makedirs(output_dir, exist_ok=True)
    ImageSimulation.simulation(
        input_dir, output_dir, corruption, sev, conflict_mode=conflict_mode
    )


def simulate_delay(input_dir, output_dir, corruption, sev, copy_mode=0):
    from corruption.simulation.delay_simulation import DelaySimulation

    DelaySimulation.simulation(
        input_dir, output_dir, corruption, sev, copy_mode=copy_mode
    )


def simulate_lidar(input_dir, output_dir, corruption, sev, conflict_mode=0):
    from corruption.simulation.lidar_simulation import LidarSimulation

    os.makedirs(output_dir, exist_ok=True)
    LidarSimulation.simulation(
        input_dir, output_dir, corruption, sev, conflict_mode=conflict_mode
    )


_SIMULATE_CALIB_PARAMETERS_SCHEMA = _ParameterSchema(
    SimulateCalibParameters,
    ("input_dir", "output_dir", "corruption", "sev", "fm", "for_depth"),
    {"fm": "%06d.txt", "for_depth": False},
    frozenset(()),
    6,
    None,
    None,
)
_SIMULATE_WEATHER_PARAMETERS_SCHEMA = _ParameterSchema(
    SimulateWeatherParameters,
    (
        "input_dir_c",
        "output_dir_c",
        "input_dir_l",
        "output_dir_l",
        "calibdir",
        "depthdir",
        "corruption",
        "sev",
        "for_depth",
        "for_tracking",
        "x2object_dir",
    ),
    {"for_depth": False, "for_tracking": False, "x2object_dir": None},
    frozenset(()),
    11,
    None,
    None,
)


def simulate_calib(*call_args, **call_kwargs):
    bound_parameters = _bind_parameters(
        _SIMULATE_CALIB_PARAMETERS_SCHEMA, call_args, call_kwargs
    )
    input_dir, output_dir, corruption, sev, fm, for_depth = bound_parameters
    from corruption.simulation.calib_simulation import CalibSimulation

    os.makedirs(output_dir, exist_ok=True)
    if for_depth:
        CalibSimulation.simulation4depth(input_dir, output_dir, corruption, sev)
    else:
        CalibSimulation.simulation(input_dir, output_dir, corruption, sev, fm=fm)


def simulate_weather(*call_args, **call_kwargs):
    params = _bind_parameters(
        _SIMULATE_WEATHER_PARAMETERS_SCHEMA, call_args, call_kwargs
    )
    from corruption.simulation.weather_simulation import WeatherSimulation

    os.makedirs(params.output_dir_c, exist_ok=True)
    os.makedirs(params.output_dir_l, exist_ok=True)
    input_camera = os.path.abspath(params.input_dir_c)
    input_lidar = os.path.abspath(params.input_dir_l)
    output_camera = os.path.abspath(params.output_dir_c)
    output_lidar = os.path.abspath(params.output_dir_l)
    if params.for_tracking:
        WeatherSimulation.simulation4tracking(
            params.x2object_dir,
            output_camera,
            input_lidar,
            output_lidar,
            params.depthdir,
            params.corruption,
            params.sev,
            conflict_mode=0,
        )
    elif params.for_depth:
        WeatherSimulation.simulation4depth(
            params.x2object_dir,
            output_camera,
            output_lidar,
            params.depthdir,
            params.corruption,
            params.sev,
            conflict_mode=0,
        )
    else:
        WeatherSimulation.simulation(
            input_camera,
            output_camera,
            input_lidar,
            output_lidar,
            params.calibdir,
            params.depthdir,
            params.corruption,
            params.sev,
        )


if __name__ == "__main__":
    fire.Fire()
