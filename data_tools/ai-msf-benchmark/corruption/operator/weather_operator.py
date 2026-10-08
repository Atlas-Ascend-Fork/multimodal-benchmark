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
@Time:2022/9/6 15:18
@Author:NianGao

"""
from typing import NamedTuple
import logging
import os


from shared.parameters import ParameterSchema as _ParameterSchema, bind_parameters as _bind_parameters


class WeatheroperatorRainParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    inputdir_c: object
    ouputdir_c: object
    inputdir_l: object
    outputdir_l: object
    severity: object
    tp: object
    conflict_mode: object


class WeatheroperatorFogParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    camerainput: object
    cameraoutput: object
    lidarinput: object
    lidaroutput: object
    severity: object
    tp: object
    conflict_mode: object


LOGGER = logging.getLogger(__name__)


def _weather_schema(parameter_type, names):
    return _ParameterSchema(
        parameter_type, names, {"tp": ("c", "l"), "conflict_mode": 0},
        frozenset(), 7, None, None,
    )


_WEATHEROPERATOR_RAIN_PARAMETERS_SCHEMA = _weather_schema(
    WeatheroperatorRainParameters,
    ("inputdir_c", "ouputdir_c", "inputdir_l", "outputdir_l", "severity", "tp", "conflict_mode"),
)
_WEATHEROPERATOR_FOG_PARAMETERS_SCHEMA = _weather_schema(
    WeatheroperatorFogParameters,
    ("camerainput", "cameraoutput", "lidarinput", "lidaroutput", "severity", "tp", "conflict_mode"),
)


class WeatherOperator(object):
    @staticmethod
    def operator_map():
        operator_map = {
            "rain": WeatherOperator.rain,
            "fog": WeatherOperator.fog,
        }
        return operator_map

    @staticmethod
    def rain(*call_args, **call_kwargs):
        bound_parameters = _bind_parameters(
            _WEATHEROPERATOR_RAIN_PARAMETERS_SCHEMA, call_args, call_kwargs
        )
        inputdir_c, ouputdir_c, inputdir_l, outputdir_l, severity, tp, conflict_mode = (
            bound_parameters
        )
        rainfall = [10, 25, 50, 75, 100][severity - 1]
        if "c" in tp:
            LOGGER.info("%s", "simulate rain for camera :{}".format(rainfall))
            os.makedirs(ouputdir_c, exist_ok=True)
            path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "3rd_parts",
                "camera",
            )
            cmd1 = "cd {}".format(path)
            if conflict_mode == 0:
                cmd2 = (
                    "python main.py --dataset kitti --output {} --intensity {}".format(
                        ouputdir_c, rainfall
                    )
                )
            else:
                cmd2 = "python main.py --dataset kitti --output {} --intensity {} --conflict_strategy skip".format(
                    ouputdir_c, rainfall
                )
            os.system("{} && {}".format(cmd1, cmd2))
        if "l" in tp:
            LOGGER.info("%s", "simulate rain for Lidar :{}".format(rainfall))
            # lidar
            os.makedirs(outputdir_l, exist_ok=True)
            path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "3rd_parts",
                "lidar",
            )
            cmd1 = "cd {}".format(path)
            cmd2 = "python main.py  rain --output {} --input {} --intensity {} ".format(
                outputdir_l, inputdir_l, rainfall
            )
            os.system("{} && {}".format(cmd1, cmd2))

    @staticmethod
    def fog(*call_args, **call_kwargs):
        bound_parameters = _bind_parameters(
            _WEATHEROPERATOR_FOG_PARAMETERS_SCHEMA, call_args, call_kwargs
        )
        (
            camerainput,
            cameraoutput,
            lidarinput,
            lidaroutput,
            severity,
            tp,
            conflict_mode,
        ) = bound_parameters
        if not (severity <= 3):
            raise AssertionError
        level = ["strong", "moderate", "low"][severity - 1]
        if "c" in tp:
            # camera
            LOGGER.info("%s", "simulate fog for camera")
            os.makedirs(cameraoutput, exist_ok=True)
            path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "3rd_parts",
                "camera",
            )
            cmd1 = "cd {}".format(path)
            cmd2 = "python fog.py  --output {} --level {} --conflict_mode {}".format(
                cameraoutput, level, conflict_mode
            )
            os.system("{} && {}".format(cmd1, cmd2))
        if "l" in tp:
            LOGGER.info("%s", "simulate fog for Lidar")
            # lidar
            os.makedirs(lidaroutput, exist_ok=True)
            path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "3rd_parts",
                "lidar",
            )
            cmd1 = "cd {}".format(path)
            cmd2 = "python main.py  fog --output {} --input {} --level {} --conflict_mode {}".format(
                lidaroutput, lidarinput, level, conflict_mode
            )
            os.system("{} && {}".format(cmd1, cmd2))
