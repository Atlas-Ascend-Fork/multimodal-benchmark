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
@Time:2022/9/6 14:05
@Author:NianGao

"""

from enum import Enum


class Modality(Enum):
    C = "C"
    L = "L"
    CL = "CL"
    A_SM = "SM"
    A_TM = "TM"


corrupions_abbr_map = {
    # weather cottuption
    "RN": "rain",  # Rain
    "FG": "fog",  # Fog
    "BR": "brightness",  # Brightness
    "DK": "darkness",  # Darknes
    # sensor cottuption
    "DT": "distortion",  # Distortion
    "MB": "motion_blur",  # Motion blur
    "DB": "defocus_blur",  # Defocus Blur
    # noise cottuption
    "GN_C": "gaussian_noise",  # Image Gaussion Nois
    "GN_L": "gaussian_noise",  # Point Cloud Gaussion Noise
    "IN_C": "impulse_noise",  # Image Impulse Noise
    "IN_L": "impulse_noise",  # Point Cloud Impulse Noise
    # sensor misalignment
    "SM_X": "rotation_x",  # Spatial misalignment i.e. calib error
    "SM_Y": "rotation_y",  # Spatial misalignment i.e. calib error
    "SM_Z": "rotation_z",  # Spatial misalignment i.e. calib error
    "SD": "delay",  # Temporal misalignment i.e. signal delay
    # signal loss
    "LP_C": "loss_partial",  # signal loss partial
    "LP_L": "loss_partial",  # signal loss partial
    "LC_C": "loss_complete",  # signal loss complete
    "LC_L": "loss_complete",  # signal loss complete
}

modality_map = {
    Modality.C: ["BR", "DK", "DT", "MB", "DB", "GN_C", "IN_C", "LP_C", "LC_C"],
    Modality.L: ["GN_L", "IN_L", "LP_L", "LC_L"],
    Modality.CL: ["RN", "FG"],
    Modality.A_SM: ["SM_X", "SM_Y", "SM_Z"],
    Modality.A_TM: ["SD"],
}

#
#
#

#
#
#
