# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""'
@Project: fusion
@Description: Siganl Delay (Temporal misalignment)
@Time:2022/9/6 15:01
@Author:NianGao
"""
import logging
import os

from natsort import natsorted
from tqdm import tqdm

from corruption.operator.delay_operator import DelayOperator

LOGGER = logging.getLogger(__name__)


class DelaySimulation(object):

    # type c or l
    # delay a source of signal (camera or Lidar)
    @staticmethod
    def simulation(inputbasedir, outputbasedir, dop, sev, copy_mode=0):  # type,
        if not (sev in range(1, 6)):
            raise AssertionError
        LOGGER.info("%s %s", "input_dir", inputbasedir)
        LOGGER.info("%s %s", "output_dir", outputbasedir)
        LOGGER.info("%s %s", "corruption", dop)
        LOGGER.info("%s %s", "severity", sev)

        inputbasedir_fns = natsorted(os.listdir(inputbasedir))
        op = DelayOperator.operator_map()[dop]
        for subinputdir in tqdm(inputbasedir_fns):
            inputdir = os.path.join(inputbasedir, subinputdir)
            outputdir = os.path.join(outputbasedir, subinputdir)
            os.makedirs(outputdir, exist_ok=True)
            op(inputdir, outputdir, sev, copy_mode)
