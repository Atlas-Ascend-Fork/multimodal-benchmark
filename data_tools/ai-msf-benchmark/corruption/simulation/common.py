# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Directory traversal shared by corruption simulators."""

import logging
import os

from natsort import natsorted
from tqdm import tqdm


class SimulationBase:
    logger = logging.getLogger(__name__)

    @classmethod
    def simulation(cls, input_dir, output_dir, dop, sev, conflict_mode=0):
        if sev not in range(1, 6):
            raise AssertionError
        cls.logger.info("%s %s", "input_dir", input_dir)
        cls.logger.info("%s %s", "output_dir", output_dir)
        cls.logger.info("%s %s", "corruption", dop)
        cls.logger.info("%s %s", "severity", sev)
        for filename in tqdm(natsorted(os.listdir(input_dir))):
            input_filename = os.path.join(input_dir, filename)
            output_filename = os.path.join(output_dir, filename)
            if conflict_mode == 1 and os.path.exists(output_filename):
                continue
            cls.simulation_one(input_filename, output_filename, dop, sev)
