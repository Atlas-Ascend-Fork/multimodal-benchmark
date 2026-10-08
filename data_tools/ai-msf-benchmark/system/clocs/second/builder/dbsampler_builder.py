# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import pickle

from second.builder import preprocess_builder
from second.core.preprocess import DataBasePreprocessor
from second.core.sample_ops import DataBaseSamplerV2


def build(sampler_config):
    cfg = sampler_config
    groups = list(cfg.sample_groups)
    prepors = [
        preprocess_builder.build_db_preprocess(c) for c in cfg.database_prep_steps
    ]
    db_prepor = DataBasePreprocessor(prepors)
    rate = cfg.rate
    grot_range = cfg.global_random_rotation_range_per_object
    groups = [dict(g.name_to_max_num) for g in groups]
    info_path = cfg.database_info_path
    with open(info_path, "rb") as f:
        db_infos = pickle.load(f)
    grot_range = list(grot_range)
    if len(grot_range) == 0:
        grot_range = None
    sampler = DataBaseSamplerV2(db_infos, groups, db_prepor, rate, grot_range)
    return sampler
