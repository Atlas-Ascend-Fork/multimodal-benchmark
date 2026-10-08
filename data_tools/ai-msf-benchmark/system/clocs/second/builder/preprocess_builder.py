# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import second.core.preprocess as prep


def build_db_preprocess(db_prep_config):
    prep_type = db_prep_config.WhichOneof("database_preprocessing_step")

    if prep_type == "filter_by_difficulty":
        cfg = db_prep_config.filter_by_difficulty
        return prep.DBFilterByDifficulty(list(cfg.removed_difficulties))
    elif prep_type == "filter_by_min_num_points":
        cfg = db_prep_config.filter_by_min_num_points
        return prep.DBFilterByMinNumPoint(dict(cfg.min_num_point_pairs))
    else:
        raise ValueError("unknown database prep type")
