# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Common operations for EasyDict-style configuration trees."""

from ast import literal_eval

import numpy as np


def _log_config_error(logger, key):
    if logger is not None:
        logger.info("Error under config key: %s", key)


def merge_config(source, target, mapping_type, logger=None):
    """Recursively merge a typed configuration mapping."""
    if not isinstance(source, mapping_type):
        return
    for key, value in source.items():
        if key not in target:
            raise KeyError("{} is not a valid config key".format(key))
        existing = target.get(key)
        if not isinstance(value, type(existing)):
            if isinstance(existing, np.ndarray):
                value = np.array(value, dtype=existing.dtype)
            else:
                raise ValueError(
                    "Type mismatch ({} vs. {}) for config key: {}".format(
                        type(existing), type(value), key
                    )
                )
        if isinstance(value, mapping_type):
            try:
                merge_config(value, existing, mapping_type, logger)
            except (KeyError, TypeError, ValueError):
                _log_config_error(logger, key)
                raise
        else:
            target[key] = value


def apply_config_list(target, config_values):
    """Apply dotted key/value pairs to a configuration mapping."""
    if len(config_values) % 2:
        raise AssertionError
    for dotted_key, raw_value in zip(config_values[0::2], config_values[1::2]):
        keys = dotted_key.split(".")
        destination = target
        for key in keys[:-1]:
            if key not in destination:
                raise AssertionError
            destination = destination.get(key)
        final_key = keys[-1]
        if final_key not in destination:
            raise AssertionError
        try:
            value = literal_eval(raw_value)
        except (SyntaxError, ValueError):
            value = raw_value
        if not isinstance(value, type(destination.get(final_key))):
            raise AssertionError(
                "type {} does not match original type {}".format(
                    type(value), type(destination.get(final_key))
                )
            )
        destination[final_key] = value
