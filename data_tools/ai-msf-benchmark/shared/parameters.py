# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import argparse
from collections import namedtuple
from functools import lru_cache
from typing import NamedTuple


class ParameterSchema(NamedTuple):
    """Description of a backward-compatible named parameter group."""

    parameter_type: type
    names: tuple
    defaults: dict
    positional_only: frozenset
    positional_limit: int
    vararg_name: object
    kwarg_name: object


def bind_parameters(schema, call_args, call_kwargs):
    """Bind a legacy positional/keyword call into its named parameter group."""
    if schema.vararg_name is None and len(call_args) > schema.positional_limit:
        raise TypeError("too many positional arguments")
    values = dict(schema.defaults)
    positional_names = schema.names[: schema.positional_limit]
    for name, value in zip(positional_names, call_args):
        values[name] = value
    extra_keywords = {}
    for name, value in call_kwargs.items():
        if name in schema.positional_only:
            raise TypeError(
                "positional-only argument passed as keyword: {}".format(name)
            )
        if name in schema.names:
            if name in positional_names[: len(call_args)]:
                raise TypeError("multiple values for argument: {}".format(name))
            values[name] = value
        elif schema.kwarg_name is not None:
            extra_keywords[name] = value
        else:
            raise TypeError("unexpected keyword argument: {}".format(name))
    if schema.vararg_name is not None:
        values[schema.vararg_name] = tuple(call_args[schema.positional_limit:])
    if schema.kwarg_name is not None:
        values[schema.kwarg_name] = extra_keywords
    missing = [name for name in schema.names if name not in values]
    if missing:
        raise TypeError("missing required arguments: {}".format(", ".join(missing)))
    return schema.parameter_type(*(values[name] for name in schema.names))


@lru_cache(maxsize=None)
def _named_parameter_type(names):
    return namedtuple("BoundParameters", names)


def bind_named(call_args, call_kwargs, names, defaults=None):
    """Bind a compact variadic signature while retaining named parameters."""
    names = tuple(names)
    schema = ParameterSchema(
        _named_parameter_type(names),
        names,
        defaults or {},
        frozenset(),
        len(names),
        None,
        None,
    )
    return bind_parameters(schema, call_args, call_kwargs)


def named_parameter_schema(names, defaults=None):
    """Build a reusable schema for callers that only need named binding."""
    names = tuple(names)
    parameter_type = _named_parameter_type(names)
    return ParameterSchema(
        parameter_type, names, defaults or {}, frozenset(), len(names), None, None
    )


def add_rcnn_output_arguments(parser):
    """Add command-line options shared by EPNet training and evaluation."""
    parser.add_argument(
        "--rcnn_eval_roi_dir", type=str, default=None,
        help="specify the saved rois for rcnn evaluation when using rcnn_offline mode",
    )
    parser.add_argument(
        "--rcnn_eval_feature_dir", type=str, default=None,
        help="specify the saved features for rcnn evaluation when using rcnn_offline mode",
    )
    parser.add_argument(
        "--set", dest="set_cfgs", default=None, nargs=argparse.REMAINDER,
        help="set extra config keys if needed",
    )
    parser.add_argument("--model_type", type=str, default="base", help="model type")


_ParameterSchema = ParameterSchema
_bind_parameters = bind_parameters
