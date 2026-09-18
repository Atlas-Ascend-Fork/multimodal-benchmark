# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Helpers for modules that reuse a canonical implementation."""

from importlib import import_module
from importlib.util import module_from_spec, spec_from_file_location
import sys


def import_from_root(module_name, root):
    """Import *module_name* after making its project root importable."""
    root = str(root)
    if root not in sys.path:
        sys.path.insert(0, root)
    return import_module(module_name)


def load_from_path(module_name, path):
    """Load a module from an explicit path without requiring it to be a package."""
    spec = spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"unable to load compatibility module from {path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
