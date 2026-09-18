# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from importlib import import_module

from google.protobuf import descriptor_pool
from google.protobuf.internal import builder


def build_module(serialized_descriptor, module_name, namespace, dependencies=()):
    """Build a generated Protobuf module from its serialized descriptor."""
    for dependency in dependencies:
        import_module(dependency)
    descriptor = descriptor_pool.Default().AddSerializedFile(serialized_descriptor)
    builder.BuildMessageAndEnumDescriptors(descriptor, namespace)
    builder.BuildTopDescriptorsAndMessages(descriptor, module_name, namespace)
    return descriptor
