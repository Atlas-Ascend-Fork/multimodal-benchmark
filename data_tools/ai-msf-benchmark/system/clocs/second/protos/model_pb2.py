# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Generated Protobuf bindings. Do not edit manually."""

from second.utils.protobuf_builder import build_module


_SERIALIZED_DESCRIPTOR = bytes.fromhex(
    "0a197365636f6e642f70726f746f732f6d6f64656c2e70726f746f120d7365636f6e642e70726f746f731a1a7365636f"
    "6e642f70726f746f732f7365636f6e642e70726f746f22440a0e446574656374696f6e4d6f64656c12290a067365636f"
    "6e6418012001280b32172e7365636f6e642e70726f746f732e566f78656c4e6574480042070a056d6f64656c62067072"
    "6f746f33"
)

_DEPENDENCIES = (
    "second.protos.second_pb2",
)

DESCRIPTOR = build_module(
    _SERIALIZED_DESCRIPTOR, __name__, globals(), _DEPENDENCIES
)
