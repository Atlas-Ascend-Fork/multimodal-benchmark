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
    "0a1c7365636f6e642f70726f746f732f706970656c696e652e70726f746f120d7365636f6e642e70726f746f731a2073"
    "65636f6e642f70726f746f732f696e7075745f7265616465722e70726f746f1a197365636f6e642f70726f746f732f6d"
    "6f64656c2e70726f746f1a197365636f6e642f70726f746f732f747261696e2e70726f746f22e8010a17547261696e45"
    "76616c506970656c696e65436f6e666967122c0a056d6f64656c18012001280b321d2e7365636f6e642e70726f746f73"
    "2e446574656374696f6e4d6f64656c12360a12747261696e5f696e7075745f72656164657218022001280b321a2e7365"
    "636f6e642e70726f746f732e496e70757452656164657212300a0c747261696e5f636f6e66696718032001280b321a2e"
    "7365636f6e642e70726f746f732e547261696e436f6e66696712350a116576616c5f696e7075745f7265616465721804"
    "2001280b321a2e7365636f6e642e70726f746f732e496e707574526561646572620670726f746f33"
)

_DEPENDENCIES = (
    "second.protos.input_reader_pb2",
    "second.protos.model_pb2",
    "second.protos.train_pb2",
)

DESCRIPTOR = build_module(
    _SERIALIZED_DESCRIPTOR, __name__, globals(), _DEPENDENCIES
)
