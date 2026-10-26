#!/bin/bash
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

SUPPORT_COMPUTE_UNIT_SHORT=(
  "ascend910b" "ascend910_93" "ascend950" "ascend910" "ascend310p"
)

ARCH_INFO=$(uname -m)
CANN_3RD_LIB_PATH="${BASE_PATH}/third_party"

# Print a stable marker before each build phase so a long CMake/OPC log can be
# correlated with the high-level pipeline in README.md.
build_phase() {
  printf '\n============================================================\n'
  printf '[BUILD] %s\n' "$1"
  printf '============================================================\n'
}

# Report an expected artifact without hiding a missing output behind a success
# message. A symlink is considered an output as well, including a broken one.
build_output() {
  local output_path="$1"
  if [[ -e "$output_path" || -L "$output_path" ]]; then
    printf '[OUTPUT] %s\n' "$output_path"
  else
    printf '[OUTPUT][MISSING] %s\n' "$output_path"
  fi
}
