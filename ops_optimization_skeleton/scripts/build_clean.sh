#!/bin/bash
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ============================================================================
#
# Build clean functions.

clean_build() {
  # Full reset used by --make_clean: remove both generated roots.
  rm -rf -- "${BUILD_PATH}" "${BUILD_OUT_PATH}"
}

clean_build_out() {
  # Keep the stable Arena exports while replacing CPack's transient output.
  mkdir -p "${BUILD_OUT_PATH}"
  find "${BUILD_OUT_PATH}" -mindepth 1 -maxdepth 1 ! -name arena \
    -exec rm -rf -- {} +
}

clean_third_party() {
  # The minimal skeleton never downloads or owns third-party sources.
  return 0
}

clean_build_binary() {
  # Rebuild generated metadata and device binaries while retaining the host
  # build directory itself; cmake_init removes the cache immediately afterward.
  rm -rf -- "${BUILD_PATH}/tbe" "${BUILD_PATH}/autogen" \
    "${BUILD_PATH}/kernel_autogen" "${BUILD_PATH}/binary" \
    "${BUILD_PATH}/ops_skeleton_kernels"
}
