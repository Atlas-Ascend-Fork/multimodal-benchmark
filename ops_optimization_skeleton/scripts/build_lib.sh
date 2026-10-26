#!/bin/bash
# Copyright (c) 2026 multimodal-bench contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# Kernel binary orchestration through CANN's target-level CMake API.

build_binary() {
  cd "${BUILD_PATH}"

  build_phase "通过 npu_op_kernel_library 编译 Kernel"
  # CANN owns source preparation, parallel compilation and runtime metadata
  # generation behind this single target.
  cmake --build . --target binary -- ${VERBOSE} -j "${THREAD_NUM}"

  build_output "${BUILD_PATH}/libcust_opmaster_rt2.0.so"
  for unit in ${COMPUTE_UNIT_SHORT//;/ }; do
    build_output "${BUILD_PATH}/ops_skeleton_kernels/binary/${unit}"
    build_output "${BUILD_PATH}/ops_skeleton_kernels/binary/config/${unit}/binary_info_config.json"
  done
}

build_package() {
  cd "${BUILD_PATH}"

  build_phase "生成 .run 安装包"
  # Remove stale installers so success can only be attributed to this build.
  clean_build_out
  # CPack stages every install() rule, then invokes the makeself bundled with
  # the already-installed CANN Toolkit. No network fetch is performed here.
  cmake --build . --target package -- ${VERBOSE} -j "${THREAD_NUM}"

  local run_files=("${BUILD_OUT_PATH}"/*.run)
  if [[ ! -e "${run_files[0]}" ]]; then
    echo "[ERROR] package target completed but no .run file was generated in ${BUILD_OUT_PATH}" >&2
    return 1
  fi

  local run_file
  for run_file in "${run_files[@]}"; do
    build_output "$run_file"
  done
}
