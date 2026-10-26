#!/bin/bash
# Copyright (c) 2026 multimodal-bench contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
set -e

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export BASE_PATH="$ROOT_DIR"
export BUILD_PATH="$ROOT_DIR/build"
export BUILD_OUT_PATH="$ROOT_DIR/build_out"
for build_script in \
  build.conf.sh \
  build_clean.sh \
  build_options.sh \
  build_cmake.sh \
  build_lib.sh; do
  source "$ROOT_DIR/scripts/$build_script"
done

build_phase "参数检查"
# Validate the requested operator, SoC and kernel-only options before touching
# the build tree. Invalid input stops here with an actionable error.
checkopts "$@"

build_phase "组装 CMake 参数"
# Convert shell options into the cache variables consumed by CMake.
assemble_cmake_args
printf '[CONFIG] CMAKE_ARGS: %s\n' "${CMAKE_ARGS}"

build_phase "清理旧的中间产物"
# Keep reusable CMake object files, but regenerate operator-specific metadata
# and device binaries.
clean_build_binary

build_phase "CMake 配置"
# Configure dependency discovery, source registration and all custom targets.
cmake_init

build_phase "编译 Host/Tiling 与 AscendC binary"
# Execute the target-level pipeline; build_lib.sh prints its inner phases.
build_binary

if [[ "$ENABLE_PACKAGE" == TRUE ]]; then
  # Package the artifacts produced above into a self-extracting installer.
  build_package
fi

build_phase "构建完成"
build_output "${BUILD_PATH}/libcust_opmaster_rt2.0.so"
for unit in ${COMPUTE_UNIT_SHORT//;/ }; do
  build_output "${BUILD_PATH}/ops_skeleton_kernels/binary/${unit}"
  build_output "${BUILD_PATH}/ops_skeleton_kernels/binary/config/${unit}/binary_info_config.json"
done
if [[ "$ENABLE_PACKAGE" == TRUE ]]; then
  for run_file in "${BUILD_OUT_PATH}"/*.run; do
    build_output "$run_file"
  done
else
  printf '[PACKAGE] --opkernel 不执行打包；需要 .run 时请使用 --pkg。\n'
fi
