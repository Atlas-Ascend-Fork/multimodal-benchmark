#!/bin/bash
# Copyright (c) 2026 multimodal-bench contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# Assemble the small set of CMake arguments used by the kernel skeleton.

assemble_cmake_args() {
  # CMake lists use semicolons; the CLI deliberately accepts the friendlier
  # comma-separated form for --ops.
  local ops="${COMPILED_OPS//,/;}"
  # These two switches enable the kernel metadata and binary target graph.
  CMAKE_ARGS="-DENABLE_BINARY=TRUE -DENABLE_CUSTOM=TRUE"
  # Select the operator(s), target SoC and host build settings.
  CMAKE_ARGS+=" -DASCEND_OP_NAME=${ops}"
  CMAKE_ARGS+=" -DASCEND_COMPUTE_UNIT=${COMPUTE_UNIT}"
  # Keep the package directory layout deterministic even for kernel-only builds.
  CMAKE_ARGS+=" -DVENDOR_NAME=${VENDOR_NAME}"
  CMAKE_ARGS+=" -DVENDOR_SUFFIX=${VENDOR_SUFFIX}"
  CMAKE_ARGS+=" -DENABLE_CCACHE=${ENABLE_CCACHE}"
  CMAKE_ARGS+=" -DCMAKE_BUILD_TYPE=${BUILD_TYPE}"
  # Pass this before find_package(ASC) runs; otherwise CANN captures CMake's
  # built-in /usr/local default instead of the repository-local package path.
  CMAKE_ARGS+=" -DCMAKE_INSTALL_PREFIX=${BUILD_OUT_PATH}"
  # Reuse the repository's third-party CANN dependency location.
  CMAKE_ARGS+=" -DCANN_3RD_LIB_PATH=${CANN_3RD_LIB_PATH}"
  # Register install/CPack rules only for the explicit packaging mode.
  if [[ "$ENABLE_PACKAGE" == TRUE ]]; then CMAKE_ARGS+=" -DENABLE_PACKAGE=TRUE"; fi
  # Optional flags are forwarded only when the user supplied them.
  if [[ -n "$BUILD_MODE" ]]; then CMAKE_ARGS+=" -DBUILD_MODE=${BUILD_MODE}"; fi
  if [[ -n "$BISHENG_FLAGS" ]]; then CMAKE_ARGS+=" -DBISHENG_FLAGS=${BISHENG_FLAGS}"; fi
  if [[ -n "$KERNEL_TEMPLATE_INPUT" ]]; then CMAKE_ARGS+=" -DKERNEL_TEMPLATE_INPUT=${KERNEL_TEMPLATE_INPUT}"; fi
  if [[ -n "$MODULE_EXT" ]]; then CMAKE_ARGS+=" -DMODULE_EXT=${MODULE_EXT}"; fi
  if [[ "$ENABLE_MSSANITIZER" == TRUE ]]; then CMAKE_ARGS+=" -DENABLE_MSSANITIZER=TRUE"; fi
  if [[ "$ENABLE_OOM" == TRUE ]]; then CMAKE_ARGS+=" -DENABLE_OOM=TRUE"; fi
  if [[ "$ENABLE_DUMP_CCE" == TRUE ]]; then CMAKE_ARGS+=" -DENABLE_DUMP_CCE=TRUE"; fi
  if [[ "$NO_FORCE" == TRUE ]]; then CMAKE_ARGS+=" -DNO_FORCE=TRUE"; fi
}

cmake_init() {
  # Create the two output roots before CMake starts generating files.
  mkdir -p "${BUILD_PATH}"
  mkdir -p "${BUILD_OUT_PATH}"
  # Remove the cache after the targeted intermediate cleanup so CMake re-evaluates
  # the current CLI cache values without deleting unrelated build directories.
  rm -f -- "${BUILD_PATH}/CMakeCache.txt"
  # Configure the project and materialize all custom targets used below.
  cmake ${CMAKE_ARGS} -S "${BASE_PATH}" -B "${BUILD_PATH}"
}
