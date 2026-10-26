#!/bin/bash
# Copyright (c) 2026 multimodal-bench contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# Minimal AscendC kernel build options.

usage() {
  cat <<'EOF'
Ascend Arena kernel skeleton build

Usage:
  bash build.sh --opkernel --ops=<op> --soc=<soc> [options]
  bash build.sh --pkg --ops=<op> --soc=<soc> [options]

Required:
  --opkernel                 Build the AscendC kernel binary
  --pkg                      Build the kernel binary and a .run package
  --ops=op1,op2              Operators to compile
  --soc=ascend910b           Target SoC

Kernel options:
  --ccache=on|off            Enable or disable ccache (default: on)
  --mssanitizer              Enable kernel sanitizer flags
  --oom                      Enable kernel OOM instrumentation
  --dump_cce                 Keep kernel precompile output
  --bisheng_flags=LIST       Extra Bisheng flags, comma-separated
  --kernel_template_input=ARGS
                             Kernel template arguments, semicolon-separated
  --module_extension=PATH    Additional operator definition directory
  --vendor_name=NAME         Package vendor prefix (default: custom)
  --vendor_suffix=FAMILY     Package family: math, nn, cv, or transformer
  --no_force                 Do not force an explicitly selected operator
  --build-type=Release|Debug C++ build type (default: Release)
  -j N, -jN                  Parallel jobs (default: 8)
  -O N, -ON                  Host optimization level 0-3
  -v                         Verbose build
  --make_clean               Remove build outputs
  -h, --help                 Show this help

--opkernel stops after host/kernel artifacts. --pkg additionally writes a
self-extracting .run installer to build_out/ without downloading dependencies.
EOF
}

require_value() {
  if [[ $# -lt 2 || -z "$2" || "$2" == -* ]]; then
    echo "[ERROR] $1 requires a value" >&2
    exit 2
  fi
}

is_supported_soc() {
  local soc
  for soc in "${SUPPORT_COMPUTE_UNIT_SHORT[@]}"; do
    [[ "$1" == "$soc" ]] && return 0
  done
  return 1
}

checkopts() {
  # Set deterministic defaults before parsing the command line.
  THREAD_NUM=8
  VERBOSE=""
  BUILD_MODE=""
  BUILD_TYPE="Release"
  COMPILED_OPS=""
  COMPUTE_UNIT="ascend910b"
  COMPUTE_UNIT_SHORT="ascend910b"
  BISHENG_FLAGS=""
  KERNEL_TEMPLATE_INPUT=""
  MODULE_EXT=""
  ENABLE_CCACHE=TRUE
  ENABLE_MSSANITIZER=FALSE
  ENABLE_OOM=FALSE
  ENABLE_DUMP_CCE=FALSE
  NO_FORCE=FALSE
  OP_KERNEL=FALSE
  ENABLE_PACKAGE=FALSE
  VENDOR_NAME="custom"
  VENDOR_SUFFIX="math"

  # Parse one option at a time so every unsupported legacy option fails loudly.
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -h|--help) usage; exit 0 ;;
      -v) VERBOSE="VERBOSE=1"; shift ;;
      -j)
        require_value "$1" "${2:-}"
        THREAD_NUM="$2"
        shift 2
        ;;
      -j*) THREAD_NUM="${1#-j}"; shift ;;
      -O)
        require_value "$1" "${2:-}"
        BUILD_MODE="-O$2"
        shift 2
        ;;
      -O[0-3]) BUILD_MODE="$1"; shift ;;
      --ops=*) COMPILED_OPS="${1#*=}"; shift ;;
      --soc=*) COMPUTE_UNIT="${1#*=}"; shift ;;
      --ccache=*)
        case "${1#*=}" in
          off|false|disable) ENABLE_CCACHE=FALSE ;;
          on|true|enable) ENABLE_CCACHE=TRUE ;;
          *) echo "[ERROR] --ccache expects on or off" >&2; exit 2 ;;
        esac
        shift
        ;;
      --bisheng_flags=*) BISHENG_FLAGS="${1#*=}"; shift ;;
      --kernel_template_input=*) KERNEL_TEMPLATE_INPUT="${1#*=}"; shift ;;
      --module_extension=*) MODULE_EXT="${1#*=}"; shift ;;
      --vendor_name=*) VENDOR_NAME="${1#*=}"; shift ;;
      --vendor_suffix=*) VENDOR_SUFFIX="${1#*=}"; shift ;;
      --build-type=*) BUILD_TYPE="${1#*=}"; shift ;;
      --mssanitizer) ENABLE_MSSANITIZER=TRUE; shift ;;
      --oom) ENABLE_OOM=TRUE; shift ;;
      --dump_cce) ENABLE_DUMP_CCE=TRUE; shift ;;
      --no_force) NO_FORCE=TRUE; shift ;;
      --opkernel) OP_KERNEL=TRUE; shift ;;
      --pkg) ENABLE_PACKAGE=TRUE; OP_KERNEL=TRUE; shift ;;
      --make_clean) clean_build; exit 0 ;;
      *)
        echo "[ERROR] Unknown option: $1" >&2
        usage >&2
        exit 2
        ;;
    esac
  done

  # Validate required mode/input before configuring CMake.
  [[ "$OP_KERNEL" == TRUE ]] || { echo "[ERROR] --opkernel or --pkg is required" >&2; exit 2; }
  [[ -n "$COMPILED_OPS" ]] || { echo "[ERROR] --opkernel/--pkg requires --ops" >&2; exit 2; }
  [[ "$VENDOR_NAME" =~ ^[A-Za-z0-9_]+$ ]] || {
    echo "[ERROR] --vendor_name may contain only letters, digits, and underscores" >&2
    exit 2
  }
  case "$VENDOR_SUFFIX" in
    math|nn|cv|transformer) ;;
    *) echo "[ERROR] --vendor_suffix must be math, nn, cv, or transformer" >&2; exit 2 ;;
  esac
  [[ "$THREAD_NUM" =~ ^[1-9][0-9]*$ ]] || { echo "[ERROR] parallel jobs must be positive" >&2; exit 2; }
  [[ -z "$BUILD_MODE" || "$BUILD_MODE" =~ ^-O[0-3]$ ]] || { echo "[ERROR] optimization level must be 0-3" >&2; exit 2; }
  [[ "$BUILD_TYPE" == Release || "$BUILD_TYPE" == Debug ]] || { echo "[ERROR] build type must be Release or Debug" >&2; exit 2; }
  [[ "$ENABLE_MSSANITIZER" != TRUE || "$ENABLE_OOM" != TRUE ]] || { echo "[ERROR] --mssanitizer and --oom are exclusive" >&2; exit 2; }

  # Normalize SoC spelling once so CMake and generated paths agree.
  COMPUTE_UNIT="$(tr '[:upper:]' '[:lower:]' <<< "$COMPUTE_UNIT")"
  is_supported_soc "$COMPUTE_UNIT" || {
    echo "[ERROR] unsupported SoC: $COMPUTE_UNIT" >&2
    echo "Supported: ${SUPPORT_COMPUTE_UNIT_SHORT[*]}" >&2
    exit 2
  }
  COMPUTE_UNIT_SHORT="$COMPUTE_UNIT"
  ENABLE_BINARY=TRUE
  ENABLE_CUSTOM=TRUE
}
