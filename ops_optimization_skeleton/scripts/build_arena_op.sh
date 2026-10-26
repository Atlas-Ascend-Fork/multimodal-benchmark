#!/bin/bash
# Copyright (c) 2026 multimodal-bench contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
manifest="${project_root}/config/ascend_arena_ops.tsv"
source_root=""
soc="ascend910b"
jobs=8
vendor_name="benchmark"
selected_op=""
build_all=false
refresh=false
skip_import=true

usage() {
  cat <<'EOF'
Usage:
  scripts/build_arena_op.sh --op <name> [options]
  scripts/build_arena_op.sh --all [options]

Options:
  --soc <soc>               Target SoC (default: ascend910b)
  --source-root <dir>       ascend_arena_e2e/dependency directory
  --vendor-name <name>      OPP vendor prefix (default: benchmark)
  --refresh                 Maintenance only: refresh sources from --source-root first
  --skip-import             Require bundled sources (default; kept for compatibility)
  -j <jobs>, -j<jobs>       Parallel build jobs (default: 8)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --op) selected_op="$2"; shift 2 ;;
    --all) build_all=true; shift ;;
    --soc) soc="$2"; shift 2 ;;
    --source-root) source_root="$2"; shift 2 ;;
    --vendor-name) vendor_name="$2"; shift 2 ;;
    --refresh) refresh=true; skip_import=false; shift ;;
    --skip-import) skip_import=true; shift ;;
    -j) jobs="$2"; shift 2 ;;
    -j*) jobs="${1#-j}"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "$build_all" != true && -z "$selected_op" ]] ||
   [[ "$build_all" == true && -n "$selected_op" ]]; then
  usage >&2
  exit 2
fi
[[ "$jobs" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid job count: $jobs" >&2; exit 2; }
[[ "$vendor_name" =~ ^[A-Za-z0-9_]+$ ]] || { echo "Invalid vendor name: $vendor_name" >&2; exit 2; }
if [[ "$refresh" == true && -z "$source_root" ]]; then
  echo "--refresh requires an explicit --source-root" >&2
  exit 2
fi

lookup_operator() {
  local wanted="$1"
  local row_op
  while IFS=$'\t' read -r row_op row_repo row_source_path row_aclnn_type \
    row_dependency_paths row_dependency_api_sources row_expected_symbols row_kernel_options; do
    [[ -z "$row_op" || "$row_op" == \#* ]] && continue
    if [[ "$row_op" == "$wanted" ]]; then
      op="$row_op"
      repo="$row_repo"
      source_path="$row_source_path"
      expected_symbols="$row_expected_symbols"
      return 0
    fi
  done < "$manifest"
  return 1
}

validate_symbols() {
  local vendor_root="$1"
  local expected_csv="$2"
  [[ "$expected_csv" == "-" ]] && return 0
  if ! command -v nm >/dev/null 2>&1; then
    echo "[WARN] nm not found; ACLNN symbol validation skipped"
    return 0
  fi

  local libraries=("${vendor_root}/op_api/lib/libcust_opapi.so")
  if [[ -n "${ASCEND_HOME_PATH:-}" && -f "${ASCEND_HOME_PATH}/lib64/libopapi.so" ]]; then
    libraries+=("${ASCEND_HOME_PATH}/lib64/libopapi.so")
  fi
  local symbol_file
  symbol_file="$(mktemp "${project_root}/build_out/.arena-symbols.XXXXXX")"
  local library
  for library in "${libraries[@]}"; do
    nm -D --defined-only "$library" 2>/dev/null | awk '{print $NF}' >> "$symbol_file"
  done

  local expected symbol
  IFS=',' read -r -a expected_list <<< "$expected_csv"
  for expected in "${expected_list[@]}"; do
    for symbol in "$expected" "${expected}GetWorkspaceSize"; do
      if ! rg -x "$symbol" "$symbol_file" >/dev/null; then
        rm -f -- "$symbol_file"
        echo "[ERROR] missing ACLNN symbol: $symbol" >&2
        return 1
      fi
    done
  done
  rm -f -- "$symbol_file"
}

build_one() {
  local wanted="$1"
  if ! lookup_operator "$wanted"; then
    echo "Operator not found in manifest: $wanted" >&2
    return 1
  fi

  if [[ "$skip_import" != true ]]; then
    import_args=(--op "$op" --source-root "$source_root")
    if [[ "$refresh" == true ]]; then
      import_args+=(--refresh)
    fi
    "${project_root}/scripts/import_arena_ops.sh" "${import_args[@]}"
  elif [[ ! -d "${project_root}/${repo}/${op}" ]]; then
    echo "Imported source not found: ${project_root}/${repo}/${op}" >&2
    return 1
  fi

  printf '\n[ARENA] building %s from ops-%s (%s)\n' "$op" "$repo" "$source_path"
  (
    cd "$project_root"
    bash build.sh --pkg --no_force \
      "--soc=${soc}" \
      "--ops=${op}" \
      "--vendor_name=${vendor_name}" \
      "--vendor_suffix=${repo}" \
      -j"${jobs}"
  )

  mapfile -t run_packages < <(find "${project_root}/build_out" -maxdepth 1 -type f -name '*.run' | sort)
  if [[ ${#run_packages[@]} -ne 1 ]]; then
    echo "[ERROR] expected one .run package, found ${#run_packages[@]}" >&2
    return 1
  fi

  local extract_dir artifact_root vendor_root
  extract_dir="$(mktemp -d "${project_root}/build_out/.arena-extract.XXXXXX")"
  if ! bash "${run_packages[0]}" --noexec --extract="$extract_dir"; then
    rm -rf -- "$extract_dir"
    return 1
  fi
  artifact_root="${project_root}/build_out/arena/${op}"
  rm -rf -- "$artifact_root"
  mkdir -p "$artifact_root"
  cp -a "$extract_dir/packages" "$artifact_root/"
  cp -a "${run_packages[0]}" "$artifact_root/"
  if [[ -f "${run_packages[0]}.json" ]]; then
    cp -a "${run_packages[0]}.json" "$artifact_root/"
  fi
  rm -rf -- "$extract_dir"

  vendor_root="${artifact_root}/packages/vendors/${vendor_name}_${repo}"
  [[ -f "${vendor_root}/op_api/lib/libcust_opapi.so" ]] || {
    echo "[ERROR] libcust_opapi.so not found under $vendor_root" >&2
    return 1
  }
  [[ -d "${vendor_root}/op_impl" ]] || {
    echo "[ERROR] op_impl not found under $vendor_root" >&2
    return 1
  }
  find "${vendor_root}/op_impl" -type f -name '*.o' -path "*${op}*" -print -quit | rg -q . || {
    echo "[ERROR] kernel object for $op not found" >&2
    return 1
  }
  find "${vendor_root}/op_impl" -type f -name '*.json' -path "*${op}*" -print -quit | rg -q . || {
    echo "[ERROR] kernel metadata for $op not found" >&2
    return 1
  }
  validate_symbols "$vendor_root" "$expected_symbols"

  printf '[ARENA][OUTPUT] %s\n' "$vendor_root"
}

if [[ "$build_all" == true ]]; then
  while IFS=$'\t' read -r op_name _; do
    [[ -z "$op_name" || "$op_name" == \#* ]] && continue
    build_one "$op_name"
  done < "$manifest"
else
  build_one "$selected_op"
fi
