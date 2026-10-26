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
source_root="${project_root}/../../cann-bench-optimize/src/ascend_arena_e2e/dependency"
selected_op=""
import_all=false
refresh=false

usage() {
  cat <<'EOF'
Usage:
  scripts/import_arena_ops.sh --op <name> [--source-root <dependency-dir>] [--refresh]
  scripts/import_arena_ops.sh --all [--source-root <dependency-dir>] [--refresh]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --op) selected_op="$2"; shift 2 ;;
    --all) import_all=true; shift ;;
    --source-root) source_root="$2"; shift 2 ;;
    --refresh) refresh=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "$import_all" != true && -z "$selected_op" ]] ||
   [[ "$import_all" == true && -n "$selected_op" ]]; then
  usage >&2
  exit 2
fi

source_root="$(realpath "$source_root")"
found=false
while IFS=$'\t' read -r op repo source_path aclnn_type dependency_paths dependency_api_sources \
  expected_symbols kernel_options; do
  [[ -z "$op" || "$op" == \#* ]] && continue
  if [[ "$import_all" != true && "$op" != "$selected_op" ]]; then
    continue
  fi
  found=true
  destination="${project_root}/${repo}/${op}"
  args=(
    --repo "$repo"
    --source "${source_root}/ops-${repo}"
    --op "$source_path"
    --aclnn-type "$aclnn_type"
    --dependency-paths "$dependency_paths"
    --dependency-api-sources "$dependency_api_sources"
    --kernel-options "$kernel_options"
  )
  if [[ "$repo" == "nn" || "$repo" == "transformer" ]]; then
    args+=(--with-common)
  fi
  if [[ "$refresh" == true ]]; then
    args+=(--refresh)
  elif [[ -e "$destination" ]]; then
    printf '[SKIP] already imported: %s\n' "$op"
    continue
  fi
  "${project_root}/scripts/import_operator.sh" "${args[@]}"
done < "$manifest"

if [[ "$found" != true ]]; then
  echo "Operator not found in manifest: $selected_op" >&2
  exit 1
fi
