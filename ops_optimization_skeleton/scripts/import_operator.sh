#!/bin/bash
# Copyright (c) 2026 multimodal-bench contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/import_operator.sh --repo <math|nn|cv|transformer> --source <repo> \
    --op <category/operator> [--aclnn-type <type>] \
    [--compute-units "soc1 soc2"] [--tiling-dirs "dir1 dir2"] \
    [--dependencies "op1 op2"] [--dependency-paths "path1,path2"] \
    [--dependency-api-sources "file1,file2"] \
    [--kernel-options "option1,option2"] [--disable-in-opp] [--hostcpu] \
    [--with-common] [--refresh]

The imported operator is normalized to <repo>/<operator> and excludes graph
plugins, docs, examples, tests, and source-repository CMake wrappers. Imported
kernel/API/common dependencies are stored inside the operator directory.
EOF
}

repo=""
source_repo=""
op_path=""
aclnn_type=""
compute_units=""
tiling_dirs=""
dependencies=""
dependency_paths=""
dependency_api_sources=""
kernel_options=""
disable_in_opp=false
hostcpu=false
with_common=false
refresh=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo) repo="$2"; shift 2 ;;
    --source) source_repo="$2"; shift 2 ;;
    --op) op_path="$2"; shift 2 ;;
    --aclnn-type) aclnn_type="$2"; shift 2 ;;
    --compute-units) compute_units="$2"; shift 2 ;;
    --tiling-dirs) tiling_dirs="$2"; shift 2 ;;
    --dependencies) dependencies="$2"; shift 2 ;;
    --dependency-paths) dependency_paths="$2"; shift 2 ;;
    --dependency-api-sources) dependency_api_sources="$2"; shift 2 ;;
    --kernel-options) kernel_options="$2"; shift 2 ;;
    --disable-in-opp) disable_in_opp=true; shift ;;
    --hostcpu) hostcpu=true; shift ;;
    --with-common) with_common=true; shift ;;
    --refresh) refresh=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "$repo" != "math" && "$repo" != "nn" && "$repo" != "cv" && "$repo" != "transformer" ]]; then
  echo "--repo must be math, nn, cv, or transformer" >&2
  exit 2
fi
if [[ -z "$source_repo" || -z "$op_path" ]]; then
  usage >&2
  exit 2
fi

source_repo="$(realpath "$source_repo")"
source_op="${source_repo}/${op_path}"
op_name="$(basename "$op_path")"
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dest="${project_root}/${repo}/${op_name}"

prune_common_dependency() {
  local operator_root="$1"
  local common_root="$2"
  local family="$3"
  local current_file include_file candidate candidate_path common_file
  local queue_index=0
  local -a scan_queue=()
  local -a include_roots=(
    "${operator_root}"
    "${operator_root}/op_api"
    "${operator_root}/op_host"
    "${operator_root}/op_kernel"
    "${operator_root}/_deps/${family}"
    "${project_root}/common/inc"
    "${project_root}/common/inc/op_api"
    "${project_root}/common/inc/external"
    "${common_root}/inc"
    "${common_root}/inc/op_api"
    "${common_root}/include"
    "${common_root}/include/common"
    "${common_root}/include/external"
    "${common_root}/stub/op_api"
  )
  declare -A queued_files=()

  while IFS= read -r current_file; do
    scan_queue+=("$current_file")
    queued_files["$current_file"]=1
  done < <(find "$operator_root" -type f \
    \( -name '*.h' -o -name '*.hpp' -o -name '*.cpp' -o -name '*.cc' \
       -o -name '*.cxx' -o -name '*.inc' -o -name '*.inl' \) \
    ! -path "${common_root}/*" -print)

  while IFS= read -r current_file; do
    scan_queue+=("$current_file")
    queued_files["$current_file"]=1
  done < <(find "${common_root}/src/tiling_base" -type f 2>/dev/null)

  while (( queue_index < ${#scan_queue[@]} )); do
    current_file="${scan_queue[queue_index]}"
    ((queue_index += 1))
    while IFS= read -r include_file; do
      candidate_path="$(realpath -m "$(dirname "$current_file")/${include_file}")"
      if [[ -f "$candidate_path" ]]; then
        candidate="$candidate_path"
      else
        candidate=""
        for candidate_path in "${include_roots[@]}"; do
          if [[ -f "${candidate_path}/${include_file}" ]]; then
            candidate="${candidate_path}/${include_file}"
            break
          fi
        done
      fi
      if [[ "$candidate" == "${common_root}/"* && -z "${queued_files[$candidate]:-}" ]]; then
        scan_queue+=("$candidate")
        queued_files["$candidate"]=1
      fi
    done < <(sed -n 's/^[[:space:]]*#include[[:space:]]*[<"]\([^">]*\)[">].*/\1/p' "$current_file")
  done

  while IFS= read -r common_file; do
    if [[ -z "${queued_files[$common_file]:-}" ]]; then
      rm -f -- "$common_file"
    fi
  done < <(find "$common_root" -type f -print)
  find "$common_root" -depth -type d -empty -delete
}

if [[ ! -d "$source_op" || ! -f "$source_op/CMakeLists.txt" ]]; then
  echo "Operator source not found: $source_op" >&2
  exit 1
fi
if [[ -e "$dest" ]]; then
  if [[ "$refresh" != true ]]; then
    echo "Destination already exists: $dest" >&2
    exit 1
  fi
  if [[ ! -f "$dest/.source" ]] ||
     ! grep -qx "repository=ops-${repo}" "$dest/.source" ||
     ! grep -qx "operator=${op_path}" "$dest/.source"; then
    echo "Refusing to refresh a destination not owned by this importer: $dest" >&2
    exit 1
  fi
  rm -rf -- "$dest"
fi
if [[ "$with_common" == true && ! -d "${source_repo}/common" ]]; then
  echo "Common directory not found: ${source_repo}/common" >&2
  exit 1
fi
if [[ -n "$compute_units" || -n "$tiling_dirs" ]]; then
  if [[ -z "$compute_units" || -z "$tiling_dirs" ]]; then
    echo "--compute-units and --tiling-dirs must be specified together" >&2
    exit 2
  fi
  read -r -a compute_unit_list <<< "$compute_units"
  read -r -a tiling_dir_list <<< "$tiling_dirs"
  if [[ ${#compute_unit_list[@]} -ne ${#tiling_dir_list[@]} ]]; then
    echo "--compute-units and --tiling-dirs must have the same number of entries" >&2
    exit 2
  fi
fi

if [[ -z "$aclnn_type" ]]; then
  mapfile -t aclnn_types < <(
    grep -RhoE 'ACLNNTYPE[[:space:]]+aclnn(_inner|_exclude)?' "$source_op"/CMakeLists.txt "$source_op"/*/CMakeLists.txt 2>/dev/null \
      | awk '{print $2}' | sort -u
  )
  if [[ ${#aclnn_types[@]} -ne 1 ]]; then
    echo "Cannot infer one ACLNN type for ${op_path}; pass --aclnn-type explicitly." >&2
    exit 1
  fi
  aclnn_type="${aclnn_types[0]}"
fi

if grep -RqsE 'DISABLE_IN_OPP[[:space:]]+TRUE' --include=CMakeLists.txt "$source_op"; then
  disable_in_opp=true
fi
if grep -RqsE 'HOSTCPU[[:space:]]+TRUE' --include=CMakeLists.txt "$source_op"; then
  hostcpu=true
fi

mkdir -p "${project_root}/${repo}"
cp -a "$source_op" "$dest"
rm -rf "$dest/README.md" "$dest/docs" "$dest/examples" "$dest/tests" \
  "$dest/op_graph" "$dest/framework"

if [[ -d "$dest/op_host/op_api" ]]; then
  mkdir -p "$dest/op_api"
  cp -a "$dest/op_host/op_api/." "$dest/op_api/"
  rm -rf "$dest/op_host/op_api"
fi

find "$dest" -name CMakeLists.txt \
  ! -path "$dest/CMakeLists.txt" \
  ! -path "$dest/op_kernel/CMakeLists.txt" -delete

dependency_cmake_paths=()
if [[ -n "$dependency_paths" && "$dependency_paths" != "-" ]]; then
  IFS=',' read -r -a dependency_path_list <<< "$dependency_paths"
  for dependency_path in "${dependency_path_list[@]}"; do
    if [[ -z "$dependency_path" || "$dependency_path" == /* || "$dependency_path" == *".."* ]]; then
      echo "Invalid dependency path: $dependency_path" >&2
      exit 2
    fi
    dependency_source="${source_repo}/${dependency_path}"
    dependency_name="$(basename "$dependency_path")"
    dependency_dest="${dest}/${dependency_name}"
    if [[ ! -d "$dependency_source" ]]; then
      echo "Dependency source not found: $dependency_source" >&2
      exit 1
    fi
    if [[ ! -d "${dependency_source}/op_kernel" ]]; then
      echo "Dependency kernel directory not found: ${dependency_source}/op_kernel" >&2
      exit 1
    fi
    mkdir -p "$dependency_dest"
    cp -a "${dependency_source}/op_kernel/." "$dependency_dest/"
    find "$dependency_dest" -name CMakeLists.txt -delete
    dependency_cmake_paths+=("\${CMAKE_CURRENT_SOURCE_DIR}/${dependency_name}")
  done
fi

dependency_api_cmake_sources=()
if [[ -n "$dependency_api_sources" && "$dependency_api_sources" != "-" ]]; then
  IFS=',' read -r -a dependency_api_source_list <<< "$dependency_api_sources"
  for dependency_api_source in "${dependency_api_source_list[@]}"; do
    if [[ -z "$dependency_api_source" || "$dependency_api_source" == /* || "$dependency_api_source" == *".."* ]]; then
      echo "Invalid dependency API source: $dependency_api_source" >&2
      exit 2
    fi
    if [[ ! -f "${source_repo}/${dependency_api_source}" ]]; then
      echo "Dependency API source not found: ${source_repo}/${dependency_api_source}" >&2
      exit 1
    fi
    dependency_api_dir="$(dirname "$dependency_api_source")"
    dependency_api_dest="${dest}/_deps/${repo}/${dependency_api_dir}"
    mkdir -p "$dependency_api_dest"
    cp -a "${source_repo}/${dependency_api_source}" "$dependency_api_dest/"
    while IFS= read -r dependency_include; do
      dependency_header="${source_repo}/${dependency_api_dir}/${dependency_include}"
      if [[ "$dependency_include" != */* && -f "$dependency_header" ]]; then
        cp -a "$dependency_header" "$dependency_api_dest/"
      fi
    done < <(sed -n 's/^[[:space:]]*#include[[:space:]]*"\([^"]*\)".*/\1/p' \
      "${source_repo}/${dependency_api_source}")
    dependency_api_cmake_sources+=(
      "\${CMAKE_CURRENT_SOURCE_DIR}/_deps/${repo}/${dependency_api_source}"
    )
  done
fi

{
  echo "# Generated by scripts/import_operator.sh from ${repo}:${op_path}."
  echo "add_all_modules_sources("
  echo "  OPTYPE ${op_name}"
  echo "  ACLNNTYPE ${aclnn_type}"
  if [[ -n "$compute_units" ]]; then
    echo "  COMPUTE_UNIT ${compute_units}"
  fi
  if [[ -n "$tiling_dirs" ]]; then
    echo "  TILING_DIR ${tiling_dirs}"
  fi
  if [[ -n "$dependencies" ]]; then
    echo "  DEPENDENCIES ${dependencies}"
  fi
  if [[ ${#dependency_cmake_paths[@]} -gt 0 ]]; then
    echo "  DEPENDENCY_DIRS"
    printf '    %s\n' "${dependency_cmake_paths[@]}"
  fi
  if [[ ${#dependency_api_cmake_sources[@]} -gt 0 ]]; then
    echo "  DEPENDENCY_API_SOURCES"
    printf '    %s\n' "${dependency_api_cmake_sources[@]}"
  fi
  if [[ "$disable_in_opp" == true ]]; then
    echo "  DISABLE_IN_OPP TRUE"
  fi
  if [[ "$hostcpu" == true ]]; then
    echo "  HOSTCPU TRUE"
  fi
  echo ")"
  if [[ -d "$dest/op_kernel" && ! -f "$dest/op_kernel/CMakeLists.txt" ]]; then
    echo
    if [[ -n "$kernel_options" && "$kernel_options" != "-" ]]; then
      echo "add_kernel_sources("
      echo "  OPTIONS ${kernel_options//,/ }"
      echo ")"
    else
      echo "add_kernel_sources()"
    fi
  fi
} > "$dest/CMakeLists.txt"

if [[ "$with_common" == true ]]; then
  common_src="${source_repo}/common"
  common_dest="${dest}/_deps/common"
  mkdir -p "$common_dest"
  for common_subdir in inc include stub/op_api src/tiling_base; do
    if [[ -d "${common_src}/${common_subdir}" ]]; then
      mkdir -p "${common_dest}/$(dirname "$common_subdir")"
      cp -a "${common_src}/${common_subdir}" "${common_dest}/${common_subdir}"
    fi
  done
  rm -rf "${common_dest}/inc/framework" "${common_dest}/inc/op_graph" \
    "${common_dest}/include/framework" "${common_dest}/include/op_graph"
  find "$common_dest" -name CMakeLists.txt -delete
  find "$common_dest" -type f -name '*.cpp' \
    ! -path "${common_dest}/src/tiling_base/*" -delete
  prune_common_dependency "$dest" "$common_dest" "$repo"
fi

source_record="$(realpath --relative-to="$project_root" "$source_repo")"
cat > "$dest/.source" <<EOF
repository=ops-${repo}
source=${source_record}
operator=${op_path}
EOF

echo "Imported ${op_path} -> ${repo}/${op_name}"
