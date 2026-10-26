# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16)
# Forward the optional compiler launcher to every compile and link rule.
set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ${RULE_LAUNCH})
set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ${RULE_LAUNCH})

# The project name is also used as the host-library target suffix.
set(PKG_NAME math)
project(${PKG_NAME} VERSION 1.0.0)

# Prefer the environment selected by set_env.sh when no explicit path was set.
if(NOT ASCEND_INSTALL_PATH AND DEFINED ENV{ASCEND_HOME_PATH})
  set(ASCEND_INSTALL_PATH $ENV{ASCEND_HOME_PATH})
endif()

option(BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG "Build with an installed CANN package" ON)

if(UNIX)
  set(SYSTEM_PREFIX ${CMAKE_SYSTEM_PROCESSOR}-linux)
endif()
set(OPS_MATH_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# Kernel-only build options. The shell entrypoint enables BINARY and CUSTOM;
# these cache variables are intentionally narrower than the full upstream CLI.
option(ENABLE_BINARY "Enable build binary" OFF)
option(ENABLE_CUSTOM "Enable build custom" OFF)
# Enabled only by build.sh --pkg; --opkernel remains a binary-only path.
option(ENABLE_PACKAGE "Enable install/package rules" OFF)
option(ENABLE_MSSANITIZER "Enable mssanitizer mode" OFF)
option(ENABLE_OOM "Enable oom mode" OFF)
option(ENABLE_DUMP_CCE "Enable dump_cce mode" OFF)

# Configure the only project-bootstrap feature still needed by the kernel
# pipeline. This local setup replaces cann-cmake's init_cann_project() so a
# clean configure never needs FetchContent or network access.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(ENABLE_CCACHE)
  find_program(CCACHE_PROGRAM ccache)
  if(CCACHE_PROGRAM)
    set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
  endif()
endif()

set(BISHENG_FLAGS "" CACHE STRING "bisheng compiler flags")
set(KERNEL_TEMPLATE_INPUT "" CACHE STRING "kernel template input")
set(BUILD_MODE "" CACHE STRING "build mode -O0/O1/O2/O3")
set(ASCEND_COMPUTE_UNIT "ascend910b" CACHE STRING "soc that need to be compiled")
if(NOT CANN_3RD_LIB_PATH)
  set(CANN_3RD_LIB_PATH ${PROJECT_SOURCE_DIR}/third_party CACHE STRING "cann third party lib path")
endif()
# Restrict discovery and metadata generation to the operator(s) selected by CLI.
set(ASCEND_OP_NAME "" CACHE STRING "operators that designated for compilation")
set(VENDOR_NAME "custom" CACHE STRING "vendor name")
set(ASCEND_ALL_COMPUTE_UNIT "${ASCEND_COMPUTE_UNIT}")

if("${BUILD_MODE}" STREQUAL "-O0" OR "${BUILD_MODE}" STREQUAL "-O1" OR
   "${BUILD_MODE}" STREQUAL "-O2" OR "${BUILD_MODE}" STREQUAL "-O3"
  )
  set(COMPILE_OP_MODE ${BUILD_MODE})
else()
  set(COMPILE_OP_MODE "-O2")
endif()
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "cmake build type")

# Configure the language standard and load dependency/target helper modules.
set(CMAKE_CXX_STANDARD 17 CACHE STRING "c++17 is needed for this project")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
include(cmake/third_party/opbase.cmake) # 放在dependencies.cmake前面
include(cmake/dependencies.cmake)
include(cmake/variables.cmake)
include(cmake/opbuild.cmake)
include(cmake/func.cmake)
include(cmake/intf_pub_linux.cmake)

# Define the common interface targets consumed by every operator's host code.
add_subdirectory(common)
# Discover operator directories directly under the skeleton root.
file(GLOB operator_dirs LIST_DIRECTORIES true RELATIVE
  "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/*")
foreach(operator_dir ${operator_dirs})
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${operator_dir}/op_host" AND
     EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${operator_dir}/CMakeLists.txt")
    add_subdirectory("${operator_dir}")
  endif()
endforeach()

# Fail during configuration if the requested operator was not discovered.
check_compiled_ops()

# Create one CANN package context for both kernel-only and package builds. CPack
# itself is enabled only by --pkg.
include(cmake/package.cmake)
init_custom_package(OPS_SKELETON_PACKAGE_NAME)

# Keep the existing ACLNN source generation, but delegate Kernel metadata and
# compilation to CANN's npu_op_kernel_library() target graph.
gen_aclnn_with_opdef()

# Build the object libraries consumed by the CANN package targets.
include(cmake/symbol.cmake)
gen_cust_opapi_symbol()
gen_cust_optiling_symbol()
include(cmake/kernel.cmake)
gen_ops_kernel_api(${OPS_SKELETON_PACKAGE_NAME})
pack_custom(${OPS_SKELETON_PACKAGE_NAME})
