# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

# Ascend mode
if(DEFINED ENV{ASCEND_HOME_PATH})
  set(ASCEND_DIR $ENV{ASCEND_HOME_PATH})
else()
  if ("$ENV{USER}" STREQUAL "root")
    if(EXISTS /usr/local/Ascend/ascend-toolkit/latest)
      set(ASCEND_DIR /usr/local/Ascend/ascend-toolkit/latest)
    else()
      set(ASCEND_DIR /usr/local/Ascend/latest)
    endif()
  else()
    if(EXISTS $ENV{HOME}/Ascend/ascend-toolkit/latest)
      set(ASCEND_DIR $ENV{HOME}/Ascend/ascend-toolkit/latest)
    else()
      set(ASCEND_DIR $ENV{HOME}/Ascend/latest)
    endif()
  endif()
endif()
set(ASCEND_CANN_PACKAGE_PATH ${ASCEND_DIR})
message(STATUS "Search libs under install path ${ASCEND_DIR}")

set(CMAKE_PREFIX_PATH ${ASCEND_DIR}/)

set(CMAKE_MODULE_PATH
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules
  ${ASCEND_DIR}/include/ge/cmake
  ${CMAKE_MODULE_PATH}
)
message(STATUS "CMAKE_MODULE_PATH            :${CMAKE_MODULE_PATH}")

set(OPS_MATH_CXX_FLAGS)
string(APPEND OPS_MATH_CXX_FLAGS " ${COMPILE_OP_MODE}")
string(APPEND OPS_MATH_CXX_FLAGS " -Wall")
string(APPEND OPS_MATH_CXX_FLAGS " -Wextra")
string(APPEND OPS_MATH_CXX_FLAGS " -Wshadow")
string(APPEND OPS_MATH_CXX_FLAGS " -Wformat=2")
string(APPEND OPS_MATH_CXX_FLAGS " -fno-common")
string(APPEND OPS_MATH_CXX_FLAGS " -fPIC")
string(APPEND OPS_MATH_CXX_FLAGS " -Wl,-z,relro,-z,now")
string(APPEND OPS_MATH_CXX_FLAGS " -fstack-protector-all")
string(APPEND OPS_MATH_CXX_FLAGS " -ftrapv")
string(APPEND OPS_MATH_CXX_FLAGS " -Wno-write-strings")
string(APPEND OPS_MATH_CXX_FLAGS " -Wno-unused-but-set-parameter")
if(NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    string(APPEND OPS_MATH_CXX_FLAGS " -Wno-error")
  else()
    string(APPEND OPS_MATH_CXX_FLAGS " -Werror")
  endif()
  string(APPEND OPS_MATH_CXX_FLAGS " -Wno-error=deprecated-declarations")
  string(APPEND OPS_MATH_CXX_FLAGS " -Wno-class-memaccess")
  string(APPEND OPS_MATH_CXX_FLAGS " -Wformat-signedness")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OPS_MATH_CXX_FLAGS}")
message(STATUS "compile option:${CMAKE_CXX_FLAGS}")

if(BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG)
  find_package(ASC REQUIRED HINTS ${ASCEND_DIR}/${SYSTEM_PREFIX}/tikcpp/ascendc_kernel_cmake)
endif()
find_package(dlog MODULE REQUIRED)
find_package(securec MODULE)
find_package(OPBASE MODULE REQUIRED)
find_package(platform MODULE REQUIRED)
find_package(metadef MODULE REQUIRED)
find_package(runtime MODULE REQUIRED)
find_package(nnopbase MODULE REQUIRED)
find_package(tilingapi MODULE REQUIRED)
