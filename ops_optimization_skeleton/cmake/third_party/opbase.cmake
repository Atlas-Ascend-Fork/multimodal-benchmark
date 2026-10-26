# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
set(OPBASE_TAG_ID 0c5e25790f67e2c6f60c1df36d74d28f0e08ecfc)

if(NOT OPBASE_SOURCE_PATH AND DEFINED ENV{OPBASE_SOURCE_PATH})
  set(OPBASE_SOURCE_PATH "$ENV{OPBASE_SOURCE_PATH}")
endif()
if(NOT OPBASE_SOURCE_PATH)
  if(EXISTS "${PROJECT_SOURCE_DIR}/../../ops-base")
    set(OPBASE_SOURCE_PATH "${PROJECT_SOURCE_DIR}/../../ops-base")
  else()
    set(OPBASE_SOURCE_PATH "${CANN_3RD_LIB_PATH}/opbase")
  endif()
endif()
get_filename_component(OPBASE_SOURCE_PATH "${OPBASE_SOURCE_PATH}" REALPATH)
if(NOT EXISTS "${OPBASE_SOURCE_PATH}/.git")
  message(FATAL_ERROR
    "A pinned opbase Git checkout is required at ${OPBASE_SOURCE_PATH}. "
    "Set OPBASE_SOURCE_PATH to revision ${OPBASE_TAG_ID}; see README.md for offline setup.")
endif()

find_package(Git REQUIRED)
execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${OPBASE_SOURCE_PATH}" rev-parse HEAD
                OUTPUT_VARIABLE OPBASE_REVISION OUTPUT_STRIP_TRAILING_WHITESPACE
                RESULT_VARIABLE OPBASE_GIT_RESULT)
if(NOT OPBASE_GIT_RESULT EQUAL 0 OR NOT OPBASE_REVISION STREQUAL OPBASE_TAG_ID)
  message(FATAL_ERROR "opbase must be checked out at ${OPBASE_TAG_ID}; found ${OPBASE_REVISION}")
endif()
execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${OPBASE_SOURCE_PATH}" status --porcelain --untracked-files=all
                OUTPUT_VARIABLE OPBASE_CHANGES OUTPUT_STRIP_TRAILING_WHITESPACE
                RESULT_VARIABLE OPBASE_GIT_RESULT)
if(NOT OPBASE_GIT_RESULT EQUAL 0 OR NOT OPBASE_CHANGES STREQUAL "")
  message(FATAL_ERROR "opbase checkout must be clean: ${OPBASE_SOURCE_PATH}")
endif()
message(STATUS "Using opbase ${OPBASE_REVISION}: ${OPBASE_SOURCE_PATH}")
