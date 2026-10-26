# Minimal helpers for the AscendC kernel skeleton.

function(add_opbase_modules)
  # Compile the small subset of opbase utilities needed by InferShape/Tiling.
  if(TARGET opbase_infer_objs OR TARGET opbase_tiling_objs OR TARGET opbase_util_objs)
    return()
  endif()

  # Locate shared host implementations supplied by the CANN opbase package.
  file(GLOB OPS_BASE_INFER_SRC
    ${OPBASE_SOURCE_PATH}/src/op_common/op_host/infershape_broadcast_util.cpp
    ${OPBASE_SOURCE_PATH}/src/op_common/op_host/infershape_elewise_util.cpp
    ${OPBASE_SOURCE_PATH}/src/op_common/op_host/infershape_reduce_util.cpp)
  file(GLOB_RECURSE OPS_BASE_TILING_SRC
    ${OPBASE_SOURCE_PATH}/src/op_common/atvoss/elewise/*.cpp
    ${OPBASE_SOURCE_PATH}/src/op_common/atvoss/broadcast/*.cpp
    ${OPBASE_SOURCE_PATH}/src/op_common/atvoss/reduce/*.cpp
    ${OPBASE_SOURCE_PATH}/src/op_common/op_host/tiling_util.cpp)
  file(GLOB_RECURSE OPS_BASE_UTIL_SRC
    ${OPBASE_SOURCE_PATH}/src/op_common/op_host/util/*.cpp
    ${OPBASE_SOURCE_PATH}/src/op_common/log/*.cpp)

  # Object libraries avoid linking the same common implementation repeatedly.
  if(OPS_BASE_INFER_SRC)
    add_library(opbase_infer_objs OBJECT ${OPS_BASE_INFER_SRC})
    target_include_directories(opbase_infer_objs PRIVATE ${OP_PROTO_INCLUDE})
    target_compile_options(opbase_infer_objs PRIVATE -DDISABLE_COMPILE_V1 -Dgoogle=ascend_private -fvisibility=hidden)
    target_link_libraries(opbase_infer_objs PRIVATE intf_pub_cxx17 dlog_headers)
  endif()
  if(OPS_BASE_TILING_SRC)
    add_library(opbase_tiling_objs OBJECT ${OPS_BASE_TILING_SRC})
    target_include_directories(opbase_tiling_objs PRIVATE ${OP_TILING_INCLUDE})
    target_compile_options(opbase_tiling_objs PRIVATE -DDISABLE_COMPILE_V1 -Dgoogle=ascend_private
      -fvisibility=hidden -fno-strict-aliasing)
    target_link_libraries(opbase_tiling_objs PRIVATE intf_pub_cxx17 dlog_headers tiling_api)
  endif()
  if(OPS_BASE_UTIL_SRC)
    add_library(opbase_util_objs OBJECT ${OPS_BASE_UTIL_SRC})
    target_include_directories(opbase_util_objs PRIVATE ${OP_TILING_INCLUDE} ${ASCEND_DIR}/pkg_inc)
    target_compile_options(opbase_util_objs PRIVATE -DDISABLE_COMPILE_V1 -Dgoogle=ascend_private -fvisibility=hidden)
    target_link_libraries(opbase_util_objs PRIVATE intf_pub_cxx17 dlog_headers)
  endif()
endfunction()

function(add_infer_modules)
  # Create the host InferShape object target once, then append each operator's
  # *_infershape.cpp source during category discovery.
  if(TARGET ${OPHOST_NAME}_infer_obj)
    return()
  endif()
  add_library(${OPHOST_NAME}_infer_obj OBJECT)
  target_include_directories(${OPHOST_NAME}_infer_obj PRIVATE ${OP_PROTO_INCLUDE})
  target_compile_definitions(${OPHOST_NAME}_infer_obj PRIVATE
    OPS_UTILS_LOG_SUB_MOD_NAME="OP_PROTO" OP_SUBMOD_NAME="OPS_SKELETON"
    LOG_CPP OPS_MATH_KERNEL_ONLY)
  target_compile_options(${OPHOST_NAME}_infer_obj PRIVATE -DDISABLE_COMPILE_V1 -Dgoogle=ascend_private -fvisibility=hidden)
  target_link_libraries(${OPHOST_NAME}_infer_obj PRIVATE intf_pub_cxx17 dlog_headers
    $<$<TARGET_EXISTS:opbase_util_objs>:$<TARGET_OBJECTS:opbase_util_objs>>
    $<$<TARGET_EXISTS:opbase_infer_objs>:$<TARGET_OBJECTS:opbase_infer_objs>>)
endfunction()

function(add_tiling_modules)
  # Create the host Tiling object target; OPC loads the final shared library.
  if(TARGET ${OPHOST_NAME}_tiling_obj)
    return()
  endif()
  add_library(${OPHOST_NAME}_tiling_obj OBJECT)
  target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE ${OP_TILING_INCLUDE})
  target_compile_definitions(${OPHOST_NAME}_tiling_obj PRIVATE
    OPS_UTILS_LOG_SUB_MOD_NAME="OP_TILING" OP_SUBMOD_NAME="OPS_SKELETON"
    LOG_CPP OPS_MATH_KERNEL_ONLY)
  target_compile_options(${OPHOST_NAME}_tiling_obj PRIVATE -DDISABLE_COMPILE_V1 -Dgoogle=ascend_private
    -fvisibility=hidden -fno-strict-aliasing)
  target_link_libraries(${OPHOST_NAME}_tiling_obj PRIVATE intf_pub_cxx17 dlog_headers
    $<$<TARGET_EXISTS:opbase_util_objs>:$<TARGET_OBJECTS:opbase_util_objs>>
    $<$<TARGET_EXISTS:opbase_tiling_objs>:$<TARGET_OBJECTS:opbase_tiling_objs>> tiling_api)
endfunction()

function(add_opapi_modules)
  if(TARGET ${OPHOST_NAME}_opapi_obj)
    return()
  endif()
  add_library(${OPHOST_NAME}_opapi_obj OBJECT)
  target_include_directories(${OPHOST_NAME}_opapi_obj PRIVATE ${OPAPI_INCLUDE})
  target_compile_options(${OPHOST_NAME}_opapi_obj PRIVATE
    -Dgoogle=ascend_private -DACLNN_LOG_FMT_CHECK -DOP_LOG_LIBOPAPI_ONLY)
  target_link_libraries(${OPHOST_NAME}_opapi_obj
    PUBLIC intf_pub_cxx17
    PRIVATE dlog_headers
      $<$<TARGET_EXISTS:opbase_util_objs>:$<TARGET_OBJECTS:opbase_util_objs>>)
endfunction()

function(find_value_by_key keys values key result)
  list(LENGTH keys key_len)
  list(LENGTH values value_len)
  if(NOT key_len EQUAL value_len)
    message(FATAL_ERROR "compute-unit and tiling-directory lists must have equal length")
  endif()
  list(FIND keys "${key}" index)
  if(index GREATER -1)
    list(GET values ${index} value)
  else()
    set(value "")
  endif()
  set(${result} "${value}" PARENT_SCOPE)
endfunction()

function(add_tiling_sources tiling_dir disable_in_opp)
  # Select architecture-specific or root-level *_tiling.cpp files for the SoC.
  if(NOT BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG AND disable_in_opp)
    return()
  endif()
  if(tiling_dir)
    file(GLOB sources ${CMAKE_CURRENT_SOURCE_DIR}/op_host/*_tiling*.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/op_host/${tiling_dir}/*_tiling*.cpp)
  else()
    file(GLOB sources ${CMAKE_CURRENT_SOURCE_DIR}/op_host/*_tiling*.cpp)
  endif()
  if(sources)
    add_tiling_modules()
    target_sources(${OPHOST_NAME}_tiling_obj PRIVATE ${sources})
    target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/op_host ${CMAKE_CURRENT_SOURCE_DIR}/..)
  endif()
endfunction()

function(add_dependency_kernel_dirs dependency_dirs)
  foreach(dependency_dir IN LISTS dependency_dirs)
    if(NOT IS_ABSOLUTE "${dependency_dir}")
      set(dependency_dir "${CMAKE_SOURCE_DIR}/${dependency_dir}")
    endif()
    if(NOT IS_DIRECTORY "${dependency_dir}")
      message(FATAL_ERROR "Imported dependency directory does not exist: ${dependency_dir}")
    endif()

    get_property(registered_dirs GLOBAL PROPERTY OPS_SKELETON_DEPENDENCY_DIRS)
    if(dependency_dir IN_LIST registered_dirs)
      continue()
    endif()
    set_property(GLOBAL APPEND PROPERTY OPS_SKELETON_DEPENDENCY_DIRS "${dependency_dir}")

    list(APPEND DEPENDENCY_KERNEL_DIRS "${dependency_dir}")
    list(REMOVE_DUPLICATES DEPENDENCY_KERNEL_DIRS)
    set(DEPENDENCY_KERNEL_DIRS "${DEPENDENCY_KERNEL_DIRS}"
      CACHE INTERNAL "Kernel-only dependency directories" FORCE)
  endforeach()
endfunction()

function(add_dependency_api_sources dependency_sources)
  foreach(dependency_source IN LISTS dependency_sources)
    if(NOT EXISTS "${dependency_source}")
      message(FATAL_ERROR "Imported dependency API source does not exist: ${dependency_source}")
    endif()
    get_property(registered_sources GLOBAL PROPERTY OPS_SKELETON_DEPENDENCY_API_SOURCES)
    if(dependency_source IN_LIST registered_sources)
      continue()
    endif()
    set_property(GLOBAL APPEND PROPERTY OPS_SKELETON_DEPENDENCY_API_SOURCES "${dependency_source}")
    get_filename_component(dependency_api_dir "${dependency_source}" DIRECTORY)
    add_opapi_modules()
    target_sources(${OPHOST_NAME}_opapi_obj PRIVATE "${dependency_source}")
    target_include_directories(${OPHOST_NAME}_opapi_obj PRIVATE "${dependency_api_dir}")
    file(GLOB dependency_api_headers "${dependency_api_dir}/aclnn_*.h")
    if(dependency_api_headers)
      target_sources(${OPHOST_NAME}_aclnn_exclude_headers INTERFACE ${dependency_api_headers})
    endif()
  endforeach()
endfunction()

function(add_operator_private_dependencies source_dir)
  set(private_include_dirs
    "${source_dir}/_deps/math"
    "${source_dir}/_deps/nn"
    "${source_dir}/_deps/transformer"
    "${source_dir}/_deps/common/inc"
    "${source_dir}/_deps/common/inc/op_api"
    "${source_dir}/_deps/common/include"
    "${source_dir}/_deps/common/include/common"
    "${source_dir}/_deps/common/include/external"
    "${source_dir}/_deps/common/stub/op_api")
  foreach(source_target
    ${OPHOST_NAME}_opapi_obj ${OPHOST_NAME}_infer_obj ${OPHOST_NAME}_tiling_obj)
    if(TARGET ${source_target})
      target_include_directories(${source_target} PRIVATE ${private_include_dirs})
    endif()
  endforeach()
endfunction()

function(add_operator_private_tiling_sources source_dir)
  file(GLOB private_tiling_sources
    "${source_dir}/_deps/common/src/tiling_base/*.cpp")
  if(NOT private_tiling_sources)
    return()
  endif()

  set(unique_tiling_sources)
  get_property(registered_hashes GLOBAL PROPERTY OPS_SKELETON_PRIVATE_TILING_HASHES)
  foreach(private_tiling_source IN LISTS private_tiling_sources)
    file(SHA256 "${private_tiling_source}" source_hash)
    if(source_hash IN_LIST registered_hashes)
      continue()
    endif()
    list(APPEND registered_hashes ${source_hash})
    list(APPEND unique_tiling_sources ${private_tiling_source})
  endforeach()
  if(NOT unique_tiling_sources)
    return()
  endif()
  set_property(GLOBAL PROPERTY OPS_SKELETON_PRIVATE_TILING_HASHES ${registered_hashes})
  add_tiling_modules()
  target_sources(${OPHOST_NAME}_tiling_obj PRIVATE ${unique_tiling_sources})
endfunction()

macro(add_all_modules_sources)
  # Register one imported operator: host sources, opdef metadata and kernel
  # subdirectory. This macro is called from each operator CMakeLists.txt.
  set(oneValueArgs DISABLE_IN_OPP HOSTCPU)
  set(multiValueArgs OPTYPE ACLNNTYPE DEPENDENCIES DEPENDENCY_DIRS DEPENDENCY_API_SOURCES COMPUTE_UNIT TILING_DIR)
  cmake_parse_arguments(MODULE "" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
  get_filename_component(OP_NAME ${SOURCE_DIR} NAME)

  if(NEED_COMPILE_OPS AND NOT OP_NAME IN_LIST NEED_COMPILE_OPS)
    return()
  endif()
  if(OP_NAME IN_LIST COMPILED_OPS)
    return()
  endif()

  list(APPEND COMPILED_OPS ${OP_NAME})
  list(APPEND COMPILED_OP_DIRS ${SOURCE_DIR})
  set(COMPILED_OPS "${COMPILED_OPS}" CACHE STRING "Operators discovered during configure" FORCE)
  set(COMPILED_OP_DIRS "${COMPILED_OP_DIRS}" CACHE STRING "Operator directories discovered during configure" FORCE)

  # Shared opbase objects must exist before operator objects link against them.
  add_opbase_modules()
  add_operator_private_tiling_sources("${SOURCE_DIR}")
  add_dependency_kernel_dirs("${MODULE_DEPENDENCY_DIRS}")
  add_dependency_api_sources("${MODULE_DEPENDENCY_API_SOURCES}")

  # A hand-written op_api is optional. If absent, op_build generates the public
  # two-stage ACLNN API from the selected *_def.cpp source.
  file(GLOB opapi_sources ${SOURCE_DIR}/op_api/*.cpp)
  if(opapi_sources)
    add_opapi_modules()
    target_sources(${OPHOST_NAME}_opapi_obj PRIVATE ${opapi_sources})
    target_include_directories(${OPHOST_NAME}_opapi_obj PRIVATE ${SOURCE_DIR}/op_api)
  endif()
  file(GLOB opapi_headers ${SOURCE_DIR}/op_api/aclnn_*.h ${SOURCE_DIR}/op_api/acl_*.h)
  if(opapi_headers)
    target_sources(${OPHOST_NAME}_aclnn_exclude_headers INTERFACE ${opapi_headers})
  endif()

  # InferShape is compiled into the host object consumed by opbuild/OPC.
  file(GLOB infer_sources ${SOURCE_DIR}/op_host/*_infershape*.cpp)
  if(infer_sources)
    add_infer_modules()
    target_sources(${OPHOST_NAME}_infer_obj PRIVATE ${infer_sources})
  endif()

  # Map the selected SoC to the corresponding architecture-specific tiling dir.
  find_value_by_key("${MODULE_COMPUTE_UNIT}" "${MODULE_TILING_DIR}" "${ASCEND_COMPUTE_UNIT}" tiling_dir)
  add_tiling_sources("${tiling_dir}" "${MODULE_DISABLE_IN_OPP}")

  list(LENGTH MODULE_OPTYPE type_count)
  list(LENGTH MODULE_ACLNNTYPE aclnn_count)
  if(NOT type_count EQUAL aclnn_count)
    message(FATAL_ERROR "OPTYPE and ACLNNTYPE must have equal length")
  endif()
  math(EXPR last_index "${type_count} - 1")
  if(last_index GREATER -1)
    foreach(index RANGE ${last_index})
      list(GET MODULE_OPTYPE ${index} op_type)
      list(GET MODULE_ACLNNTYPE ${index} aclnn_type)
      if(NOT aclnn_type STREQUAL "aclnn" AND
         NOT aclnn_type STREQUAL "aclnn_inner" AND
         NOT aclnn_type STREQUAL "aclnn_exclude")
        message(FATAL_ERROR "ACLNNTYPE must be aclnn, aclnn_inner or aclnn_exclude")
      endif()
      file(GLOB def_sources ${SOURCE_DIR}/op_host/${op_type}_def*.cpp)
      if(MODULE_EXT)
        get_filename_component(parent_name ${SOURCE_DIR} DIRECTORY)
        get_filename_component(parent_name ${parent_name} NAME)
        file(GLOB extra_defs ${MODULE_EXT}/${parent_name}/${op_type}/op_host/${op_type}_*_def*.cpp)
        list(APPEND def_sources ${extra_defs})
      endif()
      if(def_sources)
        target_sources(${OPHOST_NAME}_opdef_${aclnn_type}_obj INTERFACE ${def_sources})
      endif()
    endforeach()
  endif()

  # Kernel CMake contributes source/options to the later binary metadata pass.
  if(EXISTS "${SOURCE_DIR}/op_kernel/CMakeLists.txt")
    add_subdirectory(${SOURCE_DIR}/op_kernel)
  endif()

  add_operator_private_dependencies("${SOURCE_DIR}")

  file(RELATIVE_PATH source_relative "${CMAKE_SOURCE_DIR}" "${SOURCE_DIR}")
  string(REGEX MATCH "^[^/]+" source_family "${source_relative}")
  if(NOT source_family STREQUAL "math")
    foreach(source_target
      ${OPHOST_NAME}_opapi_obj ${OPHOST_NAME}_infer_obj ${OPHOST_NAME}_tiling_obj)
      if(TARGET ${source_target})
        target_compile_options(${source_target} PRIVATE -Wno-error)
      endif()
    endforeach()
  endif()
endmacro()

function(get_op_type_from_op_name op_name result)
  file(GLOB_RECURSE def_files "${OPS_MATH_DIR}/*/${op_name}_def.cpp")
  set(op_type "")
  foreach(def_file IN LISTS def_files)
    file(READ "${def_file}" contents)
    string(REGEX MATCH "OP_ADD[ \t]*\\([ \t]*([A-Za-z0-9_]+)" match "${contents}")
    if(match)
      set(op_type "${CMAKE_MATCH_1}")
      break()
    endif()
  endforeach()
  set(${result} "${op_type}" PARENT_SCOPE)
endfunction()

function(get_op_type_from_binary_json binary_json result)
  file(READ "${binary_json}" contents)
  string(REGEX MATCH "\\\"op_type\\\"[ \t]*:[ \t]*\\\"([^\\\"]+)\\\"" match "${contents}")
  set(op_type "${CMAKE_MATCH_1}")
  set(${result} "${op_type}" PARENT_SCOPE)
endfunction()

function(check_op_supported op_name op_dir compute_unit result)
  file(GLOB defs "${op_dir}/op_host/${op_name}_def.cpp")
  set(supported FALSE)
  foreach(def_file IN LISTS defs)
    file(READ "${def_file}" contents)
    string(FIND "${contents}" "\"${compute_unit}\"" match)
    if(match GREATER -1)
      set(supported TRUE)
    endif()
  endforeach()
  set(${result} "${supported}" PARENT_SCOPE)
endfunction()

function(check_compiled_ops)
  # Configuration must fail early instead of producing an empty kernel target.
  if(NOT COMPILED_OPS)
    message(FATAL_ERROR "No selected operator was found")
  endif()
  foreach(op_name IN LISTS NEED_COMPILE_OPS)
    if(NOT op_name IN_LIST COMPILED_OPS)
      message(FATAL_ERROR "Selected operator not found: ${op_name}")
    endif()
  endforeach()
endfunction()

function(add_kernel_sources)
  # Record the source mapping consumed by npu_op_kernel_sources() after CANN
  # has generated the operator metadata at the top level.
  set(oneValueArgs KERNEL_SRC SIMPLIFIED_KEY AUTO_SYNC IMPL_MODE)
  set(multiValueArgs COMPUTE_UNITS OPTIONS)
  cmake_parse_arguments(MODULE "" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  get_filename_component(current_dir_name "${CMAKE_CURRENT_SOURCE_DIR}" NAME)
  if(current_dir_name STREQUAL "op_kernel")
    get_filename_component(op_dir "${CMAKE_CURRENT_SOURCE_DIR}" DIRECTORY)
  else()
    set(op_dir "${CMAKE_CURRENT_SOURCE_DIR}")
  endif()
  get_filename_component(op_name "${op_dir}" NAME)
  get_op_type_from_op_name("${op_name}" op_type)
  if(NOT op_type)
    return()
  endif()
  if(MODULE_COMPUTE_UNITS AND NOT ASCEND_COMPUTE_UNIT IN_LIST MODULE_COMPUTE_UNITS)
    return()
  endif()

  if(MODULE_KERNEL_SRC)
    get_filename_component(kernel_file "${MODULE_KERNEL_SRC}" NAME)
    get_filename_component(kernel_subdir "${MODULE_KERNEL_SRC}" DIRECTORY)
    if(kernel_subdir STREQUAL "")
      set(kernel_dir "${op_dir}/op_kernel")
    else()
      set(kernel_dir "${op_dir}/op_kernel/${kernel_subdir}")
    endif()
  else()
    set(kernel_file "${op_name}.cpp")
    set(kernel_dir "${op_dir}/op_kernel")
  endif()
  if(NOT EXISTS "${kernel_dir}/${kernel_file}")
    message(FATAL_ERROR "Kernel source does not exist: ${kernel_dir}/${kernel_file}")
  endif()
  file(RELATIVE_PATH kernel_dir "${CMAKE_SOURCE_DIR}" "${kernel_dir}")
  list(APPEND KERNEL_API_SOURCE_INFO "${op_type}|${kernel_dir}|${kernel_file}")
  list(REMOVE_DUPLICATES KERNEL_API_SOURCE_INFO)
  set(KERNEL_API_SOURCE_INFO "${KERNEL_API_SOURCE_INFO}"
    CACHE INTERNAL "CANN kernel API source registrations" FORCE)

  set(kernel_options ${MODULE_OPTIONS})
  string(TOLOWER "${MODULE_AUTO_SYNC}" auto_sync)
  if(auto_sync STREQUAL "false")
    list(APPEND kernel_options --cce-auto-sync=off)
  endif()
  if(kernel_options)
    set_property(GLOBAL APPEND PROPERTY
      OPS_SKELETON_KERNEL_OPTIONS_${op_type} ${kernel_options})
  endif()

  set(simplified_key "${MODULE_SIMPLIFIED_KEY}")
  if(NOT simplified_key)
    set(simplified_key 0)
  endif()
  if(simplified_key STREQUAL "none" OR simplified_key STREQUAL "None")
    set(simplified_key None)
  endif()
  set(impl_mode "${MODULE_IMPL_MODE}")
  if(NOT impl_mode)
    set(impl_mode "high_performance,optional")
  endif()
  string(JOIN "," options ${MODULE_OPTIONS})

  list(APPEND SIMPLIFIED_KEY_LIST "${op_type} ${ASCEND_COMPUTE_UNIT} ${simplified_key}")
  list(APPEND IMPL_MODE_LIST "${op_type} ${ASCEND_COMPUTE_UNIT} ${impl_mode}")
  set(SIMPLIFIED_KEY_LIST "${SIMPLIFIED_KEY_LIST}" CACHE INTERNAL "Kernel simplified-key list")
  set(IMPL_MODE_LIST "${IMPL_MODE_LIST}" CACHE INTERNAL "Kernel implementation-mode list")
  if(auto_sync STREQUAL "false")
    list(APPEND AUTO_SYNC_LIST "${op_type} ${ASCEND_COMPUTE_UNIT} false")
    set(AUTO_SYNC_LIST "${AUTO_SYNC_LIST}" CACHE INTERNAL "Kernel auto-sync list")
  endif()
  if(options)
    list(APPEND OPTION_LIST "${op_type} ${ASCEND_COMPUTE_UNIT} ${options}")
    set(OPTION_LIST "${OPTION_LIST}" CACHE INTERNAL "Kernel option list")
  endif()
  if(MODULE_KERNEL_SRC)
    string(REGEX REPLACE "\\.cpp$" "" kernel_src "${MODULE_KERNEL_SRC}")
    list(APPEND KERNEL_SRC_LIST "${op_type} ${ASCEND_COMPUTE_UNIT} ${kernel_src}")
    set(KERNEL_SRC_LIST "${KERNEL_SRC_LIST}" CACHE INTERNAL "Kernel source list")
  endif()
endfunction()
