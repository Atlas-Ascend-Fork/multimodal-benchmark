# Generate operator metadata and, for ACLNN-enabled operators, the public
# two-stage API sources consumed by libcust_opapi.so.

function(gen_opbuild_target)
  set(oneValueArgs TARGET PREFIX GEN_ACLNN OUT_DIR OUT_SUB_DIR)
  set(multiValueArgs IN_SRCS OUT_SRCS OUT_HEADERS)
  cmake_parse_arguments(OPBUILD "" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  if(NOT OPBUILD_IN_SRCS)
    return()
  endif()

  add_library(gen_op_host_${OPBUILD_PREFIX} SHARED ${OPBUILD_IN_SRCS})
  target_link_libraries(gen_op_host_${OPBUILD_PREFIX} PRIVATE
    intf_pub_cxx17 exe_graph register c_sec)
  target_compile_options(gen_op_host_${OPBUILD_PREFIX} PRIVATE -fno-common)

  string(REPLACE ";" "\;" product_name "${ASCEND_COMPUTE_UNIT}")
  add_custom_command(
    OUTPUT ${OPBUILD_OUT_SRCS} ${OPBUILD_OUT_HEADERS}
    COMMAND ${CMAKE_COMMAND} -E make_directory
      "${OPBUILD_OUT_DIR}/${OPBUILD_OUT_SUB_DIR}"
    COMMAND ${CMAKE_COMMAND} -E env
      OPS_PROTO_SEPARATE=1
      OPS_PROJECT_NAME=${OPBUILD_PREFIX}
      OPS_ACLNN_GEN=${OPBUILD_GEN_ACLNN}
      OPS_PRODUCT_NAME=${product_name}
      LD_LIBRARY_PATH=${ASCEND_DIR}/${SYSTEM_PREFIX}/lib64:$ENV{LD_LIBRARY_PATH}
      ${OP_BUILD_TOOL} $<TARGET_FILE:gen_op_host_${OPBUILD_PREFIX}>
      "${OPBUILD_OUT_DIR}/${OPBUILD_OUT_SUB_DIR}"
    DEPENDS gen_op_host_${OPBUILD_PREFIX}
    VERBATIM)
  add_custom_target(${OPBUILD_TARGET}
    DEPENDS ${OPBUILD_OUT_SRCS} ${OPBUILD_OUT_HEADERS})
endfunction()

function(gen_aclnn_classify host_target prefix output_sources output_headers)
  get_target_property(def_sources ${host_target} INTERFACE_SOURCES)
  if(NOT def_sources)
    set(${output_sources} "${${output_sources}}" PARENT_SCOPE)
    set(${output_headers} "${${output_headers}}" PARENT_SCOPE)
    return()
  endif()

  if(prefix STREQUAL "aclnn")
    set(sub_dir "")
    set(file_prefix aclnn)
    set(generate_api 1)
  elseif(prefix STREQUAL "aclnnInner")
    set(sub_dir inner)
    set(file_prefix aclnnInner)
    set(generate_api 1)
  elseif(prefix STREQUAL "aclnnExc")
    set(sub_dir exc)
    set(file_prefix aclnn)
    set(generate_api 0)
  else()
    message(FATAL_ERROR "Unsupported ACLNN classification: ${prefix}")
  endif()

  set(output_dir "${ASCEND_AUTOGEN_PATH}/${sub_dir}")
  file(MAKE_DIRECTORY "${output_dir}")
  set(generated_sources)
  set(generated_headers)
  foreach(def_file IN LISTS def_sources)
    get_filename_component(def_name "${def_file}" NAME_WE)
    string(REGEX REPLACE "_def$" "" op_name "${def_name}")
    list(APPEND generated_sources "${output_dir}/${file_prefix}_${op_name}.cpp")
    list(APPEND generated_headers "${output_dir}/${file_prefix}_${op_name}.h")
  endforeach()

  gen_opbuild_target(
    TARGET opbuild_gen_${prefix}
    PREFIX ${prefix}
    GEN_ACLNN ${generate_api}
    IN_SRCS "${def_sources}"
    OUT_SRCS "${generated_sources}"
    OUT_HEADERS "${generated_headers}"
    OUT_DIR "${ASCEND_AUTOGEN_PATH}"
    OUT_SUB_DIR "${sub_dir}")

  if(prefix STREQUAL "aclnnExc")
    get_target_property(excluded_headers
      ${OPHOST_NAME}_aclnn_exclude_headers INTERFACE_SOURCES)
    if(excluded_headers)
      set(${output_headers} "${${output_headers}};${excluded_headers}" PARENT_SCOPE)
    endif()
  else()
    set(${output_sources} "${${output_sources}};${generated_sources}" PARENT_SCOPE)
    if(prefix STREQUAL "aclnn")
      set(${output_headers} "${${output_headers}};${generated_headers}" PARENT_SCOPE)
    endif()
  endif()
endfunction()

function(gen_aclnn_master_header output_header headers)
  set(aclnn_master_header_name "aclnn_ops_math_${VENDOR_NAME}")
  string(REGEX REPLACE "[^a-zA-Z0-9_]" "_" aclnn_master_header_name
    "${aclnn_master_header_name}")
  string(TOUPPER "${aclnn_master_header_name}" aclnn_master_header_name)
  set(aclnn_all_header_include_content "")
  foreach(header_file IN LISTS headers)
    get_filename_component(header_name "${header_file}" NAME)
    string(APPEND aclnn_all_header_include_content
      "#include \"${header_name}\"\n")
  endforeach()
  configure_file("${CMAKE_SOURCE_DIR}/cmake/aclnn_ops_math.h.in"
    "${output_header}" @ONLY)
endfunction()

function(gen_aclnn_with_opdef)
  set(generated_sources)
  set(generated_headers)
  gen_aclnn_classify(${OPHOST_NAME}_opdef_aclnn_obj aclnn
    generated_sources generated_headers)
  gen_aclnn_classify(${OPHOST_NAME}_opdef_aclnn_inner_obj aclnnInner
    generated_sources generated_headers)
  gen_aclnn_classify(${OPHOST_NAME}_opdef_aclnn_exclude_obj aclnnExc
    generated_sources generated_headers)

  set(master_header "${CMAKE_BINARY_DIR}/aclnn_ops_math_${VENDOR_NAME}.h")
  gen_aclnn_master_header("${master_header}" "${generated_headers}")
  if(ENABLE_PACKAGE)
    install(FILES ${generated_headers} "${master_header}"
      DESTINATION ${ACLNN_INC_INSTALL_DIR} OPTIONAL)
  endif()

  set(generation_targets)
  foreach(target opbuild_gen_aclnn opbuild_gen_aclnnInner opbuild_gen_aclnnExc)
    if(TARGET ${target})
      list(APPEND generation_targets ${target})
    endif()
  endforeach()
  if(NOT generation_targets)
    return()
  endif()

  string(JOIN "," kernel_sources ${KERNEL_SRC_LIST})
  add_custom_target(opbuild_custom_gen_aclnn_all
    COMMAND ${ASCEND_PYTHON_EXECUTABLE}
      ${CMAKE_SOURCE_DIR}/scripts/util/insert_kernel_src.py
      "${kernel_sources}" "${ASCEND_AUTOGEN_PATH}" "${ASCEND_COMPUTE_UNIT}"
    COMMAND ${ASCEND_PYTHON_EXECUTABLE}
      ${CMAKE_SOURCE_DIR}/scripts/util/insert_kernel_src.py
      "${kernel_sources}" "${ASCEND_AUTOGEN_PATH}/inner" "${ASCEND_COMPUTE_UNIT}"
    COMMAND ${ASCEND_PYTHON_EXECUTABLE}
      ${CMAKE_SOURCE_DIR}/scripts/util/insert_kernel_src.py
      "${kernel_sources}" "${ASCEND_AUTOGEN_PATH}/exc" "${ASCEND_COMPUTE_UNIT}"
    DEPENDS ${generation_targets}
    VERBATIM)

  if(generated_sources)
    set_source_files_properties(${generated_sources} PROPERTIES GENERATED TRUE)
    add_library(opbuild_gen_aclnn_all OBJECT ${generated_sources})
    add_dependencies(opbuild_gen_aclnn_all opbuild_custom_gen_aclnn_all)
    target_include_directories(opbuild_gen_aclnn_all PRIVATE ${OPAPI_INCLUDE})
    target_compile_options(opbuild_gen_aclnn_all PRIVATE
      -Dgoogle=ascend_private -DACLNN_LOG_FMT_CHECK -DOP_LOG_LIBOPAPI_ONLY)
    target_link_libraries(opbuild_gen_aclnn_all PUBLIC intf_pub_cxx17)
  endif()
endfunction()
