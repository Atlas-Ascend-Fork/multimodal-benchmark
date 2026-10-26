# Configure the CANN 9 kernel target graph. This path lets CANN compile every
# operator as one target without opc_cmd.sh or a second CMake configure.
function(gen_ops_kernel_api package_name)
  set(opdef_sources)
  foreach(opdef_target
      ${OPHOST_NAME}_opdef_aclnn_obj
      ${OPHOST_NAME}_opdef_aclnn_inner_obj
      ${OPHOST_NAME}_opdef_aclnn_exclude_obj)
    get_target_property(target_sources ${opdef_target} INTERFACE_SOURCES)
    if(target_sources)
      list(APPEND opdef_sources ${target_sources})
    endif()
  endforeach()
  list(REMOVE_DUPLICATES opdef_sources)
  if(NOT opdef_sources)
    message(FATAL_ERROR "No OpDef source was registered for kernel code generation")
  endif()
  if(NOT KERNEL_API_SOURCE_INFO)
    message(FATAL_ERROR "No kernel source was registered")
  endif()

  set(codegen_options)
  foreach(include_dir IN LISTS OP_PROTO_INCLUDE)
    list(APPEND codegen_options "-I${include_dir}")
  endforeach()
  npu_op_code_gen(
    SRC ${opdef_sources}
    PACKAGE ${package_name}
    OUT_DIR ${ASCEND_KERNEL_API_AUTOGEN_PATH}
    COMPILE_OPTIONS ${codegen_options}
    OPTIONS
      OPS_PROJECT_NAME ops_skeleton
      OPS_ACLNN_GEN 0
      OPS_PROTO_SEPARATE 1
  )
  file(TOUCH
    "${ASCEND_KERNEL_API_AUTOGEN_PATH}/${CUSTOM_COMPILE_OPTIONS}"
    "${ASCEND_KERNEL_API_AUTOGEN_PATH}/${CUSTOM_OPC_OPTIONS}")

  set(kernel_types)
  foreach(kernel_info IN LISTS KERNEL_API_SOURCE_INFO)
    string(REPLACE "|" ";" kernel_parts "${kernel_info}")
    list(GET kernel_parts 0 op_type)
    list(GET kernel_parts 1 kernel_dir)
    list(GET kernel_parts 2 kernel_file)
    list(APPEND kernel_types ${op_type})
    npu_op_kernel_sources(ops_skeleton_kernels
      OP_TYPE ${op_type}
      KERNEL_DIR ${kernel_dir}
      KERNEL_FILE ${kernel_file}
    )
  endforeach()
  list(REMOVE_DUPLICATES kernel_types)

  foreach(op_type IN LISTS kernel_types)
    get_property(kernel_options GLOBAL PROPERTY OPS_SKELETON_KERNEL_OPTIONS_${op_type})
    if(KERNEL_TEMPLATE_INPUT)
      list(APPEND kernel_options "--kernel-template-input=${KERNEL_TEMPLATE_INPUT}")
    endif()
    if(kernel_options)
      npu_op_kernel_options(ops_skeleton_kernels ${op_type}
        COMPUTE_UNIT ${ASCEND_COMPUTE_UNIT}
        OPTIONS ${kernel_options})
    endif()
  endforeach()

  set(op_debug_configs)
  if(BISHENG_FLAGS)
    list(APPEND op_debug_configs ${BISHENG_FLAGS})
  else()
    if(ENABLE_MSSANITIZER)
      list(APPEND op_debug_configs sanitizer)
    endif()
    if(ENABLE_OOM)
      list(APPEND op_debug_configs oom)
    endif()
    if(ENABLE_DUMP_CCE)
      list(APPEND op_debug_configs dump_cce)
    endif()
  endif()
  if(op_debug_configs)
    string(JOIN "," op_debug_config ${op_debug_configs})
    foreach(op_type IN LISTS kernel_types)
      file(APPEND "${ASCEND_KERNEL_API_AUTOGEN_PATH}/${CUSTOM_OPC_OPTIONS}"
        "${op_type}@${ASCEND_COMPUTE_UNIT}@--op_debug_config=${op_debug_config}\n")
    endforeach()
  endif()

  npu_op_kernel_library(ops_skeleton_kernels
    SRC_BASE ${CMAKE_SOURCE_DIR}
    TILING_LIBRARY cust_opmaster)
endfunction()
