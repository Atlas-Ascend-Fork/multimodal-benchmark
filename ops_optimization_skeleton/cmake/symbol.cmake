# Link the host/tiling objects required by the OPC tiling stage.
function(gen_ophost_symbol)
  # Link all discovered InferShape/Tiling object libraries into the host .so
  # that the OPC-generated kernel invokes for tiling decisions.
  add_library(${OPHOST_NAME} SHARED
    $<$<TARGET_EXISTS:${OPHOST_NAME}_infer_obj>:$<TARGET_OBJECTS:${OPHOST_NAME}_infer_obj>>
    $<$<TARGET_EXISTS:${OPHOST_NAME}_tiling_obj>:$<TARGET_OBJECTS:${OPHOST_NAME}_tiling_obj>>
    $<$<TARGET_EXISTS:opbase_util_objs>:$<TARGET_OBJECTS:opbase_util_objs>>
    $<$<TARGET_EXISTS:opbase_infer_objs>:$<TARGET_OBJECTS:opbase_infer_objs>>
    $<$<TARGET_EXISTS:opbase_tiling_objs>:$<TARGET_OBJECTS:opbase_tiling_objs>>)

  # Keep registry symbols available while resolving CANN runtime dependencies.
  target_link_libraries(${OPHOST_NAME} PRIVATE
    intf_pub_cxx17
    c_sec
    -Wl,--no-as-needed
    register
    -Wl,--as-needed
    -Wl,--whole-archive
    rt2_registry_static
    -Wl,--no-whole-archive
    tiling_api
    -Wl,-Bsymbolic
    runtime
    unified_dlog
    mmpa)
  # Resolve CANN runtime libraries from the active toolkit installation.
  target_link_directories(${OPHOST_NAME} PRIVATE ${ASCEND_DIR}/${SYSTEM_PREFIX}/lib64)
endfunction()

function(gen_cust_opapi_symbol)
  if(NOT TARGET ${OPHOST_NAME}_opapi_obj AND NOT TARGET opbuild_gen_aclnn_all)
    message(FATAL_ERROR "ACLNN package requested but no generated or hand-written API source exists")
  endif()

  npu_op_library(cust_opapi ACLNN)
  target_sources(cust_opapi PUBLIC
    $<$<TARGET_EXISTS:${OPHOST_NAME}_opapi_obj>:$<TARGET_OBJECTS:${OPHOST_NAME}_opapi_obj>>
    $<$<TARGET_EXISTS:opbuild_gen_aclnn_all>:$<TARGET_OBJECTS:opbuild_gen_aclnn_all>>
    $<$<TARGET_EXISTS:opbase_util_objs>:$<TARGET_OBJECTS:opbase_util_objs>>)
  target_link_libraries(cust_opapi PUBLIC intf_pub_cxx17)
endfunction()

function(gen_cust_optiling_symbol)
  if(NOT TARGET ${OPHOST_NAME}_tiling_obj)
    message(FATAL_ERROR "Kernel compilation requested but no Tiling source was registered")
  endif()

  npu_op_library(cust_opmaster TILING)
  target_sources(cust_opmaster PUBLIC
    $<TARGET_OBJECTS:${OPHOST_NAME}_tiling_obj>
    $<$<TARGET_EXISTS:opbase_util_objs>:$<TARGET_OBJECTS:opbase_util_objs>>
    $<$<TARGET_EXISTS:opbase_tiling_objs>:$<TARGET_OBJECTS:opbase_tiling_objs>>)
  target_link_libraries(cust_opmaster PUBLIC intf_pub_cxx17 unified_dlog)
endfunction()
