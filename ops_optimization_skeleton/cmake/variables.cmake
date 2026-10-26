# Paths and cache values shared by the kernel-only CMake pipeline.
# Target names are kept stable because build_lib.sh invokes them explicitly.
set(COMMON_NAME common_${PKG_NAME})
set(OPHOST_NAME ophost_${PKG_NAME})
set(OPAPI_NAME opapi_${PKG_NAME})

# Preserve the CLI selection in the cache so a second configure sees the same
# operator list while CMake creates the generated binary targets.
set(NEED_COMPILE_OPS "${ASCEND_OP_NAME}" CACHE STRING "Operators to compile" FORCE)
set(COMPILED_OPS "" CACHE STRING "Operators discovered during configure" FORCE)
set(COMPILED_OP_DIRS "" CACHE STRING "Operator directories discovered during configure" FORCE)

# These lists are filled by operator CMakeLists and consumed by metadata scripts.
set(KERNEL_SRC_LIST "" CACHE INTERNAL "Kernel source list")
set(SIMPLIFIED_KEY_LIST "" CACHE INTERNAL "Kernel simplified-key list")
set(IMPL_MODE_LIST "" CACHE INTERNAL "Kernel implementation-mode list")
set(AUTO_SYNC_LIST "" CACHE INTERNAL "Kernel auto-sync list")
set(OPTION_LIST "" CACHE INTERNAL "Kernel option list")
set(DEPENDENCY_KERNEL_DIRS "" CACHE INTERNAL "Kernel-only dependency directories")
set(KERNEL_API_SOURCE_INFO "" CACHE INTERNAL "CANN kernel API source registrations")

# Tools used for ACLNN generation and kernel source metadata.
set(ASCEND_PYTHON_EXECUTABLE python3 CACHE STRING "Python used by build helpers")
set(OP_BUILD_TOOL "${ASCEND_DIR}/tools/opbuild/op_build")

# Generated opbuild/OPC inputs are kept separate from source and final binaries.
set(ASCEND_AUTOGEN_PATH "${CMAKE_BINARY_DIR}/autogen")
set(ASCEND_KERNEL_API_AUTOGEN_PATH "${CMAKE_BINARY_DIR}/kernel_autogen")
set(ASCEND_TBE_BUILD_PATH "${CMAKE_BINARY_DIR}/tbe")
set(ASCEND_KERNEL_SRC_DST "${ASCEND_TBE_BUILD_PATH}/ascendc")
set(ASCEND_KERNEL_CONF_DST "${ASCEND_TBE_BUILD_PATH}/config")
# Ensure every later custom command has its destination directory available.
file(MAKE_DIRECTORY "${ASCEND_AUTOGEN_PATH}" "${ASCEND_KERNEL_API_AUTOGEN_PATH}"
                    "${ASCEND_KERNEL_SRC_DST}" "${ASCEND_KERNEL_CONF_DST}")

set(CUSTOM_COMPILE_OPTIONS custom_compile_options.ini)
set(CUSTOM_OPC_OPTIONS custom_opc_options.ini)
# Start each configure with fresh generated option files; they are inputs, not
# source-controlled configuration.
file(REMOVE "${ASCEND_AUTOGEN_PATH}/${CUSTOM_COMPILE_OPTIONS}"
           "${ASCEND_AUTOGEN_PATH}/${CUSTOM_OPC_OPTIONS}")
file(TOUCH "${ASCEND_AUTOGEN_PATH}/${CUSTOM_COMPILE_OPTIONS}"
           "${ASCEND_AUTOGEN_PATH}/${CUSTOM_OPC_OPTIONS}")

# CPack stages this CANN custom-operator layout before makeself creates .run.
set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/build_out" CACHE PATH "Package output directory")
if(NOT VENDOR_SUFFIX)
  set(VENDOR_SUFFIX math)
endif()
set(PATH_NAME "${VENDOR_NAME}_${VENDOR_SUFFIX}")
set(OPS_INFO_INSTALL_DIR
    packages/vendors/${PATH_NAME}/op_impl/ai_core/tbe/config)
set(IMPL_INSTALL_DIR
    packages/vendors/${PATH_NAME}/op_impl/ai_core/tbe/${PATH_NAME}_impl/ascendc)
set(IMPL_DYNAMIC_INSTALL_DIR
    packages/vendors/${PATH_NAME}/op_impl/ai_core/tbe/${PATH_NAME}_impl/dynamic)
set(BIN_KERNEL_INSTALL_DIR
    packages/vendors/${PATH_NAME}/op_impl/ai_core/tbe/kernel)
set(BIN_KERNEL_CONFIG_INSTALL_DIR
    packages/vendors/${PATH_NAME}/op_impl/ai_core/tbe/kernel/config)
set(OPTILING_INSTALL_DIR
    packages/vendors/${PATH_NAME}/op_impl/ai_core/tbe/op_tiling)
set(ACLNN_INC_INSTALL_DIR
    packages/vendors/${PATH_NAME}/op_api/include)
set(ACLNN_LIB_INSTALL_DIR
    packages/vendors/${PATH_NAME}/op_api/lib)

set(OPAPI_INCLUDE
  ${C_SEC_INCLUDE}
  ${PLATFORM_INC_DIRS}
  ${METADEF_INCLUDE_DIRS}
  ${NNOPBASE_INCLUDE_DIRS}
  ${OPBASE_INC_DIRS}
  ${NPURUNTIME_INC_DIRS}
  ${ASCEND_DIR}/${SYSTEM_PREFIX}/pkg_inc/aicpu
  ${OPS_MATH_DIR}
  ${OPS_MATH_DIR}/common/inc
  ${OPS_MATH_DIR}/common/inc/op_api
  ${OPS_MATH_DIR}/common/inc/external
  ${ASCEND_AUTOGEN_PATH}
  ${ASCEND_AUTOGEN_PATH}/inner
)

set(OP_TILING_INCLUDE
  ${C_SEC_INCLUDE}
  ${PLATFORM_INC_DIRS}
  ${METADEF_INCLUDE_DIRS}
  ${TILINGAPI_INC_DIRS}
  ${NPURUNTIME_INC_DIRS}
  ${ASCEND_DIR}/${SYSTEM_PREFIX}/pkg_inc/op_common
  ${OPBASE_INC_DIRS}
  ${OPS_MATH_DIR}
  ${OPS_MATH_DIR}/common/inc
)

set(OP_PROTO_INCLUDE
  ${C_SEC_INCLUDE}
  ${METADEF_INCLUDE_DIRS}
  ${OPBASE_INC_DIRS}
  ${NPURUNTIME_INC_DIRS}
  ${OPS_MATH_DIR}
  ${OPS_MATH_DIR}/common/inc
)

set(ASCEND_ALL_COMPUTE_UNIT "${ASCEND_COMPUTE_UNIT}")
