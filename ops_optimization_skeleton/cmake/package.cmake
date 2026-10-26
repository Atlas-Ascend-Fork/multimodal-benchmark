# Build a custom-operator .run with the packaging implementation shipped in the
# installed CANN Toolkit. find_package(ASC) has already loaded npu_op_package(),
# CPack integration, install scripts, and a local makeself implementation.

function(init_custom_package result)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    set(ARCH x86_64)
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|arm")
    set(ARCH aarch64)
  else()
    message(FATAL_ERROR
      "Unsupported package architecture: ${CMAKE_SYSTEM_PROCESSOR}")
  endif()

  set(PACK_CUSTOM_NAME "cann-ops-${VENDOR_SUFFIX}-${VENDOR_NAME}_linux-${ARCH}")

  # TYPE RUN establishes CANN's custom OPP layout. --pkg additionally enables
  # the bundled CPack/makeself implementation.
  npu_op_package(${PACK_CUSTOM_NAME}
    TYPE RUN
    CONFIG
      ENABLE_SOURCE_PACKAGE True
      ENABLE_BINARY_PACKAGE True
      INSTALL_PATH ${CMAKE_INSTALL_PREFIX}
      VENDOR_NAME ${PATH_NAME}
      ENABLE_DEFAULT_PACKAGE_NAME_RULE False
      ENABLE_CPACK ${ENABLE_PACKAGE}
  )

  set(${result} ${PACK_CUSTOM_NAME} PARENT_SCOPE)
endfunction()

function(pack_custom package_name)
  npu_op_package_add(${package_name}
    LIBRARY cust_opmaster cust_opapi ops_skeleton_kernels)
endfunction()
