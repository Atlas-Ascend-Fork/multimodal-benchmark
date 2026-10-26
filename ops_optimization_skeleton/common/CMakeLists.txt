# Interface targets used by the kernel-only opdef/tiling pipeline.
set(COMMON_NAME common_${PKG_NAME})
set(OPHOST_NAME ophost_${PKG_NAME})

add_library(${OPHOST_NAME}_opdef_aclnn_obj INTERFACE)
add_library(${OPHOST_NAME}_opdef_aclnn_inner_obj INTERFACE)
add_library(${OPHOST_NAME}_opdef_aclnn_exclude_obj INTERFACE)
add_library(${OPHOST_NAME}_aclnn_exclude_headers INTERFACE)
