# 编译骨架源码来源

编译骨架基于 [ascend-arena-ops-skeleton 的固定快照](https://gitcode.com/qq_45366264/ascend-arena-ops-skeleton/tree/a3c4f059b712e34614a4ead3b7a70b054042eca5)，
并维护构建集成、错误处理及生成工具。各文件保留原有版权与许可证声明。

## CANN 构建工具

本地 CMake 集成、公共头文件和 `scripts/util/insert_kernel_src.py` 的直接来源
为上述骨架固定快照；文件保留原有版权与许可证声明，许可证全文见本目录 `LICENSE`。

Kernel 元数据生成、编译和安装包由环境中的 CANN Toolkit 提供，通过
`find_package(ASC)` 加载 `npu_op_code_gen()`、`npu_op_kernel_library()` 和
`npu_op_package()`。部署方须按 Toolkit 的许可及安装要求准备该依赖。

## opbase

外部依赖来自 [cann/opbase](https://gitcode.com/cann/opbase/tree/0c5e25790f67e2c6f60c1df36d74d28f0e08ecfc)，
固定提交为 `0c5e25790f67e2c6f60c1df36d74d28f0e08ecfc`。
依赖保留官方完整源码、Git 元数据及原始 `LICENSE`，构建时核验提交号和工作区状态。
依赖许可证为 CANN Open Software License Agreement Version 2.0；准备及离线传递方法见 `README.md`。
