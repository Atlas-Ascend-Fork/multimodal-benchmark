# 编译骨架清单

骨架的发布闭包按目录维护：

| 目录 | 内容 | 是否参与评估构建 |
| --- | --- | --- |
| `cmake/` | CANN 查找、Host/API、Kernel 和打包规则 | 是 |
| `common/` | 公共编译接口与头文件 | 是 |
| `scripts/build*.sh` | 构建入口 | 是 |
| `scripts/util/insert_kernel_src.py` | Kernel 源码路径元数据 | 是 |
| `config/`、`scripts/import*.sh`、`scripts/build_arena_op.sh` | 算子资产维护工具 | 否 |
| 外部 opbase Git 检出目录 | 固定版本构建依赖；准备方法见 README | 是 |

算子源码属于 `bench_lab/ascend_arena_bench/<operator>/<operator>/`，不属于本清单。
`build/`、`build_out/` 和 `tbe/` 均为生成物。
