# 算子优化编译骨架

此目录保存算子优化评测所用的共享编译逻辑，不包含具体算子实现。使用时将算子源码
复制到独立的编译目录中：

```text
<build-dir>/<operator>/
```

在 `<build-dir>` 执行：

```bash
bash build.sh --pkg --no_force \
  --soc=ascend910b \
  --ops=<operator> \
  --vendor_name=benchmark \
  --vendor_suffix=<family>
```

算子资产和编译系统可分别维护；评测时在临时编译目录中组装，不修改此目录。

## 目录职责

| 路径 | 作用 | 当前流程 |
| --- | --- | --- |
| `CMakeLists.txt`、`cmake/` | 发现注入的算子，生成 Host、ACLNN、Tiling、Kernel 和 `.run` | 必需 |
| `common/` | 提供骨架公共接口和头文件 | 必需 |
| `scripts/build*.sh` | 参数解析、CMake 配置和构建入口 | 必需 |
| `scripts/util/insert_kernel_src.py` | 为生成的 INI 写入选定 SoC 的 Kernel 源码路径 | 必需 |
| 外部 opbase Git 检出目录 | Host 推导、tiling 和日志公共实现 | 必需 |
| `config/`、导入脚本 | 从上游维护算子资产时使用 | 评估不调用 |

opbase 通过外部 Git 检出目录提供，使用固定提交
`0c5e25790f67e2c6f60c1df36d74d28f0e08ecfc`。构建配置会检查提交号及工作区状态，
要求工作区干净（包括未跟踪文件）。依赖目录应放在构建目录之外，供独立编译目录复用。

## 准备 opbase

在联网机器上准备固定版本：

```bash
git clone https://gitcode.com/cann/opbase.git /opt/cann-deps/opbase
git -C /opt/cann-deps/opbase checkout --detach 0c5e25790f67e2c6f60c1df36d74d28f0e08ecfc
export OPBASE_SOURCE_PATH=/opt/cann-deps/opbase
```

`OPBASE_SOURCE_PATH` 支持环境变量及 CMake `-DOPBASE_SOURCE_PATH=...`。
环境变量会传递给 `build.sh` 以及启动独立编译目录的评测进程。依赖保留其 Git 元数据、
完整源码及原始 `LICENSE`；许可证为 CANN Open Software License Agreement Version 2.0。

离线环境可用 Git bundle 传递完整版本：

```bash
# 联网机器，产物写在依赖目录之外
git -C /opt/cann-deps/opbase bundle create /tmp/opbase.bundle HEAD
(cd /tmp && sha256sum opbase.bundle > opbase.bundle.sha256)
# 将这两个文件复制至离线机器的 /tmp 后执行
(cd /tmp && sha256sum -c opbase.bundle.sha256)
git clone /tmp/opbase.bundle /opt/cann-deps/opbase
git -C /opt/cann-deps/opbase checkout --detach 0c5e25790f67e2c6f60c1df36d74d28f0e08ecfc
export OPBASE_SOURCE_PATH=/opt/cann-deps/opbase
```

CMake 配置不访问网络。未指定路径时，沿用相邻 `../../ops-base` 或
`${CANN_3RD_LIB_PATH}/opbase` 的查找方式，并执行相同版本校验。

`tbe/` 是构建产物目录，由 CMake 在 `build/tbe/` 下自动生成，源码树无需预建空目录。

## 构建与打包

Kernel 元数据生成、二进制编译和 `.run` 安装包由已安装的 CANN Toolkit
`npu_op_code_gen()`、`npu_op_kernel_library()` 和 `npu_op_package()` 提供。
本地工具负责算子发现、Host/ACLNN 集成和 Kernel 源码路径选择。

`--opkernel` 构建内核，`--pkg` 同时生成安装包。两种入口均使用动态 ACLNN
库和独立 Kernel 二进制；依赖和 Toolkit 就绪后，配置及构建无需联网。

## 外部依赖

- 已安装并加载环境的 CANN Toolkit 与对应芯片算子包。
- 按上述方式准备并设置路径的固定版本 opbase Git 检出目录。
- `cmake`、`make`、C++ 编译器和项目 Python 环境。

调用方负责准备算子源码、编译目录和评测流程；`build.sh` 负责构建及打包。
