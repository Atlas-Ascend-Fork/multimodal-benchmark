# cann-bench 反作弊触发条件表

cann-bench 评测框架包含多重反作弊检测机制，**触发任一条件都会导致整算子性能得分清零（0分）**。

## 触发条件一览

| 编号 | score_error_code | 触发条件 | 根因 | 怎么避免 |
|------|-----------------|---------|------|---------|
| A1 | `no_npu_kernel_detected` | profiler 的 kernel_details.csv 中没有提交者自己的 NPU kernel | 算子用 torch/aclnn 现成算子拼接，没有 `@pl.jit` 自定义 kernel | **必须用 `@pl.jit` 编写核心计算** |
| A2 | `no_npu_kernel_detected` | profiler 有 csv 但全是 aclnn 辅助算子（PadV3、Conv2D 等） | 核心计算逻辑不在自定义 kernel 内，被 torch 算子替代 | 核心计算（如 conv+sigmoid 融合）必须在 `@pl.jit` kernel 内实现 |
| A3 | `no_npu_kernel_detected` | editable install（`pip install -e .`）在评测子进程中加载失败 | 评测框架用 subprocess 启动子进程，editable finder 不生效 | 使用 `pip install . --force-reinstall --no-deps`（非 editable） |
| A4 | `no_npu_kernel_detected` | `impl/build/` 缓存目录干扰了包加载，子进程加载到旧版本 | editable install 的 build/lib/ 副本覆盖了真实代码 | 每次安装前清理 `impl/build/`、`impl/dist/`、`*.egg-info/` |
| A5 | `no_npu_kernel_detected` | PYTHONPATH 中旧版 pypto（0.2.1）优先于新版，`pypto_pro` 导入失败 | 评测子进程的 PYTHONPATH 没有把 `/usr/local/Ascend/cann/python/site-packages` 放在最前面 | 确保 PYTHONPATH 新版路径在前 |
| A6 | （case 级 failed） | `module 'pypto.pypto_impl' has no attribute 'codegen'` | 同 A5，旧版 pypto 没有 codegen 模块 | 同 A5 |
| A7 | （case 级 failed） | `无法找到算子 XXX` | `import cann_bench` 后 `getattr(cann_bench, func_name)` 失败 | 算子函数名必须是 proto.yaml 中 `schema` 字段解析出的名字（如 `fused_conv_sigmoid`） |
| A8 | （case 级 failed） | `aclnnXXX not found in libcust_opapi.so` | 代码中走了 aclnn 自定义算子路径但没有编译注册 | 不要混用 aclnn 路径，统一用 pypto_pro JIT |
| A9 | （case 级 failed） | 输出 tensor non-contiguous | 算子返回了 view/transpose 产生的非连续 tensor | 返回前调用 `.contiguous()` |
| A10 | `no_npu_kernel_detected` | 有自定义 `@pl.jit` kernel 但 profiler csv 中还有大量非自定义算子 | host 侧用了 PyTorch 算子做预处理（如 F.pad/transpose/zeros），这些操作到 NPU 上变成额外的 kernel 调用，自定义 kernel 只占少数（如 3/36），cann-bench 判定为"未检测到 NPU 算子" | profiler csv 中除了自定义 `@pl.jit` kernel 和评测框架的 CacheClean 之外，不应该有任何其他算子。所有 host 侧预处理必须全部移入 `@pl.jit` kernel 内部用 pypto_pro tile API 完成。禁止用缓存/sleep 等绕道方案 |

## 自检清单（stage2/3 开发时逐项确认）

1. ✅ 核心计算在 `@pl.jit` kernel 内（不是 torch 算子拼接）
2. ✅ `import cann_bench; cann_bench.<func_name>` 能找到（函数名和 proto.yaml schema 一致）
3. ✅ 用 `pip install .`（非 editable）安装
4. ✅ 安装前清理了 `impl/build/`、`dist/`、`*.egg-info/`
5. ✅ 输出 tensor 是 contiguous 的
6. ✅ 没有调用 `torch.nn.functional.*` 或 `torch.ops.aten.*` 做核心计算
7. ✅ NPU 设备识别通过（自测 TC3）
8. ✅ host 侧 aclnn 辅助算子数量不超过自定义 kernel 数量（防止 A10 触发）
