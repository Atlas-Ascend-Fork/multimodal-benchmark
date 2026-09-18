# NPU Profiling 数据解读指南

> 本文档帮助你理解 cann-bench 性能评测产出的 profiler 数据，知道每个文件是什么、怎么看。

## 目录结构

cann-bench 对每个 case 跑完性能评测后，在 `ASCEND_PROFILER_OUTPUT/` 目录下产出以下文件：

```
ASCEND_PROFILER_OUTPUT/
├── kernel_details.csv          ← 最重要：每个 kernel 的详细耗时
├── op_statistic.csv            ← 算子类型统计（按类型汇总耗时）
├── task_time.csv               ← 任务级耗时（和 kernel_details 类似但更底层）
├── api_statistic.csv           ← host 侧 API 调用统计
├── step_trace_time.csv         ← 整步耗时分解（计算/通信/空闲）
├── trace_view.json             ← 可视化时间线（可用 Chrome Trace Viewer 打开）
├── ascend_pytorch_profiler.db  ← SQLite 数据库（profiler 原始数据）
├── analysis.db                 ← SQLite 数据库（分析结果）
└── analyse.done                ← 标记文件（表示分析完成）
```

## 各文件详解

### 1. kernel_details.csv（最重要，优先看）

**大白话**：记录了你调一次算子时，NPU 芯片上实际跑了哪些计算任务、每个花了多久。

**打个比方**：你去餐厅点了一道"番茄炒蛋"（融合算子），厨房监控记录下来厨师做这道菜的每一步：洗番茄 3 秒、打蛋 2 秒、热锅 5 秒、炒菜 10 秒、装盘 2 秒。kernel_details.csv 就是这个监控记录。

**关键列**：

| 列名 | 含义 | 怎么看 |
|------|------|--------|
| Name | kernel 的名字 | 你写的 `@pl.jit` kernel 会有你起的函数名（如 `fcs_kernel_fcsfp16`）；PyTorch 操作产生的会带 `aclnn` 前缀（如 `aclnnConstantPadNd_PadV3`）；评测框架的清缓存叫 `CannBenchCacheClean` |
| Type | kernel 类型 | 你的 kernel 显示为函数名或 `PyPTO`；框架算子显示为算子名（如 `PadV3`、`Transpose`） |
| Duration(us) | 耗时（微秒） | 这个 kernel 在 NPU 上跑了多久。1μs = 百万分之一秒 |
| Accelerator Core | 跑在哪个核上 | `AI_CORE` = Cube 核（矩阵运算）、`AI_VECTOR_CORE` = Vector 核（逐元素运算）、`MIX_AIC` = Cube+Vector 混合、`AI_CPU` = 控制核 |
| Block Num | 用了几个核 | 比如 28 = 用了 28 个 AI Core 并行 |
| Wait Time(us) | 等待时间 | 这个 kernel 在排队等上一个 kernel 完成的时间。如果这个值很大，说明流水线有气泡 |
| OP State | 编译状态 | `static` = 静态编译（PyPTO Pro SPMD）、`dynamic` = 动态编译（PyPTO frontend MPMD） |

**怎么判断性能好不好**：
- 看你的 kernel 的 Duration 占总 Duration 的比例——越高越好
- 如果非你的 kernel（aclnn/Transpose/PadV3 等）耗时占比很高，说明预处理太重，需要优化
- Wait Time 大说明流水线没排好，kernel 之间在互相等
- 正常的融合算子：csv 里应该只有你的 kernel + CacheClean，没有其他东西

### 2. op_statistic.csv（算子类型统计）

**大白话**：把 kernel_details.csv 按类型分组汇总。比如你有 3 次 `fcs_kernel`，每次 7μs，这里就显示 `fcs_kernel: Count=3, Total=21μs, Avg=7μs`。

**关键列**：
- `OP Type`：算子类型名
- `Count`：执行了几次
- `Total Time(us)`：总耗时
- `Ratio(%)`：占全部算子耗时的百分比

**怎么看**：快速判断哪类算子最耗时。如果 `PyPTO` 类型的 Ratio 很低，说明你的 kernel 不是性能主体。

### 3. task_time.csv（任务级耗时）

**大白话**：和 kernel_details.csv 类似，但更底层——记录每个 NPU 任务的启动时间和结束时间。

**用途**：当 kernel_details.csv 不够详细时（比如想看精确的时间戳排列），可以用这个文件还原执行时间线。

### 4. api_statistic.csv（host 侧 API 统计）

**大白话**：记录 CPU（host）侧调了哪些 CANN 底层 API、调了几次、每次花多久。

**关键场景**：如果这里出现 `aclrtMemcpy`（内存拷贝），说明有数据在 CPU 和 NPU 之间来回搬运——这通常是性能杀手，也是反作弊检测项之一。

### 5. step_trace_time.csv（整步耗时分解）

**大白话**：把一次算子调用的总时间拆成几块：计算时间（Computing）、通信时间（Communication）、空闲时间（Free）。

**怎么看**：
- `Computing` 占比高 → 好，大部分时间在算
- `Free` 占比高 → 不好，NPU 在空转等数据或等同步

### 6. trace_view.json（可视化时间线）

**大白话**：一个可以用浏览器打开的时间线图。在 Chrome 里打开 `chrome://tracing`，把这个文件拖进去，就能看到每个 kernel 在时间线上的排列——谁先谁后、有没有重叠、哪里有空隙。

**用途**：直观看流水线效率。如果 kernel 之间有大段空白（气泡），说明同步/调度有问题。

### 7. ascend_pytorch_profiler.db + analysis.db

**大白话**：profiler 的原始数据库，上面的 csv 都是从这里导出的。一般不需要直接读——除非你要做自定义查询。

### 8. analyse.done

**大白话**：标记文件，表示 profiler 数据分析已完成。空文件，没有内容。

## 实战例子：判断 exp 算子性能

看 exp 算子的 kernel_details.csv：

```
CacheClean        140μs  （评测框架清缓存，不算分）
exp_kernel_fp16     5μs  （你的 kernel，算分）
CacheClean        130μs  （第2次清缓存）
exp_kernel_fp16     5μs  （第2次跑你的 kernel）
CacheClean        125μs  （第3次清缓存）
exp_kernel_fp16     5μs  （第3次跑你的 kernel）
```

干干净净——只有你的 kernel 和 CacheClean。这就是正确的融合算子应有的样子。

对比 fused_conv_sigmoid 的问题版本：

```
CacheClean        126μs
Transpose           3μs  ← host 侧 torch.transpose 产生的
ZerosLike           2μs  ← host 侧 torch.zeros_like 产生的
ZerosLike           1μs
ViewCopy            3μs  ← host 侧 tensor.view 产生的
ViewCopy            3μs
ViewCopy            3μs
ZerosLike           1μs
Fill                2μs  ← host 侧 tensor.fill_ 产生的
BroadcastTo         2μs
PadV3               3μs  ← host 侧 F.pad 产生的
PadV3               3μs
PadV3               3μs
fcs_kernel          6μs  ← 你的 kernel（终于轮到了）
Transpose           3μs
...（重复3次）
```

你的 kernel 只有 6μs，但前面排了 11 个预处理操作共 26μs——你的 kernel 占比只有 18%，剩下 82% 都是 PyTorch 操作。这就是反作弊触发的原因。
