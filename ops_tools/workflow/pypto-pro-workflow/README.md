# PyPTO Pro Workflow

> 多智能体协作的 PyPTO Pro 算子自动开发与迭代优化框架

## 📖 项目简介

PyPTO Pro Workflow 是一个面向昇腾 NPU 的**多智能体算子自动开发框架**，基于 PyPTO Pro 编程范式，通过 3 个 AI Agent（CANNBot 写代码、Kerminal 编译评测、Hermes 搜索优化）和 1 个确定性控制平面（orchestrator）协同工作，实现算子从需求分析→代码编写→编译部署→精度验证→性能优化→报告生成的**全流程自动化迭代**。

| 维度 | 说明 |
|------|------|
| 目标框架 | PyPTO Pro（SPMD 模式，昇腾达芬奇架构） |
| 评测工具 | cann-bench（精度+性能评测） |
| Agent | CANNBot（GLM-5.3-Flash）、Kerminal（kernelcat1.0）、Hermes（GLM-5.3-Flash） |
| 迭代阶段 | 10 个 stage，最多 20 轮自动迭代 |
| 核心特性 | 跨轮记忆（history）、Tech Lead 方向指导、反作弊知识库、代码版本追溯 |

## ✨ 核心特性

- **10 阶段流水线**：需求分析→首版实现→编译部署→精度评测→性能评测→Profiling→搜索优化→Tech Lead 总结→代码修改→最终报告
- **跨轮记忆（history 机制）**：Tech Lead 每轮提炼 insights/ledger，追踪最慢 case，指导下一轮方向，避免重复失败
- **融合方向追踪**：`fusion_kernel_strategy` 累加字段记录每轮融合方案的尝试（真融合/假融合/片上直传 vs HBM 中间读写），stage9 填写，stage3 读取
- **动态退出条件**：性能达标后不直接退出，Tech Lead 验证融合方案是否"真融合"后才允许退出
- **已有实现迭代模式**：`--init-impl` 从已有基线代码开始优化，`--optimize-hint` 注入优化方向，不影响正常新写流程
- **反作弊知识库**：10 条触发条件（A1-A10）+ 8 条自检清单，kernel_csv 自动分析
- **芯片自动检测**：从 CANN 编译器 tbe API 直接读取 UB/L1/L0A/L0B/L0C/核数真实值，注入所有 stage，禁止写死 tiling 参数
- **代码版本追溯**：每轮迭代前自动备份 impl 到 `operator_iter/`
- **性能评测代码直执**：stage6 由 orchestrator 直接 subprocess 调用 cann-bench，不经 agent，自动重试+旧报告过滤

## 🧭 适用场景

- 昇腾 NPU 上的 PyPTO Pro 融合算子开发与性能优化
- cann-bench 评测驱动的算子迭代优化
- 多 Agent 协作的算子开发流程自动化研究

## 🔧 芯片自动检测

orchestrator 启动时自动检测当前 NPU 芯片的真实硬件参数，写入 `<work>/device_info.json`，并注入到所有 stage 的 prompt 中，确保 agent 按实际硬件规格设计 tiling 参数。**不使用任何硬编码映射表**——所有参数从设备实际读取，读不到则退出。

检测流程：

1. 通过 `torch_npu.npu.get_device_properties(0)` 获取 SoC 型号（如 `Ascend950PR_9579`）、Cube/Vector 核数、L2 Cache 大小
2. 通过 CANN 编译器内部接口 `tbe.common.platform.get_soc_spec` 读取片上内存真实值（UB/L1/L0A/L0B/L0C/CORE_NUM），这是编译器编译 kernel 时依赖的同一份数据
3. 根据 SoC 型号补充芯片系列名和架构代号（如 `Ascend 950` / `dav-3510`，纯名称映射不涉及硬件参数）
4. 校验关键参数（soc_version/UB/L1/CORE_NUM/L0A）全部获取到，任一缺失则打印排查步骤并 `sys.exit(1)` 退出

手动验证命令：
```bash
# 查看 SoC 型号
python3 -c "import torch,torch_npu; print(torch_npu.npu.get_device_properties(0).name)"

# 查看片上内存参数（需要 CANN Toolkit）
python3 -c "
from tbe.common.platform import get_soc_spec, set_current_compile_soc_info
set_current_compile_soc_info('Ascend950PR_9579')  # 替换为你的 SoC 型号
for k in ['UB_SIZE','L1_SIZE','L0A_SIZE','L0B_SIZE','L0C_SIZE','CORE_NUM']:
    v = get_soc_spec(k)
    print(f'{k} = {v} ({v//1024} KB)' if isinstance(v,int) and v>1024 else f'{k} = {v}')
"
```

如果自动检测失败（如 CANN 未安装完整），可在 `config.yaml` 的 `hardware` 段手动填写全部参数。

## 启动

```bash
# 正常模式：从零开始写算子
source ~/.bashrc
cd /workspace/workflow_mutilagent_operator_develop/pypto-pro-workflow
python orchestrator.py --task-dir /path/to/cann-bench/tasks/level1/exp --max-iter 10

# 新写算子 + 指定融合方向（stage1/2 会按此方向分析和实现）
python orchestrator.py --task-dir /path/to/task \
  --optimize-hint "Conv+Sigmoid融合：单kernel内Cube算完conv后L0C→UB直传做sigmoid，零HBM中间读写" \
  --max-iter 20

# 已有实现迭代模式：从基线代码开始优化
python orchestrator.py --task-dir /path/to/task \
  --init-impl /path/to/existing/impl \
  --optimize-hint "当前是半融合(K1写HBM再K2读)，改为L0C→UB直传真融合" \
  --max-iter 20

# 断点续跑
python orchestrator.py --task-dir /path/to/task --work-dir /path/to/work/dir
```

### --optimize-hint 方向提示词

`--optimize-hint` 在新写和续写两种场景下含义不同：

| 场景 | 命令 | stage1 注入 | stage2 注入 |
|------|------|-----------|-----------|
| 新写算子（无 --init-impl） | `--optimize-hint "融合方向..."` | 💡 融合方向提示 + 请分析可行性 | 💡 请按此方向设计数据流 |
| 续写已有实现 | `--init-impl ... --optimize-hint "修改方向..."` | ⚡ 已有基线 + 修改方向 | ⚡ 请按此方向修改，不从零重写 |
| 不传 --optimize-hint | — | 无额外提示 | 无额外提示（行为不变） |

### --init-impl 模式说明

当提供 `--init-impl` 时，orchestrator 将已有算子实现复制到新 work/impl/ 目录，然后：
- stage1：分析需求时注入修改方向提示词
- stage2：变为"首次修改"模式（不从零写，在基线上改）
- 后续迭代正常进行，history 从零开始累积
- `--init-impl` 和 `--work-dir` 不能同时使用（前者新建 work，后者断点续跑）

不传 `--init-impl` 时行为完全不变。

### 退出条件（动态判断）

性能达标（所有 case speedup ≥ 1.0）后**不直接退出**，而是：
1. orchestrator 调 stage9 Tech Lead，注入"性能达标，请验证融合方案"
2. Tech Lead 检查实际代码+设计文档+kernel_csv，判断融合是否"真融合"
3. `exit_decision=true`（真融合）→ 退出迭代 | `exit_decision=false`（假融合）→ 继续优化融合方案
4. `max_iterations` 硬上限兜底

## 项目结构

```
pypto-pro-workflow/
├── orchestrator.py          ← 主控制平面（唯一入口）
├── config.yaml              ← 统一配置（agent CLI 路径、迭代参数、cann-bench 路径）
├── diff_geak.md             ← GEAK 框架对比分析
├── 测试用例.csv              ← 测试用例记录
├── roles/                   ← 角色 prompt（每个阶段一个 md）
│   ├── n1_stage1_requirements_analysis.md   ← 阶段1 需求分析（cannbot）
│   ├── n1_stage2_first_impl.md              ← 阶段2 首版实现（cannbot）
│   ├── n1_stage3_fix_and_optimize.md        ← 阶段3 修改优化（cannbot）
│   ├── n2_stage4_build.md                   ← 阶段4 编译部署（kerminal）
│   ├── n2_stage5_precision_eval.md          ← 阶段5 精度评测（kerminal）
│   ├── n2_stage6_perf_eval.md               ← 阶段6 性能评测（kerminal）
│   ├── n2_stage7_kerminal_profile.md        ← 阶段7 profiling（kerminal）
│   ├── n3_stage8_search.md                  ← 阶段8 搜索优化（hermes）
│   ├── n4_stage9_tech_lead_guide.md         ← 阶段9 经验提炼（kerminal/tech_lead）
│   └── n5_stage10_kerminal_report.md        ← 阶段10 最终报告（kerminal）
├── lib/                     ← 工具库
│   ├── agent_runner.py      ← agent CLI 调用 + PyPTO Pro 环境注入（PYTHONPATH/LD_LIBRARY_PATH）
│   ├── bench_parser.py      ← cann-bench 评测结果解析 + 反作弊 kernel_csv 自动分析
│   ├── history_manager.py   ← 跨轮记忆管理（history/ledger/fix_plan/rounds 保护）
│   ├── state.py             ← .state.json 状态管理
│   ├── logger.py            ← 日志系统（workflow/state/history/N1-N5 共 8 个日志）
│   └── handoff.py           ← 文件读写工具
├── knowledge/               ← 知识库（不对 agent 直接展示，通过 orchestrator 注入）
│   ├── anti_cheat_reference.md  ← 反作弊触发条件表（A1-A10）+ 自检清单
│   └── profiling_guide.md       ← Profiling 数据解读指南
├── docs/                    ← 文档
│   └── pypto_build_guide.md     ← PyPTO 源码编译安装教程
├── examples/                ← 算子实现示例（含代码 + 迭代报告）
│   ├── fused_conv_tanh/         ← 融合卷积+Tanh 算子
│   │   ├── impl/                    ← 算子代码（setup.py + cann_bench/）
│   │   ├── ITER_REPORT.html         ← 迭代优化报告（可视化）
│   │   ├── ANALYSIS.md              ← 需求分析
│   │   └── FINAL_REPORT.md          ← 最终评测报告
│   ├── fused_conv_sigmoid_v2/   ← 融合卷积+Sigmoid 算子（14 轮迭代版）
│   │   ├── impl/                    ← 算子代码
│   │   ├── ITER_REPORT.html         ← 迭代优化报告
│   │   ├── ANALYSIS.md              ← 需求分析
│   │   └── WORK_RECORD.md           ← 工作留痕
│   └── fused_conv_sigmoid_v1/   ← 融合卷积+Sigmoid 算子（9 轮迭代版）
│       ├── impl/                    ← 算子代码
│       ├── ITER_REPORT.html         ← 迭代优化报告
│       ├── ANALYSIS.md              ← 需求分析
│       ├── FINAL_REPORT.md          ← 最终评测报告
│       └── WORK_RECORD.md           ← 工作留痕
└── work/                    ← 运行时工作目录（自动生成，.gitignore 已排除）
```

## 运行时 work 目录

每次运行自动创建 `work/<op>_<timestamp>/`：

```
work/exp_20260828_232950/
├── task/                  ← 软链接 → cann-bench task 目录（只读）
├── example/               ← 软链接 → pypto_pro_example（只读）
├── impl/                  ← 全局：当前最新算子代码
├── ANALYSIS.md            ← 全局：阶段1 需求分析
├── WORK_RECORD.md         ← 全局：工作留痕
├── FINAL_REPORT.md        ← 全局：阶段9 最终报告
├── .state.json            ← 全局：状态机（iteration/stage/history）
├── knowledge/
│   ├── history.json           ← 跨轮记忆（insights/ledger/rounds/suggest_next/fusion_kernel_strategy）
│   ├── proven_patterns.md     ← 成功经验（性能提升≥5%时自动记录，stage9填内容+程序填数字）
│   ├── regression_patterns.md ← 失败教训（性能退步≥5%时自动记录，同上）
│   ├── tech_lead_pitfalls.md  ← 决策错题本（cannbot反馈→tech_lead裁定，✅误判/❌驳回）
│   ├── _rounds_snapshot.json  ← rounds 保护快照（stage9 前备份，防模型篡改）
│   └── _fusion_snapshot.json  ← fusion_kernel_strategy 保护快照（同上）
├── operator_iter/             ← 每轮 impl 备份（代码追溯用，不对 agent 展示）
│   ├── iter1/
│   └── iter{N}/
├── develop/
│   ├── iter0/
│   │   ├── design_rationale.md     ← 阶段2 首版设计思路
│   │   └── self_test_report.md     ← 阶段2 自测报告（部署+import+功能验证）
│   ├── iter1/
│   │   ├── design_rationale.md     ← 阶段3 第1轮优化设计思路
│   │   └── self_test_report.md     ← 阶段3 第1轮自测报告
│   └── iter{N}/...
├── build/
│   ├── iter1/build.log
│   └── iter2/build.log
├── eval/
│   ├── iter1/
│   │   ├── precision_result.json   ← 精度判定
│   │   ├── precision_reports/      ← 软链接到 cann-bench 精度报告
│   │   ├── perf_result.json        ← 性能判定
│   │   ├── perf_reports/           ← 软链接到 cann-bench 性能报告
│   │   └── prof_data/              ← 深拷贝：20 个 case 的 profiler 数据
│   │       ├── 1/...kernel_details.csv
│   │       ├── 2/...
│   │       └── 20/...
│   └── iter2/
│       ├── ...（同上）
│       └── prof_data/              ← 每轮独立快照，不被下一轮覆盖
├── profile/
│   ├── iter1/bottleneck_analysis.md
│   └── iter2/bottleneck_analysis.md
├── search/
│   ├── iter1/
│   │   ├── SEARCH_REPORT.md
│   │   └── FIX_DIRECTIVE.md
│   └── iter2/...
└── log/
    ├── workflow.log             ← 全局日志
    ├── state_transitions.log   ← 状态切换专用
    ├── history.log             ← history 变化日志（每轮完整 JSON 快照）
    ├── N1.log                  ← cannbot 节点日志
    ├── N2.log                  ← kerminal 节点日志
    ├── N3.log                  ← hermes 节点日志
    ├── N4.log                  ← tech_lead 节点日志
    └── N5.log                  ← 报告节点日志
```

说明：
- `develop/iter0/` 是阶段2 首版实现的设计思路和自测报告，`iter{N}/` 是阶段3 每轮优化后的设计思路和自测报告
- `eval/iter{N}/prof_data/` 是每轮性能评测后从 cann-bench 深拷贝的 profiler 原始数据，保证历轮数据不丢失
- 阶段7 读取的 design_rationale 是上一轮的（fallback 查找最近存在的，处理 build_fail 轮无产出的情况）
- `build_fail`/`precision_fail` 只触发 bug 修复，不产出 design_rationale（设计未变）

## 一次迭代流程（以 exp 算子为例）

```
启动 → 阶段1(cannbot分析task) → 阶段2(cannbot写PyPTO Pro代码)
     → 进入迭代循环：
       阶段4: kerminal 执行 pip install . --force-reinstall 并验证 import
              → FAILED → stage9(tech_lead总结) → 阶段3(cannbot修复) → 重回阶段4
              → SUCCESS ↓
       阶段5: kerminal 跑 cann-bench --no-perf 精度评测
              → precision_fail → stage9(tech_lead总结) → 阶段3(cannbot修精度) → 重回阶段4
              → true ↓
       阶段6: orchestrator 直接执行 cann-bench 性能评测（不经 agent）
              → 反作弊零分 → stage9(tech_lead反作弊分析) → 阶段3 → 重回阶段4
              → 所有case speedup≥1.0 → stage9(tech_lead验证融合方案)
                   → exit_decision=true（真融合）→ 阶段10(报告) → 结束
                   → exit_decision=false（假融合）→ 阶段3(改进融合) → 重回阶段4
              → 有case<1.0 ↓
       阶段7: kerminal 分析 profiling 数据，定位瓶颈
       阶段8: hermes 搜索优化方案，输出 FIX_DIRECTIVE.md
       阶段9: kerminal(tech_lead) 提炼经验 + 融合方向追踪，更新 history.json
       阶段3: cannbot 按 FIX_DIRECTIVE + insights + fusion_kernel_strategy 修改代码
     → 回到阶段4，下一轮迭代
```

**实际运行示例**（exp 算子，10 轮，最近一次完整运行）：

| 迭代 | 精度 | overall_score | performance_score | avg_speedup | 状态 |
|------|------|--------------|-------------------|-------------|------|
| iter1 | ✅ | 61.41 | 11.41 | 0.49 | 初版实现 |
| iter2 | ✅ | 61.84 | 11.84 | 0.51 | 小幅提升 |
| iter3 | ✅ | 60.87 | 10.87 | 0.47 | 性能退化 |
| iter4 | ✅ | 60.79 | 10.79 | 0.47 | 继续退化 |
| iter5 | ✅ | 61.79 | 11.79 | 0.50 | 回升 |
| iter6 | ✅ | 61.11 | 11.11 | 0.47 | 波动 |
| iter7 | ✅ | 61.66 | 11.66 | 0.50 | 波动 |
| iter8 | ✅ | 61.03 | 11.03 | 0.47 | 波动 |
| iter9 | ✅ | 62.02 | 12.02 | 0.52 | 最佳轮次 |
| iter10 | ✅ | 62.00 | 12.00 | 0.52 | max_iterations 退出 |

> 性能在 0.47~0.52 震荡，根因：无跨轮记忆，每轮 agent 不知道之前试过什么。已实现 GEAK 风格的 insights/ledger 机制（见下方）。

## 跨轮记忆（history 机制）

每轮迭代结束后，阶段9 tech_lead 分析本轮结果 + 所有历轮 design_rationale，更新 `knowledge/history.json`。下一轮阶段3 cannbot 的 prompt 中注入 history 摘要，避免重复失败方向。

**history.json 7 个字段**：

| 字段 | 谁写 | 更新方式 | 给谁看 |
|------|------|---------|--------|
| `rounds` | 程序（stage6后自动） | 每轮追加 | tech_lead |
| `ledger` | 程序写硬数据 + tech_lead 补方向名 | 每轮追加 | tech_lead + cannbot |
| `insights` | tech_lead（阶段9） | 每轮整体重写 | cannbot（最关键） |
| `bottleneck_now` | tech_lead（阶段9） | 每轮替换 | cannbot |
| `suggest_next` | tech_lead（阶段9） | 每轮替换（**P0/P1/P2 分优先级列表**） | cannbot（最优先执行） |
| `worst_cases_tracker` | tech_lead（阶段9） | 每轮替换 | cannbot |
| `fusion_kernel_strategy` | tech_lead（阶段9） | **每轮累加追加**，不覆盖旧条目 | cannbot + tech_lead |

**suggest_next 示例**（分优先级，P0 内部排序：融合方向 > 屡教不改 > 其他）：
```json
"suggest_next": [
  {"priority": "P0", "action": "把wsT从HBM改为L0C→UB片上直传", "reason": "融合天花板"},
  {"priority": "P0", "action": "（上轮建议未执行）去掉K2转置", "reason": "屡教不改"},
  {"priority": "P1", "action": "tile 64→128", "reason": "减少launch"},
  {"priority": "P2", "action": "双缓冲", "reason": "DMA有空间"}
]
```
cannbot 收到的渲染效果：`🔴[P0]` > `🟡[P1]` > `🟢[P2]`

**fusion_kernel_strategy 示例**（累加追踪融合方案演变）：
```json
"fusion_kernel_strategy": [
  {"iter": 1, "direction": "K1写HBM再K2读：半融合", "evidence": "csv有两个kernel", "status": "❌无效"},
  {"iter": 3, "direction": "L0C→UB直传：真融合", "evidence": "csv只有1个kernel", "status": "✅有效"}
]
```

**cannbot 阶段3 收到的历史经验（按优先级排列）**：
1. `★ 本轮修改指令（P0/P1/P2）` — 🔴融合方向最高优先 > 🔴屡教不改 > 🟡应该做 > 🟢锦上添花
2. `历史经验知识库(insights)` — ❌已否决的方向禁止再试
3. `当前性能瓶颈` — 优化要针对这个瓶颈
4. `融合算子策略追踪` — ❌无效的融合方式不要重复，✅有效的要保持
5. `已验证的成功经验` — 延续有效方向（knowledge/proven_patterns.md）
6. `已验证的失败教训` — 绝对不要重复（knowledge/regression_patterns.md）
7. `决策错题本` — tech_lead 历轮误判，写代码时避开（knowledge/tech_lead_pitfalls.md）
8. `最慢case追踪` — 硬件限制的 case 不浪费时间
9. `假设追踪账本(ledger)` — regression/no_change 的方向不要重复
10. `性能趋势(rounds)` — 每轮 avg_speedup 变化

## 知识积累（proven_patterns / regression_patterns）

程序在 stage6 后自动计算性能 diff（本轮 vs 上一轮有效数据），根据变化幅度触发知识积累：

| 条件 | 触发 | 文件 | 谁写内容 | 谁写数字 |
|------|------|------|---------|---------|
| avg_speedup 提升 ≥5% | 记录成功经验 | `knowledge/proven_patterns.md` | stage9 tech_lead 填 what_changed / why_it_worked | 程序填 speedup_before/after/delta_pct |
| avg_speedup 退步 ≤-5% | 记录失败教训 | `knowledge/regression_patterns.md` | stage9 tech_lead 填 what_changed / why_it_failed | 程序填 speedup_before/after/delta_pct |
| -5% < delta < +5% | 不记录 | — | — | — |

流程：程序先判断有没有提升/退步 → 注入 diff 数据（逐 case 对比 + 两轮 design_rationale 路径）到 stage9 prompt → tech_lead 对着数据总结"为什么有效/退步" → 程序写入 md 文件。

这两个 knowledge 文件注入到 **stage2/3/7/8/9** 的 prompt 中，所有写代码和分析的角色都能看到。

## 下级反馈 + 错题本（question.md / tech_lead_pitfalls.md）

一个"下级反馈 → 上级裁定 → 沉淀 → 反哺"的闭环，防止 tech_lead 反复给出错误建议。

**触发场景**：stage3 cannbot 实施 tech_lead 建议时，发现某条建议在硬件/框架层面确实不可行（有硬证据）。

**完整闭环**：
```
iter N:  stage9(给建议) → stage3(实施, 发现建议有误)
                              ↓ 严格判断+硬证据
                            写 develop/iterN/question.md
iter N+1: stage9 读 iterN/question.md + 对应 history 意见
            ↓ 裁定
            ├─ ✅confirmed(确认误判) → 承认错误, 给正确做法
            └─ ❌rejected(驳回)      → cannbot 理解错了, 说明正确认知
          程序: 裁定结论写回 question.md + append 到错题本
          之后 stage9 提意见前必读错题本, 不再重犯
```

**严格把关**（防止 cannbot 拿 question 偷懒）：
- 必须真的尝试过 + 有硬证据（编译错误/日志/profiler/官方文档限制）
- 只能是客观不可行，不是"我觉得没必要"
- 提 question 前先查错题本，已收录的不重复提
- 无论建议对错，**仍需用替代方案完成本轮任务**——question 是附带反馈，不是拒绝执行

**错题本文件** `knowledge/tech_lead_pitfalls.md`：
| verdict | 含义 | 记录内容 |
|---------|------|---------|
| ✅confirmed | tech_lead 确实建议错了 | 误判原因 + 正确做法 |
| ❌rejected | cannbot 理解错了 | 驳回原因 + 正确认知 |

- 谁写：tech_lead 裁定后输出 `pitfall` 字段 → 程序 append（累加不覆盖）
- 注入范围：**stage2/3/7/9**（写代码的、分析的、决策的都能看到），文件不存在时返回空不影响流程
- 兜底：如果迭代提前退出导致 question.md 从未被裁定，stage10 会扫描并在报告中提示

## stage6 后路由判断

stage6 性能评测完成后，程序按以下顺序做 3 个判断（**顺序不能调换**）：

```
stage6 完成 → 程序写入 rounds/ledger → 程序算 perf_diff
  ↓
  ① 反作弊？(score_error_code 存在 或 avg_speedup=0)
  │  YES → stage9(反作弊分析) → stage3 → 回 stage4
  │         日志: [判断] 反作弊触发: no_npu_kernel_detected
  │
  ② 性能达标？(所有 case speedup ≥ 1.0)
  │  YES → stage9(融合验证: exit_decision)
  │         ├─ exit_decision=true  → stage10(报告) → 结束
  │         │  日志: [判断] Tech Lead exit_decision=True: 真融合 → 退出
  │         └─ exit_decision=false → stage3(fusion_improve) → 回 stage4
  │            日志: [判断] Tech Lead exit_decision=False: 假融合 → 继续
  │
  ③ 正常性能不达标
     → stage7(profiling) → stage8(搜索) → stage9(perf_optimize) → stage3 → 回 stage4
       日志: [判断] 性能不达标: avg_speedup=0.5, perf_pass=False
```

perf_diff 数据（程序算一次，stage7/8/9 共享）：
- 提升 ≥5%：注入 `📊 性能提升检测` + 逐 case 对比 + "请总结成功经验"
- 退步 ≤-5%：注入 `⚠️ 性能退步检测` + 逐 case 对比 + "请分析退步原因"
- ±5% 以内：不注入 diff 文本

## state_transitions.log 日志格式

每条状态切换带原因，关键判断点带 `[判断]` / `[知识积累]` 标签：

```
iter1_stage4 → iter1_stage5 | 编译成功，开始精度评测
[判断] 精度失败: passed=18/24
iter1_stage5 → iter1_stage9 | tech_lead经验总结(precision_fail)
iter1_stage9 → iter1_stage3 | 回退修改代码(precision_fail)

iter2_stage6 后:
[判断] 性能提升: avg_speedup 0.5→1.5 (+200%), iter1→iter2
[判断] 性能不达标: avg_speedup=1.5, perf_pass=False → stage7→8→9→3
[知识积累] 成功经验写入 proven_patterns.md: +200%

iter5_stage6 后:
[判断] 性能退步: avg_speedup 3.0→2.5 (-16.7%), iter4→iter5
[知识积累] 退步教训写入 regression_patterns.md: -16.7%

iter9_stage6 后:
[判断] 性能达标: avg_speedup=4.5, 所有case speedup≥1.0 → 进入融合验证
[判断] Tech Lead exit_decision=True: 真融合+性能达标 → 退出迭代
```
tail -10 work/exp_xxx/log/N4.log   # tech_lead
```

## 三个 Agent

| Agent | 节点 | 模型 | CLI 路径 | 配置文件 |
|-------|------|------|---------|---------|
| cannbot | N1（写代码） | GLM-5.3-Flash | `/home/developer/.nvm/versions/node/v18.19.1/bin/cannbot` | `/root/.config/opencode/opencode.jsonc` |
| kerminal | N2（编译评测）| kernelcat1.0 | `/root/.local/bin/kerminal` | `/root/.kerminal/config.toml` |
| hermes | N3（搜索） | GLM-5.3-Flash | `/usr/local/bin/hermes` → `~/.hermes/venvs/hermes-dev/bin/hermes` | `~/.hermes/config.yaml` + `~/.hermes/.env` |

## 新环境安装指南

### 1. cannbot（Node.js）

```bash
# 前提：Node.js >= 18
npm install -g cannbot @cannbot-ai/install-helper

# 验证
cannbot --version   # 应显示 1.1.2+

# 配置 API key
mkdir -p ~/.config/opencode
cat > ~/.config/opencode/opencode.jsonc << 'EOF'
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "glm": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "GLM (ZhipuAI)",
      "options": {
        "baseURL": "https://open.bigmodel.cn/api/paas/v4",
        "apiKey": "<你的智谱API key>"
      },
      "models": {
        "glm-5.3-flash": { "name": "GLM-5.3-Flash" }
      }
    }
  },
  "model": "glm/glm-5.3-flash"
}
EOF
```

### 2. kerminal（二进制）

```bash
# 安装（从官方安装脚本，会下载到 ~/.local/bin/kerminal）
curl -fsSL https://kerminal.cn/install.sh | bash

# 验证
kerminal --version

# 配置
mkdir -p ~/.kerminal
cat > ~/.kerminal/config.toml << 'EOF'
model = "kernelcat1.0"
model_provider = "autokernel"
thinking_enabled = true
thinking_budget_tokens = 10000
show_thinking = true
show_raw_agent_reasoning = true

[features]
skills = true

[model_providers.autokernel]
name = "autokernel"
wire_api = "messages"
experimental_bearer_token = "<你的 autokernel token>"
EOF
```

### 3. hermes（Python venv）

```bash
# 安装（官方一键脚本，自带 Python 3.11 venv）
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash

# 或从源码安装（如果有 hermes-agent-main/）
cd hermes-agent-main && pip install -e .

# 验证
hermes status

# 如果 /usr/local/bin/hermes 报 ModuleNotFoundError，做软链接：
ln -sf ~/.hermes/venvs/hermes-dev/bin/hermes /usr/local/bin/hermes

# 配置模型
cat > ~/.hermes/config.yaml << 'EOF'
model:
  default: glm/glm-5.3-flash
EOF

# 配置 API key + 搜索引擎
cat > ~/.hermes/.env << 'EOF'
GLM_API_KEY=<你的智谱API key>
GLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4
TAVILY_API_KEY=<你的tavily key，可选，没有则用免费ddgs>
EOF
```

### 4. workflow 配置

安装完三个 Agent 后，修改 `pypto-pro-workflow/config.yaml` 里的 CLI 路径：

```yaml
agents:
  cannbot:
    cli: "/home/developer/.nvm/versions/node/v18.19.1/bin/cannbot"  # 改成你的 cannbot 路径
  kerminal:
    cli: "/root/.local/bin/kerminal"  # 改成你的 kerminal 路径
  hermes:
    cli: "/usr/local/bin/hermes"  # 改成你的 hermes 路径
```

### 5. CANN 环境（NPU 必需）

```bash
# ~/.bashrc 中加入（路径按实际安装目录调整）
export LD_LIBRARY_PATH=/home/developer/Ascend/cann-9.1.0/x86_64-linux/lib64:$LD_LIBRARY_PATH
source ~/.bashrc

# 验证
python3 -c "import torch, torch_npu; print(torch.npu.is_available())"  # 应输出 True
```
