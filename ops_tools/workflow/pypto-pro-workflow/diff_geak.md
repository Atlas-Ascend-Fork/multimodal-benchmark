# GEAK 设计优势对比分析（diff_geak.md）

> 对比 GEAK 框架与我们现有 pypto-pro-workflow，列出 GEAK 的所有设计优点。

---

## 1. 角色系统（11 个角色，职责边界清晰）

**GEAK**：每个角色只做一件事，绝不越界。所有角色都是同一个 Claude Code，只是每次调用传入不同的 role prompt。

| GEAK 角色 | 我们的对应 | Agent | 有/缺 | 说明 |
|-----------|-----------|-------|--------|------|
| tech_lead（规划+记忆） | ❌ 无 | - | **缺** | 我们没有方向规划和经验提炼角色 |
| engineer（写代码） | n1_stage2 / n1_stage3 | cannbot | ✅ 有 | 功能一致 |
| verify_engineer（独立验证） | n2_stage5 / n2_stage6 | kerminal | ✅ 有 | 我们的验证不是全新 workspace，但功能一致 |
| profile_engineer（profiling） | n2_stage7 | kerminal | ✅ 有 | 功能一致 |
| benchmark_engineer（建 baseline） | ❌ 无 | - | **缺** | 我们依赖 cann-bench 外部提供 baseline |
| researcher（搜索） | n3_stage8 | hermes | ✅ 有 | 我们是单阶段搜索，GEAK 是多阶段 DRA |
| integrator（合并 patch） | ❌ 无 | - | **缺** | 我们单方向不需要合并 |
| update_experience（知识积累） | ❌ 无 | - | **缺** | 我们没有经验提炼 |
| director（环境+最终验证） | n1_stage1 + n4_stage9 | cannbot/kerminal | ✅ 有 | 分散在阶段1和阶段9 |
| author_engineer（初始编写） | n1_stage2 | cannbot | ✅ 有 | 功能一致 |
| deep_engineer（整体重写） | ❌ 无 | - | **缺** | 我们没有"大改"模式 |

**共同点**：都是同一个 agent 带不同 prompt 扮演不同角色，每次独立进程、不续接 session。

**我们缺的关键角色及规划**：

| 缺失角色 | 应放在哪个节点 | 应放在哪个阶段 | 用哪个 agent | 说明 |
|----------|--------------|--------------|-------------|------|
| tech_lead（规划方向） | N1 | 每轮迭代开始前（阶段3.5） | cannbot | 看 history 决定本轮往哪个方向优化，输出给 coder |
| tech_lead（提炼经验） | N1 | 每轮迭代结束后（阶段8.5） | cannbot | 总结本轮结果，更新 insights/ledger |
| update_experience（知识积累） | N1 | 全部迭代结束后（阶段9之前） | cannbot | 把整轮 run 的经验提炼成一张知识卡 |
| benchmark_engineer（建 COMMANDMENT） | N2 | 阶段1.5（首次 baseline 建立） | kerminal | 跑一次 baseline，冻结为不可变约束文件 |
| deep_engineer（整体重写） | N1 | 连续 N 轮无进步时触发（替代普通阶段3） | cannbot | 不是修 patch，而是从头重写 kernel |

**最优先加的**：tech_lead 规划 + 经验提炼。加了之后 cannbot 就知道"上轮试了 X 没用，这轮换 Y"。注意：tech_lead 提炼的 insights 仅作为参考，真正可信的是 ledger 里的硬数据（试了什么方向、实际 speedup 变化了多少）。

---

## 2. 上下文管理（GEAK 详细机制）

GEAK 的上下文管理由 5 个模块组成，全部在控制平面（JS）中维护，每轮通过 prompt 注入给不同 agent。

### 2.1 history 对象（控制平面维护的核心状态）

```python
history = {
    "insights": [],       # 字符串数组，每条是一句话洞察
    "ledger": [],         # 对象数组，假设追踪账本（累积追加，不替换）
    "rounds": [],         # 每轮完整记录（directions/results/winner/improved/cumulative）
    "bottleneck_now": "", # 当前瓶颈描述
    "suggest_next": "",   # tech_lead 对下一轮的建议
}
```

**谁写哪个字段**：

| 字段 | 谁写 | 怎么更新 |
|------|------|---------|
| insights | **大模型**（tech_lead:update_memory 输出） | 每轮整体替换（重写全部） |
| ledger | **大模型**（tech_lead:update_memory 输出），但基于程序提供的硬数据 | 每轮累积追加（不删旧的） |
| rounds | **程序**（orchestrator 自动写入） | 每轮追加一条（纯硬数据，不经过大模型） |
| bottleneck_now | **大模型**（tech_lead:update_memory 输出） | 每轮替换 |
| suggest_next | **大模型**（tech_lead:update_memory 输出） | 每轮替换 |

**数据流向**：
- 初始化为空 → 每轮结束后程序写 rounds + 大模型写 insights/ledger/bottleneck/suggest → 下一轮注入给 tech_lead:plan_round + 所有 engineer

**大白话例子**（以 exp 算子优化 3 轮为例）：

第 1 轮结束后，tech_lead 总结本轮，history 变成：
```python
history = {
    "insights": [
        "标量循环是根本瓶颈：把 for i in range(N) 改成 vf.exp 向量化后 speedup 从 0.007 涨到 0.49"
    ],
    "ledger": [
        {"iter": 1, "direction": "标量循环→向量化", "speedup_before": 0.007, "speedup_after": 0.49, "verdict": "big_win"}
    ],
    "rounds": [
        {"iter": 1, "avg_speedup": 0.49, "improved": true}
    ],
    "bottleneck_now": "向量化完成后，小 shape case 的 kernel launch overhead 成为新瓶颈",
    "suggest_next": "尝试调大 tile_size 减少 launch 次数"
}
```

第 2 轮，cannbot 按 suggest_next 调了 tile_size 64→128。结束后 tech_lead 再次总结：
```python
history = {
    "insights": [
        "标量循环是根本瓶颈：向量化后 speedup 从 0.007→0.49（已解决，不要回退）",
        "tile_size 128 比 64 稍好但提升有限：0.49→0.51，这条路接近天花板"
    ],
    "ledger": [
        {"iter": 1, "direction": "标量循环→向量化", "speedup_before": 0.007, "speedup_after": 0.49, "verdict": "big_win"},
        {"iter": 2, "direction": "tile_size 64→128", "speedup_before": 0.49, "speedup_after": 0.51, "verdict": "small_win"}
    ],
    "rounds": [
        {"iter": 1, "avg_speedup": 0.49, "improved": true},
        {"iter": 2, "avg_speedup": 0.51, "improved": true}
    ],
    "bottleneck_now": "tile_size 调整收益递减，大 shape 已经 OK，小 shape 仍被 launch overhead 卡住",
    "suggest_next": "不要继续调 tile_size，换方向：尝试多个小 case 合并一次 kernel launch"
}
```

第 3 轮，cannbot 试了 tile_size 256（没听 suggest_next），结果退化。tech_lead 总结：
```python
history = {
    "insights": [
        "标量循环→向量化是正确的（已解决，不要回退）",
        "tile_size 最优在 128 附近：256 会导致 UB 不够用反而退化",
        "tile_size 方向已穷尽，不要再调了"
    ],
    "ledger": [
        {"iter": 1, "direction": "标量循环→向量化", "speedup_before": 0.007, "speedup_after": 0.49, "verdict": "big_win"},
        {"iter": 2, "direction": "tile_size 64→128", "speedup_before": 0.49, "speedup_after": 0.51, "verdict": "small_win"},
        {"iter": 3, "direction": "tile_size 128→256", "speedup_before": 0.51, "speedup_after": 0.47, "verdict": "regression"}
    ],
    "rounds": [
        {"iter": 1, "avg_speedup": 0.49, "improved": true},
        {"iter": 2, "avg_speedup": 0.51, "improved": true},
        {"iter": 3, "avg_speedup": 0.47, "improved": false}
    ],
    "bottleneck_now": "tile_size 方向已死，需要完全不同的优化方向",
    "suggest_next": "尝试减少 kernel launch 次数，或者用双缓冲隐藏数据搬运延迟"
}
```

**关键观察**：
- `insights` 每轮被 tech_lead **整体重写**（第3轮删掉了"tile 128 比 64 稍好"的旧结论，改成了更准确的"tile 最优在 128 附近"）
- `ledger` 只**追加**不删除（3 轮的记录都在，是事实不能改）
- `rounds` 由**程序自动追加**（不经过大模型，纯硬数据）
- 第 4 轮的 cannbot 拿到这个 history 后，就知道"tile_size 方向已死、要换新方向"——不会再去试 tile 256

**如果没有 history**（我们现在的情况）：第 4 轮的 cannbot 看不到前 3 轮的经验，可能又去试 tile_size 256，跟第 3 轮干一样的事。这就是为什么我们 10 轮迭代性能在 0.47~0.52 之间反复震荡。

**→ 挪到我们框架**：在 orchestrator.py 中维护一个 `history` dict，每轮迭代结束后更新，下一轮构建 prompt 时注入。存储位置：`<work>/knowledge/history.json`。

### 2.2 insights[]（大模型提炼的洞察）

- **谁写**：tech_lead 在 `update_memory` 阶段输出
- **什么时候写**：每轮结束后
- **怎么更新**：**整体替换**（不是追加）。tech_lead 每轮重写全部 insights，可以删掉过时的、合并重复的
- **内容格式**：字符串数组，每条一句话（如 "tile 256 在 case_5 上比 128 快 1.3x"）
- **注入给谁**：
  - `tech_lead:plan_round`（通过 HISTORY 对象）
  - 每个 engineer（通过 INSIGHTS 字段）
  - integrator（通过 INSIGHTS 字段）

**本质**：大模型对历史经验的**主观总结**。可能不完全准确，但能引导方向。

**→ 挪到我们框架**：新增 role `tech_lead_memory.md`（cannbot 扮演），在阶段8完成后、阶段3之前调用。输入：本轮 perf_result.json 中的 avg_speedup + 上一轮的 avg_speedup + bottleneck_analysis.md。输出：更新后的 insights[] 写入 `<work>/knowledge/insights.json`。阶段3 的 prompt 中注入 insights 列表。

### 2.3 ledger[]（假设追踪账本）

- **谁写**：tech_lead 在 `update_memory` 阶段输出
- **什么时候写**：每轮结束后
- **怎么更新**：**累积追加**（concat，不替换）。历史 ledger 永远保留
- **内容格式**：对象数组，每条是一个假设条目（schema 未强制字段，tech_lead 自由填写，通常包含方向名、结果、原因）
- **注入给谁**：仅通过 HISTORY 传给 tech_lead 自身和 update_experience，**不直接注入给 engineer**

**与 insights 的关键区别**：
| | insights | ledger |
|--|---------|--------|
| 更新方式 | 整体替换 | 累积追加 |
| 给谁看 | tech_lead + engineer + integrator | 仅 tech_lead |
| 内容 | 提炼的一句话结论 | 完整的假设追踪（试了什么、结果如何） |
| 可靠性 | 主观总结，可能不对 | 基于硬数据的追踪，相对可靠 |

**→ 挪到我们框架**：orchestrator 每轮结束后自动追加一条 ledger 到 `<work>/knowledge/ledger.jsonl`（一行一条 JSON）。内容由程序生成（不用大模型）：`{"iter": 3, "direction": "从FIX_DIRECTIVE提取", "avg_speedup_before": 0.48, "avg_speedup_after": 0.47, "delta": -0.01, "verdict": "no_improve"}`。下一轮 tech_lead_memory 读 ledger 时能看到所有历史尝试。

### 2.4 ROUND_RESULTS（传给 update_memory 的硬数据）

orchestrator 在每轮结束后，把本轮的**硬数据**打包传给 tech_lead：

```python
ROUND_RESULTS = [
    {
        "id": "r3_d0",                # direction id
        "title": "优化 tile size",     # 方向标题（来自 plan）
        "specialty": "memory",         # 类别（来自 plan）
        "expected": 1.5,               # tech_lead 预期的加速比（来自 plan）
        "claimed": 1.3,                # engineer 自报的加速比（可能不准）
        "verified": 1.25,              # verify_engineer 独立测量的加速比（可信）
        "status": "verified",          # 验证状态
        "notes": "changed tile from 64 to 128"  # engineer 备注
    },
    ...
]
```

**数据来源**：`expected` 来自 tech_lead plan；`claimed` 来自 engineer 自报；`verified` 来自 verify_engineer 独立测量（最可信）。

tech_lead 拿到这些硬数据后，总结出 insights 和 ledger 条目。所以 ledger 虽然是大模型写的，但**基于的是 verified 硬数据**，不是凭空编的。

**→ 挪到我们框架**：orchestrator 在阶段6完成后，自动构建 ROUND_RESULTS 传给 tech_lead_memory：`{"iter": N, "avg_speedup": 从perf_result读, "overall_score": 从perf_result读, "direction_summary": 从FIX_DIRECTIVE.md提取第一行}`。这些全是程序从文件读的硬数据，不是大模型编的。

### 2.5 COMMANDMENT.md（不可变约束）

- **谁创建**：benchmark_engineer 在 Phase 1 一次性写入
- **包含什么**：正确性检测命令、完整 benchmark 命令、profile 命令、baseline 数据
- **谁读它**：所有 agent（engineer、verify_engineer、profiler、director）
- **为什么不可变**：保证所有轮次的评测标准一致。如果 COMMANDMENT 被改了，不同轮的数据就没法比较

**→ 挪到我们框架**：我们已经有类似的——cann-bench 的 task 目录（desc.md + cases.yaml + golden.py）就是不可变的评测标准。但我们缺一个显式的"约束文件"让所有 agent 读。可以在阶段1完成后由 orchestrator 生成一个 `<work>/COMMANDMENT.md`，内容从 task/ 提取（算子签名、精度阈值、性能目标 speedup≥1.0、禁止修改函数签名），后续每个 agent prompt 都注入这个文件路径。

### 2.6 其他上下文

- **cumulative_speedup**：当前最佳加速比，只在 commit（提升 >2%）时更新。注入给 tech_lead:plan_round
- **bestPerCase**：每个 case 的最佳性能数组，commit 后更新。注入给 tech_lead:plan_round
- **codebase_context.md**：tech_lead 在 analyze 阶段写的代码分析文档，传给所有 engineer 读

**→ 挪到我们框架**：我们已有 `ANALYSIS.md`（阶段1产物）作为代码分析文档。cumulative_speedup 和 bestPerCase 可以直接从 `.state.json` 的 `history[]` 数组中读取（每轮的 speedup 已记录）。只需在阶段3的 prompt 中额外注入当前最佳 avg_speedup 值即可。

### 2.7 上下文注入总图

```
每轮开始:
  tech_lead:plan_round 收到:
    - HISTORY（含 insights + ledger + rounds + bottleneck_now + suggest_next）
    - CUMULATIVE_SPEEDUP
    - CURRENT_BEST_PER_CASE
    - PROFILE_SUMMARY
    → 输出: 本轮方向 directions[]

每个 engineer 收到:
    - INSIGHTS（从 history 中提取）
    - COMMANDMENT
    - codebase_context.md
    → 输出: patch + claimed speedup

每轮结束:
  tech_lead:update_memory 收到:
    - ROUND_RESULTS（硬数据：expected/claimed/verified/status）
    - PRIOR_HISTORY（当前 history）
    → 输出: 更新后的 insights[] + 新的 ledger 条目 + bottleneck_now + suggest_next
    → orchestrator 合并到 history
```

**我们现在**：无跨轮记忆。每轮 cannbot 从零开始理解问题。FIX_DIRECTIVE.md 只包含当前轮的搜索结果，不包含历史经验。

**→ 挪到我们框架（总结）**：
```
orchestrator 每轮结束后:
  1. 从 perf_result.json 读硬数据 → 自动写入 ledger.jsonl（程序写，不经过大模型）
  2. 调 cannbot（tech_lead_memory role）→ 输入 ledger + 本轮结果 → 输出 insights.json（大模型写）
  
orchestrator 每轮开始（阶段3 prompt 构建时）:
  3. 读 insights.json + ledger.jsonl → 注入到 cannbot 的 prompt 中
  4. cannbot 看到历史经验 → 不会重复尝试失败方向
```
新增文件：`<work>/knowledge/insights.json` + `<work>/knowledge/ledger.jsonl` + `roles/tech_lead_memory.md`

---

## 3. 并行探索（一轮多方向 + pipeline 验证）

**GEAK 一轮内**：
1. tech_lead 规划 N 个方向（每个有 specialty: algorithm/memory/compute/host_runtime）
2. N 个 engineer 并行执行（每个独立 workspace，互不干扰）
3. 完成后 pipeline 到 verify_engineer 并行验证
4. 多个 verified patch → integrator 合并
5. 合并后的 winner commit

**GPU 调度**：
- `pool` 模式：flock 抢占空闲 GPU
- `pin` 模式：direction i 绑定 GPU_LIST[i % n]

**我们现在**：一轮只有一个 cannbot 改一次，串行。

---

## 4. 独立验证（verify_engineer 门禁）

**GEAK 的验证流程**：
1. engineer 写完 patch → 不直接 commit
2. verify_engineer 在**全新 workspace** 中：`git checkout -- . → git apply patch → 跑 COMMANDMENT correctness → 跑完整 benchmark`
3. 通过门禁（3 条件全满足）：
   - status 以 "verified" 开头
   - correctness 以 "pass" 开头
   - speedup > CANDIDATE_FLOOR（默认 1.0）
4. 只有 verified 的 patch 才能进入 winner 竞选

**关键设计**：verify_engineer **不是 engineer 自己验证**——是另一个独立 agent 在干净环境中重跑。防止"自己写自己验"。

**我们现在**：cannbot 写完代码后直接交给 kerminal 评测——但 kerminal 可能在同一个污染环境中跑。

---

## 5. 知识积累（update_experience + knowledge/learned/）

**触发条件**（4 个全满足）：
- update_experience 开关 = on
- 最终验证 validation_status = accepted
- 最终 speedup > 1.0
- 非共享机器（BOX_QUIET）

**内容规则**：
- 只写比例（speedup 2.3x），不写绝对数字（ms）
- 从 ledger 提取 pitfalls（什么方向失败了为什么）
- 一次 run 只产出一张知识卡
- ADD-only，不删不改已有卡

**KB 读取预算**（防止知识库过度主导）：
- 每轮最多 KB_DIR_CAP=1 个方向可以引用知识卡
- 必须有 ≥1 个"纯 profile 驱动"方向（不看 KB，作为对照组）

**我们现在**：无知识积累机制。

---

## 6. Commit 门禁（2% 相对提升才 commit）

**GEAK**：
```
improved = winner.geomean > cumulative * (1 + MIN_IMPROVE)  // MIN_IMPROVE = 0.02
```

只有 `improved=true` 才：
- `git apply` patch 到 CANONICAL
- 重新 profile
- 重置 noImprove

**意义**：防止微小波动（噪声）被误认为提升。只有实质性进步才更新代码。

**我们现在**：只看 perf_pass（true/false），没有相对提升判定。

---

## 7. Profiling 策略（只在 commit 后才 reprofile）

**GEAK**：
- baseline profile：Phase 1 跑一次
- reprofile：**只在 improved=true（commit 后）**才重新 profile
- 传入 `PREVIOUS_METRICS` 做 shift 分析

**我们现在**：每轮都跑 profiling，浪费时间。

---

## 8. DRA 深度研究（多阶段搜索）

**GEAK 的 4 阶段搜索**：
1. **research_plan**：提取事实 + 生成 ≤8 个排序的研究问题
2. **research_question**（并行）：每个问题独立搜索
3. **research_blindspot**（可选）：批判式审查，找盲点，再补充搜索
4. **research_synthesize**：综合所有搜索结果 → 写 `deep_search_brief.md`

**输出**：喂给 tech_lead.plan_round 的 `DEEP_SEARCH_BRIEF`，影响方向规划。

**我们现在**：hermes 一次性搜索，没有多阶段、没有盲点审查、没有综合提炼。

---

## 9. Agent 超时与重试（agentBounded + safeAgent）

**GEAK**：
- `agentBounded`：每个 agent 调用有超时守卫（默认 120 分钟），超时解析为 null
- `safeAgent`：包装 agentBounded，3 次重试，全失败降级为 null
- 调用方对 null 有 null-guard（跳过/继续）

**关键设计**：单个 agent 挂掉不杀整个 run。降级 + 继续。

**我们现在**：有连续 5 次 API 错误杀进程，但没有单 agent 超时降级。

---

## 10. 文件交接（StructuredOutput + 原子 commit）

**GEAK**：
- 每个 agent 调用强制 `schema: JSON_SCHEMA`——输出必须是结构化 JSON
- patch 用 `git diff > best_patch.diff` 标准格式
- commit 用 `git apply → git add -A → git commit` 原子操作
- workspace 隔离用 tar-copy（不是软链接）

**我们现在**：agent 写文件格式不固定，kerminal 有时编造数据。

---

## 11. Baseline 冻结（防作弊）

**GEAK**：
- baseline 在 Phase 1 由 benchmark_engineer 冻结
- 如果 `baseline_frozen !== true`，workflow 直接 abort
- 防止候选算子跟自己的 scaffold 比

**我们现在**：baseline 由 cann-bench 提供（外部固定），不存在作弊风险。但没有显式的冻结验证。

---

## 12. 进程安全（PROCESS_SAFETY）

**GEAK**：每个 agent prompt 注入 PROCESS_SAFETY 约束：
- 禁止 `pkill -f`、`killall`、`kill -- -PGID`
- 只能 kill 自己启动的进程（by pid）
- 防止 agent 误杀 orchestrator

**我们现在**：`--dangerously-skip-permissions` + `-a never` 跳过了所有权限检查，没有进程安全约束。

---

## 总结：我们缺什么

| 优点 | 重要程度 | 我们有吗 |
|------|---------|---------|
| 跨轮记忆（insights+ledger） | ★★★★★ | ❌ |
| 方向规划（tech_lead） | ★★★★★ | ❌ |
| 独立验证 | ★★★★☆ | ❌（同一个 kerminal） |
| 提升才 commit（2% 门禁） | ★★★★☆ | ❌ |
| 多阶段搜索（DRA） | ★★★☆☆ | ❌ |
| 并行探索 | ★★★☆☆ | ❌ |
| 知识积累 | ★★★☆☆ | ❌ |
| 只在 commit 后 reprofile | ★★☆☆☆ | ❌（每轮都跑） |
| 进程安全 | ★★☆☆☆ | ❌ |
| StructuredOutput 强制格式 | ★★☆☆☆ | ❌ |
| Agent 超时降级 | ★★☆☆☆ | 部分有 |
| Baseline 冻结 | ★☆☆☆☆ | 不需要（cann-bench 外部提供） |
