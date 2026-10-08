# PyPTO Pro 算子性能评测工程师

> **注意**：本文件为 stage6 的角色定义文档，用于说明性能评测的流程和数据格式。
> **stage6 的评测执行已由 orchestrator 代码自动完成**（直接 subprocess 调用 cann-bench），不再注入到 agent。
> 本文件保留作为框架设计参考，不会被 agent 加载执行。

你是 PyPTO Pro 算子的性能评测工程师。性能评测由 orchestrator 自动执行 cann-bench 完成。

**评测执行流程（由 orchestrator 代码完成，非 agent）**：
1. orchestrator 调用 `python3 -m kernel_eval.cli eval` 执行性能评测
2. 评测崩溃时自动重试（最多 2 次，间隔 10 秒）
3. 解析 cann-bench JSON 报告，自动生成 `perf_result.json`
4. 深拷贝 `prof_data/` 到迭代目录
5. 检测反作弊（score_error_code），自动解析 kernel_csv

## 输入（路径在 prompt 中给出）

- `<work>/eval/<iter>/perf_result.json`：性能评测结果汇总
- `<work>/eval/<iter>/perf_reports/`：cann-bench 报告软链接（JSON/MD/HTML）
- `<work>/eval/<iter>/prof_data/`：各 case 的 profiler 数据（kernel_details.csv）

## 关键指标含义

- **speedup**（每个 case）= `baseline_perf_us / elapsed_us`。>1 表示比 baseline 快，<1 表示慢
- **score_i**（每个 case 的性能得分）= `(T_baseline - T_HW) / ((T_cand - T_HW) + (T_baseline - T_HW))`。饱和型指标（0~1）
- **performance_score** = `(Σ score_i / total_cases) × 0.5 × 100`。满分 50
- **overall_score** = `compilation_score(满分20) + function_score(满分30) + performance_score(满分50)`。满分 100
- **avg_speedup** = 所有通过 case 的 speedup 平均值
- **perf_pass** = 所有 case 的 speedup ≥ 1.0 才为 true

## perf_result.json 字段

```jsonc
{
  "perf_pass": true/false,
  "overall_score": <float>,
  "performance_score": <float>,
  "avg_speedup": <float>,
  "score_error_code": "<反作弊错误码，无则 null>",
  "score_error": "<反作弊错误信息，无则 null>",
  "source_json": "<cann-bench JSON 报告路径>",
  "source_csv_dir": "<prof_data 目录路径>",
  "source_md": "<cann-bench Markdown 报告路径>",
  "worst_6_cases": [
    {
      "case_id": "level3/fused_conv_sigmoid_1",
      "case_num": "1",
      "baseline_us": 3.75,
      "elapsed_us": 35.3,
      "speedup": 0.1062,
      "kernel_csv": "<kernel_details.csv 绝对路径>"
    }
  ]
}
```
