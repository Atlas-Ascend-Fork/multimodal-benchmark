# multimodal-bench

# multimodal-bench

> CANN Multimodal-AI SIG — 工具与评测层仓库

多模态 AI 算子与模型的自动化工具链及评测基准。包含算子级工具（模式识别与归纳抽象）、模型级工具（计算图匹配与自动替换）和统一评测框架（正确性、精度、性能、回归）。

## 模块概览

| 模块 | 目录 | 功能 | 说明 |
|------|------|------|------|
| **Ops Tools** | `ops_tools/` | 算子模式识别 · 归纳抽象 | 扫描模型计算图，自动识别可融合的多模态算子模式，归纳抽象为通用算子模板，输出到 multimodal-ops |
| **Model Tools** | `model_tools/` | 计算图匹配 · 自动替换 | 匹配模型中的标准子图模式，自动替换为 multimodal-ops 提供的高性能算子实现 |
| **Benchmark** | `benchmark/` | 正确性 · 精度 · 性能 · 回归 | 统一评测框架，支持算子级和模型级的全维度评测，持续追踪性能回归 |

## 目录结构

```
multimodal-bench/
├── ops_tools/                  # 算子级工具
│   ├── patterns/               #   模式定义（可融合子图模式库）
│   └── miners/                 #   模式挖掘（自动扫描 · 归纳抽象）
├── model_tools/                # 模型级工具
│   ├── matchers/               #   子图匹配引擎
│   └── rewriters/              #   自动改写（标准子图 → 高性能算子）
├── benchmark/                  # 统一评测框架
│   ├── cases/                  #   测试用例定义
│   ├── datasets/               #   评测数据集索引与预处理
│   ├── metrics/                #   评测指标（精度 / 性能 / 内存 / 功