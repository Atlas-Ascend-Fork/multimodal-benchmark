# multimodal-bench

> CANN Multimodal-AI SIG — 工具与评测层仓库

多模态 AI 算子与模型的自动化工具链及评测基准。包含算子级工具（模式识别与归纳抽象）、模型级工具（计算图匹配与自动替换）和统一评测框架（正确性、精度、性能、回归）。

## 源码下载

```bash
git clone https://gitcode.com/cann/multimodal-bench.git
cd multimodal-bench
```

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
│   ├── metrics/                #   评测指标（精度 / 性能 / 内存 / 功耗）
│   ├── runners/                #   评测执行器（单算子 / 端到端 / 回归）
│   ├── baselines/              #   基线数据（冻结参照，跨版本可比）
│   └── reports/                #   评测报告输出
└── README.md
```

## 贡献指南

欢迎参与贡献，请参阅 [CONTRIBUTING.md](CONTRIBUTING.md) 了解贡献流程与规范。

## 安全声明

请参阅 [SECURITY.md](SECURITY.md) 了解安全相关信息。

## 许可证

本项目采用 [CANN Open Software License Agreement Version 2.0](LICENSE) 许可证。

## 所属 SIG

本仓库隶属于 CANN 社区 [multimodal-ai SIG](https://gitcode.com/cann/community/blob/master/CANN/sigs/multimodal-ai/README.md)。

## 问题反馈

如有问题或建议，请通过 [Issue](../../issues) 提交反馈。
