# multimodal-benchmark

> CANN Multimodal-AI SIG — 工具与评测层仓库

多模态 AI 算子与模型的自动化工具链、数据生成工具及评测基准。本仓库的核心思路是**用代码生成数据、用工具驱动算子开发、用评测反馈优化迭代**。

## 源码下载

```bash
git clone https://gitcode.com/cann/multimodal-benchmark.git
cd multimodal-benchmark
```

## 模块概览

| 模块 | 目录 | 功能 | 说明 |
|------|------|------|------|
| **Ops Tools — Workflow** | `ops_tools/workflow/` | 算子自动开发 | 多智能体协作框架，自动完成算子从需求分析→代码编写→编译部署→精度/性能评测→迭代优化的全流程 |
| **Ops Tools — Patterns** | `ops_tools/patterns/` | 算子模式识别 | 扫描模型计算图，自动识别可融合的多模态算子模式（待共建） |
| **Ops Tools — Miners** | `ops_tools/miners/` | 归纳抽象 | 归纳抽象为通用算子模板（待共建） |
| **Data Tools — AI-MSF-Benchmark** | `data_tools/ai-msf-benchmark/` | 传感器数据腐蚀生成 | 多种腐蚀模式（天气/传感器/噪声/空间错位/时序延迟），支持 KITTI/Waymo/NuScenes，附带 7 个 MSF 系统评测 |
| **Data Tools — Image Noise Pattern** | `data_tools/image-noise-pattern/` | ISP 图像噪声生成 | 18 种真实世界 ISP 链路缺陷模拟（白平衡/坏点/CFA/色彩空间/镜头等） |
| **Model Tools** | `model_tools/` | 计算图匹配 · 自动替换 | 匹配标准子图并替换为高性能算子（待共建） |
| **Benchmark** | `benchmark/` | 统一评测框架 | 算子级和模型级全维度评测（待共建） |

## 工具使用说明

### PyPTO Pro Workflow — 多智能体算子自动开发

> 目录：`ops_tools/workflow/pypto-pro-workflow/`

面向昇腾 NPU 的多智能体算子自动开发框架。3 个 AI Agent（CANNBot 写代码、Kerminal 编译评测、Hermes 搜索优化）协同工作，通过 10 阶段流水线和最多 20 轮自动迭代，实现融合算子从需求分析到最终报告的全流程自动化。

**核心特性**：
- 10 阶段流水线：需求分析→首版实现→编译部署→精度评测→性能评测→Profiling→搜索优化→经验提炼→代码修改→最终报告
- 跨轮记忆（history 机制）：Tech Lead 每轮提炼 insights/ledger，追踪最慢 case，避免重复失败方向
- 芯片自动检测：从 CANN 编译器直接读取 UB/L1/L0A/L0B/L0C/核数真实值
- 动态退出条件：性能达标 + Tech Lead 验证融合方案为"真融合"后才退出

**快速启动**：
```bash
cd ops_tools/workflow/pypto-pro-workflow
# 从零开始写算子
python orchestrator.py --task-dir /path/to/cann-bench/tasks/level1/exp --max-iter 10

# 从已有实现开始迭代优化
python orchestrator.py --task-dir /path/to/task \
  --init-impl /path/to/existing/impl \
  --optimize-hint "Conv+Sigmoid融合：单kernel内Cube算完conv后L0C→UB直传做sigmoid" \
  --max-iter 20
```

**已验证算子**：fused_conv_sigmoid（Cube-Vector 片上直传融合，2.87× 加速）、fused_conv_tanh（双 Kernel 流水融合，3.92× 加速）。

详细文档：[ops_tools/workflow/pypto-pro-workflow/README.md](ops_tools/workflow/pypto-pro-workflow/README.md)

### AI-MSF-Benchmark — 多传感器融合数据腐蚀与评测

> 目录：`data_tools/ai-msf-benchmark/`

面向多传感器融合（MSF）系统的鲁棒性评测基准（ESEC/FSE 2023 论文）。提供多种腐蚀模式生成器，可对干净数据集施加真实世界退化，生成腐蚀后的相机图像、LiDAR 点云、标定文件和时序数据，用于评测 MSF 系统的鲁棒性。

**已实现的腐蚀类型**（5 大类）：
- **天气**：rain、fog
- **传感器**：brightness、darkness、motion_blur、defocus_blur、distortion
- **噪声**：gaussian_noise、impulse_noise
- **空间错位**：rotation_x/y/z
- **时序延迟**：delay

> 腐蚀模式支持图像和 LiDAR 两条通道，部分模式同时支持。完整腐蚀类型与参数配置见 `corruption/corruption_configs.py`。

**快速使用**：
```bash
cd data_tools/ai-msf-benchmark

# 生成亮度腐蚀图像（严重度 3）
python -m corruption.main simulate_image \
  --input_dir "./corruption/example/input_dir/image" \
  --output_dir "./corruption/example/output_dir/image/" \
  --corruption "brightness" --sev 3

# 生成高斯噪声点云（严重度 5）
python -m corruption.main simulate_lidar \
  --input_dir "./corruption/example/input_dir/velodyne" \
  --output_dir "./corruption/example/output_dir/velodyne/" \
  --corruption "gaussian_noise" --sev 5

# 模拟传感器空间错位
python -m corruption.main simulate_calib \
  --input_dir "./corruption/example/input_dir/calib" \
  --output_dir "./corruption/example/output_dir/calib/" \
  --corruption "rotation_x" --sev 3
```

**内置 MSF 系统**（7 个，覆盖 3 种任务 × 3 种融合机制）：CLOCs、EPNet、FConv、MDANet、DFMOT、JMODT、TWISE。

详细文档：[data_tools/ai-msf-benchmark/readme.md](data_tools/ai-msf-benchmark/readme.md)

### Image Noise Pattern — ISP 真实世界图像噪声生成

> 目录：`data_tools/image-noise-pattern/`

模拟相机 ISP（图像信号处理）链路各阶段缺陷的数据生成工具。每个 Python 模块对应一种 ISP 缺陷，可独立调用生成退化图像。

**18 种 ISP 腐蚀模式**：

| 类别 | 模块 | 缺陷 |
|------|------|------|
| 传感器 | `CCD-D_CMOS-D.py` `sensorBroken.py` `SE.py` | CCD/CMOS 损伤、坏像素、曝光异常 |
| ISP 前处理 | `BPC-D.py` `I-BLC_E-BLC.py` `LSC-D.py` | 坏点校正、黑电平校正、镜头阴影校正缺陷 |
| ISP 核心 | `CFAI-D.py` `AWB-D.py` `CSC-D.py` `GC-d.py` `colorCorrection.py` | CFA 插值、白平衡、色彩空间转换、Gamma 校正缺陷 |
| 光学/环境 | `FM-D.py` `LO.py` `F.py` `generateRain.py` `THE.py` `ME.py` | 失焦、镜头遮挡、雾、雨、热效应、运动模糊 |
| 后处理 | `postProcessing.py` | 后处理压缩伪影 |

各模块独立运行，使用方式参考模块内注释和示例。

详细文档：[data_tools/image-noise-pattern/README.md](data_tools/image-noise-pattern/README.md)

## 版本配套

| 项目 | 版本 |
|------|------|
| 产品型号 | Ascend950PR |
| NPU 芯片 | Ascend 950 |
| CANN 版本 | 9.1.0 |
| 操作系统 | Ubuntu 24.04.3 LTS (x86_64) |
| Python | >= 3.10（验证版本 3.12.3） |
| Node.js | >= 18（验证版本 18.19.1，workflow 的 CANNBot 依赖） |
| Kerminal | 0.8.12（workflow 的编译评测 Agent） |

## 目录结构

```
multimodal-benchmark/
├── ops_tools/                           # 算子级工具
│   ├── workflow/                        #   多智能体算子自动开发框架
│   │   └── pypto-pro-workflow/          #     orchestrator + agents + roles + knowledge
│   ├── patterns/                        #   模式定义（待共建）
│   └── miners/                          #   模式挖掘（待共建）
├── data_tools/                          # 数据生成工具
│   ├── ai-msf-benchmark/               #   多传感器融合数据腐蚀 + 7 个 MSF 系统评测
│   └── image-noise-pattern/            #   18 种 ISP 链路图像噪声生成
├── model_tools/                         # 模型级工具（待共建）
│   ├── matchers/                        #   子图匹配引擎
│   └── rewriters/                       #   自动改写
├── benchmark/                           # 统一评测框架（待共建）
│   ├── cases/                           #   测试用例定义
│   ├── datasets/                        #   评测数据集索引
│   ├── metrics/                         #   评测指标
│   ├── runners/                         #   评测执行器
│   ├── baselines/                       #   基线数据
│   └── reports/                         #   评测报告输出
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
