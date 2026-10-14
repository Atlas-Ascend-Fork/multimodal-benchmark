# multimodal-bench

> CANN Multimodal-AI SIG — Tools & Benchmark Repository

Automated toolchain, data generation utilities, and evaluation benchmarks for multimodal AI operators and models. The core philosophy of this repository is to **generate data with code, drive operator development with tools, and optimize iteratively with evaluation feedback**.

## Source Code Download

```bash
git clone https://gitcode.com/cann/multimodal-bench.git
cd multimodal-bench
```

## Module Overview

| Module | Directory | Function | Description |
|--------|-----------|----------|-------------|
| **Ops Tools — Workflow** | `ops_tools/workflow/` | Automated operator development | Multi-agent collaboration framework that automates the full operator pipeline from requirement analysis → code generation → compilation & deployment → accuracy/performance evaluation → iterative optimization |
| **Ops Tools — Patterns** | `ops_tools/patterns/` | Operator pattern recognition | Scans model computation graphs to automatically identify fusible multimodal operator patterns (planned for community co-development) |
| **Ops Tools — Miners** | `ops_tools/miners/` | Inductive abstraction | Abstracts identified patterns into general-purpose operator templates (planned for community co-development) |
| **Data Tools — AI-MSF-Benchmark** | `data_tools/ai-msf-benchmark/` | Sensor data corruption generation | Multiple corruption modes (weather / sensor / noise / spatial misalignment / temporal delay), supports KITTI/Waymo/NuScenes, includes evaluation of 7 MSF systems |
| **Data Tools — Image Noise Pattern** | `data_tools/image-noise-pattern/` | ISP image noise generation | 18 real-world ISP pipeline defect simulations (white balance / dead pixel / CFA / color space / lens, etc.) |
| **Model Tools** | `model_tools/` | Computation graph matching & automatic replacement | Matches standard subgraphs and replaces them with high-performance operators (planned for community co-development) |
| **Benchmark** | `benchmark/` | Unified evaluation framework | Operator-level and model-level comprehensive evaluation (planned for community co-development) |

## Tool Usage Guide

### PyPTO Pro Workflow — Multi-Agent Automated Operator Development

> Directory: `ops_tools/workflow/pypto-pro-workflow/`

A multi-agent automated operator development framework for Ascend NPUs. Three AI Agents (CANNBot for code generation, Kerminal for compilation & evaluation, Hermes for search & optimization) work collaboratively through a 10-stage pipeline with up to 20 automatic iterations, achieving full-process automation from requirement analysis to final report for fused operators.

**Core Features**:
- 10-stage pipeline: Requirement Analysis → Initial Implementation → Compilation & Deployment → Accuracy Evaluation → Performance Evaluation → Profiling → Search & Optimization → Insight Extraction → Code Modification → Final Report
- Cross-iteration memory (history mechanism): Tech Lead extracts insights/ledger each iteration, tracks the slowest cases, and avoids repeating failed approaches
- Automatic chip detection: Reads real UB/L1/L0A/L0B/L0C/core-count values directly from the CANN compiler
- Dynamic exit conditions: Exits only after performance targets are met AND Tech Lead confirms the fused implementation is a "true fusion"

**Quick Start**:
```bash
cd ops_tools/workflow/pypto-pro-workflow
# Develop an operator from scratch
python orchestrator.py --task-dir /path/to/cann-bench/tasks/level1/exp --max-iter 10

# Iteratively optimize an existing implementation
python orchestrator.py --task-dir /path/to/task \
  --init-impl /path/to/existing/impl \
  --optimize-hint "Conv+Sigmoid fusion: after conv on Cube, direct L0C→UB transfer for sigmoid within a single kernel" \
  --max-iter 20
```

**Validated Operators**: fused_conv_sigmoid (Cube-Vector on-chip direct transfer fusion, 2.87× speedup), fused_conv_tanh (dual-kernel pipeline fusion, 3.92× speedup).

Detailed documentation: [ops_tools/workflow/pypto-pro-workflow/README.md](ops_tools/workflow/pypto-pro-workflow/README.md)

### AI-MSF-Benchmark — Multi-Sensor Fusion Data Corruption & Evaluation

> Directory: `data_tools/ai-msf-benchmark/`

A robustness evaluation benchmark for Multi-Sensor Fusion (MSF) systems (ESEC/FSE 2023 paper). Provides multiple corruption mode generators that apply real-world degradations to clean datasets, producing corrupted camera images, LiDAR point clouds, calibration files, and temporal data for evaluating MSF system robustness.

**Implemented Corruption Types** (5 categories):
- **Weather**: rain, fog
- **Sensor**: brightness, darkness, motion_blur, defocus_blur, distortion
- **Noise**: gaussian_noise, impulse_noise
- **Spatial Misalignment**: rotation_x/y/z
- **Temporal Delay**: delay

> Corruption modes support both image and LiDAR channels; some modes support both simultaneously. See `corruption/corruption_configs.py` for the full list of corruption types and parameter configurations.

**Quick Start**:
```bash
cd data_tools/ai-msf-benchmark

# Generate brightness-corrupted images (severity 3)
python -m corruption.main simulate_image \
  --input_dir "./corruption/example/input_dir/image" \
  --output_dir "./corruption/example/output_dir/image/" \
  --corruption "brightness" --sev 3

# Generate Gaussian-noise-corrupted point clouds (severity 5)
python -m corruption.main simulate_lidar \
  --input_dir "./corruption/example/input_dir/velodyne" \
  --output_dir "./corruption/example/output_dir/velodyne/" \
  --corruption "gaussian_noise" --sev 5

# Simulate sensor spatial misalignment
python -m corruption.main simulate_calib \
  --input_dir "./corruption/example/input_dir/calib" \
  --output_dir "./corruption/example/output_dir/calib/" \
  --corruption "rotation_x" --sev 3
```

**Built-in MSF Systems** (7 systems, covering 3 task types × 3 fusion mechanisms): CLOCs, EPNet, FConv, MDANet, DFMOT, JMODT, TWISE.

Detailed documentation: [data_tools/ai-msf-benchmark/readme.md](data_tools/ai-msf-benchmark/readme.md)

### Image Noise Pattern — Real-World ISP Image Noise Generation

> Directory: `data_tools/image-noise-pattern/`

A data generation tool that simulates defects at each stage of the camera ISP (Image Signal Processing) pipeline. Each Python module corresponds to a specific ISP defect and can be invoked independently to generate degraded images.

**18 ISP Corruption Modes**:

| Category | Module | Defect |
|----------|--------|--------|
| Sensor | `CCD-D_CMOS-D.py` `sensorBroken.py` `SE.py` | CCD/CMOS damage, dead pixels, exposure anomalies |
| ISP Pre-processing | `BPC-D.py` `I-BLC_E-BLC.py` `LSC-D.py` | Bad pixel correction, black level correction, lens shading correction defects |
| ISP Core | `CFAI-D.py` `AWB-D.py` `CSC-D.py` `GC-d.py` `colorCorrection.py` | CFA interpolation, white balance, color space conversion, gamma correction defects |
| Optics/Environment | `FM-D.py` `LO.py` `F.py` `generateRain.py` `THE.py` `ME.py` | Defocus, lens obstruction, fog, rain, thermal effects, motion blur |
| Post-processing | `postProcessing.py` | Post-processing compression artifacts |

Each module runs independently. Refer to the in-module comments and examples for usage instructions.

Detailed documentation: [data_tools/image-noise-pattern/README.md](data_tools/image-noise-pattern/README.md)

## Version Compatibility

| Item | Version |
|------|---------|
| Product | Ascend950PR |
| NPU Chip | Ascend 950 |
| CANN | 9.1.0 |
| OS | Ubuntu 24.04.3 LTS (x86_64) |
| Python | >= 3.10 (validated with 3.12.3) |
| Node.js | >= 18 (validated with 18.19.1, required by CANNBot in workflow) |
| Kerminal | 0.8.12 (compilation & evaluation agent in workflow) |

## Directory Structure

```
multimodal-bench/
├── ops_tools/                           # Operator-level tools
│   ├── workflow/                        #   Multi-agent automated operator development framework
│   │   └── pypto-pro-workflow/          #     orchestrator + agents + roles + knowledge
│   ├── patterns/                        #   Pattern definitions (planned for community co-development)
│   └── miners/                          #   Pattern mining (planned for community co-development)
├── data_tools/                          # Data generation tools
│   ├── ai-msf-benchmark/               #   Multi-sensor fusion data corruption + 7 MSF system evaluations
│   └── image-noise-pattern/            #   18 ISP pipeline image noise generators
├── model_tools/                         # Model-level tools (planned for community co-development)
│   ├── matchers/                        #   Subgraph matching engine
│   └── rewriters/                       #   Automatic rewriting
├── benchmark/                           # Unified evaluation framework (planned for community co-development)
│   ├── cases/                           #   Test case definitions
│   ├── datasets/                        #   Evaluation dataset index
│   ├── metrics/                         #   Evaluation metrics
│   ├── runners/                         #   Evaluation runners
│   ├── baselines/                       #   Baseline data
│   └── reports/                         #   Evaluation report output
└── README.md
```

## Contributing

Contributions are welcome! Please refer to [CONTRIBUTING.md](CONTRIBUTING.md) for the contribution workflow and guidelines.

## Security

Please refer to [SECURITY.md](SECURITY.md) for security-related information.

## License

This project is licensed under the [CANN Open Software License Agreement Version 2.0](LICENSE).

## SIG Affiliation

This repository belongs to the CANN community [multimodal-ai SIG](https://gitcode.com/cann/community/blob/master/CANN/sigs/multimodal-ai/README.md).

## Issue Reporting

For questions or suggestions, please submit feedback via [Issues](../../issues).
