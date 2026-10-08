# AI-MSF-Benchmark 数据指南

本文档说明仓库中未包含的外部依赖、数据集和生成产物的位置。

## 仓库未包含的内容

以下内容因体积原因未纳入仓库，需要使用时自行获取：

| 类别 | 路径 | 大小 | 说明 |
|------|------|------|------|
| 三方依赖 — rain-rendering | `corruption/3rd_parts/camera/` | ~30000 文件 | 雨雾天气的相机图像渲染库（可选，仅 `simulate_weather` 需要） |
| 三方依赖 — LISA | `corruption/3rd_parts/lidar/` | ~500 文件 | 雨雾天气的 LiDAR 点云模拟库（可选，仅 `simulate_weather` 需要） |
| 模型权重 | 各 `system/` 子目录 | 67MB+ | 7 个 MSF 系统的预训练权重（`.pth`） |
| 编译产物 | `system/FConv/ops/pybind11/build/` 等 | ~50MB | C++ 扩展的编译中间文件（`.o` `.so`） |
| 可视化输出 | 各 `system/` 子目录 | ~88MB | 跟踪/检测结果的可视化图片（`.png` `.jpg`）和视频（`.avi`） |
| 论文与文档 | `document/` | ~3MB | 论文 PDF、安装说明 PDF |

## 获取外部依赖

### 三方依赖（仅使用 `simulate_weather` 时需要）

```bash
# rain-rendering（相机雨雾渲染）
cd corruption/3rd_parts/
mkdir -p camera && cd camera
git clone https://github.com/cv-rits/rain-rendering.git .
# 按 rain-rendering 的 README 安装依赖、生成深度图、下载粒子模拟文件

# LISA（LiDAR 雨雾模拟）
cd ../
mkdir -p lidar && cd lidar
git clone https://github.com/velatkilic/LISA.git .
# 按 LISA 的 README 安装依赖
```

### 数据集（KITTI 为主）

```bash
# KITTI 目标检测数据集
# 下载地址：https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d
# 放置路径按 config.py 中 corruption_base_public_dir 配置，默认结构：
#   <base_dir>/object/training/image_2/      ← 相机图像
#   <base_dir>/object/training/velodyne/     ← LiDAR 点云
#   <base_dir>/object/training/calib/        ← 标定文件
#   <base_dir>/object/training/label_2/      ← 标注文件
```

使用前需修改 `config.py` 中的路径配置：

```python
path_config = {
    "project_dir": "/your/project/path",              # 项目根目录
    "corruption_base_public_dir": "/your/kitti/path",  # KITTI 数据集根目录
    "dataset": "kitti"
}
```

### 各 MSF 系统的权重和环境

7 个 system 各有独立的环境要求和预训练权重，按各自目录下的 README 操作：

| 系统 | 任务 | 权重获取 |
|------|------|---------|
| CLOCs | 目标检测 | 见 `system/clocs/README.md` |
| EPNet | 目标检测 | 见 `system/EPNet/README.md` |
| FConv | 目标检测/深度补全 | 见 `system/FConv/README.md`，C++ 扩展需本地编译 |
| MDANet | 深度补全 | 见 `system/MDANet/README.md` |
| DFMOT | 多目标跟踪 | 见 `system/DFMOT/README.md` |
| JMODT | 多目标跟踪 | 见 `system/JMODT/README.md` |
| TWISE | 深度补全 | 见 `system/TWISE/README.md` |

## 生成产物说明

### 腐蚀数据生成

运行 `corruption/main.py` 后，生成的退化数据放在你指定的 `--output_dir` 中：

```
<output_dir>/
├── image/                   ← simulate_image 产出：退化后的相机图像（.png）
├── velodyne/                ← simulate_lidar 产出：退化后的 LiDAR 点云（.bin）
├── calib/                   ← simulate_calib 产出：偏移后的标定文件（.txt）
├── delay_image/             ← simulate_delay 产出：时序延迟的图像序列
│   └── 0000/               #   按序列号组织
└── rain/                    ← simulate_weather 产出（需要 3rd_parts）
    ├── image/
    │   ├── rainy_image/     #   叠加雨滴后的图像
    │   └── rain_mask/       #   雨滴遮罩
    └── velodyne/            #   雨天退化的点云
```

每种腐蚀模式支持 severity 1-5 五个等级，等级越高退化越严重。

### MSF 系统评测

将生成的腐蚀数据软链接到各 system 的数据目录（`config.py` 中 `work_dir` 自动处理），然后按各系统的评测流程运行。评测产出：

| 产出 | 位置 | 说明 |
|------|------|------|
| 检测/跟踪结果 | 各 system 的输出目录 | 边界框、跟踪轨迹（.txt） |
| 评测指标 | 各 system 的 eval 目录 | AP/MOTA/HOTA 等（.txt/.json） |
| 可视化 | 各 system 的 visualization 目录 | 检测框叠加图（.png）、跟踪视频（.avi） |
| 聚合结果 | `tools/aggerate_bounding_box.py` 产出 | RQ4 分支聚合结果 |

### 完整工作流

```
干净数据集（KITTI）
    │
    ▼  corruption/main.py（5 条管线 × severity 1-5）
腐蚀数据集
    │
    ▼  软链接到各 system 数据目录
7 个 MSF 系统分别评测
    │
    ▼  对比干净 vs 腐蚀的指标差距
鲁棒性评测结论
```
