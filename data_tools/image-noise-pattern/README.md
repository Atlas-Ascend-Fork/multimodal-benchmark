# Image Noise Pattern — ISP 真实世界图像噪声生成

模拟相机 ISP（图像信号处理）链路各阶段缺陷的数据生成工具。每个 Python 模块对应一种 ISP 缺陷，可独立调用生成退化图像。

## 模块列表

| 类别 | 模块 | 缺陷 |
|------|------|------|
| 传感器 | `CCD-D_CMOS-D.py` `sensorBroken.py` `SE.py` | CCD/CMOS 损伤、坏像素、曝光异常 |
| ISP 前处理 | `BPC-D.py` `I-BLC_E-BLC.py` `LSC-D.py` | 坏点校正、黑电平校正、镜头阴影校正缺陷 |
| ISP 核心 | `CFAI-D.py` `AWB-D.py` `CSC-D.py` `GC-d.py` `colorCorrection.py` | CFA 插值、白平衡、色彩空间转换、Gamma 校正缺陷 |
| 光学/环境 | `FM-D.py` `LO.py` `F.py` `generateRain.py` `THE.py` `ME.py` | 失焦、镜头遮挡、雾、雨、热效应、运动模糊 |
| 后处理 | `postProcessing.py` | 后处理压缩伪影 |
| 辅助 | `RGB2RAW.py` `least_object.py` | RGB 转 RAW、最少目标筛选 |

## 使用方式

各模块独立运行，直接调用对应 Python 脚本即可。例如：

```bash
python AWB-D.py --input <input_image> --output <output_image>
```

具体参数见各模块内注释。

## 依赖

- numpy
- opencv-python
- Pillow
