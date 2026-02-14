# PyStar

PyStar 是一个用于空间转录组学（Spatial Transcriptomics）图像处理和分析的 Python 管道。它提供了从原始显微镜图像到基因表达矩阵的完整流程，包括图像预处理、配准、斑点检测和解码。

## 功能特点

- **图像预处理**：中值滤波、高斯模糊、直方图匹配、背景扣除等
- **图像配准**：支持全局对齐和局部变形校正（光流法、B-spline、Demons 3D）
- **斑点检测**：支持 Spotiflow、Blob DoG、Peak Local Max 等多种算法
- **基因解码**：支持自定义编码规则和多色通道解码
- **GPU 加速**：支持 CUDA 加速的图像处理
- **并行处理**：基于 Dask 的分布式计算

## 安装

### 环境要求

- Python >= 3.9
- CUDA 12.x（可选，用于 GPU 加速）

### 使用 pip 安装

```bash
pip install pystar
```

### 使用 pixi 安装（推荐）

```bash
pixi install
```

### 从源码安装

```bash
git clone https://github.com/yourusername/pystar.git
cd pystar
pip install -e .
```

## 快速开始

### 1. 准备配置文件

创建 `experiment_config.yaml` 文件，配置数据集路径、图像结构、通道定义等。

示例配置请参考 `config/experiment_config.yaml`

### 2. 运行预处理

```python
from pystar.preprocessing import DataSanitizer
from pystar.infrastructure import ExperimentConfig

# 加载配置
config = ExperimentConfig.from_yaml("config/experiment_config.yaml")

# 创建处理器
sanitizer = DataSanitizer(config)

# 处理单个 FOV
sanitizer.sanitize_fov(fov_id=1)
```

### 3. 运行配准

```python
from pystar.registration import RegistrationEngine

engine = RegistrationEngine(config)
engine.register_all_fovs()
```

### 4. 斑点检测和解码

```python
from pystar.spot_finding import SpotFinder
from pystar.decoding import Decoder

# 斑点检测
finder = SpotFinder(config)
spots = finder.find_spots(fov_id=1)

# 解码
decoder = Decoder(config)
results = decoder.decode(spots)
```

## 项目结构

```
pystar/
├── pystar/              # 核心模块
│   ├── __init__.py
│   ├── preprocessing.py    # 图像预处理
│   ├── registration.py     # 图像配准
│   ├── spot_finding.py     # 斑点检测
│   ├── decoding.py         # 基因解码
│   ├── io.py               # 输入输出
│   ├── infrastructure.py   # 基础设施/配置
│   ├── mining.py           # 数据挖掘
│   └── visualization.py    # 可视化
├── notebooks/           # Jupyter notebooks
├── scripts/             # 批处理脚本
├── config/              # 配置文件示例
├── env/                 # 环境配置
├── pyproject.toml       # 项目配置
├── README.md
└── LICENSE
```

## 依赖

核心依赖：
- numpy
- pandas
- tifffile
- dask
- scipy
- scikit-image
- opencv-python
- tqdm

可选依赖：
- cupy (GPU 加速)
- dask-cuda (GPU 分布式计算)
- spotiflow (深度学习斑点检测)

## 支持的编码方案

- **3-color**：AT/GT/TT/AG/GG/TG/AA/GA/TA
- **4-color**：16 种双碱基组合
- **自定义**：可根据实验设计灵活配置

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 作者

- Zhurui Zenglab

## 贡献

欢迎提交 Issue 和 Pull Request！

## 引用

如果您在研究中使用了 PyStar，请引用：

```
PyStar: A Python Pipeline for Spatial Transcriptomics Analysis
```
