# PyStar

PyStar 是一个面向空间转录组图像处理的 Python 管道，覆盖从原始显微镜图像到解码结果的主要处理链路。当前默认入口仍以 **Python-native PyStar 流程** 为主线；仓库中同时保留 MATLAB 兼容/提供者相关实现。MATLAB provider 可以通过 PyStar 调用，并可在 runtime manifest、transform artifact、sidecar、scope metadata 与 schema 合同全部验证通过时参与 release-valid `image_warp` 工作流。

## 当前支持状态

### 受支持
- Python-native PyStar 预处理、配准、Spot finding、信号提取与解码流程
- 当前推荐的 native spot finding baseline（`algorithm: peak_local_max`）
- 本次同步纳入的非 MATLAB 运行时改进
- 以 `config/experiment_config.yaml` 为示例入口、在本仓库内运行的 Python 管道
- 基于合同验证的 MATLAB provider 路线：当 MATLAB runtime manifest、entrypoint、transform manifest、`flow_3d` sidecar、field semantics、scope metadata 与 spot/intensity schema 均通过 fail-loud 校验时，MATLAB preprocessing/registration/spot finding/extraction 可用于 release-valid `image_warp` provenance。

### 仍需注意
- Python-native 示例仍是默认入口；MATLAB-backed 路线需要用户显式配置 `provider: matlab` 并准备可用的 MATLAB Engine 与仓库内 runtime 资源。
- `coordinate_mapping` 仍是 legacy diagnostic 路线，不随 MATLAB provider 的 `image_warp` 合同提升而变成 release-valid。
- MATLAB provider 不做静默 fallback：MATLAB runtime、entrypoint、sidecar 或输出 schema 不满足合同时会直接失败，而不是自动回退到 native provider。

> MATLAB 相关代码和 `matlab_runtime/` 资源保留在仓库中，目的是让开发、验证和兼容性工作保持可见。它们可以被 PyStar provider seam 调用；release-valid 与否由持久化 artifact 合同决定，而不是由 provider 名称本身决定。

## 安装

### 从源码安装（当前推荐）

```bash
git clone https://github.com/1508324011/pystar.git
cd pystar
pip install -e .
```

### 使用 pixi 环境

本仓库沿用旧版发布仓库布局，pixi manifest 位于 `env/pixi.toml`：

```bash
pixi install --manifest-path env/pixi.toml
pixi run --manifest-path env/pixi.toml -e pystar python -c "import pystar; print('ok')"
```

## 快速开始

### 1. 第一次运行前检查

- 示例配置：`config/experiment_config.yaml`
- MATLAB-provider parity 示例：`config/experiment_config_matlab_provider.yaml`（实验性；算法/provider/tiling 参数参照 2026-04-28 Experiment 1 MATLAB-provider as-run 配置，用于对齐 all-MATLAB/STATES 风格流程；release-valid 取决于 artifact 合同校验）
- 配置说明：`config/README.md`
- 当前示例配置默认走 **Python-native** 主线（`provider: native`）
- 当前示例使用 `pipeline.spot_finding.algorithm: peak_local_max`，参考阈值为 `threshold_rel: 0.1`
- 若你手动启用 MATLAB provider，请将其视为测试功能，而不是默认生产路径，并确认 MATLAB Engine、runtime manifest、entrypoint、transform sidecar 与 schema 合同都可用；MATLAB-provider 示例中的 local registration 使用 full-FOV 覆盖下的 4x4 subtile tiling。示例中的数据路径仍是发布站点占位路径，运行前必须按实际数据修改。

建议不要直接改仓库内的示例文件，而是复制一份自己的配置：

```bash
cp config/experiment_config.yaml my_config.yaml
```

运行前至少修改三类路径：

```yaml
dataset:
  raw_data_path: "/path/to/raw_data"

codebook:
  gene_list: "/path/to/raw_data/genes.csv"

pipeline:
  output:
    directory: "/path/to/output_pystar"
```

第一次 smoke test 建议只跑一个 FOV，例如：

```yaml
dataset:
  fov_list: [1]
```

默认示例的文件匹配模式是：

```yaml
filename_pattern: "round{round:03d}/Position{fov}/*_ch{ch}.tif"
```

因此原始图像目录应能匹配类似结构：

```text
/path/to/raw_data/
  round001/
    Position1/
      image_ch0.tif
      image_ch1.tif
      image_ch2.tif
      image_ch3.tif
  round002/
    Position1/
      image_ch0.tif
      image_ch1.tif
      image_ch2.tif
      image_ch3.tif
```

如果你的文件命名或目录层级不同，请先调整 `dataset.filename_pattern`，再运行 pipeline。

可以先只验证配置能被当前 PyStar schema 读取：

```bash
pixi run --manifest-path env/pixi.toml -e pystar \
  python -c "from pystar.infrastructure import load_config; cfg = load_config('my_config.yaml'); print(cfg.dataset.parsed_fovs[:5])"
```

### 2. 本地单 FOV smoke test

没有 SLURM 时，直接调用 worker 脚本跑一个 FOV：

```bash
pixi run --manifest-path env/pixi.toml -e pystar \
  python scripts/batch_pystar.py \
  --config my_config.yaml \
  --task_id 1
```

`--task_id` 是 `dataset.fov_list` 的 **1-based 索引**，不一定等于 FOV 编号。例如：

```yaml
dataset:
  fov_list: [7, 9, 12]
```

对应关系是：

```text
--task_id 1 -> FOV 7
--task_id 2 -> FOV 9
--task_id 3 -> FOV 12
```

确认单个 FOV 输出正常后，再扩大 `fov_list` 范围。

### 3. Python API 方式运行完整单个 FOV 流程

```python
from pystar.infrastructure import load_config
from pystar.preprocessing import DataSanitizer
from pystar.registration import RegistrationEngine
from pystar.spot_finding import SpotFinder
from pystar.mining import SignalMiner
from pystar.decoding import Decoder
from pystar.io import ImageLoader

cfg = load_config("my_config.yaml")

fov_id = cfg.dataset.parsed_fovs[0]

sanitizer = DataSanitizer(cfg)
sanitizer.sanitize_fov(fov_id)

loader = ImageLoader(cfg)
data = loader.load_fov(fov_id)

reg_engine = RegistrationEngine(cfg)
reg_engine.register_fov(data, fov_id)

finder = SpotFinder(cfg)
finder.find_spots_in_fov(fov_id)

miner = SignalMiner(cfg)
miner.mine_fov(fov_id)

decoder = Decoder(cfg)
decoder.decode_fov(fov_id)
```

### 4. 使用 SLURM 批处理脚本

```bash
bash scripts/run_pystar.sh my_config.yaml
```

脚本会通过 `env/pixi.toml` 环境提交 `scripts/batch_pystar.py`，这是当前发布仓库布局下的集群批处理入口。

> `scripts/run_pystar.sh` 是 SLURM 模板，不是通用本地运行脚本。使用前请按你的集群修改 partition、qos、time、`Batch_FOV`、`CPUS_PER_FOV` 等资源设置。若你不在 SLURM 环境中，请直接使用上面的单 FOV smoke test 命令或 Python API。

## 仓库结构

```text
repo-root/
├── pystar/
│   ├── infrastructure.py
│   ├── io.py
│   ├── preprocessing.py
│   ├── registration.py
│   ├── spot_finding.py
│   ├── extraction_utils.py
│   ├── mining.py
│   ├── decoding.py
│   ├── visualization.py
│   ├── tiling.py
│   ├── matlab_engine_bootstrap.py
│   ├── matlab_preprocessing.py
│   ├── matlab_registration.py
│   ├── matlab_spot_finding.py
│   └── matlab_extraction.py
├── matlab_runtime/          # MATLAB provider 运行时资源（release 合同由 artifact 校验决定）
├── config/
├── scripts/
├── env/
├── notebooks/
├── sitecustomize.py         # 可选 MATLAB Engine 启动辅助（可用 PYSTAR_DISABLE_MATLAB_ENGINE_BOOTSTRAP=1 禁用）
├── pyproject.toml
├── README.md
└── CHANGELOG.md
```

## 依赖

核心运行依赖包括：
- numpy
- pandas
- scipy
- tifffile
- dask / distributed
- xarray
- scikit-image
- SimpleITK
- matplotlib / seaborn
- pydantic
- PyYAML
- opencv-python-headless

可选/按需依赖：
- cupy / dask-cuda（GPU 相关实验）
- spotiflow（特定 spot-finding 算法）
- MATLAB Engine for Python（仅在显式启用 MATLAB provider 时需要）

## MATLAB 相关说明

仓库内包含以下 MATLAB 相关资源：
- `pystar/matlab_*.py`
- `matlab_runtime/`
- `scripts/check_matlab_engine.py`
- `sitecustomize.py`

这些内容用于**开发、验证和兼容性工作**。当前发布说明下：
- 它们可以通过 PyStar provider seam 被调用
- 它们不是默认路径，需要显式配置 MATLAB provider
- 它们的 release-valid 状态由 runtime manifest、transform artifact、sidecar、field semantics、scope metadata 和 schema 合同决定
- 它们失败时应显式报错，而不是静默回退成 native 路径

## 许可证

本项目采用 MIT 许可证。详见 `LICENSE`。
