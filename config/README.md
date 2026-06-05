# PyStar 配置说明

`experiment_config.yaml` 是当前发布仓库的示例入口。它采用当前开发工作区的可读版式：用分段标题组织模块、用行内列表表示短参数、并在关键参数旁直接写清楚可修改项和含义。`pipeline` 段已经同步到当前最接近 MATLAB/STATE 行为的 Python-native 参考参数。

运行前通常只需要先改三类路径：

- `dataset.raw_data_path`：原始 TIFF 数据根目录。
- `codebook.gene_list`：基因表/码本 CSV。
- `pipeline.output.directory`：PyStar 输出目录。

本目录提供两个入口：

- `experiment_config.yaml`：默认 Python-native 示例，适合作为发布版主线入口。
- `experiment_config_matlab_provider.yaml`：MATLAB-provider parity 示例，保留同一数据/码本结构，但把 preprocessing、registration、spot finding 和 extraction 都切到 `provider: matlab`。算法/provider/tiling 参数参照 2026-04-28 Experiment 1 MATLAB-provider as-run 配置（Position1 gene-level Pearson 约 `0.9971`，gene-state Pearson 约 `0.9943`）；数据路径仍是发布示例路径，运行前需要改成自己的数据位置。

## `dataset`

- `raw_data_path`：原始输入根目录。所有原始图像路径都应从这里和 `filename_pattern` 推导，不要在代码里硬编码本地路径。
- `filename_pattern`：每张图像的相对路径模板，支持 `{round}`、`{fov}`、`{ch}` 占位符。
- `pixel_size_xy_nm` / `pixel_size_z_nm`：像素物理尺寸，用于记录显微镜几何信息。
- `dimensions`：单个 FOV 的体数据尺寸，当前 Leica 示例为 `z=42`、`height=2048`、`width=2048`。
- `io_chunk_size`：读写大图像时使用的 chunk 尺寸。
- `fov_list`：要处理的 FOV 范围，例如 `1-324`。
- `round_structure`：每轮可用 channel 列表。示例用 `1: [0, 1, 2, 3]` 这种逐轮展开写法，便于检查输入数据结构。
- `channel_roles`：每个 channel 的角色，`seq` 表示测序 channel，`anchor` 表示 anchor channel。

## `codebook`

- `gene_list`：基因/条码 CSV 路径。
- `channel_base_index`：码本中 channel 编码的起始编号。当前 Leica/STATE 相关数据使用 `1`。
- `encoding_tables`：碱基组合到 channel 的映射表。
- `topology.func`：条码拓扑变换函数，当前示例使用 `reverse_string`。
- `topology.structure`：各条码片段的长度、轮次、CSV 切片、anchor base 和编码表。
- `topology.physical_order`：物理读段顺序。

## `providers`

`providers.matlab` 记录 MATLAB provider seam 的 runtime 路径和入口函数。默认 pipeline 仍使用 `provider: native`；当用户显式选择 MATLAB provider 且 runtime manifest、entrypoint、transform artifact、`flow_3d` sidecar、scope metadata 与 schema 合同都验证通过时，MATLAB provider 路线可以参与 release-valid `image_warp` provenance。

MATLAB provider 不做静默 fallback：MATLAB runtime、MATLAB Engine、entrypoint、sidecar 或输出 schema 不可用/不合法时应直接失败，而不是自动回退到 native provider。

`providers.matlab.shared_session` 是可选的 MATLAB Engine 复用配置，当前由 `scripts/batch_pystar.py` 使用：

- `enabled`：默认 `false`，保持旧的每个 MATLAB backend 自管 Engine 生命周期。设为 `true` 后，批处理会创建一个显式共享 owner，并把同一个 owner 注入 preprocessing、registration、spot finding 和 extraction。
- `name`：可填具体共享 session 名；为 `null` 时自动生成 `pystar_{config_stem}_{config_hash8}_{run_id}`。Slurm array 任务使用 `slurm_{SLURM_JOB_ID}_{SLURM_ARRAY_TASK_ID}` 作为 `run_id`，普通进程使用 `pid_{os.getpid()}`，避免同一 YAML 的并行 worker 误连同一个 MATLAB session。
- `lifetime`：`run` 或 `fov`，默认 `run`。当前批处理 worker 一次处理一个 FOV，因此两者都在该 worker 边界释放 PyStar-owned session；`fov` 保留给隔离/调试。
- `health_check_timeout_s`：共享 session attach/start 后的健康检查超时。

共享 session 只按确定名称连接：不会调用未命名 `connect_matlab()`，不会连接“第一个可用 session”，健康检查或 sentinel 身份不匹配时会 fail loudly，也不会静默启动替代 session 或回退到 native provider。

## `pipeline`

- `scope_mode: full_fov`：以完整 FOV 为处理单位，保持当前 Python-native 参考路径。
- `accelerator: cpu`：默认 CPU。GPU 相关路径仍是实验性能力。
- `field_semantics`：位移场语义记录。当前为 residual field，按 global 后 local 的顺序组合，状态仍标记为 provisional。

### `pipeline.preprocessing`

`sequence` 按顺序执行预处理步骤：

- `none`：占位步骤，保持原始输入进入统一步骤链。
- `min_max_normalize`：强度归一化。
- `histogram_match` with `scope: inter_round`：跨 round 直方图匹配。
- `histogram_match` with `scope: intra_round`：round 内 channel 直方图匹配。
- `morpho_reconstruction_contrast`：形态学重建增强；当前参考参数为 `radius: 6`、`downsample_factor: 0.25`。

默认每步 `provider: native`，对应纯 Python 实现。`native_volume_workers` 是可选的 native FOV-volume 调度开关，省略或设为 `1` 时保持历史串行执行；设为正整数时，只有在 inter-round / intra-round 参考图已经物化后，eligible volume 才会以有界 worker 数并行处理。该开关只改变调度，不改变 `histogram_match` 语义、clean TIFF 文件名或 production provenance schema。

`native_volume_workers` 的真实数据验证计划：在声明任何端到端加速前，必须用同一验证目录、同一 config、同一 FOV / round / channel 范围分别运行基线提交 `fec32e7` 和候选提交；报告需记录 PyStar 源路径与 commit hash、验证 config、输出 artifact 路径、clean TIFF 文件 SHA/数组等价性、Stage29 histogram real-match/no-reference attribution，以及 paired wall-time/profile 对比。如果 clean 输出不等价，或只验证了不同数据面，不得宣称速度提升。

### `pipeline.registration`

- `reference_round: 1`：配准参考 round。为保持 MATLAB/STATE parity，除非有新证据，不建议改动。
- `source.method: mip_all_channels`：使用指定测序 channel 的 MIP 作为配准源。
- `source.mip_channels`：参与 MIP 的 channel，当前为 `0, 1, 2`。
- `global`：全局 3D phase correlation，当前 `provider: native`，`max_shift: 200`。
- `local`：局部 `demons_3d` refinement，当前 `num_iter: 50`、`smoothing_sigma: 1.0`。
- 默认 native 示例没有开启 local tiling；MATLAB-provider 示例显式开启 `local.params.demons_3d.use_tiling: true`、`sqrt_pieces: 4`、`tiling_layout_policy: matlab_subtile`，表示 full-FOV 覆盖下按 4x4 subtile 计算 local demons 后 stitch 回完整 flow。
- `guards.reject_if_correlation_worse: true`：局部 refinement 如果让相关性变差，应拒绝该 refinement。
- `save_displacement_fields`：保存位移场，便于后续 signal extraction 和调试。
- `save_registered_images`：是否保存 registered image；默认关闭以减少输出体积。

### `pipeline.spot_finding`

- `algorithm: peak_local_max`：当前推荐的 native spot finding 算法入口。
- `provider: native`：默认走纯 Python spot finding。
- `reference_round: 1`：spot finding 的参考 round。
- `method: max_intensity`：按最大强度聚合候选点。
- `peak_local_max.threshold_rel: 0.1`：当前参考阈值，与 MATLAB/STATE max3d 的 `intensity_threshold=0.1` 对齐。
- `peak_local_max.min_distance: 2` 和 `exclude_border: true`：spot finding 的默认空间参数，通常先保持示例值。
- `spotiflow` / `blob_dog`：备用算法参数，不是当前推荐参考路径。

### `pipeline.extraction`

- `method: box_sum`：以候选点附近 box 进行信号积分。
- `provider: native`：默认走纯 Python extraction。
- `integration_box: [3, 3, 3]`：积分窗口尺寸。
- `handle_out_of_bounds: pad_zero`：边界外按 0 padding。
- `transform_application_mode: image_warp`：使用图像 warp 后的体数据进行 extraction；这是当前参考路径。

### `pipeline.output` 与 `pipeline.qc`

- `output.directory`：输出根目录。
- `output.save_qc_images`：是否保存 QC 图像。
- `qc.enable`：是否启用 QC 输出。
- `qc.alignment_check` / `qc.correlation_plot`：配准和相关性 QC 图的参数。

## MATLAB provider 状态

当前 PyStar 可以通过 MATLAB provider 调用 MATLAB-backed preprocessing/registration/spot finding/extraction seam；内部 benchmark 中，全 MATLAB-provider 路线已经能接近 STATE。这个能力按合同验证发布：provider 名称本身不会自动把 release contract 降级为 `debug_only`，但 runtime manifest、transform manifest、`flow_3d` sidecar、field semantics、scope metadata、spot/intensity schema 任一不满足合同都会 fail loudly，`image_warp` 也仍要求 `release_gate.status == "valid"`。

如果需要复现 all-MATLAB provider 路线，请从 `experiment_config_matlab_provider.yaml` 开始改路径。该示例的算法参数对齐 2026-04-28 Experiment 1 MATLAB-provider as-run 配置；它保留 `scope_mode: full_fov`，但 local registration 使用 MATLAB subtile 4x4 tiling；这和“只处理一个 tile_local”不同，输出仍覆盖完整 Position。`coordinate_mapping` 仍是 legacy diagnostic 路线，不随 MATLAB provider 的 `image_warp` 合同提升而变成 release-valid。
