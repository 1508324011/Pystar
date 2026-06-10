import hashlib
import re
import yaml
from numbers import Integral
from pathlib import Path
from typing import List, Dict, Union, Any, Optional, Literal, Tuple
from pydantic import BaseModel, model_validator, Field, ValidationError, ConfigDict


REQUIRED_PIPELINE_RELEASE_FIELDS = (
    "scope_mode",
    "accelerator",
)
REQUIRED_EXTRACTION_RELEASE_FIELDS = ("transform_application_mode",)
SUPPORTED_PREPROCESSING_METHODS = {
    "median_filter",
    "gaussian_blur",
    "histogram_match",
    "gamma_correction",
    "difference_of_gaussians",
    "clip_percentile",
    "clahe",
    "morpho_reconstruction_contrast",
    "min_max_normalize",
    "none",
}
SUPPORTED_NATIVE_PREPROCESSING_METHODS = set(SUPPORTED_PREPROCESSING_METHODS)
SUPPORTED_MATLAB_PREPROCESSING_METHODS = {
    "none",
    "min_max_normalize",
    "histogram_match",
    "morpho_reconstruction_contrast",
}
SUPPORTED_PREPROCESSING_PROVIDERS = {"native", "matlab"}
SUPPORTED_SPOT_FINDING_PROVIDERS = {"native", "matlab"}
SUPPORTED_EXTRACTION_PROVIDERS = {"native", "matlab"}
SUPPORTED_GLOBAL_REGISTRATION_METHODS = {"phase_corr_3d"}
SUPPORTED_LOCAL_REGISTRATION_METHODS = {"optical_flow", "bspline", "demons_3d"}
SUPPORTED_REGISTRATION_PROVIDERS = {"native", "matlab"}

# --- 辅助函数 ---

def parse_range_string(s: Union[str, int, List[int]]) -> List[int]:
    """
    解析 FOV 列表配置。
    支持: 1, "1-56", [1, 2, 3], "1,2,3"
    """
    if isinstance(s, list):
        return s
    if isinstance(s, int):
        return [s]
    if isinstance(s, str):
        s = s.strip()
        try:
            # 处理 "1-56" 这种范围
            if '-' in s:
                start, end = map(int, s.split('-'))
                return list(range(start, end + 1))
            # 处理 "1,2,3" 这种逗号分隔
            elif ',' in s:
                return [int(x.strip()) for x in s.split(',')]
            else:
                return [int(s)]
        except (ValueError, AttributeError):
            raise ValueError(f"Invalid FOV range format: '{s}'. Use list or 'start-end' string.")
    raise ValueError(f"Unknown type for fov_list: {type(s)}")

# --- Pydantic Models (数据校验) ---

class DatasetConfig(BaseModel):
    """Raw image input contract for one experiment.

    This model describes how PyStar discovers raw TIFF stacks before any
    pipeline stage runs. `filename_pattern` is formatted with `round`, `fov`
    and `ch`; `round_structure` states which physical channels exist in each
    imaging round; `channel_roles` declares which channels are sequencing
    signal versus auxiliary channels. Runtime code must derive all raw input
    paths from these fields, never from hard-coded local paths.
    """

    raw_data_path: Path
    filename_pattern: str
    pixel_size_xy_nm: float
    pixel_size_z_nm: float
    dimensions: Dict[str, int]
    io_chunk_size: Dict[str, int]
    
    # Raw input from yaml
    # Pydantic 2.0 对 Union 的解析更严格，我们允许任意类型进入，然后手动校验
    fov_list: Union[str, List[int], int]     
    parsed_fovs: List[int] = []              
    
    # Explicit structure
    round_structure: Dict[int, List[int]]
    channel_roles: Dict[int, str]

    @model_validator(mode='after')
    def parse_fovs(self):
        """
        Pydantic V2 风格验证器。
        自动把 '1-56' 转换成列表。
        """
        raw = self.fov_list
        try:
            # 修改模型内部的 parsed_fovs 属性
            self.parsed_fovs = parse_range_string(raw)
        except Exception as e:
            raise ValueError(f"Error parsing fov_list: {e}")
        return self

class BlueprintSegment(BaseModel):
    """One logical barcode segment in the codebook topology.

    A segment maps a slice of the gene-list sequence (`csv_slice`, 1-based
    inclusive in the YAML) to one or more physical imaging rounds. The encoder
    named by `encoding_table` converts bases in that slice into color/channel
    indices. `anchor_base`, when present, provides start/end bases used by the
    decoder to validate barcode pattern semantics.
    """

    id: str
    rounds: List[int]       # 物理轮次，如 [1, 2, 3, 4, 5]
    csv_slice: List[int]    # CSV 里的切片 [Start, End] (1-based physical index recommended for config, logic handles conversion)
    anchor_base: Optional[List[str]] = None   # 每个 segment 的 anchor bases
    encoding_table: str     # 使用哪张表
    
    
    
class TopologyConfig(BaseModel):
    """Barcode assembly blueprint shared by codebook compilation and decoding.

    `structure` defines each named segment; `physical_order` says how those
    segments are concatenated to match the round order emitted by extraction.
    `func` is a global sequence transform applied before segment slicing, for
    example `reverse_string` for data whose gene-list sequence orientation is
    opposite to imaging order.
    """

    func: str = "none" # "reverse_string" etc.
    structure: List[BlueprintSegment]
    physical_order: List[str] # 决定最终 barcode 拼接的顺序

class CodebookConfig(BaseModel):
    """Codebook and color-encoding configuration.

    `gene_list` points to the gene/sequence CSV. `encoding_tables` map base
    pairs or short motifs to color-channel IDs. `channel_base_index` tells
    PyStar whether the table is authored with 0-based or 1-based channel
    numbers; runtime decoding normalizes to Python 0-based color indices.
    """

    gene_list: Path
    channel_base_index: int  # 用户填 0 或 1
    encoding_tables: Dict[str, Dict[str, int]]
    topology: TopologyConfig

    # 禁止在实例初始化后修改数据，强制不可变
    model_config = ConfigDict(frozen=True) 

    @property
    def normalized_encoding_map(self) -> Dict[str, Dict[str, int]]:
        """Runtime 转换，不污染 Config 状态"""
        base = self.channel_base_index
        if base == 0:
            return self.encoding_tables
        return {
            table_name: {pair: color - base for pair, color in mapping.items()}
            for table_name, mapping in self.encoding_tables.items()
        }

class PreprocessingStep(BaseModel):
    """
    定义流水线中的单一步骤。
    对应 YAML 中的:
      - method: "median_filter"
        params: {kernel_size: 3}
    """
    method: str
    provider: Literal["native", "matlab"] = "native"
    # 允许 params 为空，默认为空字典。
    # params 里的值可能是 int, float, str，所以用 Any
    params: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode='after')
    def validate_method_provider_pair(self) -> 'PreprocessingStep':
        if self.method not in SUPPORTED_PREPROCESSING_METHODS:
            raise ValueError(
                f"Unsupported preprocessing method: {self.method!r}. "
                f"Supported methods: {sorted(SUPPORTED_PREPROCESSING_METHODS)}"
            )

        if self.provider not in SUPPORTED_PREPROCESSING_PROVIDERS:
            raise ValueError(
                f"Unsupported preprocessing provider: {self.provider!r}. "
                f"Supported providers: {sorted(SUPPORTED_PREPROCESSING_PROVIDERS)}"
            )

        if self.provider == "native" and self.method not in SUPPORTED_NATIVE_PREPROCESSING_METHODS:
            raise ValueError(
                f"Preprocessing method {self.method!r} is not supported by provider='native'"
            )

        if self.provider == "matlab" and self.method not in SUPPORTED_MATLAB_PREPROCESSING_METHODS:
            raise ValueError(
                f"Preprocessing method {self.method!r} is not supported by provider='matlab'. "
                f"Supported MATLAB methods: {sorted(SUPPORTED_MATLAB_PREPROCESSING_METHODS)}"
            )

        return self
    
    model_config = ConfigDict(frozen=True)

class PreprocessingConfig(BaseModel):
    """
    对应 YAML 中的:
    preprocessing:
      enable: true
      save_path: "clean_data"
      sequence: [...]
    """
    enable: bool = True
    
    # 这里是关键：我们强制要求 sequence 是一个 PreprocessingStep 的列表
    # Pydantic 会自动遍历列表，验证每一项都符合结构
    sequence: List[PreprocessingStep] = Field(default_factory=list)

    # Native provider FOV-volume scheduler.  The default keeps the historical
    # serial round/channel loop.  Values greater than one opt into bounded
    # reference-aware parallel execution for volumes whose calibration
    # references have already been materialized.
    native_volume_workers: int = 1

    @model_validator(mode='before')
    @classmethod
    def reject_invalid_native_volume_workers(cls, data: Any) -> Any:
        if isinstance(data, dict):
            raw_workers = data.get("native_volume_workers", 1)
            if isinstance(raw_workers, bool) or not isinstance(raw_workers, Integral):
                raise ValueError(
                    "pipeline.preprocessing.native_volume_workers must be a positive integer"
                )
        return data

    @model_validator(mode='after')
    def validate_native_volume_workers(self) -> 'PreprocessingConfig':
        if isinstance(self.native_volume_workers, bool) or int(self.native_volume_workers) <= 0:
            raise ValueError("pipeline.preprocessing.native_volume_workers must be a positive integer")
        return self
    
    model_config = ConfigDict(frozen=True)


class MatlabMorphologyConfig(BaseModel):
    """MATLAB morphology parameters for MATLAB-backed preprocessing.

    These values are forwarded to the repo-local MATLAB preprocessing runtime.
    They only affect preprocessing steps that explicitly select
    `provider: matlab`; native preprocessing ignores this section.
    """

    method: Literal["2d", "2d_thres", "3d"] = "2d"
    radius: int = Field(default=2, gt=0)
    height: int = Field(default=3, gt=0)

    model_config = ConfigDict(frozen=True, extra='forbid')


class MatlabPreprocessingProviderConfig(BaseModel):
    """Boundary contract for MATLAB-backed preprocessing.

    `runtime_path` must stay inside the PyStar repository and point to the
    MATLAB functions shipped with this package. The dtype/loader fields record
    how arrays cross the Python/MATLAB boundary so that benchmark runs can be
    reproduced and audited.
    """

    runtime_path: Path = Path("matlab_runtime/pystar_preprocessing")
    entrypoint: str = "pystar_preprocess_entry"
    loader_input_format: Literal["uint8"] = "uint8"
    loader_output_dtype: Literal["uint8"] = "uint8"
    use_gpu: Literal[False] = False
    morphology: MatlabMorphologyConfig = Field(default_factory=MatlabMorphologyConfig)

    @model_validator(mode='after')
    def validate_entrypoint(self) -> 'MatlabPreprocessingProviderConfig':
        if not self.entrypoint.strip():
            raise ValueError("providers.matlab.preprocessing.entrypoint must be a non-empty MATLAB function name")
        return self

    model_config = ConfigDict(frozen=True, extra='forbid')


class MatlabRegistrationProviderConfig(BaseModel):
    """Boundary contract for MATLAB-backed registration.

    Global and local registration use separate MATLAB entrypoints. The current
    local MATLAB seam is intentionally narrow: it supports the `demons_3d`
    path used for MATLAB parity experiments and records transfer/dtype choices
    explicitly instead of hiding them behind fallback behavior.
    """

    runtime_path: Path = Path("matlab_runtime/pystar_registration")
    entrypoint: str = "pystar_register_global_entry"
    local_entrypoints: Dict[str, str] = Field(
        default_factory=lambda: {"demons_3d": "pystar_register_local_demons_entry"}
    )
    volume_transfer_mode: Literal["temporary_tiff"] = "temporary_tiff"
    input_volume_dtype: Literal["uint8"] = "uint8"
    use_gpu: Literal[False] = False

    @model_validator(mode='after')
    def validate_entrypoint(self) -> 'MatlabRegistrationProviderConfig':
        if not self.entrypoint.strip():
            raise ValueError("providers.matlab.registration.entrypoint must be a non-empty MATLAB function name")

        for method_name, entrypoint in self.local_entrypoints.items():
            if method_name not in SUPPORTED_LOCAL_REGISTRATION_METHODS:
                raise ValueError(
                    f"providers.matlab.registration.local_entrypoints declares unsupported local method {method_name!r}"
                )
            if not isinstance(entrypoint, str) or not entrypoint.strip():
                raise ValueError(
                    f"providers.matlab.registration.local_entrypoints[{method_name!r}] must be a non-empty MATLAB function name"
                )
        return self

    model_config = ConfigDict(frozen=True, extra='forbid')


class MatlabSpotFindingProviderConfig(BaseModel):
    """Boundary contract for MATLAB-backed spot finding.

    The MATLAB runtime receives one clean 3D channel volume at a time and
    returns canonical `z, y, x, intensity` spot rows. The Python side keeps the
    output schema identical to native spot finding, while backend metadata
    records the MATLAB entrypoint and transfer mode.
    """

    runtime_path: Path = Path("matlab_runtime/pystar_spotfinding")
    entrypoint: str = "pystar_spotfind_entry"
    volume_transfer_mode: Literal["temporary_tiff"] = "temporary_tiff"
    input_volume_dtype: Literal["uint8"] = "uint8"

    @model_validator(mode='after')
    def validate_entrypoint(self) -> 'MatlabSpotFindingProviderConfig':
        if not self.entrypoint.strip():
            raise ValueError("providers.matlab.spot_finding.entrypoint must be a non-empty MATLAB function name")
        return self

    model_config = ConfigDict(frozen=True, extra='forbid')


class MatlabExtractionProviderConfig(BaseModel):
    """Boundary contract for MATLAB-backed signal extraction.

    Extraction sends registered or moving image volumes plus coordinate CSVs to
    MATLAB and expects one intensity vector back. The contract is deliberately
    explicit because small dtype/coordinate differences directly affect gene
    calls downstream.
    """

    runtime_path: Path = Path("matlab_runtime/pystar_extraction")
    entrypoint: str = "pystar_extract_entry"
    volume_transfer_mode: Literal["temporary_tiff"] = "temporary_tiff"
    coords_transfer_mode: Literal["temporary_csv"] = "temporary_csv"
    input_volume_dtype: Literal["uint8", "float32"] = "float32"

    @model_validator(mode='after')
    def validate_entrypoint(self) -> 'MatlabExtractionProviderConfig':
        if not self.entrypoint.strip():
            raise ValueError("providers.matlab.extraction.entrypoint must be a non-empty MATLAB function name")
        return self

    model_config = ConfigDict(frozen=True, extra='forbid')


class MatlabSharedSessionConfig(BaseModel):
    """Optional named MATLAB Engine session sharing policy.

    The shared-session feature is opt-in so existing configs keep the historical
    per-backend MATLAB Engine lifecycle.  When enabled, the batch runner creates
    one explicit owner with a deterministic name and MATLAB-capable stages borrow
    that owner instead of independently cold-starting sessions.
    """

    enabled: bool = False
    name: Optional[str] = None
    lifetime: Literal["run", "fov"] = "run"
    health_check_timeout_s: float = Field(default=30.0, gt=0)

    @model_validator(mode='after')
    def validate_session_name(self) -> 'MatlabSharedSessionConfig':
        if self.name is None:
            return self
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("providers.matlab.shared_session.name must be a non-empty string when provided")
        name = self.name.strip()
        if len(name) > 63:
            raise ValueError(
                "providers.matlab.shared_session.name must be at most 63 characters for MATLAB "
                "namelengthmax/shareEngine compatibility"
            )
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
            raise ValueError(
                "providers.matlab.shared_session.name must contain only ASCII letters, digits, and underscores, "
                "and must start with a letter"
            )
        object.__setattr__(self, "name", name)
        return self

    model_config = ConfigDict(frozen=True, extra='forbid')


class MatlabProviderConfig(BaseModel):
    """Top-level switch and runtime sections for all MATLAB provider seams.

    Setting `enabled: true` does not force MATLAB execution by itself. A stage
    must also select `provider: matlab`. This separation lets a single config
    document both native and MATLAB-capable environments without silently
    changing which stages run through MATLAB.
    """

    enabled: bool = True
    shared_session: MatlabSharedSessionConfig = Field(default_factory=MatlabSharedSessionConfig)
    preprocessing: MatlabPreprocessingProviderConfig = Field(default_factory=MatlabPreprocessingProviderConfig)
    registration: MatlabRegistrationProviderConfig = Field(default_factory=MatlabRegistrationProviderConfig)
    spot_finding: MatlabSpotFindingProviderConfig = Field(default_factory=MatlabSpotFindingProviderConfig)
    extraction: MatlabExtractionProviderConfig = Field(default_factory=MatlabExtractionProviderConfig)

    model_config = ConfigDict(frozen=True, extra='forbid')


class ProvidersConfig(BaseModel):
    """External runtime provider configuration namespace."""

    matlab: MatlabProviderConfig = Field(default_factory=MatlabProviderConfig)

    model_config = ConfigDict(frozen=True, extra='forbid')

class BsplineConfig(BaseModel):
    """Native B-spline local-registration tuning parameters."""

    grid_spacing: int = 50  # 控制点间距，单位像素
    num_iter: int = 50      # 优化迭代次数
    model_config = ConfigDict(frozen=True)
    
class OpticalFlowConfig(BaseModel):
    """
    专门管理光流法的参数。
    """
    attachment: float = 15.0  # 越小越紧跟数据(容易受噪点影响)，越大越平滑
    tightness: float = 0.3    # 平滑项权重
    num_warp: int = 5         # 图像金字塔每层的变形次数
    num_iter: int = 10        # 迭代次数
    tol: float = 0.0001       # 收敛容差
    prefilter: bool = False   # 是否预过滤

    model_config = ConfigDict(frozen=True)
    
class Demons3DConfig(BaseModel):
    """
    专门管理 3D Demons 配准的参数。
    对应 MATLAB 的 imregdemons 函数。
    """
    num_iter: int = 50  # 对应 MATLAB 的 Iterations 参数
    smoothing_sigma: float = 1.0  # 对应 MATLAB 的 AccumulatedFieldSmoothing
    
    # 多分辨率金字塔参数。
    # MATLAB provider forwards positive integers to MATLAB PyramidLevels;
    # None preserves MATLAB runtime auto-computation. Native SimpleITK
    # DemonsRegistrationFilter has no multi-level pyramid API here, so
    # registration.py accepts only None/1 for native execution and fails loudly
    # for values greater than one.
    pyramid_levels: Optional[int] = None
    
    # 是否使用分块处理（针对大图像）
    # 默认保持关闭；开启后默认按 MATLAB sqrt_pieces=4 的 subtile 合同构建 4x4 tiles。
    # 如果显式给出 tile_grid_shape_yx，则它优先于 sqrt_pieces；当 layout_policy=matlab_subtile 时必须保持正方形。
    use_tiling: bool = False
    tile_size: int = 512
    tile_overlap: Optional[int] = None
    sqrt_pieces: Optional[int] = 4
    tile_grid_shape_yx: Optional[Tuple[int, int]] = None
    tiling_layout_policy: Literal["matlab_subtile", "even_split"] = "matlab_subtile"

    @model_validator(mode='before')
    @classmethod
    def reject_invalid_pyramid_levels(cls, data: Any) -> Any:
        if isinstance(data, dict):
            raw_levels = data.get("pyramid_levels", None)
            if raw_levels is not None and (
                isinstance(raw_levels, bool) or not isinstance(raw_levels, Integral)
            ):
                raise ValueError(
                    "registration.local.params.demons_3d.pyramid_levels must be a positive integer when provided"
                )
        return data

    @model_validator(mode='after')
    def validate_tiling(self) -> 'Demons3DConfig':
        if self.num_iter <= 0:
            raise ValueError("registration.local.params.demons_3d.num_iter must be positive")
        if self.smoothing_sigma < 0:
            raise ValueError("registration.local.params.demons_3d.smoothing_sigma must be non-negative")
        if self.pyramid_levels is not None and self.pyramid_levels <= 0:
            raise ValueError("registration.local.params.demons_3d.pyramid_levels must be positive when provided")
        if self.tile_size <= 0:
            raise ValueError("registration.local.params.demons_3d.tile_size must be positive")
        if self.tile_overlap is not None and self.tile_overlap < 0:
            raise ValueError("registration.local.params.demons_3d.tile_overlap must be non-negative when provided")

        if self.sqrt_pieces is not None and self.sqrt_pieces <= 0:
            raise ValueError("registration.local.params.demons_3d.sqrt_pieces must be positive when provided")

        if self.tile_grid_shape_yx is not None:
            grid_shape = tuple(int(value) for value in self.tile_grid_shape_yx)
            if len(grid_shape) != 2:
                raise ValueError(
                    "registration.local.params.demons_3d.tile_grid_shape_yx must contain exactly two integers"
                )
            if any(value <= 0 for value in grid_shape):
                raise ValueError(
                    "registration.local.params.demons_3d.tile_grid_shape_yx must contain positive integers"
                )
            if self.tiling_layout_policy == "matlab_subtile" and grid_shape[0] != grid_shape[1]:
                raise ValueError(
                    "registration.local.params.demons_3d.tile_grid_shape_yx must stay square when tiling_layout_policy='matlab_subtile'"
                )

        if self.tiling_layout_policy == "matlab_subtile":
            sqrt_value = int(self.sqrt_pieces) if self.sqrt_pieces is not None else None
            if sqrt_value is None and self.tile_grid_shape_yx is None:
                raise ValueError(
                    "registration.local.params.demons_3d.matlab_subtile layout requires sqrt_pieces or tile_grid_shape_yx"
                )
        return self
    
    model_config = ConfigDict(frozen=True)


class FieldSemanticsConfig(BaseModel):
    """显式声明位移场语义合同。

    这里不试图判定“真相”，只记录当前 runtime 声称自己在使用什么语义。
    """

    representation: Literal["residual", "total", "unknown"] = "unknown"
    composition: Literal["sequential_global_then_local", "independent", "unknown"] = "unknown"
    status: Literal["settled", "provisional", "unknown"] = "unknown"

    model_config = ConfigDict(frozen=True)

    def is_unknown(self) -> bool:
        return (
            self.representation == "unknown"
            and self.composition == "unknown"
            and self.status == "unknown"
        )

    def as_dict(self) -> Dict[str, str]:
        return {
            "representation": self.representation,
            "composition": self.composition,
            "status": self.status,
        }


class RegistrationInputConfig(BaseModel):
    """Select which clean channels form the registration volume.

    `mip_all_channels` builds a max-intensity projection over selected channel
    IDs for each round. `single_channel` uses one channel directly. The choice
    changes registration evidence, not downstream decoding channels.
    """

    method: Literal["mip_all_channels", "single_channel"] = "mip_all_channels"
    single_channel_id: Optional[int] = None
    mip_channels: Optional[List[int]] = None

    @model_validator(mode='after')
    def validate_input_mode(self) -> 'RegistrationInputConfig':
        if self.method == "single_channel" and self.single_channel_id is None:
            raise ValueError("registration.source.single_channel_id is required when method='single_channel'")
        if self.method == "mip_all_channels" and self.mip_channels is None:
            raise ValueError("registration.source.mip_channels is required when method='mip_all_channels'")
        return self

    model_config = ConfigDict(frozen=True, extra='forbid')


class GlobalRegistrationParams(BaseModel):
    """Parameters for phase-correlation global 3D shift estimation."""

    use_gpu: bool = False
    downsample_factor: int = 4
    max_shift: int = 200

    model_config = ConfigDict(frozen=True, extra='forbid')


class GlobalRegistrationStageConfig(BaseModel):
    """Global registration stage configuration.

    The current runtime requires global registration to stay enabled. It
    estimates a coarse `[dz, dy, dx]` shift from each moving round to the
    reference round before any optional local deformation is computed.
    """

    enabled: bool = True
    method: Literal["phase_corr_3d"] = "phase_corr_3d"
    provider: Literal["native", "matlab"] = "native"
    params: GlobalRegistrationParams = Field(default_factory=GlobalRegistrationParams)

    @model_validator(mode='after')
    def validate_stage(self) -> 'GlobalRegistrationStageConfig':
        if not self.enabled:
            raise ValueError("registration.global.enabled must remain true for the current runtime contract")
        if self.method not in SUPPORTED_GLOBAL_REGISTRATION_METHODS:
            raise ValueError(
                f"Unsupported registration.global.method {self.method!r}. "
                f"Supported methods: {sorted(SUPPORTED_GLOBAL_REGISTRATION_METHODS)}"
            )
        if self.provider not in SUPPORTED_REGISTRATION_PROVIDERS:
            raise ValueError(
                f"Unsupported registration.global.provider {self.provider!r}. "
                f"Supported providers: {sorted(SUPPORTED_REGISTRATION_PROVIDERS)}"
            )
        return self

    model_config = ConfigDict(frozen=True, extra='forbid')


class LocalRegistrationParams(BaseModel):
    """Container for all local-registration method parameter blocks."""

    bspline: BsplineConfig = Field(default_factory=BsplineConfig)
    optical_flow: OpticalFlowConfig = Field(default_factory=OpticalFlowConfig)
    demons_3d: Demons3DConfig = Field(default_factory=Demons3DConfig)

    model_config = ConfigDict(frozen=True, extra='forbid')


class LocalRegistrationStageConfig(BaseModel):
    """Optional local registration stage configuration.

    Local registration refines the global shift with a dense deformation field.
    Native mode can select optical flow, B-spline or 3D Demons; MATLAB mode is
    restricted to the Demons path used for parity validation.
    """

    enabled: bool = False
    method: Literal["optical_flow", "bspline", "demons_3d"] = "optical_flow"
    provider: Literal["native", "matlab"] = "native"
    params: LocalRegistrationParams = Field(default_factory=LocalRegistrationParams)

    @model_validator(mode='after')
    def validate_stage(self) -> 'LocalRegistrationStageConfig':
        if self.method not in SUPPORTED_LOCAL_REGISTRATION_METHODS:
            raise ValueError(
                f"Unsupported registration.local.method {self.method!r}. "
                f"Supported methods: {sorted(SUPPORTED_LOCAL_REGISTRATION_METHODS)}"
            )
        if self.provider not in SUPPORTED_REGISTRATION_PROVIDERS:
            raise ValueError(
                f"Unsupported registration.local.provider {self.provider!r}. "
                f"Supported providers: {sorted(SUPPORTED_REGISTRATION_PROVIDERS)}"
            )
        if self.provider == "matlab" and self.method != "demons_3d":
            raise ValueError(
                "registration.local.provider='matlab' currently supports only method='demons_3d'"
            )
        return self

    model_config = ConfigDict(frozen=True, extra='forbid')


class RegistrationGuardsConfig(BaseModel):
    """Safety thresholds for accepting registration results."""

    skip_if_global_corr_below: float = 0.2
    reject_if_correlation_worse: bool = True

    model_config = ConfigDict(frozen=True, extra='forbid')


class MatlabLocalParallelConfig(BaseModel):
    """Opt-in process-level execution for MATLAB local demons tiles.

    This does not change demons parameters or tiling geometry.  It only controls
    whether already-independent MATLAB local demons tile jobs may be scheduled in
    separate worker processes.
    """

    enabled: bool = False
    workers: int = 2
    strict_equivalence_audit: bool = True

    @model_validator(mode='before')
    @classmethod
    def reject_bool_worker_values(cls, data: Any) -> Any:
        if isinstance(data, dict) and isinstance(data.get("workers"), bool):
            raise ValueError("registration.matlab_local_parallel.workers must be an integer, not a boolean")
        return data

    @model_validator(mode='after')
    def validate_parallel_workers(self) -> 'MatlabLocalParallelConfig':
        if self.enabled and (isinstance(self.workers, bool) or int(self.workers) <= 0):
            raise ValueError("registration.matlab_local_parallel.workers must be a positive integer when enabled")
        return self

    model_config = ConfigDict(frozen=True, extra='forbid')


class RegistrationConfig(BaseModel):
    """Full registration configuration used by `RegistrationEngine`.

    Registration writes a per-round transform bundle under `transforms/`.
    The bundle contains global shifts, optional local fields, and explicit
    `_semantics`/`_scope` metadata consumed later by extraction. Keeping these
    semantics in config prevents coordinate-mapping and image-warp code from
    silently interpreting the same field in different ways.
    """

    reference_round: int
    source: RegistrationInputConfig = Field(default_factory=RegistrationInputConfig)
    global_stage: GlobalRegistrationStageConfig = Field(default_factory=GlobalRegistrationStageConfig, alias="global")
    local: LocalRegistrationStageConfig = Field(default_factory=LocalRegistrationStageConfig)
    guards: RegistrationGuardsConfig = Field(default_factory=RegistrationGuardsConfig)
    matlab_local_parallel: MatlabLocalParallelConfig = Field(default_factory=MatlabLocalParallelConfig)

    # 位移场语义：registration producer 对外声明当前 field 的表示/组合方式
    field_semantics: FieldSemanticsConfig = Field(default_factory=FieldSemanticsConfig)
    
    # 质量控制参数
    min_peak_intensity: float = 10.0
    min_correlation: float = 0.2
    
    # 输出参数
    save_displacement_fields: bool = True
    save_registered_images: bool = False

    @property
    def method(self) -> str:
        return self.source.method

    @property
    def single_channel_id(self) -> Optional[int]:
        return self.source.single_channel_id

    @property
    def mip_channels(self) -> Optional[List[int]]:
        return self.source.mip_channels

    @property
    def use_gpu(self) -> bool:
        return self.global_stage.params.use_gpu

    @property
    def downsample_factor(self) -> int:
        return self.global_stage.params.downsample_factor

    @property
    def global_max_shift(self) -> int:
        return self.global_stage.params.max_shift

    @property
    def enable_local(self) -> bool:
        return self.local.enabled

    @property
    def local_method(self) -> str:
        return self.local.method

    @property
    def global_provider(self) -> str:
        return self.global_stage.provider

    @property
    def local_provider(self) -> Optional[str]:
        if not self.local.enabled:
            return None
        return self.local.provider

    @property
    def bspline(self) -> BsplineConfig:
        return self.local.params.bspline

    @property
    def optical_flow(self) -> OpticalFlowConfig:
        return self.local.params.optical_flow

    @property
    def demons_3d(self) -> Demons3DConfig:
        return self.local.params.demons_3d

    @model_validator(mode='after')
    def align_guard_defaults(self) -> 'RegistrationConfig':
        if self.min_correlation != self.guards.skip_if_global_corr_below:
            self.min_correlation = self.guards.skip_if_global_corr_below
        if self.matlab_local_parallel.enabled:
            if not self.local.enabled:
                raise ValueError(
                    "registration.matlab_local_parallel.enabled=true requires registration.local.enabled=true"
                )
            if self.local.provider != "matlab" or self.local.method != "demons_3d":
                raise ValueError(
                    "registration.matlab_local_parallel.enabled=true requires "
                    "registration.local.provider='matlab' and registration.local.method='demons_3d'"
                )
            if not self.local.params.demons_3d.use_tiling:
                raise ValueError(
                    "registration.matlab_local_parallel.enabled=true requires "
                    "registration.local.params.demons_3d.use_tiling=true"
                )
        return self

    model_config = ConfigDict(extra='ignore', populate_by_name=True)

class SpotiflowConfig(BaseModel):
    """Spotiflow model and probability threshold parameters."""

    model_name: str = "general"
    prob_thresh: float = 0.5
    use_gpu: bool = True
    model_config = ConfigDict(frozen=True)

class BlobDogConfig(BaseModel):
    """Difference-of-Gaussians blob detector parameters."""

    min_sigma: Union[List[float], float] = Field(default_factory=lambda: [0.5, 0.5, 0.5])
    max_sigma: Union[List[float], float] = Field(default_factory=lambda: [2.0, 5.0, 5.0])
    threshold: float = 0.05
    overlap: float = 0.5

    @model_validator(mode='after')
    def normalize_sigmas(self) -> 'BlobDogConfig':
        """
        无论用户输入 0.5 还是 [0.5, 0.5, 0.5]，
        在模型初始化后，它们全部变成 [0.5, 0.5, 0.5]。
        """
        def to_list(val: Union[List[float], float]) -> List[float]:
            if isinstance(val, (int, float)):
                return [float(val)] * 3
            return [float(item) for item in val]

        min_sigma = to_list(self.min_sigma)
        max_sigma = to_list(self.max_sigma)

        # 简单的校验：确保列表长度是 3
        if len(min_sigma) != 3 or len(max_sigma) != 3:
            raise ValueError("Sigma must be a single float or a list of 3 floats (Z, Y, X)")

        object.__setattr__(self, "min_sigma", min_sigma)
        object.__setattr__(self, "max_sigma", max_sigma)
            
        return self

    model_config = {"frozen": True} # 保持实用主义，配置一旦生成就不该被后面的人乱改

class PeakLocalMaxConfig(BaseModel):
    """Native Max3D regional-maxima threshold parameters.

    The `peak_local_max` algorithm name is the public config value for PyStar's
    native Max3D detector. The current implementation detects 26-connected
    regional maxima and collapses each plateau to a centroid; `threshold_rel`
    is interpreted relative to the image dtype range.
    """

    min_distance: int = 3
    threshold_rel: float = 0.05
    exclude_border: bool = True
    model_config = ConfigDict(frozen=True)

class SpotFindingConfig(BaseModel):
    """Spot-finding stage configuration.

    The stage consumes clean reference-round sequencing channels and writes
    `spots_fov_<id>.csv` with reference-frame `z, y, x` coordinates. Native and
    MATLAB providers share the same output schema so that extraction and
    decoding do not need provider-specific branches.
    """

    # 核心开关
    algorithm: Literal["spotiflow", "blob_dog", "peak_local_max"] = "peak_local_max"  # 默认值
    provider: Literal["native", "matlab"] = "native"
    
    # 通用参数
    reference_round: int = 1
    method: str = "max_intensity"
    smooth_sigma: float = 1.0
    
    # 嵌套的子配置对象
    # 就算 YAML 里没写具体的子项，它们也会以默认值存在
    spotiflow: SpotiflowConfig = SpotiflowConfig()
    blob_dog: BlobDogConfig = BlobDogConfig()
    peak_local_max: PeakLocalMaxConfig = PeakLocalMaxConfig()

    @model_validator(mode='after')
    def validate_provider_algorithm_pair(self) -> 'SpotFindingConfig':
        if self.provider not in SUPPORTED_SPOT_FINDING_PROVIDERS:
            raise ValueError(
                f"Unsupported spot_finding.provider {self.provider!r}. "
                f"Supported providers: {sorted(SUPPORTED_SPOT_FINDING_PROVIDERS)}"
            )

        if self.provider == "matlab" and self.algorithm != "peak_local_max":
            raise ValueError(
                "spot_finding.provider='matlab' currently supports only algorithm='peak_local_max' "
                "(mapped to the MATLAB max3d-style local-maxima kernel)"
            )

        return self

    model_config = ConfigDict(extra='ignore')
    
class ExtractionConfig(BaseModel):
    """Signal extraction configuration.

    Extraction converts spot coordinates into an intensity matrix with shape
    `(N_spots, N_rounds, N_seq_channels)`. `image_warp` first registers each
    moving volume into reference-frame pixels and then sums fixed boxes;
    `coordinate_mapping` maps spot coordinates into each moving image and is
    retained as a diagnostic/legacy path.
    """

    method: str = "box_sum"
    provider: Literal["native", "matlab"] = "native"
    integration_box: List[int] = [1, 3, 3]
    handle_out_of_bounds: str = "pad_zero"
    transform_application_mode: Literal["coordinate_mapping", "image_warp"] = "image_warp"

    @model_validator(mode='after')
    def validate_provider_method_pair(self) -> 'ExtractionConfig':
        if self.provider not in SUPPORTED_EXTRACTION_PROVIDERS:
            raise ValueError(
                f"Unsupported extraction.provider {self.provider!r}. "
                f"Supported providers: {sorted(SUPPORTED_EXTRACTION_PROVIDERS)}"
            )

        if self.provider == "matlab" and self.method != "box_sum":
            raise ValueError(
                "extraction.provider='matlab' currently supports only method='box_sum'"
            )

        return self

class DecodingRuleConfig(BaseModel):
    """One optional post-calling rule in the decoding rule pipeline."""

    name: str
    stage: Literal["spot", "barcode"]
    enabled: bool = True
    hard: bool = True
    weight: float = 1.0
    params: Dict[str, Any] = Field(default_factory=dict)

class WeightedRescueConfig(BaseModel):
    """Experimental weighted-nearest-codebook rescue parameters."""

    enable: bool = False
    target: Literal["background", "all"] = "background"
    max_weighted_distance: float = 1.5
    min_second_gap: float = 0.0
    round_weights: Optional[List[float]] = None

class DecodingConfig(BaseModel):
    """Barcode decoding and gating configuration."""

    quality_threshold: float = 0.5
    gating_mode: Literal["pattern_first", "legacy_membership_first"] = "pattern_first"
    max_soft_penalty: Optional[float] = None
    rules: List[DecodingRuleConfig] = Field(default_factory=list)
    round_channel_bias: Dict[int, List[float]] = Field(default_factory=dict)
    weighted_rescue: WeightedRescueConfig = Field(default_factory=WeightedRescueConfig)

class OutputConfig(BaseModel):
    """Canonical output root and QC-image switch."""

    directory: str
    save_qc_images: bool = True

class QCConfig(BaseModel):
    """Global QC-report switch."""

    enable: bool = True


class PipelineConfig(BaseModel):
    """All runnable pipeline-stage configuration.

    This model is the contract between the YAML file and runtime stages. It
    keeps preprocessing, registration, spot finding, extraction and decoding in
    one immutable object, and exposes small helper methods for provider-mode
    decisions used by validation and MATLAB boundary checks.
    """

    scope_mode: Literal["full_fov", "tile_local"] = "full_fov"
    accelerator: Literal["cpu"] = "cpu"
    field_semantics: FieldSemanticsConfig = Field(default_factory=FieldSemanticsConfig)
    preprocessing: PreprocessingConfig
    registration: RegistrationConfig  # 使用具体的类
    spot_finding: SpotFindingConfig
    extraction: ExtractionConfig
    decoding: DecodingConfig = Field(default_factory=DecodingConfig)
    output: OutputConfig 
    qc: QCConfig

    @model_validator(mode='after')
    def align_field_semantics(self) -> 'PipelineConfig':
        pipeline_semantics = self.field_semantics
        registration_semantics = self.registration.field_semantics

        if pipeline_semantics.is_unknown() and not registration_semantics.is_unknown():
            self.field_semantics = registration_semantics.model_copy(deep=True)
            return self

        if registration_semantics.is_unknown() and not pipeline_semantics.is_unknown():
            self.registration = self.registration.model_copy(
                update={"field_semantics": pipeline_semantics.model_copy(deep=True)}
            )
            return self

        if pipeline_semantics != registration_semantics:
            raise ValueError(
                "pipeline.field_semantics and pipeline.registration.field_semantics must match when both are explicit"
            )

        return self

    def qc_images_enabled(self) -> bool:
        """Backward-compatible effective switch for QC image generation.

        `pipeline.qc.enable` remains the semantic master switch, while
        `pipeline.output.save_qc_images` keeps acting as the legacy output-level
        guard. QC image generation is enabled only when both are true.
        """
        return bool(self.qc.enable) and bool(self.output.save_qc_images)

    def preprocessing_providers_used(self) -> List[str]:
        used = {step.provider for step in self.preprocessing.sequence}
        return sorted(used) if used else ["native"]

    def uses_matlab_preprocessing(self) -> bool:
        return any(step.provider == "matlab" for step in self.preprocessing.sequence)

    def uses_matlab_spot_finding(self) -> bool:
        return self.spot_finding.provider == "matlab"

    def uses_matlab_extraction(self) -> bool:
        return self.extraction.provider == "matlab"

    def preprocessing_provider_mode(self) -> str:
        used = set(self.preprocessing_providers_used())
        if used == {"native"}:
            return "native_only"
        if used == {"matlab"}:
            return "matlab_only"
        return "mixed"

    def uses_matlab_registration(self) -> bool:
        return self.registration.global_provider == "matlab" or self.registration.local_provider == "matlab"

    def registration_provider_mode(self) -> str:
        global_provider = self.registration.global_provider
        local_provider = self.registration.local_provider
        if global_provider == "native" and local_provider in {None, "native"}:
            return "native_only"
        if global_provider == "matlab" and local_provider in {None, "matlab"}:
            return "matlab_only"
        return "mixed"


def _matlab_registration_local_entrypoint_for_method(
    providers: ProvidersConfig,
    method: str,
) -> Optional[str]:
    return providers.matlab.registration.local_entrypoints.get(method)

class ExperimentConfig(BaseModel):
    """Validated experiment configuration returned by `load_config`.

    The object is the single source of truth passed to all runtime stages. It
    stores the parsed YAML plus non-serialized provenance (`config_source_path`
    and `config_sha256`) so reports can identify exactly which config was used.
    """

    dataset: DatasetConfig
    codebook: CodebookConfig
    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)
    pipeline: PipelineConfig
    config_source_path: Optional[Path] = Field(default=None, exclude=True)
    config_sha256: Optional[str] = Field(default=None, exclude=True)

    @model_validator(mode='after')
    def validate_provider_runtime_contracts(self) -> 'ExperimentConfig':
        matlab_fields = set(self.providers.matlab.model_fields_set)

        if self.pipeline.uses_matlab_preprocessing() and "preprocessing" not in matlab_fields:
            raise ValueError(
                "providers.matlab.preprocessing must be explicitly configured when any preprocessing.sequence step selects provider='matlab'"
            )

        if self.pipeline.uses_matlab_preprocessing() and not self.providers.matlab.enabled:
            raise ValueError(
                "providers.matlab.enabled must be true when any preprocessing.sequence step selects provider='matlab'"
            )

        if self.pipeline.registration.global_provider == "matlab" and "registration" not in matlab_fields:
            raise ValueError(
                "providers.matlab.registration must be explicitly configured when registration.global.provider='matlab'"
            )

        if self.pipeline.registration.global_provider == "matlab" and not self.providers.matlab.enabled:
            raise ValueError(
                "providers.matlab.enabled must be true when registration.global.provider='matlab'"
            )

        if self.pipeline.registration.local_provider == "matlab":
            if "registration" not in matlab_fields:
                raise ValueError(
                    "providers.matlab.registration must be explicitly configured when registration.local.provider='matlab'"
                )
            if not self.providers.matlab.enabled:
                raise ValueError(
                    "providers.matlab.enabled must be true when registration.local.provider='matlab'"
                )
            local_method = self.pipeline.registration.local_method
            if _matlab_registration_local_entrypoint_for_method(self.providers, local_method) is None:
                raise ValueError(
                    f"providers.matlab.registration.local_entrypoints is missing an entrypoint for local method {local_method!r}"
                )

        if self.pipeline.uses_matlab_spot_finding() and "spot_finding" not in matlab_fields:
            raise ValueError(
                "providers.matlab.spot_finding must be explicitly configured when spot_finding.provider='matlab'"
            )

        if self.pipeline.uses_matlab_spot_finding() and not self.providers.matlab.enabled:
            raise ValueError(
                "providers.matlab.enabled must be true when spot_finding.provider='matlab'"
            )

        if self.pipeline.uses_matlab_extraction() and "extraction" not in matlab_fields:
            raise ValueError(
                "providers.matlab.extraction must be explicitly configured when extraction.provider='matlab'"
            )

        if self.pipeline.uses_matlab_extraction() and not self.providers.matlab.enabled:
            raise ValueError(
                "providers.matlab.enabled must be true when extraction.provider='matlab'"
            )

        return self


def _validate_release_facing_fields(raw_data: Any) -> None:
    if not isinstance(raw_data, dict):
        raise ValueError("Config root must be a mapping")

    pipeline = raw_data.get("pipeline")
    if not isinstance(pipeline, dict):
        raise ValueError("Config missing required 'pipeline' mapping")

    missing_pipeline_fields = [
        field_name for field_name in REQUIRED_PIPELINE_RELEASE_FIELDS if field_name not in pipeline
    ]
    if missing_pipeline_fields:
        raise ValueError(
            "Missing required release-facing pipeline fields: "
            + ", ".join(missing_pipeline_fields)
        )

    extraction = pipeline.get("extraction")
    if not isinstance(extraction, dict):
        raise ValueError("Config missing required 'pipeline.extraction' mapping")

    missing_extraction_fields = [
        field_name for field_name in REQUIRED_EXTRACTION_RELEASE_FIELDS if field_name not in extraction
    ]
    if missing_extraction_fields:
        raise ValueError(
            "Missing required release-facing extraction fields: "
            + ", ".join(missing_extraction_fields)
        )

# --- 加载器 ---

def load_config(config_path: str) -> ExperimentConfig:
    """
    加载并严格验证 YAML 配置文件。
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    raw_text = path.read_text(encoding='utf-8')
    try:
        raw_data = yaml.safe_load(raw_text)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML format: {e}")

    _validate_release_facing_fields(raw_data)

    try:
        config = ExperimentConfig(**raw_data)
        config.config_source_path = path.resolve()
        config.config_sha256 = f"sha256:{hashlib.sha256(raw_text.encode('utf-8')).hexdigest()}"
        
        # 简单的业务逻辑检查
        if config.dataset.pixel_size_xy_nm <= 0:
            raise ValueError("Pixel size must be positive!")
            
        return config
    except ValidationError as e:
        print("\n Configuration Error: Your yaml file is garbage.")
        # Pydantic V2 的错误输出格式可能会有所不同，但这能工作
        print(e)
        raise


# Test it immediately
if __name__ == "__main__":
    print("Testing config loader...")
    try:
        # 尝试
        config = load_config("experiment_config.yaml")
        
        print(" SUCCESS! Config loaded and validated.")
        print("-" * 40)
        
        # 验证一下读取到的数据是不是我们写的
        # dataset 是对象，所以用 .dataset
        # dimensions 是我们在 model 里定义为 Dict 的字段，所以用 ['z']
        print(f"Dimensions: Z={config.dataset.dimensions['z']}, W={config.dataset.dimensions['width']}")
        
        # 2. 访问 pipeline (Object) -> registration (Object) -> method (Attribute)
        # 这里的 registration 是 RegistrationConfig 的实例，不是字典！
        print(f"Registration Method: {config.pipeline.registration.method}")
        print(f"FOVs to process: {len(config.dataset.parsed_fovs)} positions")
        print(f"First 5 FOVs: {config.dataset.parsed_fovs[:5]}...")
        print(f"Round 1 Channels: {config.dataset.round_structure[1]}")
        
        # 比如测试一个没填的字段 (它应该报错或返回None，取决于定义，但这里我们填全了)
        print("-" * 40)

    except Exception as e:
        print(" FAILED! The loader crashed.")
        print(e)
