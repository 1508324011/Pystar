import yaml
from pathlib import Path
from typing import List, Dict, Union, Any, Optional
from pydantic import BaseModel, model_validator, Field, ValidationError, ConfigDict

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
    id: str
    rounds: List[int]       # 物理轮次，如 [1, 2, 3, 4, 5]
    csv_slice: List[int]    # CSV 里的切片 [Start, End] (1-based physical index recommended for config, logic handles conversion)
    anchor_base: Optional[List[str]] = None   # 每个 segment 的 anchor bases
    encoding_table: str     # 使用哪张表
    
    
    
class TopologyConfig(BaseModel):
    func: str = "none" # "reverse_string" etc.
    structure: List[BlueprintSegment]
    physical_order: List[str] # 决定最终 barcode 拼接的顺序

class CodebookConfig(BaseModel):
    gene_list: Path
    channel_base_index: int  # 用户填 0 或 1
    encoding_tables: Dict[str, Dict[str, int]]
    topology: TopologyConfig

    # 禁止在实例初始化后修改数据，强制不可变
    model_config = ConfigDict(frozen=True) 

    @property
    def normalized_encoding_map(self) -> Dict[str, int]:
        """Runtime 转换，不污染 Config 状态"""
        base = self.channel_base_index
        if base == 0: return self.encoding_tables
        return {k: v - base for k, v in self.encoding_tables.items()}

class PreprocessingStep(BaseModel):
    """
    定义流水线中的单一步骤。
    对应 YAML 中的:
      - method: "median_filter"
        params: {kernel_size: 3}
    """
    method: str
    # 允许 params 为空，默认为空字典。
    # params 里的值可能是 int, float, str，所以用 Any
    params: Dict[str, Any] = Field(default_factory=dict)
    
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
    
    model_config = ConfigDict(frozen=True)

class BsplineConfig(BaseModel):
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
    
    # 多分辨率金字塔参数
    # MATLAB 自动计算: pyd_level = floor(log2(obj.dimZ))
    # 这里允许手动覆盖，None 表示自动计算
    pyramid_levels: Optional[int] = None
    
    # 是否使用分块处理（针对大图像）
    use_tiling: bool = False
    tile_size: int = 512
    tile_overlap: int = 64
    
    model_config = ConfigDict(frozen=True)

class RegistrationConfig(BaseModel):
    reference_round: int
    method: str  # "mip_all_channels" or "single_channel"
    
    single_channel_id: Optional[int] = None
    mip_channels: Optional[List[int]] = None
    
    # 粗对齐参数
    use_gpu: bool = False
    downsample_factor: int = 4
    global_max_shift: int = 200
    
    # 精细对齐参数
    enable_local: bool = False
    local_method: str = "optical_flow"  # "optical_flow" | "bspline" | "demons_3d"
    
    #bspline参数
    bspline: BsplineConfig = BsplineConfig()
    
    # 嵌套的光流配置 (即使 enable_local=False, 这个对象也存在，这是为了结构清晰)
    optical_flow: OpticalFlowConfig = OpticalFlowConfig()
    
    # 3D Demons 参数
    demons_3d: Demons3DConfig = Demons3DConfig()
    
    # 质量控制参数
    min_peak_intensity: float = 10.0
    min_correlation: float = 0.2
    
    # 输出参数
    save_displacement_fields: bool = True
    save_registered_images: bool = False

    model_config = ConfigDict(extra='ignore')

class SpotiflowConfig(BaseModel):
    model_name: str = "general"
    prob_thresh: float = 0.5
    use_gpu: bool = True
    model_config = ConfigDict(frozen=True)

class BlobDogConfig(BaseModel):
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
        def to_list(val):
            if isinstance(val, (int, float)):
                return [float(val)] * 3
            return val

        # 使用 __dict__ 直接修改，避免引发额外的验证循环
        self.__dict__['min_sigma'] = to_list(self.min_sigma)
        self.__dict__['max_sigma'] = to_list(self.max_sigma)
        
        # 简单的校验：确保列表长度是 3
        if len(self.min_sigma) != 3 or len(self.max_sigma) != 3:
            raise ValueError("Sigma must be a single float or a list of 3 floats (Z, Y, X)")
            
        return self

    model_config = {"frozen": True} # 保持实用主义，配置一旦生成就不该被后面的人乱改

class PeakLocalMaxConfig(BaseModel):
    min_distance: int = 3
    threshold_rel: float = 0.05
    exclude_border: bool = True
    model_config = ConfigDict(frozen=True)

class SpotFindingConfig(BaseModel):
    # 核心开关
    algorithm: str = "peak_local_max"  # 默认值
    
    # 通用参数
    reference_round: int = 1
    method: str = "max_intensity"
    smooth_sigma: float = 1.0
    
    # 嵌套的子配置对象
    # 就算 YAML 里没写具体的子项，它们也会以默认值存在
    spotiflow: SpotiflowConfig = SpotiflowConfig()
    blob_dog: BlobDogConfig = BlobDogConfig()
    peak_local_max: PeakLocalMaxConfig = PeakLocalMaxConfig()

    model_config = ConfigDict(extra='ignore')
    
class ExtractionConfig(BaseModel):
    method: str = "box_sum"
    integration_box: List[int] = [1, 3, 3]
    handle_out_of_bounds: str = "pad_zero"

class OutputConfig(BaseModel):
    directory: str
    save_qc_images: bool = True

class QCConfig(BaseModel):
    enable: bool = True


class PipelineConfig(BaseModel):
    preprocessing: PreprocessingConfig
    registration: RegistrationConfig  # 使用具体的类
    spot_finding: SpotFindingConfig
    extraction: ExtractionConfig
    output: OutputConfig 
    qc: QCConfig
class ExperimentConfig(BaseModel):
    dataset: DatasetConfig
    codebook: CodebookConfig
    pipeline: PipelineConfig

# --- 加载器 ---

def load_config(config_path: str) -> ExperimentConfig:
    """
    加载并严格验证 YAML 配置文件。
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, 'r') as f:
        try:
            raw_data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML format: {e}")

    try:
        config = ExperimentConfig(**raw_data)
        
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