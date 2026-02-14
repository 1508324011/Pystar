import numpy as np
import pandas as pd
import tifffile
from pathlib import Path
from scipy import ndimage
from skimage.feature import peak_local_max, blob_dog
from .io import ImageLoader
from .io import get_fov_output_structure
from .visualization import inspect_spots_interactive

class SpotFinder:
    def __init__(self, config):
        self.cfg = config
        self.spot_cfg = config.pipeline.spot_finding
        self.loader = ImageLoader(config)
        
        # 预留模型槽位，不要在初始化时乱占显存
        self._model = None

    def _get_spotiflow_model(self):
        """延迟加载模型：只有真正开始挖矿时，才启动大型机械。"""
        if self._model is None:
            try:
                from spotiflow.model import Spotiflow
                model_name = self.spot_cfg.spotiflow.model_name
                print(f" [SpotFinder] 正在从硬盘调取 Spotiflow 模型: {model_name}...")
                self._model = Spotiflow.from_pretrained(model_name)
            except ImportError:
                print(" 错误: 未安装Spotiflow库")
                print(" 运行: pip install spotiflow")
                raise
        return self._model

    def find_spots_in_fov(self, fov_id: int):
        """
        主入口：把图像变成坐标。
        直接读取 clean_data 里的单独通道文件进行寻点。
        """
        base_dir = Path(self.cfg.pipeline.output.directory)
        paths = get_fov_output_structure(base_dir, fov_id)
        
        ref_round = self.spot_cfg.reference_round
        # 获取所有由 'seq' 定义的通道 (通常 0,1,2,3)
        roles = self.cfg.dataset.channel_roles
        channels = sorted([c for c, role in roles.items() if role == 'seq'])
        
        print(f" [SpotFinding] Mining FOV {fov_id} using Clean Data (Ref Round {ref_round})...")
        print(f" [SpotFinding] Target Channels: {channels}")

        all_spots_dfs = []
        
        # 用于 QC 可视化的容器 (Channel -> Image)
        qc_images = {}

        for c in channels:
            # 1. 加载 Clean Data (单通道)
            try:
                # 使用 loader 从 clean_data 目录读取
                vol = self.loader.load_clean_image(fov_id, ref_round, c)
                print(f"Data type: {vol.dtype}")
                print(f"Value range: [{vol.min():.2f}, {vol.max():.2f}]")
                print(f"Mean: {vol.mean():.2f}, Std: {vol.std():.2f}")
            except FileNotFoundError:
                print(f" !!! Skip Channel {c}: Clean data not found. Run Sanitizer first!")
                continue

            # 2. 归一化 (Normalization)
            # 我们的 Clean Data 是 float32，范围大概在 0.0 - 255.0 之间 (取决于 Gain)
            # 算法通常喜欢 0-1 范围
            # 注意：如果你的 Gain 很大，值可能超过 255。这里简单除以 255 是一种折中，
            # 保证参数 (threshold) 的物理意义和之前 RAW 流程保持一致。
            #vol_norm = vol.astype(np.float32) / 255.0

            # 3. 收集 QC 图像 (只存中间层，节省内存)
            z_mid = vol.shape[0] // 2
            qc_images[c] = vol[z_mid].copy()

            # 4. 选择算法
            algo = self.spot_cfg.algorithm
            
            # 运行具体算法
            if algo == "spotiflow":
                df_c = self._run_spotiflow(vol)
            elif algo == "blob_dog":
                df_c = self._run_blob_dog(vol)
            elif algo == "peak_local_max":
                df_c = self._run_peak_local_max(vol)
            else:
                raise ValueError(f"Unknown algorithm: {algo}")
            
            # 5.标签注入
            df_c['channel'] = c
            all_spots_dfs.append(df_c)
            print(f"   > Channel {c}: found {len(df_c)} spots")
            
        # 4. 合并结果
        if not all_spots_dfs:
            print(" [SpotFinding] No spots found in any channel!")
            return pd.DataFrame()

        df = pd.concat(all_spots_dfs, ignore_index=True)

        # 注入元数据
        df['fov'] = fov_id
        df['algo'] = algo

        # 固化结果
        out_csv = paths["spots"] / f"spots_fov_{fov_id}.csv"
        df.to_csv(out_csv, index=False)
        print(f" [SpotFinding] Finished. Total: {len(df)} spots. Saved to {out_csv.name}")
        
        if self.cfg.pipeline.output.save_qc_images:
                qc_dir = paths["qc"]
                
                # 构造一个伪 4D 数组 (C, 1, H, W)
                h, w = list(qc_images.values())[0].shape
                viz_stack = np.zeros((len(channels), 1, h, w), dtype=np.float32)
                
                for idx, c in enumerate(channels):
                    if c in qc_images:
                        viz_stack[idx, 0, :, :] = qc_images[c]
                inspect_spots_interactive(
                    viz_stack, df, 
                    z_plane=0, 
                    roi_size=128,
                    output_path=qc_dir / f"spot_finding_qc_fov_{fov_id}.png"
                )
        
        return df

    def _run_spotiflow(self, vol_3d):
        """Spotiflow 挖掘逻辑：精准回归。"""
        model = self._get_spotiflow_model()
        params = self.spot_cfg.spotiflow
        
        print(f" [Spotiflow] 正在预测，概率阈值: {params.prob_thresh}")
        # Spotiflow 的 predict 非常简洁，直接返回亚像素坐标
        coords, details = model.predict(
            vol_3d, 
            prob_thresh=params.prob_thresh,
            subpix=True
        )
        
        # 修正：维度自适应 (Good Taste: 消除特殊情况)
        # 无论返回 (N, 3) 还是 (N, 2)，这里的逻辑都能跑
        ndim = coords.shape[1] 
        cols = ['z', 'y', 'x'][-ndim:] # 自动取后N个标签
        
        df = pd.DataFrame(coords, columns=cols)
        
        # 修正：防御性地获取 details
        # 有些版本返回对象，有些可能是字典，我们做一个简单的兼容处理
        # (假设这里它是对象，和你的原始代码一致，但请在运行时确认)
        if hasattr(details, 'intens'):
            df['intensity'] = details.intens
            df['prob'] = details.prob
        else:
            # 如果是字典的情况
            df['intensity'] = details.get('intens', 0)
            df['prob'] = details.get('prob', 0)
            
        return df

    def _run_blob_dog(self, vol_3d):
        """DoG 挖掘逻辑：数学形状匹配。"""
        params = self.spot_cfg.blob_dog
        blobs = blob_dog(
            vol_3d,
            min_sigma=params.min_sigma,
            max_sigma=params.max_sigma,
            threshold=params.threshold,
            overlap=params.overlap
        )
        if blobs.shape[1] == 4:
            cols = ['z', 'y', 'x', 'sigma']
        elif blobs.shape[1] == 6:
            cols = ['z', 'y', 'x', 'sigma_z', 'sigma_y', 'sigma_x']
        else:
            # 万一以后有更奇怪的输出，直接按索引给个默认列名，而不是崩溃
            cols = [f'col_{i}' for i in range(blobs.shape[1])]

        df = pd.DataFrame(blobs, columns=cols)
        return df

    def _run_peak_local_max(self, vol_3d):
        """Max3D 挖掘逻辑：快速但粗糙。"""
        params = self.spot_cfg.peak_local_max
    
        # 1. 极大值检测（布尔掩码，内存占用 = 原图的 1/8）
        footprint = np.ones((3, 3, 3), dtype=bool)
        max_filtered = ndimage.maximum_filter(
            vol_3d, 
            footprint=footprint, 
            mode='constant',
            cval=0  # 边界外视为 0，避免边界效应
        )
        is_max = (vol_3d == max_filtered)
        
        # 2. 阈值
        if vol_3d.dtype in [np.uint8, np.uint16]:
            dtype_max = 255 if vol_3d.dtype == np.uint8 else 65535
        else:
            dtype_max = vol_3d.max()
        
        abs_thresh = params.threshold_rel * dtype_max
        mask = is_max & (vol_3d > abs_thresh)
        
        # 3. 快速连通性分析
        labeled, n_spots = ndimage.label(mask, structure=np.ones((3,3,3)))
        
        if n_spots == 0:
            return pd.DataFrame(columns=['z', 'y', 'x', 'intensity'])
        
        # 4. 向量化批量计算（比循环快 10-100 倍）
        indices = np.arange(1, n_spots + 1)
        
        # 一次性计算所有质心和强度
        centroids = ndimage.center_of_mass(vol_3d, labeled, indices)
        max_intensities = ndimage.maximum(vol_3d, labeled, indices)
        
        # 5. 快速构造（避免列表推导）
        coords = np.array(centroids)
        df = pd.DataFrame(coords, columns=['z', 'y', 'x'])
        df['intensity'] = max_intensities
        
        return df
        

    
def _run_algo_on_channels(vol, run_fn):
    """
    [Internal Helper] 
    通用包装器：处理维度的统一逻辑。
    让所有算法自动支持 3D (Single) 和 4D (Multi-channel) 输入。
    """
    # 1. 维度归一化
    if vol.ndim == 3:
        # (Z, Y, X) -> (1, Z, Y, X)
        vol_4d = vol[np.newaxis, ...]
    elif vol.ndim == 4:
        # (C, Z, Y, X)
        vol_4d = vol
    else:
        raise ValueError(f"Input must be 3D or 4D, got {vol.ndim}D")

    all_dfs = []
    n_ch = vol_4d.shape[0]

    # 2. 遍历通道
    for c in range(n_ch):
        # 提取单通道 3D 数据
        vol_3d = vol_4d[c]
        
        # 运行具体的算法函数
        df = run_fn(vol_3d)
        
        # 注入 Channel ID (这在 Notebook 调试多通道时至关重要)
        df['channel'] = c
        all_dfs.append(df)
    
    # 3. 合并
    return pd.concat(all_dfs, ignore_index=True)

def detect_spots_max3d(vol_3d, threshold=0.05, min_dist=3):
    """
    这是一个轻量级的辅助函数，专门在 Notebook 里做实验用的。
    不用读文件，直接传内存里的数组就行。
    """
    def _logic(vol_3d):
        coords = peak_local_max(
            vol_3d,
            min_distance=min_dist,
            threshold_rel=threshold,
            exclude_border=True
        )
        return pd.DataFrame(coords, columns=['z', 'y', 'x'])

    return _run_algo_on_channels(vol_3d, _logic)

def detect_spots_blob_dog(vol_3d, min_sigma=(0.5, 0.5, 0.5), max_sigma=3.0, threshold=0.05, overlap=0.5):
    """
    这是一个轻量级的辅助函数，专门在 Notebook 里做实验用的。
    不用读文件，直接传内存里的数组就行。
    """
    def _logic(vol_3d):
        blobs = blob_dog(
            vol_3d,
            min_sigma=min_sigma,
            max_sigma=max_sigma,
            threshold=threshold,
            overlap=overlap
        )
        if blobs.shape[1] == 4:
            cols = ['z', 'y', 'x', 'sigma']
        elif blobs.shape[1] == 6:
            cols = ['z', 'y', 'x', 'sigma_z', 'sigma_y', 'sigma_x']
        else:
            cols = [f'col_{i}' for i in range(blobs.shape[1])]

        return pd.DataFrame(blobs, columns=cols)

    return _run_algo_on_channels(vol_3d, _logic)

def detect_spots_spotiflow(vol_3d, model_name="general", prob_thresh=0.5, use_gpu=True):
    """
    这是一个轻量级的辅助函数，专门在 Notebook 里做实验用的。
    不用读文件，直接传内存里的数组就行。
    """
    try:
        from spotiflow.model import Spotiflow
    except ImportError:
        print(" 错误: 未安装Spotiflow库")
        print(" 运行: pip install spotiflow")
        raise
    
    print(f" [Spotiflow] 正在加载模型: {model_name}...")
    model = Spotiflow.from_pretrained(model_name)
    
    def _logic(vol_3d):
        print(f" [Spotiflow] 正在预测，概率阈值: {prob_thresh}")
        coords, details = model.predict(
            vol_3d, 
            prob_thresh=prob_thresh,
            subpix=True
        )
        
        ndim = coords.shape[1] 
        cols = ['z', 'y', 'x'][-ndim:] 
        
        df = pd.DataFrame(coords, columns=cols)
        
        if hasattr(details, 'intens'):
            df['intensity'] = details.intens
            df['prob'] = details.prob
        else:
            df['intensity'] = details.get('intens', 0)
            df['prob'] = details.get('prob', 0)
            
        return df
    return _run_algo_on_channels(vol_3d, _logic)