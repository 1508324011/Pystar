# pystar/mining.py
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.ndimage import map_coordinates
from pathlib import Path
from .infrastructure import ExperimentConfig
from .io import ImageLoader
from .io import get_fov_output_structure
# visualization 模块保留引用，按需导入即可
from .visualization import plot_spot_traces, plot_spot_extraction_check

def map_spot_coordinates(
    ref_coords: np.ndarray,
    transform_data: dict
) -> np.ndarray:
    """
    将参考坐标映射到目标轮次 (保持 float32 精度以支持亚像素配准)
    """
    if transform_data is None:
        return ref_coords.astype(np.float32)

    mapped = ref_coords.copy().astype(np.float32)

    # 1. Global Shift
    global_shift = transform_data.get('global_shift_3d', np.zeros(3))
    mapped -= global_shift

    # 2. Local Flow
    flow_2d = transform_data.get('flow_2d')
    flow_3d = transform_data.get('flow_3d')
    flow = flow_2d if flow_2d is not None else flow_3d

    if flow is not None:
        # 这里的 map_coordinates 是必须的，因为 flow 场本身是连续的
        # 但这一步只对 N 个点做，开销很小
        if flow.ndim == 3:  # 2D flow
            h, w = flow.shape[1], flow.shape[2]
            sample_y = np.clip(mapped[:, 1], 0, h-1)
            sample_x = np.clip(mapped[:, 2], 0, w-1)
            
            dy = map_coordinates(flow[0], [sample_y, sample_x], order=1, mode='nearest')
            dx = map_coordinates(flow[1], [sample_y, sample_x], order=1, mode='nearest')
            mapped[:, 1] += dy
            mapped[:, 2] += dx

        elif flow.ndim == 4:  # 3D flow
            d, h, w = flow.shape[1:]
            sample_z = np.clip(mapped[:, 0], 0, d-1)
            sample_y = np.clip(mapped[:, 1], 0, h-1)
            sample_x = np.clip(mapped[:, 2], 0, w-1)
            
            dz = map_coordinates(flow[0], [sample_z, sample_y, sample_x], order=1, mode='nearest')
            dy = map_coordinates(flow[1], [sample_z, sample_y, sample_x], order=1, mode='nearest')
            dx = map_coordinates(flow[2], [sample_z, sample_y, sample_x], order=1, mode='nearest')
            
            mapped[:, 0] += dz
            mapped[:, 1] += dy
            mapped[:, 2] += dx
            
    return mapped

def extract_box_sum_integer(
    img_vol: np.ndarray, 
    coords: np.ndarray, 
    box_size: tuple = (1, 3, 3)
) -> np.ndarray:
    """    
    img_vol: (Z, Y, X)
    coords: (N, 3) float32 -> 将被四舍五入为 int
    """
    D, H, W = img_vol.shape
    bz, by, bx = box_size
    rz, ry, rx = bz // 2, by // 2, bx // 2

    n_spots = len(coords)
    intensities = np.zeros(n_spots, dtype=np.float32)

    # 1. 四舍五入到最近整数 (Round to Nearest Integer)
    # 这与 MATLAB 的索引逻辑一致（MATLAB extents 是基于整数位置）
    # 使用 rint 比 astype(int) 更准确，避免 3.99 -> 3 的截断误差
    coords_int = np.rint(coords).astype(np.int32)
    
    ic_z = coords_int[:, 0]
    ic_y = coords_int[:, 1]
    ic_x = coords_int[:, 2]

    # 2. 遍历 Box 内的所有偏移量
    # 相比生成巨大的 index grid，简单的 Python 循环处理小 box (3x3=9次循环) 反而更快
    # 因为它避免了分配巨大的中间数组
    for dz in range(-rz, rz + 1):
        for dy in range(-ry, ry + 1):
            for dx in range(-rx, rx + 1):
                # 计算当前偏移下的 absolute coordinates
                cur_z = ic_z + dz
                cur_y = ic_y + dy
                cur_x = ic_x + dx
                
                # 3. 极速边界检查 (Vectorized)
                # 虽然增加了逻辑，但直接索引越界的代价是崩溃或 wrap-around
                # 我们只选取有效的点
                valid_mask = (
                    (cur_z >= 0) & (cur_z < D) &
                    (cur_y >= 0) & (cur_y < H) &
                    (cur_x >= 0) & (cur_x < W)
                )
                
                # 4. 累加强度
                # 利用 Boolean Masking 进行部分更新
                if np.any(valid_mask):
                    # 只读取有效坐标的像素值
                    val = img_vol[cur_z[valid_mask], cur_y[valid_mask], cur_x[valid_mask]]
                    intensities[valid_mask] += val

    return intensities

class SignalMiner:
    def __init__(self, config: ExperimentConfig):
        self.cfg = config
        self.loader = ImageLoader(config)
        
    def _load_transforms(self, fov_id):
        base_dir = Path(self.cfg.pipeline.output.directory)
        paths = get_fov_output_structure(base_dir, fov_id)
        path = paths["transforms"] / f"transforms_fov_{fov_id}.npy"
        if not path.exists(): return {} 
        return np.load(path, allow_pickle=True).item()

    def mine_fov(self, fov_id: int):
        print(f"[{'='*20} Mining FOV {fov_id} {'='*20}]")
        base_dir = Path(self.cfg.pipeline.output.directory)
        paths = get_fov_output_structure(base_dir, fov_id)
        # 1. Load Metadata & Transforms
        spots_df = pd.read_csv(paths["spots"] / f"spots_fov_{fov_id}.csv")
        transforms = self._load_transforms(fov_id)
        
        ref_coords = spots_df[['z', 'y', 'x']].values.astype(np.float32)
        n_spots = len(ref_coords)

        # Filters channels
        roles = self.cfg.dataset.channel_roles
        all_channels = sorted(list(roles.keys()))
        channels = [c for c in all_channels if roles.get(c) == 'seq']
        
        print(f" [Miner] Channels to extract: {channels}")
        
        rounds = sorted(list(self.cfg.dataset.round_structure.keys()))
        
        # Pre-allocate
        intensity_matrix = np.zeros((n_spots, len(rounds), len(channels)), dtype=np.float32)
        
        # Box Size
        box_size = self.cfg.pipeline.extraction.integration_box 

        # 2. Main Loop
        # 优化点：外层循环是 Round，内层是 Channel。
        # 我们在这里引入 tqdm 显示总进度
        total_steps = len(rounds) * len(channels)
        
        with tqdm(total=total_steps, desc="Extracting Signals") as pbar:
            for r_idx, r_id in enumerate(rounds):
                # Pre-calculate coordinates for this round ONCE
                trans_data = transforms.get(r_id, {'global_shift_3d': np.zeros(3), 'flow_2d': None})
                
                # 这一步计算浮点坐标
                target_coords = map_spot_coordinates(ref_coords, trans_data)

                current_round_channels = self.cfg.dataset.round_structure[r_id]
                
                for c_idx, c_id in enumerate(channels):
                    if c_id not in current_round_channels:
                        pbar.update(1)
                        continue
                    
                    # Load Image - 这是主要的 IO 开销
                    # 确保是 clean data
                    img_vol = self.loader.load_clean_image(fov_id, r_id, c_id) 
                    
                    # Extraction - The Optimized Part
                    vals = extract_box_sum_integer(img_vol, target_coords, tuple(box_size))
                    
                    intensity_matrix[:, r_idx, c_idx] = vals
                    
                    # 显式删除引用，帮助 GC 
                    del img_vol
                    pbar.update(1)

        # 4. Save
        out_name = paths["extraction"] / f"intensity_matrix_fov_{fov_id}.npy"
        np.save(out_name, intensity_matrix)
        print(f" [Miner] Saved extraction matrix to {out_name.name} | Shape: {intensity_matrix.shape}")
        
        # 5. QC (Optional visualization code kept minimal here for speed)
        self._generate_qc(intensity_matrix, spots_df, rounds, channels, fov_id)

    def _generate_qc(self, matrix, spots_df, rounds, channels, fov_id):
        # 剥离出来的 QC 逻辑，保持主流程清晰
        if not self.cfg.pipeline.output.save_qc_images:
            return
            
        print(f" [QC] Generating extraction QC plots...")
        base_dir = Path(self.cfg.pipeline.output.directory)
        paths = get_fov_output_structure(base_dir, fov_id)
        qc_dir = paths["qc"]
            
        # Trace Plots
        total_intensity = matrix.sum(axis=(1, 2))
        top_indices = np.argsort(total_intensity)[-5:] 
        random_indices = np.random.choice(len(matrix), 5, replace=False)
        selected_indices = np.concatenate([top_indices, random_indices])
            
        plot_spot_traces(
            matrix, selected_indices, 
            rounds, channels,
            output_path=qc_dir / f"spot_traces_fov_{fov_id}.png"
        )
        # Debug CSV
        self._save_debug_csv(matrix, spots_df, rounds, channels, fov_id)

    def _save_debug_csv(self, matrix, spots_df, rounds, channels, fov_id):
        n_debug = min(100, len(spots_df))
        cols = []
        for r in rounds:
            for c in channels:
                cols.append(f"R{r}_C{c}")
        flat_mat = matrix[:n_debug].reshape(n_debug, -1)
        df_debug = spots_df.iloc[:n_debug].copy()
        df_vals = pd.DataFrame(flat_mat, columns=cols, index=df_debug.index)
        final = pd.concat([df_debug, df_vals], axis=1)
        base_dir = Path(self.cfg.pipeline.output.directory)
        paths = get_fov_output_structure(base_dir, fov_id)
        final.to_csv(paths["extraction"] / f"debug_intensities_fov_{fov_id}.csv", index=False)