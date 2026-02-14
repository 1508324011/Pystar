# pystar/visualization.py

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.patches as patches
from tqdm import tqdm
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Tuple
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

from .io import ImageLoader

def compare_preprocessing_5regions(
    raw_vol: np.ndarray, 
    clean_vol: np.ndarray, 
    z_plane: int, 
    fov_id: int, 
    round_id: int,
    roi_size: int = 200,
    output_path: Optional[Path] = None
):
    """
    [Notebook Tool] 预处理效果 5 点采样对比 (5-Point Inspection).
    在指定的 Z 层上选取 Centre, TL, TR, BL, BR 五个区域。
    横向对比每个通道及 Merge 后的 Raw vs Clean。
    
    Parameters
    ----------
    raw_vol : (C, Z, Y, X) 原始数据的 numpy 数组
    clean_vol : (C, Z, Y, X) 处理后数据的 numpy 数组
    z_plane : int 切片层数
    roi_size : int 裁剪窗口大小 (像素), 默认 200
    """
    import matplotlib.pyplot as plt
    
    # 1. 维度检查与切片
    # 确保是 (C, Z, Y, X)
    if raw_vol.ndim == 3: # (Z, Y, X) -> (1, Z, Y, X)
        raw_vol = raw_vol[np.newaxis, ...]
        clean_vol = clean_vol[np.newaxis, ...]
        
    n_channels = raw_vol.shape[0]
    h, w = raw_vol.shape[2:]
    
    # 取出指定 Z 层 -> (C, Y, X)
    raw_slice = raw_vol[:, z_plane, :, :]
    clean_slice = clean_vol[:, z_plane, :, :]
    
    # 2. 定义 5 个采样位置 (中心坐标 y, x)
    positions = {
        "Center": (h//2, w//2),
        "Top-Left": (h//4, w//4),
        "Top-Right": (h//4, 3*w//4),
        "Btm-Left": (3*h//4, w//4),
        "Btm-Right": (3*h//4, 3*w//4)
    }
    
    # 3. 定义颜色 (Cyan, Yellow, Magenta, Red, Green)
    # 用于 Merge 显示
    colors = [
        np.array([0, 1, 1]), # Ch0: Cyan
        np.array([1, 1, 0]), # Ch1: Yellow
        np.array([1, 0, 1]), # Ch2: Magenta
        np.array([1, 0, 0]), # Ch3: Red
        np.array([0, 1, 0])  # Ch4: Green
    ]
    
    # 辅助：着色函数 (单通道灰度 -> RGB)
    def _colorize(img, color_arr):
        # 鲁棒归一化 (1-99 percentile) 避免热像素毁掉对比度
        vmin, vmax = np.percentile(img, [1, 99.5])
        if vmax <= vmin: return np.zeros((*img.shape, 3))
        norm = np.clip((img - vmin) / (vmax - vmin), 0, 1)
        return norm[..., np.newaxis] * color_arr

    # 4. 绘图布局
    # 行数 = 5 (区域), 列数 = 2 (Merge对比) + 2 * N_channels (单通道对比)
    n_rows = 5
    n_cols = 2 + (n_channels * 2)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 2.5 + 1), constrained_layout=True)
    if n_rows == 1: axes = axes[np.newaxis, ...] # 防御性编程
    
    row_names = list(positions.keys())
    
    for r, (pos_name, (cy, cx)) in enumerate(positions.items()):
        # 计算 ROI 边界
        y0, y1 = max(0, cy - roi_size//2), min(h, cy + roi_size//2)
        x0, x1 = max(0, cx - roi_size//2), min(w, cx + roi_size//2)
        
        # 容器：用于合成 Merge 图像
        merge_raw = np.zeros((y1-y0, x1-x0, 3))
        merge_clean = np.zeros((y1-y0, x1-x0, 3))
        
        col_idx = 0
        
        # --- A. 遍历通道 (绘制单通道对比并累加 Merge) ---
        for c in range(n_channels):
            # 获取 ROI 数据
            r_roi = raw_slice[c, y0:y1, x0:x1].astype(np.float32)
            c_roi = clean_slice[c, y0:y1, x0:x1].astype(np.float32)
            color = colors[c % len(colors)]
            
            # --- 核心修改：手动着色逻辑 ---
            def to_rgb(img_data, vmin, vmax, color_vec):
                """将单通道灰度归一化并染成指定 RGB"""
                if vmax <= vmin: return np.zeros((*img_data.shape, 3))
                # 1. 归一化到 0-1
                norm = np.clip((img_data - vmin) / (vmax - vmin), 0, 1)
                # 2. 升维并染色 (H, W, 1) * (3,) -> (H, W, 3)
                return norm[..., np.newaxis] * color_vec

            # 累加到 Merge (使用之前定义的 _colorize，它内部有自己的鲁棒 norm)
            merge_raw += _colorize(r_roi, color)
            merge_clean += _colorize(c_roi, color)
            
            # --- 绘制 Raw Ch ---
            ax_raw = axes[r, 2 + c*2]
            # Raw 数据比较脏，使用 1% - 99.5% 截断，去掉极值热点
            v_min_r, v_max_r = np.percentile(r_roi, [1, 99.5])
            rgb_raw = to_rgb(r_roi, v_min_r, v_max_r, color)
            
            ax_raw.imshow(rgb_raw) # 传入 RGB 数据，不需要 cmap
            if r == 0: ax_raw.set_title(f"RAW Ch{c}", color=color, fontweight='bold')
            ax_raw.axis('off')

            # --- 绘制 Clean Ch ---
            ax_clean = axes[r, 2 + c*2 + 1]
            # Clean 数据背景很黑，使用 0.1% - 99.9% 尽量保留细节
            v_min_c, v_max_c = np.percentile(c_roi, [0.1, 99.9])
            rgb_clean = to_rgb(c_roi, v_min_c, v_max_c, color)
            
            ax_clean.imshow(rgb_clean)
            
            # 边框装饰保持不变
            for spine in ax_clean.spines.values():
                spine.set_edgecolor(color) # Good Taste: 边框也改成对应通道颜色，而不是统一绿色
                spine.set_linewidth(1)
            
            if r == 0: ax_clean.set_title(f"CLEAN Ch{c}", color=color, fontweight='bold')
            ax_clean.axis('off')
            
        # --- B. 绘制 Merge (放在最左边) ---
        # 1. Raw Merge
        ax_m_raw = axes[r, 0]
        ax_m_raw.imshow(np.clip(merge_raw, 0, 1))
        ax_m_raw.set_ylabel(pos_name, fontsize=12, fontweight='bold') # 行标
        if r == 0: ax_m_raw.set_title("RAW MERGE", fontsize=12)
        # 去掉刻度但保留 Label
        ax_m_raw.set_xticks([])
        ax_m_raw.set_yticks([])
        
        # 2. Clean Merge
        ax_m_clean = axes[r, 1]
        ax_m_clean.imshow(np.clip(merge_clean, 0, 1))
        if r == 0: ax_m_clean.set_title("CLEAN MERGE", fontsize=12)
        ax_m_clean.axis('off')
        
    # 全局标题
    fig.suptitle(f"Preprocessing QC: FOV {fov_id} | Round {round_id} | Z={z_plane}", fontsize=16, y=1.02)
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"saved to {output_path}")
        plt.close()
    else:
        plt.show()

def _ensure_2d(img: np.ndarray) -> np.ndarray:
    """内部辅助函数：确保图像是 2D 的。如果是 3D，做最大投影(MIP)。"""
    if img.ndim == 3:
        # 假设顺序是 (Z, Y, X)，沿 Z 轴压缩
        return img.max(axis=0)
    elif img.ndim == 2:
        return img
    else:
        # 极端的防御：如果是 1D 或者 >3D，抛错或者返回切片
        raise ValueError(f"Image dimension {img.ndim} not supported for visualization.")

def normalize_for_display(img: np.ndarray) -> np.ndarray:
    """
    Robust normalization for 8-bit display.
    Handles outliers and zero-division.
    """
    if img.size == 0: return img
    
    # 确保是 2D
    img = _ensure_2d(img)
    
    # 鲁棒的归一化：忽略极值点 (1% - 99%)
    vmin, vmax = np.percentile(img, (1, 99))
    
    if vmax <= vmin: # 防止除以 0
        return np.zeros_like(img, dtype=np.float32)
        
    img_clip = np.clip(img, vmin, vmax)
    return (img_clip - vmin) / (vmax - vmin)

def overlay_images(ref: np.ndarray, mov: np.ndarray) -> np.ndarray:
    """
    经典的 Magenta/Green 叠加图。
    Ref = Green
    Mov = Magenta (Red + Blue)
    白色 = 对齐
    分离 = 没对齐
    """
    # 1. 维度安全检查 (这是关键修改！)
    ref_2d = _ensure_2d(ref)
    mov_2d = _ensure_2d(mov)

    # 形状对齐检查
    if ref_2d.shape != mov_2d.shape:
        # 如果形状不匹配，裁剪到最小尺寸 (极端的防崩溃)
        h = min(ref_2d.shape[0], mov_2d.shape[0])
        w = min(ref_2d.shape[1], mov_2d.shape[1])
        ref_2d = ref_2d[:h, :w]
        mov_2d = mov_2d[:h, :w]

    h, w = ref_2d.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    
    r_norm = normalize_for_display(ref_2d)
    m_norm = normalize_for_display(mov_2d)
    
    rgb[..., 0] = m_norm  # Red (Mov)
    rgb[..., 1] = r_norm  # Green (Ref)
    rgb[..., 2] = m_norm  # Blue (Mov)
    
    return rgb

def save_registration_qc(
    ref_img: np.ndarray, 
    mov_original: np.ndarray,
    mov_registered: np.ndarray, 
    round_id: int,
    score_before: float,
    score_after: float,
    output_dir: Path,
    fov_id: int
):
    """
    生成三联图：原始叠加 -> 配准后叠加
    """
    # 这里不需要显式调用 _ensure_2d，因为 overlay_images 内部会处理
    # 这样代码更简洁
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # 1. Before
    ax0 = axes[0]
    ax0.imshow(overlay_images(ref_img, mov_original))
    ax0.set_title(f"Before: Round {round_id} vs Ref\nCorr: {score_before:.4f}")
    ax0.axis('off')

    # 2. After
    ax1 = axes[1]
    ax1.imshow(overlay_images(ref_img, mov_registered))
    ax1.set_title(f"After: Round {round_id} vs Ref\nCorr: {score_after:.4f}")
    ax1.axis('off')
    
    out_name = output_dir / f"alignment_qc_fov{fov_id}_round{round_id}.jpg"
    plt.tight_layout()
    plt.savefig(out_name, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   [QC] Saved visual report: {out_name.name}")

def plot_vector_field(flow: np.ndarray, step: int = 20):
    """
    用于 Notebook 调试：画出流场箭头。
    flow shape: (2, H, W)
    """
    if flow is None:
        print("Flow is None, skipping plot.")
        return

    dy, dx = flow[0], flow[1]
    h, w = dy.shape
    
    y, x = np.mgrid[0:h:step, 0:w:step]
    u = dx[0:h:step, 0:w:step]
    v = dy[0:h:step, 0:w:step]
    
    plt.figure(figsize=(10, 10))
    # 黑色背景看起来更酷，也更清楚
    plt.imshow(np.zeros((h, w)), cmap='gray') 
    plt.quiver(x, y, u, v, angles='xy', scale_units='xy', scale=0.5, color='yellow')
    plt.gca().invert_yaxis() # 图片坐标系Y轴向下
    plt.title("Deformation Field (Optical Flow)")
    plt.axis('off')
    plt.show()
    
def overlay_rgb(img_ref: np.ndarray, img_mov: np.ndarray) -> np.ndarray:
    """
    [Helper] 创建高质量的红绿叠加图 (Green=Ref, Magenta=Mov)。
    
    自动处理归一化，确保显示效果清晰。
    Ref (Green) + Mov (Magenta: Red+Blue) -> White (Perfect Match)
    
    Parameters
    ----------
    img_ref : 2D Array
    img_mov : 2D Array
    
    Returns
    -------
    rgb_img : (H, W, 3) float32 array, range [0, 1]
    """
    # 1. 安全检查
    if img_ref.shape != img_mov.shape:
        raise ValueError(f"Shape mismatch: {img_ref.shape} vs {img_mov.shape}")
        
    # 2. 鲁棒归一化 (Robust Normalization)
    # 使用 percentile 避免极亮噪点导致的整体变暗
    def normalize(img):
        vmin, vmax = np.percentile(img, [1, 99])
        if vmax - vmin < 1e-9:
            return np.zeros_like(img, dtype=np.float32)
        return np.clip((img - vmin) / (vmax - vmin), 0, 1).astype(np.float32)

    n_ref = normalize(img_ref)
    n_mov = normalize(img_mov)
    
    # 3. 构建 RGB
    # Green Channel = Reference
    # Red + Blue = Magenta = Moving
    h, w = img_ref.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    
    rgb[..., 0] = n_mov  # R (part of Magenta)
    rgb[..., 1] = n_ref  # G (Reference)
    rgb[..., 2] = n_mov  # B (part of Magenta)
    
    return rgb



def inspect_spots_interactive(img, spots_df, z_plane, center=None, roi_size=200,output_path: Optional[Path] = None):
    """
    1. 居中裁剪一个小的 ROI (Region of Interest)，不要看全图，全图什么都看不清。
    2. 只画出 Z 轴在当前层附近的点 (Z +/- 1)。
    3. 用空心圈而不是实心叉，这样能看到圈里面到底有没有东西。
    如果是 4D，会横向排列展示所有 Channel 的寻点结果。
    """
    
    # 1. 维度标准化 (Normalize to 4D: C, Z, Y, X)
    if img.ndim == 3:
        img_4d = img[np.newaxis, ...] # (1, Z, Y, X)
    elif img.ndim == 4:
        img_4d = img
    else:
        raise ValueError(f"Image must be 3D or 4D, got {img.ndim}D")
        
    n_channels = img_4d.shape[0]
    
    # 2. 准备画布 (1 行 N 列)
    fig, axes = plt.subplots(1, n_channels, figsize=(6 * n_channels, 6), constrained_layout=True)
    if n_channels == 1: axes = [axes] # 列表化，方便迭代
    
    # 3. 确定 ROI 物理边界 (所有通道共用一个 ROI)
    h, w = img_4d.shape[2:] # Y, X
    cy, cx = center if center else (h // 2, w // 2)
    y0, y1 = max(0, cy - roi_size//2), min(h, cy + roi_size//2)
    x0, x1 = max(0, cx - roi_size//2), min(w, cx + roi_size//2)

    # 4. 遍历通道绘图
    for c, ax in enumerate(axes):
        # A. 提取当前通道的 ROI 图像
        # shape: (C, Z, Y, X) -> [c, z, y:y, x:x]
        img_roi = img_4d[c, z_plane, y0:y1, x0:x1]
        
        # B. 过滤点 (Space + Channel)
        # 必须同时满足: 在 ROI 范围内 + Z轴邻域 + 当前 Channel
        # 注意: 如果 spots_df 没有 'channel' 列 (旧数据)，默认视为 Channel 0
        if 'channel' in spots_df.columns:
            mask_c = (spots_df['channel'] == c)
        else:
            mask_c = (c == 0) # Fallback

        mask_pos = (spots_df['y'] >= y0) & (spots_df['y'] < y1) & \
                    (spots_df['x'] >= x0) & (spots_df['x'] < x1) & \
                    (np.abs(spots_df['z'] - z_plane) <= 1)
        
        sub_spots = spots_df[mask_c & mask_pos].copy()
        
        # C. 绘图 (图像)
        vmin, vmax = np.percentile(img_roi, [5, 99.5])
        ax.imshow(img_roi, cmap='gray', vmin=vmin, vmax=vmax, interpolation='nearest')
        
        # D. 绘图 (点)
        local_y = sub_spots['y'] - y0
        local_x = sub_spots['x'] - x0
        on_plane = np.isclose(sub_spots['z'], z_plane, atol=0.5)

        # On Plane (Red)
        ax.scatter(local_x[on_plane], local_y[on_plane], 
                    s=100, facecolors='none', edgecolors='red', linewidth=1.5, label='On Plane')
        # Neighbor (Yellow Dashed)
        ax.scatter(local_x[~on_plane], local_y[~on_plane], 
                    s=60, facecolors='none', edgecolors='yellow', linewidth=1, linestyle='--', label='Neighbor')
        
        ax.set_title(f"CH {c} | Z={z_plane}\nSpots: {len(sub_spots)}")
        if c == 0: # 只在第一个图显示图例，避免遮挡
            ax.legend(loc='upper right', fontsize='small')
        ax.axis('off')

    plt.suptitle(f"Inspection: Z={z_plane} | ROI Center: ({cy}, {cx})", fontsize=14)
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f" [QC] Saved spot finding QC to {output_path.name}")
    else:
        plt.show()
    


def compare_spot_algorithms(img, algo_results, channel_id, z_plane, center=None, roi_size=200):
    """
    横向对比不同寻点算法的结果。
    algo_results: 字典，例如 {'Max3D': df1, 'DoG': df2, 'Spotiflow': df3}
    """
    # 1. 维度处理：只取特定的 Channel 进行对比
    if img.ndim == 4:
        img_3d = img[channel_id]
    else:
        img_3d = img # Assume 3D
        if channel_id > 0:
            print(f"Warning: Image is 3D but channel_id={channel_id} requested. Ignoring channel_id.")

    # 2. 数据过滤：只取对应 Channel 的点
    # 我们需要在进入绘图循环前，把所有 DataFrame 都过滤一遍
    filtered_results = {}
    for name, df in algo_results.items():
        if 'channel' in df.columns:
            filtered_results[name] = df[df['channel'] == channel_id]
        else:
            # 如果没有 channel 列，假设它是单通道结果，直接用
            filtered_results[name] = df

    # 3. 确定 ROI
    h, w = img_3d.shape[1:]
    cy, cx = center if center else (h // 2, w // 2)
    y0, y1 = max(0, cy - roi_size//2), min(h, cy + roi_size//2)
    x0, x1 = max(0, cx - roi_size//2), min(w, cx + roi_size//2)
    
    img_roi = img_3d[z_plane, y0:y1, x0:x1]
    vmin, vmax = np.percentile(img_roi, [5, 99.5])

    # 4. 绘图循环 (N Algos)
    n_algos = len(filtered_results)
    fig, axes = plt.subplots(1, n_algos, figsize=(6 * n_algos, 6), constrained_layout=True)
    if n_algos == 1: axes = [axes]

    for ax, (algo_name, df) in zip(axes, filtered_results.items()):
        ax.imshow(img_roi, cmap='gray', vmin=vmin, vmax=vmax, interpolation='nearest')
        
        mask = (df['y'] >= y0) & (df['y'] < y1) & \
                (df['x'] >= x0) & (df['x'] < x1) & \
                (np.abs(df['z'] - z_plane) <= 1)
        
        sub_spots = df[mask].copy()
        local_y = sub_spots['y'] - y0
        local_x = sub_spots['x'] - x0
        on_plane = np.isclose(sub_spots['z'], z_plane, atol=0.5)

        ax.scatter(local_x[on_plane], local_y[on_plane], 
                    s=100, facecolors='none', edgecolors='red', linewidth=1.5, label='On Plane')
        ax.scatter(local_x[~on_plane], local_y[~on_plane], 
                    s=60, facecolors='none', edgecolors='yellow', linewidth=1, linestyle='--', label='Neighbor')
        
        ax.set_title(f"Algo: {algo_name}\nCH {channel_id} | Count: {len(sub_spots)}")
        ax.axis('off')
    
    plt.suptitle(f"Algorithm Comparison on Channel {channel_id} | Z={z_plane}", fontsize=16)
    plt.show()
    
def plot_spot_traces(
    intensity_matrix: np.ndarray,
    spot_indices: np.ndarray,
    rounds: List[int],
    channels: List[int],
    output_path: Optional[Path] = None
):
    """
    绘制选定 Spot 在所有轮次的光强变化轨迹 (Intensity Trace)。
    这用于诊断信号质量：好的信号应该有清晰的峰值，坏的信号是杂乱无章的。

    Parameters
    ----------
    intensity_matrix : np.ndarray
        Shape (N_spots, N_rounds, N_channels)
    spot_indices : list or np.ndarray
        要检查的 Spot 索引列表，e.g. [0, 100, 5050]
    rounds : list
        轮次列表，作为 X 轴刻度，e.g. [1, 2, 3, 4]
    channels : list
        通道列表，作为 Legend，e.g. [0, 1, 2, 3]
    output_path : Path, optional
        如果提供，保存图片而不是显示。
    """
    n_spots = len(spot_indices)
    if n_spots == 0:
        print("Warning: No spots selected for tracing.")
        return

    # 限制最大绘图数量，防止传进来 100000 个点炸掉内存
    if n_spots > 10:
        print(f"Warning: Too many spots ({n_spots}). Truncating to first 10.")
        spot_indices = spot_indices[:10]
        n_spots = 10    

    fig, axes = plt.subplots(n_spots, 1, figsize=(12, 2.5 * n_spots), sharex=True)
    if n_spots == 1: axes = [axes]

    # 定义颜色循环，确保不同通道颜色区分明显
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'] # Matplotlib default

    for ax, idx in zip(axes, spot_indices):
        # Shape: (N_rounds, N_channels)
        spot_data = intensity_matrix[idx]
        
        # 绘制每个通道的折线
        for c_idx, c_id in enumerate(channels):
            color = colors[c_idx % len(colors)]
            ax.plot(rounds, spot_data[:, c_idx], marker='o', linewidth=2, 
                    label=f"CH {c_id}", color=color, alpha=0.8)
            
        ax.set_title(f"Spot Index: {idx} | Intensity Trace", fontsize=10, pad=5)
        ax.set_ylabel("Intensity")
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # 优化 X 轴
        ax.set_xticks(rounds)
        
        # 只在第一个图显示图例，避免乱
        if idx == spot_indices[0]:
            ax.legend(loc='upper right', frameon=True, fontsize='small')

    plt.xlabel("Round ID")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f" [QC] Saved spot traces to {output_path.name}")
    else:
        plt.show()
        
def plot_channel_correlation(
    intensity_matrix: np.ndarray,
    round_index: int, 
    channel_labels: List[str],
    sample_size: int = 5000,
    output_path: Optional[Path] = None
):
    """
    绘制某一轮次内，不同通道之间的相关性矩阵 (Corner Plot)。
    用于检查 Crosstalk (串色) 或 信号正交性。
    
    Parameters
    ----------
    intensity_matrix : np.ndarray
        (N_spots, N_rounds, N_channels)
    round_index : int
        我们要检查哪一轮？ 注意这是 0-based index，不是 Round ID。
    channel_labels : list
        通道名称列表，e.g. ["CH0", "CH1", "CH2", "CH3"]
    """
    if sns is None:
        print("Error: Seaborn is not installed. Run `pip install seaborn`.")
        return

    # Extract data for specific round
    # Shape: (N_spots, N_channels)
    data_round = intensity_matrix[:, round_index, :]

    # Subsample (Plotting 100k points makes pairplot slow)
    n_total = data_round.shape[0]
    if n_total > sample_size:
        indices = np.random.choice(n_total, sample_size, replace=False)
        data_subset = data_round[indices]
    else:
        data_subset = data_round

    # Create DataFrame for Seaborn
    df_plot = pd.DataFrame(data_subset, columns=channel_labels)

    # Plot
    # corner=True 只画左下角，如果想看全图可以去掉，但通常对称的没必要
    g = sns.pairplot(df_plot, diag_kind="kde", plot_kws={'alpha': 0.3, 's': 10}, corner=True)
    g.fig.suptitle(f"Channel Correlation Check (Round Index {round_index})", y=1.02)

    if output_path:
        g.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f" [QC] Saved correlation plot to {output_path.name}")
    else:
        plt.show()
        
def plot_spot_extraction_check(
    loader,
    fov_id: int,
    spot_info: dict,
    rounds: list,
    channels: list,
    config_obj,  # 传入整个 config 对象以便读取 integration_box
    view_radius: int = 25,
    output_path: str = None
):
    """
    [QC] 绘制单个 Spot 的多通道彩色提取检查图。
    
    改进点：
    1. 移除无用的 Donut，只保留 Box。
    2. Box 尺寸自动从 Config 读取。
    3. 只有 Merged 列显示彩色叠加，单通道显示灰度但带颜色边框（保持清晰度），或者单通道也着色。
       这里我们采用：单通道做伪彩处理，Merged 做加法混合。
    
    spot_info 需要包含: 
    - 'coords_per_round': {r_id: [z, y, x]}
    - 'spot_id': int
    - 'gene': str (Decoded result)
    - 'barcode': str
    - 'quality': float
    """
    
    # 1. 准备颜色映射 (Fixed High-Contrast Palette)
    # 对应 Channel 0, 1, 2, 3...
    # 常用搭配: 0:Cyan, 1:Yellow, 2:Magenta, 3:Red (适合黑色背景)
    CHANNEL_COLORS = [
        np.array([0, 1, 1]),   # Ch0: Cyan
        np.array([1, 1, 0]),   # Ch1: Yellow
        np.array([1, 0, 1]),   # Ch2: Magenta
        np.array([1, 0, 0]),   # Ch3: Red
        np.array([0, 1, 0]),   # Ch4: Green (fallback)
    ]
    
    # 2. 获取 Box Size (取 YX 维度，假设是正方形)
    # config.integration_box is [z, y, x]
    int_box = config_obj.pipeline.extraction.integration_box
    box_yx = int_box[1] 
    
    n_rows = len(rounds)
    n_cols = len(channels) + 1 
    
    # 布局调整
    # 增加 figsize 的高度，给标题腾地方
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*2.5, n_rows*2.5 + 1.5))

    if n_rows == 1: axes = np.array([axes])
    if n_cols == 1: axes = axes[:, np.newaxis]
    
    # 解析元数据
    spot_id = spot_info.get('spot_id', 'Unknown')
    gene_name = spot_info.get('gene', 'N/A')
    barcode = spot_info.get('barcode', 'N/A')
    quality = spot_info.get('quality', 0.0)
    spot_coords_map = spot_info['coords_per_round']

    # 辅助：绘制 Box
    def draw_box(ax, cx, cy, color='white'):
        box_half = box_yx / 2.0
        # Matplotlib Rectangle 是 (left, bottom), width, height
        rect = patches.Rectangle(
            (cx - box_half, cy - box_half),
            box_yx, box_yx,
            linewidth=1.5, edgecolor=color, facecolor='none', linestyle='-'
        )
        ax.add_patch(rect)

    # 辅助：归一化并着色
    def colorize(img, color_arr):
        # Robust Norm
        vmin, vmax = np.percentile(img, [2, 99.8])
        if vmax <= vmin: return np.zeros((*img.shape, 3))
        norm = np.clip((img - vmin) / (vmax - vmin), 0, 1)
        # (H,W) -> (H,W,1) * (3,) -> (H,W,3)
        return norm[..., np.newaxis] * color_arr

    # --- 主循环 ---
    for r_idx, r_id in enumerate(rounds):
        coords = spot_coords_map.get(r_id)
        
        # 用于 Merged 的累加器
        merged_rgb = np.zeros((view_radius*2+1, view_radius*2+1, 3))
        has_data = False

        # 如果这轮没坐标，跳过
        if coords is None:
            for c_idx in range(n_cols):
                axes[r_idx, c_idx].axis('off')
                axes[r_idx, c_idx].text(0.5, 0.5, "No Coords", ha='center', color='white')
            continue

        cz, cy, cx = coords
        
        # 遍历通道
        for c_idx, c_id in enumerate(channels):
            ax = axes[r_idx, c_idx]
            color = CHANNEL_COLORS[c_idx % len(CHANNEL_COLORS)]
            
            try:
                # 惰性加载 + 裁剪 (和之前逻辑一致)
                # 这里为了性能及简洁，直接调用 loader 的 protected 方法是合理的实用主义
                path = loader._get_path(fov_id, r_id, c_id)
                # 注意：这里我们只读一个小块，不要把整个图读进来
                # 但 TIFFFile 通常不支持小块读取，除非是 Tile 格式。
                # Dask 切片是目前最好的方案。
                dask_img = loader._lazy_load_tiff(path)
                D, H, W = dask_img.shape
                
                # 计算切片范围
                y0, y1 = max(0, int(cy) - view_radius), min(H, int(cy) + view_radius + 1)
                x0, x1 = max(0, int(cx) - view_radius), min(W, int(cx) + view_radius + 1)
                z0, z1 = max(0, int(cz) - 1), min(D, int(cz) + 2) # MIP thickness 3
                
                # Compute (IO happens here)
                vol_crop = dask_img[z0:z1, y0:y1, x0:x1].compute()
                
                if vol_crop.size == 0:
                    mip = np.zeros((view_radius*2+1, view_radius*2+1))
                else:
                    mip = np.max(vol_crop, axis=0) # MIP
                    
                    # Pad 如果在边缘
                    target_h = view_radius*2+1
                    target_w = view_radius*2+1
                    ph = target_h - mip.shape[0]
                    pw = target_w - mip.shape[1]
                    if ph > 0 or pw > 0:
                        mip = np.pad(mip, ((0, ph), (0, pw)))

                # 1. 绘制单通道伪彩 (更容易看清来源)
                rgb_single = colorize(mip, color)
                ax.imshow(rgb_single, origin='upper')
                
                # 2. 累加到 Merged
                merged_rgb += rgb_single
                has_data = True
                
                # Box 永远画在正中心 (因为我们是基于 cy, cx 裁剪的)
                draw_box(ax, view_radius, view_radius, color='white')
                
                # 装饰
                if r_idx == 0: 
                    ax.set_title(f"CH{c_id}", color=color, fontweight='bold')
                
            except Exception as e:
                ax.text(0.5, 0.5, "Err", ha='center', color='red')
                print(f"Debug: {e}")
            
            ax.axis('off')

        # --- 绘制 Merged Image ---
        ax_merge = axes[r_idx, -1]
        
        if has_data:
            # 简单的 Gamma 校正让暗部更清楚，不做复杂的 Tone mapping 并防止过曝
            final_merge = np.clip(merged_rgb, 0, 1) 
            ax_merge.imshow(final_merge, origin='upper')
            
            # 画一个显眼的 Box
            draw_box(ax_merge, view_radius, view_radius, color='white')
        else:
            ax_merge.text(0.5, 0.5, "No Data", ha='center', color='white')

        # Merged 这一列的标题
        if r_idx == 0:
            ax_merge.set_title("MERGED", color='black', fontweight='bold')
        
        # 左侧显示轮次信息
        axes[r_idx, 0].text(-0.1, 0.5, f"R{r_id}", transform=axes[r_idx, 0].transAxes, 
                            color='black', ha='right', va='center', fontweight='bold')
    # 全局标题 (包含解码结果)
    # 用深色背景的 fig text 会更好看，这里简单设置
    #fig.patch.set_facecolor('#1c1c1c') # 像 VSCode 一样的深灰色背景，专业
    #正常白色背景
    fig.patch.set_facecolor('white')
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # 标题放到顶部区域
    title_str = (f"SPOT INSPECTOR | ID: {spot_id} | GENE: {gene_name}\n"
                 f"BARCODE: {barcode} | QUALITY: {quality:.2f}")
    
    plt.suptitle(title_str, color='black', y=0.98, va='top', fontsize=16, fontweight='bold')
    
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close()
    else:
        plt.show()

class SpotInspector:
    def __init__(self, config, fov_id: int):
        self.cfg = config
        self.fov_id = fov_id
        self.loader = ImageLoader(config)
        
        # 预加载位移场 (Transforms)，避免每次查询都读硬盘
        self.transforms = self._load_transforms()
        
        # 预加载 Spots DataFrame (可选，如果内存够大)
        self.df = self._load_decoded_csv()
        
    def _load_transforms(self):
        """加载位移场数据"""
        path = Path(self.cfg.pipeline.output.directory) / "transforms" / f"transforms_fov_{self.fov_id}.npy"
        if not path.exists():
            print(f"Warning: Transform file not found: {path}")
            return {}
        return np.load(path, allow_pickle=True).item()

    def _load_decoded_csv(self):
        """加载解码后的 CSV"""
        path = Path(self.cfg.pipeline.output.directory) / "decoded" / f"decoded_fov_{self.fov_id}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Decoded CSV not found: {path}")
        return pd.read_csv(path)

    def view_gene(self, gene_name: str, index: int = 0):
        """
        交互式查看器的主入口。
        
        Parameters
        ----------
        gene_name : str
            你想看哪个基因？比如 "GAPDH"
        index : int
            你想看第几个？比如 0 (光强最高的/列表第一个)，或者 10 (随机一个)
        """
        from .mining import map_spot_coordinates
        
        # 1. 筛选基因
        # 忽略大小写
        subset = self.df[self.df['gene'].str.upper() == gene_name.upper()]
        
        if len(subset) == 0:
            print(f"Error: No spots found for gene '{gene_name}' in FOV {self.fov_id}")
            return
            
        if index >= len(subset):
            print(f"Error: Index {index} out of bounds. Found {len(subset)} spots for '{gene_name}'.")
            return
            
        # 2. 提取目标行
        # 我们可以按质量或光强排序，确保 index=0 是“最好”的那个，或者保持 CSV 顺序
        # 这里默认按 CSV 顺序
        target_row = subset.iloc[index]
        
        print(f"[{'='*10} Spot Inspector {'='*10}]")
        print(f" Gene: {target_row['gene']}")
        print(f" Ranking: {index}/{len(subset)}")
        print(f" Barcode: {target_row['barcode']}")
        print(f" Quality: {target_row['quality']:.4f}")
        print(f" Position (R1): Z={target_row['z']}, Y={target_row['y']}, X={target_row['x']}")
        
        # 3. 重建轨迹 (Reconstruct Trajectory)
        # 这是最关键的一步：我们需要把 R1 的坐标映射到所有轮次
        base_coord = np.array([target_row['z'], target_row['y'], target_row['x']], dtype=np.float32)
        
        coords_per_round = {}
        rounds = sorted(list(self.cfg.dataset.round_structure.keys()))
        
        for r_id in rounds:
            # 获取该轮的位移参数
            trans_data = self.transforms.get(r_id, {'global_shift_3d': np.zeros(3), 'flow_2d': None})
            
            # 计算映射坐标
            # map_spot_coordinates 需要 (N, 3) 输入，所以增加一个维度
            mapped = map_spot_coordinates(base_coord[np.newaxis, :], trans_data)[0]
            coords_per_round[r_id] = mapped # [z, y, x]
            
        # 4. 构建 spot_info
        spot_info = {
            'spot_id': target_row.get('spot_id', target_row.name),
            'gene': target_row['gene'],
            'barcode': target_row['barcode'],
            'quality': target_row['quality'],
            'coords_per_round': coords_per_round
        }
        
        # 5. 调用画图
        # 获取 Seq Channel
        roles = self.cfg.dataset.channel_roles
        seq_channels = sorted([c for c, role in roles.items() if role == 'seq'])
        
        plot_spot_extraction_check(
            loader=self.loader,
            fov_id=self.fov_id,
            spot_info=spot_info,
            rounds=rounds,
            channels=seq_channels,
            config_obj=self.cfg,
            view_radius=16, 
            output_path=None # None 表示直接在 Notebook 显示
        )

class ExpressionValidator:
    def __init__(self, pystar_results_path: str):
        """
        初始化验证器，加载 PyStar 的解码结果。
        """
        self.pystar_path = Path(pystar_results_path)
        if not self.pystar_path.exists():
            raise FileNotFoundError(f"PyStar results not found: {self.pystar_path}")
            
        print(f" [Validator] Loading PyStar results: {self.pystar_path.name}...")
        self.df_pystar = pd.read_csv(self.pystar_path)
        
        # 1. 聚合计数 (Aggregation)
        # 过滤掉 background
        valid_spots = self.df_pystar[self.df_pystar['gene'] != 'background'].copy()
        
        # 2. 清洗基因名 (Suffix Stripping)
        # 去除 _ntRNA 或 _rbRNA 后缀
        # 使用正则：结尾是 _ntRNA 或 _rbRNA 的替换为空
        print(" [Validator] Stripping suffixes (_ntRNA, _rbRNA)...")
        valid_spots['gene_clean'] = valid_spots['gene'].astype(str).str.replace(r'_(nt|rb)RNA$', '', regex=True)
        
        # 3. 计算 Raw Counts
        self.counts_pystar = valid_spots['gene_clean'].value_counts().reset_index()
        self.counts_pystar.columns = ['gene', 'count_pystar']
        print(f"   -> Found {len(self.counts_pystar)} unique genes in PyStar results.")

    def compare_with_reference(self, ref_path: str, gene_col: str, value_col: str, 
                            label: str = "Reference", log_transform: bool = True):
        """
        核心对比逻辑。
        ref_path: 参考文件路径 (tsv/csv)
        gene_col: 参考文件中基因名的列名
        value_col: 参考文件中数值(TPM/CPM)的列名
        """
        ref_path = Path(ref_path)
        if not ref_path.exists():
            raise FileNotFoundError(f"Reference file not found: {ref_path}")
            
        # 1. 加载参考集
        # 自动识别分隔符
        sep = '\t' if ref_path.suffix == '.tsv' else ','
        df_ref = pd.read_csv(ref_path, sep=sep)
        
        # 简单的列名检查
        if gene_col not in df_ref.columns or value_col not in df_ref.columns:
            raise ValueError(f"Reference columns '{gene_col}' or '{value_col}' not found. Available: {list(df_ref.columns)}")
            
        # 提取关键数据
        ref_subset = df_ref[[gene_col, value_col]].copy()
        ref_subset.columns = ['gene', 'value_ref']
        
        # 聚合参考集 (防止参考集里有重复基因名行)
        ref_subset = ref_subset.groupby('gene')['value_ref'].sum().reset_index()
        
        # 2. 合并数据 (Inner Join)
        merged = pd.merge(self.counts_pystar, ref_subset, on='gene', how='inner')
        n_common = len(merged)
        
        print(f"\n[{'='*10} Comparison: PyStar vs {label} {'='*10}]")
        print(f" Common Genes: {n_common}")
        
        if n_common < 5:
            print(" [Error] Not enough common genes to calculate correlation. Check gene names/suffixes.")
            return

        # 3. 数据转换 (Log1p)
        # 生物数据的相关性必须在 Log 空间计算，否则高表达基因会主导 Pearson R
        x_raw = merged['count_pystar']
        y_raw = merged['value_ref']
        
        if log_transform:
            x = np.log1p(x_raw)
            y = np.log1p(y_raw)
            axis_label_suffix = " (Log1p)"
        else:
            x, y = x_raw, y_raw
            axis_label_suffix = ""

        # 4. 计算相关性
        r_pearson, p_pearson = pearsonr(x, y)
        r_spearman, p_spearman = spearmanr(x, y)
        
        print(f" Pearson R : {r_pearson:.4f}")
        print(f" Spearman R: {r_spearman:.4f}")
        
        # 5. 绘图 (Glass Box QC)
        plt.figure(figsize=(8, 8))
        sns.regplot(x=x, y=y, scatter_kws={'alpha': 0.6, 's': 30})
        
        # 标注一些离群点 (Outliers)
        # 计算残差，标注偏离回归线最远的基因
        # 简单的距离：|y - x| (假设斜率为1，虽然regplot不是1，但作为粗略标注够了)
        # 这里为了简单，我们标注 x (PyStar count) 最高的前 5 个基因
        top_indices = np.argsort(x)[-5:]
        for idx in top_indices:
            plt.text(x[idx], y[idx], merged.iloc[idx]['gene'], 
                    fontsize=9, color='black', ha='right', weight='bold')

        plt.title(f"Correlation: PyStar vs {label}\n(n={n_common} genes)")
        plt.xlabel(f"PyStar Counts{axis_label_suffix}")
        plt.ylabel(f"{label} Values{axis_label_suffix}")
        
        # 在图上写统计值
        stats_text = (f"Pearson R = {r_pearson:.3f}\n"
                    f"Spearman ρ = {r_spearman:.3f}")
        plt.gca().text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.show()