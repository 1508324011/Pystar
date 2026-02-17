# pystar/decoding.py
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Callable
from tqdm import tqdm

from .io import get_fov_output_structure
from .infrastructure import ExperimentConfig

def compatible_base_calling(norm_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    1. 找到每个round的max值
    2. 检测平局：如果有多个channel值相等，标记为-1和Inf
    3. 计算质量分数：-log(max_val)
    4. 全局过滤：任何round有Inf就废弃整个spot
    
    Parameters:
    -----------
    norm_matrix : np.ndarray (N_spots, N_rounds, N_channels)
        L2归一化后的强度矩阵
        
    Returns:
    --------
    read_indices : np.ndarray (N_spots, N_rounds)
        颜色索引，-1表示平局
    base_scores : np.ndarray (N_spots, N_rounds)
        负对数质量分数，Inf表示无效
    is_valid : np.ndarray (N_spots,)
        bool数组，True表示所有round都有效（无平局、无Inf）
    """
    N, R, C = norm_matrix.shape
    
    # 1. 找到每个round的最大值
    max_vals = np.max(norm_matrix, axis=2)  # (N, R)
    
    # 2. 检测平局（tie-breaking）
    # Matlab逻辑: m = find(colorSeq(i,j,:) == currMax); if numel(m) ~= 1
    is_max = (norm_matrix == max_vals[:, :, np.newaxis])  # (N, R, C)
    num_max = np.sum(is_max, axis=2)  # (N, R) 每个round有几个max
    
    has_tie = (num_max > 1)  # (N, R) bool数组
    
    # 3. 计算read_indices
    #  maxColors(i,j) = m(1); 或 -1 如果平局
    read_indices = np.argmax(norm_matrix, axis=2)  # (N, R)
    read_indices[has_tie] = -1  # 平局标记为-1
    
    # 4. 计算base_scores（负对数）
    #  baseScores(i,j) = -log(currMax);
    with np.errstate(divide='ignore', invalid='ignore'):  
        # 忽略log(0)和log(nan)的警告
        base_scores = -np.log(max_vals)  # (N, R)
    
    # 平局的地方设为Inf
    #  baseScores(i,j) = Inf;
    base_scores[has_tie] = np.inf
    
    # 处理NaN（如果max_val是0或负数）
    base_scores[~np.isfinite(base_scores)] = np.inf
    
    # 5. 全局有效性检查
    #  if ~any(isinf(baseScores(i, :)))
    is_valid = ~np.any(np.isinf(base_scores), axis=1)  # (N,)
    
    return read_indices, base_scores, is_valid


def compatible_quality_filter(
    base_scores: np.ndarray, 
    threshold: float = 0.5
) -> np.ndarray:
    """
    belowScoreThresh = mean(allScores, 2) < 0.5;
    toKeep = belowScoreThresh & finiteScores;
    
    注意：score越小越好（负对数的特性）
    
    Parameters:
    -----------
    base_scores : np.ndarray (N_spots, N_rounds)
        负对数质量分数
    threshold : float
        质量阈值，默认0.5（Matlab标准）
        
    Returns:
    --------
    pass_filter : np.ndarray (N_spots,)
        bool数组，True表示通过过滤
    """
    # 只对有限值计算平均（Inf会被自动处理）
    # 但其实有Inf的spot已经在is_valid中被过滤了
    with np.errstate(invalid='ignore'):
        mean_scores = np.mean(base_scores, axis=1)  # (N,)
    
    # Matlab逻辑：mean(score) < threshold 才保留
    # 因为score越小越好（-log的特性）
    pass_filter = mean_scores < threshold
    
    return pass_filter


class Decoder:
    def __init__(self, config: ExperimentConfig):
        self.cfg = config
        
        # 加载并编译码本
        self.gene_map, self.barcode_map = self._compile_codebook()
        self.reverse_lookups = self._build_reverse_lookups()
        
    def _compile_codebook(self) -> Tuple[Dict[str, str], pd.DataFrame]:
        """
        [Critical Logic] Forward Simulation.
        不反推碱基，而是把基因表的碱基序列(ACTG)直接翻译成期望的颜色序列(0123)。
        """
        codebook_cfg = self.cfg.codebook
        gene_list_path = Path(codebook_cfg.gene_list)
        topo = codebook_cfg.topology
        
        if not gene_list_path.exists():
            raise FileNotFoundError(f"Gene list not found: {gene_list_path}")
            
        # 1. 读取基因表 (假设没有 header，或者根据实际情况修改)
        # 通常 genes.csv 结构是: GeneName, Sequence
        try:
            df_genes = pd.read_csv(gene_list_path, header=None, names=['gene', 'seq'])
        except Exception:
            # 兼容带有 header 的情况
            df_genes = pd.read_csv(gene_list_path)
            if 'gene' not in df_genes.columns: # fallback
                df_genes.columns = ['gene', 'seq']
        
        # Topology Preprocessing (Global)
        # ---------------------------------------------------
        if topo.func == "reverse_string":
            print(" [Decoder] Applying Topology: Reverse Sequence")
            # STRING REVERSE in Python
            df_genes['processed_seq'] = df_genes['seq'].apply(lambda s: s[::-1])
        else:
            df_genes['processed_seq'] = df_genes['seq']
            
        # 3. Build Encoding Functions (闭包工厂)
        # 我们把 Config 里的 mapping 转换成 Python 可调用的函数
        encoders = {}
        for table_name, mapping in codebook_cfg.encoding_tables.items():
            encoders[table_name] = self._create_encoder(mapping, codebook_cfg.channel_base_index)

        # 4. Parse Blueprint Structure
        # 建立一个字典方便按 ID 查找 segment 定义
        segment_defs = {seg.id: seg for seg in topo.structure}
        
        # 验证 physical_order 是否都定义了
        for seg_id in topo.physical_order:
            if seg_id not in segment_defs:
                raise ValueError(f"Topology physical_order references undefined segment ID: {seg_id}")

        # 5. The Assembler (核心循环)
        def assemble_barcode(seq: str) -> str:
            full_barcode = ""
            
            # 严格按照物理成像顺序 (Physical Order) 拼接
            # 因为 Miner 提取出来的矩阵是 [R1, R2 ... Rn] 排序的
            # 如果 R1-5 属于 seqD，R6-10 属于 seqF，那我们就必须先算 seqD 再算 seqF
            
            # 注意：这里的 physical_order 实际上应该是指 "Decoder Order"
            # 你的 config 里 physical_order: ['seqD', 'seqF', 'seqE']
            # 对应 Rounds: [1..5], [6..10], [11]
            # 只要这个顺序和 io.py 加载图像的顺序一致，就是对的。
            
            for seg_id in topo.physical_order:
                seg_def = segment_defs[seg_id]
                
                # A. Slicing (Config is 1-based Inclusive -> Python 0-based Exclusive)
                # Config: [1, 6] -> Python: [0 : 6] (Length 6)
                # Config: [7, 12] -> Python: [6 : 12] (Length 6)
                start_1b, end_1b = seg_def.csv_slice
                py_start = max(0, start_1b - 1) # 防止用户输入 0 导致负索引
                py_end = end_1b
                
                # 防御性截取
                if py_end > len(seq):
                    # 如果配置切片超出了序列长度，那是 Config 写错了或者 CSV 脏了
                    return "ERROR_LEN"
                
                sub_seq = seq[py_start : py_end]
                
                # B. Encoding
                encoder = encoders[seg_def.encoding_table]
                encoded_chunk = encoder(sub_seq)
                
                # C. Check Expectations
                # 编码后的长度应该等于该段对应的物理轮次数量
                expected_rounds = len(seg_def.rounds)
                if len(encoded_chunk) != expected_rounds:
                    # 这通常发生在 N 碱基或者逻辑错误
                    # 比如 seq="GC", rounds=1. encoder("GC")->"1". OK.
                    # 比如 seq="GNNNNA", rounds=5. encoder-> ".....". OK.
                    pass # 只要逻辑自洽就行，暂不报错
                    
                full_barcode += encoded_chunk
            
            return full_barcode

        # Apply Assembly
        df_genes['barcode'] = df_genes['processed_seq'].apply(assemble_barcode)
        
        # 6. Checks & Output
        # 过滤掉生成失败的
        valid_df = df_genes[df_genes['barcode'] != "ERROR_LEN"].copy()
        if len(valid_df) < len(df_genes):
            print(f" [Warning] {len(df_genes) - len(valid_df)} genes failed barcode generation (Check sequence lengths).")

        # 生成查找表
        gene_map = dict(zip(valid_df['barcode'], valid_df['gene']))
        
        # Save Debug CSV (这是给你检查切片对不对的关键文件)
        # 我们把切分后的每一段也保存下来方便肉眼Debug，这需要稍微改一下上面的逻辑，但作为Debug
        # 我们可以直接保存最终结果
        base_dir = Path(self.cfg.pipeline.output.directory)
        debug_path = base_dir / "compiled_codebook_debug.csv"
        valid_df.to_csv(debug_path, index=False)
        print(f"   -> Compiled {len(valid_df)} barcodes. Debug info saved to {debug_path.name}")
        
        return gene_map, valid_df

    def _create_encoder(self, mapping: Dict[str, int], base_idx: int) -> Callable[[str], str]:
        """
        工厂函数：生成一个这就编码字符串的函数。
        支持 Sliding Window (Window=2) 和 Direct Map (Window=Length)。
        """
        # 探测 Window Size
        keys = list(mapping.keys())
        if not keys:
            raise ValueError("Empty encoding table")
        window_size = len(keys[0]) # e.g. 2 for "AT"
        
        # 将 Config 里的 1,2,3 转换为 Python 的 0,1,2 (如果 base_index=1)
        # 这样生成的 barcode 字符串由 '0', '1', '2' 组成，对应从图像 argmax 出来的 0,1,2
        normalized_map = {k: str(v - base_idx) for k, v in mapping.items()}

        def encode(seq: str) -> str:
            res = []
            N = len(seq)
            
            # 情况 1: 序列长度正好等于窗口大小 (例如 Omics "GC" -> 1)
            # 这种情况下直接查表，不滑动
            if N == window_size:
                val = normalized_map.get(seq, ".")
                return val
            
            # 情况 2: 滑动窗口 (Standard STARmap/RIBOmap)
            # Seq: A G T C (Len 4)
            # Win=2
            # 0: AG
            # 1: GT
            # 2: TC
            # Output Len = 4 - 2 + 1 = 3 colors.
            # 这个逻辑是标准的。
            
            # 计算输出长度
            if N < window_size:
                return "." * (N) # 序列不够长，这就尴尬了，补坏点

            steps = N - window_size + 1
            for i in range(steps):
                chunk = seq[i : i + window_size]
                val = normalized_map.get(chunk, ".")
                res.append(val)
                
            return "".join(res)
            
        return encode
    
    def _build_reverse_lookups(self) -> Dict[str, Dict[Tuple[str, int], str]]:
        """
        构建所有encoding tables的反向查找表
        
        用于end bases验证：给定(前一个碱基, 颜色) -> 推断出后一个碱基
        
        例如：
        "AT": 4 -> reverse_lookup[('A', 3)] = 'T'  (假设base_idx=1)
        "TG": 3 -> reverse_lookup[('T', 2)] = 'G'
        
        Returns:
        --------
        Dict[table_name, Dict[(prev_base, color), next_base]]
        """
        reverse_lookups = {}
        
        for table_name, mapping in self.cfg.codebook.encoding_tables.items():
            reverse_lookups[table_name] = self._build_single_reverse_lookup(
                mapping, 
                self.cfg.codebook.channel_base_index
            )
        
        return reverse_lookups
    
    def _build_single_reverse_lookup(
        self, 
        encoding_table: Dict[str, int], 
        base_idx: int
    ) -> Dict[Tuple[str, int], str]:
        """
        构建单个encoding table的反向查找表
        
        Parameters:
        -----------
        encoding_table : Dict[str, int]
            碱基对 -> 颜色的映射，如 {"AT": 4, "CA": 2, ...}
        base_idx : int
            Config中的base index（0或1），用于归一化颜色值
            
        Returns:
        --------
        reverse : Dict[(prev_base, color), next_base]
            (前碱基, 归一化颜色) -> 后碱基
        """
        reverse = {}
        
        for base_pair, color in encoding_table.items():
            if len(base_pair) != 2:
                # 跳过非两碱基编码（如果有的话）
                continue
            
            prev_base = base_pair[0]
            next_base = base_pair[1]
            normalized_color = color - base_idx  # 转换为0-based
            
            key = (prev_base, normalized_color)
            
            # 检测编码冲突
            if key in reverse and reverse[key] != next_base:
                raise ValueError(
                    f"Ambiguous encoding in table: "
                    f"{key} maps to both '{reverse[key]}' and '{next_base}'"
                )
            
            reverse[key] = next_base
        
        return reverse
    
    def _decode_color_sequence(
        self, 
        color_seq: str, 
        start_base: str, 
        reverse_lookup: Dict[Tuple[str, int], str]
    ) -> str:
        """
        从颜色序列解码出碱基序列
        
        这是two-base encoding的反向过程：
        1. 知道起始碱基
        2. 根据每个颜色推断出下一个碱基
        
        Parameters:
        -----------
        color_seq : str
            颜色序列，如 "0123"
        start_base : str
            起始碱基（anchor），如 "C"
        reverse_lookup : Dict
            反向查找表 {(prev_base, color): next_base}
        
        Returns:
        --------
        base_seq : str
            解码后的碱基序列，如 "CAAAC"
            如果解码失败（无法查找或遇到坏点），返回空字符串
        
        Example:
        --------
        color_seq = "0123"
        start_base = "C"
        reverse_lookup = {('C', 0): 'A', ('A', 1): 'A', ...}
        
        过程：
        - 起始: base_seq = "C"
        - Color 0 + prev='C' -> 'A', base_seq = "CA"
        - Color 1 + prev='A' -> 'A', base_seq = "CAA"
        - Color 2 + prev='A' -> 'A', base_seq = "CAAA"
        - Color 3 + prev='A' -> 'C', base_seq = "CAAAC"
        """
        if not color_seq or not start_base:
            return ""
        
        base_seq = start_base
        prev_base = start_base
        
        for color_char in color_seq:
            # 处理坏点标记
            if color_char == '.':
                return ""
            
            # 转换为整数
            try:
                color_int = int(color_char)
            except ValueError:
                # 非法字符
                return ""
            
            # 查找下一个碱基
            key = (prev_base, color_int)
            if key not in reverse_lookup:
                # 无法解码（可能是编码表不完整或数据错误）
                return ""
            
            next_base = reverse_lookup[key]
            base_seq += next_base
            prev_base = next_base
        
        return base_seq
    
    def _validate_end_bases(self, barcode: str) -> bool:
        """
        验证单个barcode是否符合end bases规则
        
        1. 按照topology将barcode切分成segments
        2. 对每个segment：
           a. 用anchor_base[0]作为起始碱基解码颜色序列
           b. 检查解码后的序列最后一个碱基是否等于anchor_base[1]
        3. 所有segments都通过验证才返回True
        
        Parameters:
        -----------
        barcode : str
            颜色barcode，如 "01230123"
            
        Returns:
        --------
        is_valid : bool
            True表示通过验证，False表示不符合pattern
        """
        topo = self.cfg.codebook.topology
        segment_defs = {seg.id: seg for seg in topo.structure}
        
        barcode_idx = 0
        
        # 遍历每个segment（按照physical_order）
        for seg_id in topo.physical_order:
            seg_def = segment_defs[seg_id]
            
            # 提取这个segment对应的颜色序列
            seg_length = len(seg_def.rounds)
            color_seq = barcode[barcode_idx : barcode_idx + seg_length]
            barcode_idx += seg_length
            
            # 检查是否定义了anchor_base
            if seg_def.anchor_base is None:
                continue  # 如果没定义，跳过验证
            
            if len(seg_def.anchor_base) != 2:
                continue  # 配置错误，跳过
            
            start_base, end_base = seg_def.anchor_base
            
            # 获取反向查找表
            reverse_lookup = self.reverse_lookups[seg_def.encoding_table]
            
            # 解码颜色序列
            decoded_seq = self._decode_color_sequence(
                color_seq, 
                start_base, 
                reverse_lookup
            )
            
            # 验证解码是否成功
            if not decoded_seq:
                return False  # 解码失败
            
            # 验证结尾碱基
            if decoded_seq[-1] != end_base:
                return False  # 结尾不匹配
        
        return True
    
    def _calculate_box_volume(self) -> int:
        """从 Config 计算积分盒子的像素数"""
        box = self.cfg.pipeline.extraction.integration_box # [z, y, x]
        return box[0] * box[1] * box[2]

    def decode_fov(self, fov_id: int):
        print(f"[{'='*20} Decoding FOV {fov_id} {'='*20}]")
        
        base_dir = Path(self.cfg.pipeline.output.directory)
        paths = get_fov_output_structure(base_dir, fov_id)
        
        # 1. 加载数据
        raw_path = paths["extraction"] / f"intensity_matrix_fov_{fov_id}.npy"
        spots_path = paths["spots"] / f"spots_fov_{fov_id}.csv"
        
        if not raw_path.exists():
            raise FileNotFoundError(f"Intensity matrix missing: {raw_path}")
            
        # Shape: (N_spots, N_rounds, N_channels)
        raw_matrix = np.load(raw_path)
        spots_df = pd.read_csv(spots_path)
        n_spots = len(spots_df)
        
        if len(raw_matrix) != len(spots_df):
            raise ValueError("Matrix and Spots count mismatch! Pipeline broken.")
        
        # 因为 miner 已经过滤过了，raw_matrix 现在全是 seq channel
        # 我们不需要再切片，或者简单检查一下维度匹配
        raw_seq = raw_matrix
        print(f" -> Loaded matrices with shape {raw_seq.shape}. Assuming Seq channels only.")
        
        # 2. 归一化 (Normalization)
        # 我们需要在 Channel 维度做 L2 Norm，消除亮度差异，只留颜色向量
        # 加上 epsilon 防止除零
        print(" -> Normalizing intensities...")
        norms = np.linalg.norm(raw_matrix, axis=2, keepdims=True) + 1e-6
        norm_matrix = raw_matrix / norms
        
        #print(" -> Applying Normalization (Z-Score)...")
        # 形状: (1, 1, C)
        #channel_means = np.mean(raw_matrix, axis=(0, 1), keepdims=True)
        #channel_stds = np.std(raw_matrix, axis=(0, 1), keepdims=True)
        
        #print(f"    Channel Stds: {channel_stds.flatten()}")
        
        # Z-Score:让所有通道的分布都在同一个尺度上 (Mean~0, Std~1)
        #z_score_matrix = (raw_matrix - channel_means) / (channel_stds + 1e-9)
        #norm_matrix = softmax(z_score_matrix, axis=2, temperature=0.2)
        
        # 3. Base Calling (Color Calling)
        # 哪个通道最亮，就是哪个颜色
        # Shape: (N_spots, N_rounds)
        print(" -> Calling colors...")
        read_indices, base_scores, is_valid = compatible_base_calling(norm_matrix)
        
        # 统计平局和无效点
        n_ties = np.sum(~is_valid)
        print(f"   Tie/Invalid detection: {n_ties} spots flagged ({n_ties/n_spots:.2%})")
        
        print(" -> Filtering by quality score...")
        
        # Matlab的阈值是0.5，只对有效的spot计算
        # 注意：有Inf的spot已经在is_valid中被标记为False了
        quality_pass = compatible_quality_filter(
            base_scores[is_valid], 
            threshold=0.5
        )
        
        # 创建一个全局的质量过滤mask
        quality_pass_global = np.zeros(n_spots, dtype=bool)
        quality_pass_global[is_valid] = quality_pass

        final_pass = is_valid & quality_pass_global
        
        print(f"\n [Filtration Statistics]")
        print(f"   Total spots:        {n_spots}")
        print(f"   Valid (no ties):    {is_valid.sum()} ({is_valid.sum()/n_spots:.2%})")
        print(f"   Quality pass:       {quality_pass_global.sum()} ({quality_pass_global.sum()/n_spots:.2%})")
        print(f"   Final kept:         {final_pass.sum()} ({final_pass.sum()/n_spots:.2%})")
        print(f"   Removed by quality filter:  {n_spots - final_pass.sum()}")

        # 5. Fast String Construction (Vectorized)
        print(" -> Constructing barcodes...")
        
        # 只对通过过滤的spots构建barcode
        valid_indices = np.where(final_pass)[0]
        valid_read_indices = read_indices[valid_indices]
        
        # Fast vectorized string construction
        df_reads = pd.DataFrame(valid_read_indices)
        raw_barcodes = df_reads.astype(str).agg(''.join, axis=1)
        
        # 5. 序列化 (Vectorized String Conversion)
        # 这是一个 Numpy 到 Pandas 的技巧
        print(" -> Matching codebook...")
        
        sample_code = next(iter(self.gene_map.keys()))
        if raw_matrix.shape[1] != len(sample_code):
            print(f" [Warning] Imaging Rounds ({raw_matrix.shape[1]}) != Codebook Length ({len(sample_code)})")
            
        # 创建结果DataFrame（只包含通过过滤的spots）
        df_res = spots_df.iloc[valid_indices].copy()
        df_res['barcode'] = raw_barcodes.values
        
        # 计算平均质量分数（只对有限值）
        valid_base_scores = base_scores[valid_indices]
        df_res['quality'] = np.mean(valid_base_scores, axis=1)
        
        # 计算总强度（使用原始矩阵）
        valid_raw_matrix = raw_matrix[valid_indices]
        df_res['intensity'] = np.max(np.max(valid_raw_matrix, axis=2), axis=1)
        
        print(" -> Validating end bases pattern...")
        
        # 应用验证函数到每个barcode
        pattern_valid = df_res['barcode'].apply(self._validate_end_bases)
        
        n_pattern_fail = (~pattern_valid).sum()
        pattern_fail_rate = n_pattern_fail / len(df_res) if len(df_res) > 0 else 0
        
        print(f"   Pattern validation removed: {n_pattern_fail} spots ({pattern_fail_rate:.2%})")
        
        # Gene mapping
        df_res['gene'] = df_res['barcode'].map(self.gene_map).fillna('background')
        
        # 过滤掉不符合pattern的spots
        df_res_true = df_res[pattern_valid].copy()
        
        if len(df_res_true) == 0:
            print(" [ERROR] No spots left after pattern validation!")
            print(" [HINT] Check your anchor_base configuration in experiment_config.yaml")
            return pd.DataFrame()
        
        # 计算每轮的平均质量分数（用于诊断）
        valid_finite_scores = valid_base_scores.copy()
        valid_finite_scores[~np.isfinite(valid_finite_scores)] = np.nan
        
        with np.errstate(invalid='ignore'):
            avg_quality_per_round = np.nanmean(valid_finite_scores, axis=0)
        
        print("\n [Quality Diagnostics] Average -log(max) per Round:")
        for r_idx, q in enumerate(avg_quality_per_round):
            status = "✓" if q < 0.5 else "✗"
            print(f"   Round {r_idx+1}: {q:.4f} {status}")
        
        if np.nanmin(avg_quality_per_round) > 0.7:
            weakest_link = np.nanargmin(avg_quality_per_round) + 1
            print(f"   !!! WARNING: Round {weakest_link} has poor quality. Check registration!")
        
        
        n_mapped = (df_res_true['gene'] != 'background').sum()
        mapping_rate_quality = n_mapped / len(df_res) if len(df_res) > 0 else 0
        mapping_rate_pattern = n_mapped / len(df_res_true) if len(df_res_true) > 0 else 0
        
        print(f"\n [Mapping Results]")
        print(f"   Spots after quality filter: {len(df_res)}")
        print(f"   Spots after pattern check:  {len(df_res_true)}")
        print(f"   Spots after quality filter matched to genes:   {n_mapped} ({mapping_rate_quality:.2%})")
        print(f"   Spots after pattern check matched to genes:    {n_mapped} ({mapping_rate_pattern:.2%})")
        print(f"   Background/Unknown: {len(df_res) - n_mapped}")
        
        # Top genes
        if n_mapped > 0:
            top_genes = df_res_true[df_res_true['gene'] != 'background']['gene'].value_counts().head(10)
            print(f"\n [Top 10 Detected Genes]")
            for gene, count in top_genes.items():
                print(f"   {gene}: {count}")
        # 8. 保存 - 分离 decoded 和 background
        # decoded_fov_{fov_id}.csv: 只包含成功解码的 spots (非 background)
        decoded_path = paths["decoded"] / f"decoded_fov_{fov_id}.csv"
        df_decoded = df_res_true[df_res_true['gene'] != 'background'].copy()
        df_decoded.to_csv(decoded_path, index=False)
        n_decoded = len(df_decoded)
        n_background = len(df_res_true) - n_decoded
        print(f" [Decoder] Saved {n_decoded} decoded spots to {decoded_path.name}")
        
        # all_reads_fov_{fov_id}.csv: 包含所有 spots (包括 background)
        all_reads_path = paths["decoded"] / f"all_reads_fov_{fov_id}.csv"
        df_res_true.to_csv(all_reads_path, index=False)
        print(f" [Decoder] Saved {len(df_res_true)} total spots ({n_background} background) to {all_reads_path.name}")
        
        # 保留 pre_pattern_check 用于调试
        df_res.to_csv(
            paths["decoded"] / f"decoded_fov_{fov_id}_pre_pattern_check.csv", 
            index=False
        )
        
        return df_res_true

if __name__ == "__main__":
    from pystar.infrastructure import load_config
    cfg = load_config("experiment_config.yaml")
    decoder = Decoder(cfg)
    try:
        decoder.decode_fov(1)
    except Exception as e:
        print(e)