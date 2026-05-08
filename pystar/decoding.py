# pystar/decoding.py
from json import loads
from pathlib import Path
from typing import Any, Callable, Dict, Tuple, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from tqdm import tqdm

from ._artifact_schemas import (
    SpotTableSchema,
    build_intensity_matrix_spec,
    empty_decoded_table,
    intensity_matrix_metadata_expected_description,
    intensity_matrix_metadata_path,
    validate_decoded_table,
    validate_intensity_matrix,
    validate_intensity_matrix_consumer_contract,
    validate_intensity_matrix_metadata_payload,
    validate_spot_table,
    wrap_array_read_error,
    wrap_payload_read_error,
    wrap_table_read_error,
)
from .io import get_fov_output_structure
from .infrastructure import ExperimentConfig


NDArrayAny = npt.NDArray[Any]
BoolArray = npt.NDArray[np.bool_]

def softmax(x, axis=2, temperature=1.0):
    """
    计算 Softmax，带温度参数。
    Temperature 越小，分布越尖锐（Highlights winner）。
    Temperature 越大，分布越平坦。
    通常 T=1.0 即可，如果是 Z-score 输入，分布已经很标准了。
    """
    # 减去最大值防止 exp 溢出 (Numerical Stability)
    e_x = np.exp((x - np.max(x, axis=axis, keepdims=True)) / temperature)
    return e_x / e_x.sum(axis=axis, keepdims=True)

def compatible_base_calling(norm_matrix: NDArrayAny) -> tuple[NDArrayAny, NDArrayAny, BoolArray]:
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
    read_indices = np.asarray(np.argmax(norm_matrix, axis=2), dtype=np.int32)  # (N, R)
    read_indices[has_tie] = -1  # 平局标记为-1
    
    # 4. 计算base_scores（负对数）
    #  baseScores(i,j) = -log(currMax);
    with np.errstate(divide='ignore', invalid='ignore'):  
        # 忽略log(0)和log(nan)的警告
        base_scores = np.asarray(-np.log(max_vals), dtype=np.float32)  # (N, R)
    
    # 平局的地方设为Inf
    #  baseScores(i,j) = Inf;
    base_scores[has_tie] = np.inf
    
    # 处理NaN（如果max_val是0或负数）
    base_scores[~np.isfinite(base_scores)] = np.inf
    
    # 5. 全局有效性检查
    #  if ~any(isinf(baseScores(i, :)))
    is_valid = np.asarray(~np.any(np.isinf(base_scores), axis=1), dtype=bool)  # (N,)
    
    return read_indices, base_scores, is_valid


def compatible_quality_filter(
    base_scores: NDArrayAny,
    threshold: float = 0.5
) -> BoolArray:
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
    pass_filter = np.asarray(mean_scores < threshold, dtype=bool)
    
    return pass_filter


class Decoder:
    """Turn extracted per-round intensities into gene calls.

    The decoder consumes two artifacts from earlier stages: the spot table
    (`spots_fov_<id>.csv`) and the intensity tensor
    (`intensity_matrix_fov_<id>.npy`). The tensor shape is
    `(N_spots, N_rounds, N_seq_channels)`, where the spot axis is aligned with
    the rows of the spot table. Coordinates and channel provenance are carried
    through from the spot table unchanged.

    Codebook handling is deliberately forward-simulated: configured gene
    sequences are transformed into expected color barcodes, then observed color
    calls are matched to those barcodes. The final decoded CSV may contain
    `background` rows when the active gate keeps pattern-valid reads that are
    not in the codebook; callers comparing against STATE goodPoints should
    filter those rows when they need gene-only counts.
    """

    def __init__(self, config: ExperimentConfig):
        self.cfg = config
        self.output_dir = Path(self.cfg.pipeline.output.directory)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载并编译码本
        self.gene_map, self.barcode_map = self._compile_codebook()
        self.reverse_lookups = self._build_reverse_lookups()

    def _sequencing_channels(self) -> list[int]:
        roles = self.cfg.dataset.channel_roles
        return sorted([channel_id for channel_id, role in roles.items() if role == 'seq'])

    def _decoded_artifact_extra_columns(self) -> tuple[str, ...]:
        return ("channel", "fov", "algo", "pattern_valid", "in_codebook", "gating_mode")

    def _write_decoded_artifact(
        self,
        df: pd.DataFrame,
        *,
        fov_id: int,
        path: Path,
        context: str,
    ) -> pd.DataFrame:
        validated = validate_decoded_table(
            df,
            fov_id=fov_id,
            path=path,
            context=context,
        )
        validated.to_csv(path, index=False)
        return validated

    def _write_empty_decoded_family(
        self,
        *,
        fov_id: int,
        paths: dict[str, Path],
        pre_pattern_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Write the three decoded CSV artifacts with canonical empty schemas."""

        decoded_empty = empty_decoded_table(extra_columns=self._decoded_artifact_extra_columns())
        written_empty = self._write_decoded_artifact(
            decoded_empty,
            fov_id=fov_id,
            path=paths["decoded"] / f"decoded_fov_{fov_id}.csv",
            context="decoded save",
        )
        self._write_decoded_artifact(
            empty_decoded_table(extra_columns=self._decoded_artifact_extra_columns()),
            fov_id=fov_id,
            path=paths["decoded"] / f"decoded_fov_{fov_id}_goodreads.csv",
            context="decoded goodreads save",
        )
        self._write_decoded_artifact(
            decoded_empty if pre_pattern_df is None else pre_pattern_df,
            fov_id=fov_id,
            path=paths["decoded"] / f"decoded_fov_{fov_id}_pre_pattern_check.csv",
            context="decoded pre-pattern save",
        )
        return written_empty

    def _load_validated_intensity_matrix(
        self,
        *,
        fov_id: int,
        matrix_path: Path,
        matrix_spec: Any,
    ) -> NDArrayAny:
        expected = matrix_spec.expected_description()
        try:
            raw_matrix = np.load(matrix_path, allow_pickle=False)
        except Exception as exc:
            raise wrap_array_read_error(
                exc,
                "intensity matrix",
                fov_id=fov_id,
                path=matrix_path,
                context="decode load",
                expected=expected,
            ) from exc

        metadata_path = intensity_matrix_metadata_path(matrix_path)
        if metadata_path.exists():
            metadata_expected = intensity_matrix_metadata_expected_description()
            try:
                metadata_payload = loads(metadata_path.read_text(encoding="utf-8"))
            except Exception as exc:
                raise wrap_payload_read_error(
                    exc,
                    "intensity matrix metadata sidecar",
                    fov_id=fov_id,
                    path=metadata_path,
                    context="decode metadata load",
                    expected=metadata_expected,
                ) from exc
            persisted_spec = validate_intensity_matrix_metadata_payload(
                metadata_payload,
                fov_id=fov_id,
                path=metadata_path,
                context="decode metadata load",
            )
            validate_intensity_matrix_consumer_contract(
                persisted_spec,
                matrix_spec,
                path=metadata_path,
                context="decode metadata load",
                matrix_path=matrix_path,
            )
            print(
                f" -> Validating intensity matrix {matrix_path.name} against metadata sidecar {metadata_path.name} "
                f"(round_order={list(persisted_spec.rounds)}, channel_order={list(persisted_spec.channels)})"
            )
            return validate_intensity_matrix(
                raw_matrix,
                persisted_spec,
                path=matrix_path,
                context="decode load against metadata sidecar",
            )

        print(
            f" -> Intensity metadata sidecar absent at {metadata_path}; using explicit legacy config-derived "
            f"matrix validation for {matrix_path.name} without reordering, reshaping, or reinterpretation."
        )
        return validate_intensity_matrix(
            raw_matrix,
            matrix_spec,
            path=matrix_path,
            context="decode load (legacy config-derived validation; sidecar absent)",
        )
        
    def _compile_codebook(self) -> tuple[dict[str, str], pd.DataFrame]:
        """
        Compile gene sequences into expected color barcodes.

        This is the critical forward-simulation step. PyStar does not infer the
        codebook by reversing the observed reads; it takes each gene sequence,
        applies the configured topology transform, slices the sequence according
        to `BlueprintSegment.csv_slice`, encodes each segment with its declared
        color table, and concatenates segments in `physical_order`. Config
        slices are 1-based inclusive and become Python's 0-based half-open
        ranges inside this method.
        """
        codebook_cfg = self.cfg.codebook
        gene_list_path = Path(codebook_cfg.gene_list)
        topo = codebook_cfg.topology

        def _empty_gene_list_error() -> ValueError:
            return ValueError(
                f"Codebook gene-map contract error: gene list is empty at {gene_list_path}. "
                "Decoder requires at least one gene/sequence row that can compile to a barcode."
            )

        def _drop_optional_header_row(df: pd.DataFrame) -> pd.DataFrame:
            if len(df) == 0:
                return df
            first_gene = str(df.iloc[0]["gene"]).strip().lower()
            first_seq = str(df.iloc[0]["seq"]).strip().lower()
            gene_header_names = {"gene", "gene_name", "genename", "name"}
            sequence_header_names = {"seq", "sequence"}
            if first_gene in gene_header_names and first_seq in sequence_header_names:
                return cast(pd.DataFrame, df.iloc[1:].reset_index(drop=True))
            return df
        
        if not gene_list_path.exists():
            raise FileNotFoundError(f"Gene list not found: {gene_list_path}")

        # 1. 读取基因表 (假设没有 header，或者根据实际情况修改)
        # 通常 genes.csv 结构是: GeneName, Sequence
        try:
            df_genes = cast(pd.DataFrame, pd.read_csv(gene_list_path, header=None, names=['gene', 'seq']))
        except pd.errors.EmptyDataError as exc:
            raise _empty_gene_list_error() from exc
        except Exception:
            # 兼容带有 header 的情况
            try:
                df_genes = cast(pd.DataFrame, pd.read_csv(gene_list_path))
            except pd.errors.EmptyDataError as exc:
                raise _empty_gene_list_error() from exc
            if 'gene' not in df_genes.columns: # fallback
                df_genes.columns = ['gene', 'seq']

        df_genes = _drop_optional_header_row(df_genes)

        if len(df_genes) == 0:
            raise _empty_gene_list_error()

        missing_codebook_columns = [column for column in ('gene', 'seq') if column not in df_genes.columns]
        if missing_codebook_columns:
            raise ValueError(
                f"Codebook gene-map contract error at {gene_list_path}: missing required columns "
                f"{missing_codebook_columns}. Expected columns ['gene', 'seq'] or a two-column gene list."
            )

        invalid_gene_mask = df_genes['gene'].isna() | (df_genes['gene'].astype(str).str.strip() == "")
        invalid_seq_mask = df_genes['seq'].isna() | (df_genes['seq'].astype(str).str.strip() == "")
        if bool(invalid_gene_mask.any()) or bool(invalid_seq_mask.any()):
            raise ValueError(
                f"Codebook gene-map contract error at {gene_list_path}: gene and seq values must be non-empty "
                "for every row before barcode compilation."
            )

        df_genes['gene'] = df_genes['gene'].astype(str)
        df_genes['seq'] = df_genes['seq'].astype(str)
        
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
        encoding_tables = codebook_cfg.encoding_tables
        if not isinstance(encoding_tables, dict) or not encoding_tables:
            raise ValueError(
                "Codebook gene-map contract error: encoding_tables is empty. "
                "Decoder requires at least one non-empty encoding table before barcode compilation."
            )

        encoders = {}
        for table_name, mapping in encoding_tables.items():
            if not isinstance(mapping, dict) or not mapping:
                raise ValueError(
                    f"Codebook gene-map contract error: encoding table {table_name!r} is empty or malformed. "
                    "Decoder requires non-empty base-to-color mappings before barcode compilation."
                )
            encoders[table_name] = self._create_encoder(mapping, codebook_cfg.channel_base_index)

        # 4. Parse Blueprint Structure
        # 建立一个字典方便按 ID 查找 segment 定义
        segment_defs = {seg.id: seg for seg in topo.structure}
        
        # 验证 physical_order 是否都定义了
        for seg_id in topo.physical_order:
            if seg_id not in segment_defs:
                raise ValueError(f"Topology physical_order references undefined segment ID: {seg_id}")
            encoding_table_name = segment_defs[seg_id].encoding_table
            if encoding_table_name not in encoders:
                raise ValueError(
                    f"Codebook gene-map contract error: topology segment {seg_id!r} references missing "
                    f"encoding table {encoding_table_name!r}. Available encoding tables: {sorted(encoders)}"
                )

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
        valid_df = cast(pd.DataFrame, df_genes[df_genes['barcode'] != "ERROR_LEN"].copy())
        if len(valid_df) < len(df_genes):
            print(f" [Warning] {len(df_genes) - len(valid_df)} genes failed barcode generation (Check sequence lengths).")

        if len(valid_df) == 0:
            raise ValueError(
                f"Codebook gene-map contract error: gene list at {gene_list_path} compiled to zero valid barcodes. "
                "Check topology csv_slice/physical_order and gene sequence lengths before decoding."
            )

        # 生成查找表
        gene_map = dict(zip(valid_df['barcode'], valid_df['gene']))
        if not gene_map:
            raise ValueError(
                f"Codebook gene-map contract error: compiled gene map is empty for {gene_list_path}. "
                "Decoder requires at least one barcode-to-gene mapping."
            )
        
        # Save Debug CSV (这是给你检查切片对不对的关键文件)
        # 我们把切分后的每一段也保存下来方便肉眼Debug，这需要稍微改一下上面的逻辑，但作为Debug
        # 我们可以直接保存最终结果
        debug_path = self.output_dir / "compiled_codebook_debug.csv"
        valid_df.to_csv(debug_path, index=False)
        print(f"   -> Compiled {len(valid_df)} barcodes. Debug info saved to {debug_path.name}")
        
        return gene_map, valid_df

    def _create_encoder(self, mapping: Dict[str, int], base_idx: int) -> Callable[[str], str]:
        """
        Build a sequence-to-color encoder for one configured table.

        Two cases are supported by the same closure: a direct lookup when the
        sequence length equals the table key length, and a sliding-window lookup
        for standard two-base encodings. Color indices are normalized by
        `base_idx` so the emitted barcode characters match Python argmax channel
        indices (`0`, `1`, `2`, ...).
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
        Validate one observed color barcode against configured anchor bases.
        
        The barcode is split by topology segment length in physical round order.
        For each segment that declares `anchor_base`, the first anchor base is
        used as the start base for reverse decoding and the decoded end base is
        compared with the second anchor base. Segments without anchors are
        ignored. Every anchored segment must pass for the read to be pattern
        valid.
        
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
        """Decode one FOV and persist canonical decoded CSV artifacts.

        The method performs channel-wise L2 normalization per round, calls the
        brightest sequencing channel as the color, rejects ties/invalid reads,
        applies the MATLAB-like mean quality threshold, constructs barcodes,
        evaluates topology/end-base validity, maps barcodes to genes, and then
        applies the configured gating mode. Two CSVs are written: the active-gate
        output used downstream and a pre-pattern-check diagnostic table.
        """
        print(f"[{'='*20} Decoding FOV {fov_id} {'='*20}]")
        
        base_dir = Path(self.cfg.pipeline.output.directory)
        paths = get_fov_output_structure(base_dir, fov_id)
        
        # 1. 加载数据
        raw_path = paths["extraction"] / f"intensity_matrix_fov_{fov_id}.npy"
        spots_path = paths["spots"] / f"spots_fov_{fov_id}.csv"
        if hasattr(self, "gene_map"):
            early_gene_map = getattr(self, "gene_map")
            if not isinstance(early_gene_map, dict) or not early_gene_map:
                raise ValueError(
                    f"Codebook gene-map contract error: compiled gene map is empty before decoding FOV {fov_id}. "
                    "Decoder requires at least one barcode-to-gene mapping and will not interpret reads against an empty map."
                )
        
        if not raw_path.exists():
            raise FileNotFoundError(f"Intensity matrix missing: {raw_path}")
            
        # Shape: (N_spots, N_rounds, N_channels)
        spot_expected = SpotTableSchema().expected_description()
        try:
            raw_spots_df = pd.read_csv(spots_path)
        except Exception as exc:
            raise wrap_table_read_error(
                exc,
                "spot table",
                fov_id=fov_id,
                path=spots_path,
                context="decode load",
                expected=spot_expected,
            ) from exc
        spots_df = validate_spot_table(
            raw_spots_df,
            fov_id=fov_id,
            path=spots_path,
            context="decode load",
        )
        n_spots = len(spots_df)
        rounds = sorted(list(self.cfg.dataset.round_structure.keys()))
        channels = self._sequencing_channels()
        matrix_spec = build_intensity_matrix_spec(
            fov_id=fov_id,
            n_spots=n_spots,
            rounds=rounds,
            channels=channels,
        )
        raw_matrix = self._load_validated_intensity_matrix(
            fov_id=fov_id,
            matrix_path=raw_path,
            matrix_spec=matrix_spec,
        )

        if n_spots == 0:
            decoded_empty = self._write_empty_decoded_family(fov_id=fov_id, paths=paths)
            print(" [Decoder] No spots available after artifact load; wrote canonical empty decoded artifacts")
            return decoded_empty

        gene_map_raw = getattr(self, "gene_map", None)
        gene_map: dict[str, str] | None = gene_map_raw if isinstance(gene_map_raw, dict) else None
        if not gene_map:
            raise ValueError(
                f"Codebook gene-map contract error: compiled gene map is empty before decoding FOV {fov_id}. "
                "Decoder requires at least one barcode-to-gene mapping and will not interpret reads against an empty map."
            )
        
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

        if not bool(final_pass.any()):
            decoded_empty = self._write_empty_decoded_family(fov_id=fov_id, paths=paths)
            print(" [Decoder] No reads passed base-calling and quality filtering; wrote canonical empty decoded artifacts")
            return decoded_empty

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
        
        sample_code = next(iter(gene_map.keys()))
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

        gating_mode = self.cfg.pipeline.decoding.gating_mode
        print(f" -> Applying gating mode: {gating_mode}")

        # 应用验证函数到每个barcode
        pattern_valid = df_res['barcode'].apply(self._validate_end_bases)

        n_pattern_fail = (~pattern_valid).sum()
        pattern_fail_rate = n_pattern_fail / len(df_res) if len(df_res) > 0 else 0

        in_codebook = df_res['barcode'].isin(gene_map)
        n_codebook = int(in_codebook.sum())
        codebook_rate = n_codebook / len(df_res) if len(df_res) > 0 else 0

        print(f"   Pattern validation removed: {n_pattern_fail} spots ({pattern_fail_rate:.2%})")
        print(f"   In-codebook after quality filter: {n_codebook} spots ({codebook_rate:.2%})")

        # Gene mapping
        df_res['gene'] = df_res['barcode'].map(gene_map).fillna('background')

        df_res['pattern_valid'] = pattern_valid.values
        df_res['in_codebook'] = in_codebook.values
        df_res['gating_mode'] = gating_mode

        if gating_mode == 'legacy_membership_first':
            final_keep_mask = in_codebook
            print("   Using legacy membership-first gate: keeping all in-codebook reads after quality filter")
        else:
            final_keep_mask = pattern_valid
            print("   Using pattern-first gate: keeping only pattern-valid reads")

        # 过滤掉未保留的spots
        df_res_true = cast(pd.DataFrame, df_res[final_keep_mask].copy())

        if len(df_res_true) == 0:
            print(f" [ERROR] No spots left after gating mode '{gating_mode}'!")
            print(" [HINT] Check your anchor_base configuration and codebook compatibility in experiment_config.yaml")
            decoded_empty = self._write_empty_decoded_family(
                fov_id=fov_id,
                paths=paths,
                pre_pattern_df=df_res,
            )
            return decoded_empty
        
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
        print(f"   Spots after active gate:    {len(df_res_true)}")
        print(f"   Spots after pattern check:  {int(pattern_valid.sum())}")
        print(f"   Spots in codebook:          {n_codebook}")
        print(f"   Spots after quality filter matched to genes:   {n_mapped} ({mapping_rate_quality:.2%})")
        print(f"   Spots after active gate matched to genes:      {n_mapped} ({mapping_rate_pattern:.2%})")
        print(f"   Background/Unknown after active gate: {len(df_res_true) - n_mapped}")
        
        # Top genes
        if n_mapped > 0:
            gene_only_df = cast(pd.DataFrame, df_res_true[df_res_true['gene'] != 'background'].copy())
            top_genes = gene_only_df['gene'].value_counts().head(10)
            print(f"\n [Top 10 Detected Genes]")
            for gene, count in top_genes.items():
                print(f"   {gene}: {count}")
        # 8. 保存
        out_path = paths["decoded"] / f"decoded_fov_{fov_id}.csv"
        df_res_true = self._write_decoded_artifact(
            df_res_true,
            fov_id=fov_id,
            path=out_path,
            context="decoded save",
        )
        print(f" [Decoder] Saved decoded list to {out_path.name}")

        goodreads_path = paths["decoded"] / f"decoded_fov_{fov_id}_goodreads.csv"
        df_goodreads = cast(pd.DataFrame, df_res_true[df_res_true['gene'] != 'background'].copy())
        df_goodreads = self._write_decoded_artifact(
            df_goodreads,
            fov_id=fov_id,
            path=goodreads_path,
            context="decoded goodreads save",
        )
        print(f" [Decoder] Saved decoded good reads to {goodreads_path.name} ({len(df_goodreads)} rows)")
        
        self._write_decoded_artifact(
            df_res,
            fov_id=fov_id,
            path=paths["decoded"] / f"decoded_fov_{fov_id}_pre_pattern_check.csv",
            context="decoded pre-pattern save",
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
