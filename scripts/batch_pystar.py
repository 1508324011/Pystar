import sys
import time
import argparse
import logging
import traceback
from pathlib import Path

# 确保能找到 pystar 包
sys.path.append(str(Path(__file__).resolve().parents[1]))

from pystar.infrastructure import load_config
from pystar.io import get_fov_output_structure
from pystar.preprocessing import DataSanitizer
from pystar.registration import RegistrationEngine
from pystar.spot_finding import SpotFinder
from pystar.mining import SignalMiner
from pystar.decoding import Decoder

def setup_logger(fov_id, output_dir_base):
    """
    让每个 Job 自己管理自己的日志文件，而不是全部堆在 Slurm 的 error log 里。
    """
    # 复用 io.py 里的逻辑获取标准路径
    dirs = get_fov_output_structure(Path(output_dir_base), fov_id)
    log_dir = dirs['root'].parent / "log" # 根据你的目录结构调整: PositionX/log
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"pystar_fov_{fov_id}.log"
    
    # 配置 Logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'), # 每次覆盖，保持干净
            logging.StreamHandler(sys.stdout)        # 同时输出到 Slurm out
        ]
    )
    return logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="PyStar Worker Node")
    parser.add_argument("--config", required=True, help="Path to experiment_config.yaml")
    # Slurm Array ID 是从 1 开始的，我们在内部把它转为 List Index
    parser.add_argument("--task_id", required=True, type=int, help="Slurm Array Task ID (1-based)")
    args = parser.parse_args()

    # 1. 加载配置
    try:
        cfg = load_config(args.config)
    except Exception as e:
        print(f"FATAL: Config load failed: {e}")
        sys.exit(1)

    # 2. 映射 Task ID -> FOV ID
    # 你的 fov_list 可能是 "1-56" 也可能是 [1, 5, 9]
    # cfg.dataset.parsed_fovs 是最权威的列表
    fov_list = cfg.dataset.parsed_fovs
    total_jobs = len(fov_list)
    
    # 边界检查 (Never trust user input)
    if args.task_id < 1 or args.task_id > total_jobs:
        print(f"FATAL: Task ID {args.task_id} is out of range [1, {total_jobs}]")
        sys.exit(1)

    # 获取当前任务要处理的真实 FOV ID
    current_fov = fov_list[args.task_id - 1] # 0-based index

    # 3. 初始化日志
    logger = setup_logger(current_fov, cfg.pipeline.output.directory)
    logger.info(f"{'='*40}")
    logger.info(f" PyStar Worker Started")
    logger.info(f" Task ID: {args.task_id} / {total_jobs}")
    logger.info(f" Target FOV: {current_fov}")
    logger.info(f"{'='*40}")

    start_time_global = time.time()

    try:
        # === Stage 1: Preprocessing ===
        t0 = time.time()
        logger.info(">>> Stage 1: Preprocessing (Sanitization)...")
        sanitizer = DataSanitizer(cfg)
        sanitizer.sanitize_fov(current_fov)
        logger.info(f"    Done in {time.time() - t0:.2f}s")

        # === Stage 2: Registration ===
        t0 = time.time()
        logger.info(">>> Stage 2: Registration...")
        # 实例化 IO 加载数据，利用 xarray 的惰性加载机制
        # 注意：这里可能需要稍微调整 registration API 以接受 fov_id 而不是 data array
        # 为了保持你现有逻辑，我们这里做个适配
        from pystar.io import ImageLoader
        loader = ImageLoader(cfg)
        # 只要建立了索引，load_fov 不会读入内存
        data_xr = loader.load_fov(current_fov) 
        
        reg_engine = RegistrationEngine(cfg)
        # 跑配准
        reg_engine.register_fov(data_xr, current_fov)
        logger.info(f"    Done in {time.time() - t0:.2f}s")
        
        # 显式清理，防止 Dask 图过大
        del data_xr
        del loader

        # === Stage 3: Spot Finding ===
        t0 = time.time()
        logger.info(">>> Stage 3: Spot Finding...")
        finder = SpotFinder(cfg)
        finder.find_spots_in_fov(current_fov)
        logger.info(f"    Done in {time.time() - t0:.2f}s")

        # === Stage 4: Mining ===
        t0 = time.time()
        logger.info(">>> Stage 4: Signal Extraction...")
        miner = SignalMiner(cfg)
        miner.mine_fov(current_fov)
        logger.info(f"    Done in {time.time() - t0:.2f}s")

        # === Stage 5: Decoding ===
        t0 = time.time()
        logger.info(">>> Stage 5: Decoding...")
        decoder = Decoder(cfg)
        decoder.decode_fov(current_fov)
        logger.info(f"    Done in {time.time() - t0:.2f}s")

        total_time = time.time() - start_time_global
        logger.info(f"{'='*40}")
        logger.info(f" SUCCESS: FOV {current_fov} processing complete.")
        logger.info(f" Total Time: {total_time/60:.2f} minutes")
        logger.info(f"{'='*40}")

    except Exception as e:
        logger.error(f"CRITICAL FAILURE on FOV {current_fov}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()