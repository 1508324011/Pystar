#!/bin/bash

# =================================================================
# 自动读取 Config，计算 Array 长度，提交任务
# 用法: bash run_pystar.sh [config_path]
# =================================================================

# 1. 确定 Config 路径
CONFIG_FILE="${1:-config/experiment_config.yaml}"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found at $CONFIG_FILE"
    exit 1
fi

echo "--- PyStar Launcher ---"
echo "Reading config: $CONFIG_FILE"

# 用 Python 单行脚本读取 YAML 里的 FOV 数量
# 这样我们永远不需要手动修改 --array
NUM_JOBS=$(pixi run -e pystar python -c "
import sys, yaml
try:
    with open('$CONFIG_FILE') as f:
        data = yaml.safe_load(f)
    fovs = data['dataset']['fov_list']
    # 处理 '1-56' 这种字符串
    if isinstance(fovs, str) and '-' in fovs:
        start, end = map(int, fovs.split('-'))
        print(end - start + 1)
    elif isinstance(fovs, str) and ',' in fovs:
        print(len(fovs.split(',')))
    elif isinstance(fovs, list):
        print(len(fovs))
    else:
        print(1) # Fallback
except Exception as e:
    print(0)
")

if [ "$NUM_JOBS" -eq "0" ]; then
    echo "Error: Failed to parse fov_list from yaml."
    exit 1
fi

echo "Detected $NUM_JOBS FOVs to process."

# 3. 准备 log 目录 (Slurm 的 log，不是 PyStar 内部 log)
mkdir -p logs/pystar

# 4. 生成并提交任务
# 使用 Here-Doc (EOF) 动态生成 sbatch 脚本
# 将 python 计算出的 NUM_JOBS 注入到 --array 参数中

CPUS_PER_FOV=4  # 每个 FOV 分配的 CPU 数量
Batch_FOV=128 #并行FOV数

JOB_ID=$(sbatch << EOF | awk '{print $4}'
#!/bin/bash
#SBATCH -J pystar_batch
#SBATCH -o logs/pystar/%x.%A_%a.out
#SBATCH -e logs/pystar/%x.%A_%a.err
#SBATCH -p C64M512G
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS_PER_FOV}
#SBATCH --time=24:00:00
#SBATCH --array=1-${NUM_JOBS}%${Batch_FOV}
#SBATCH --no-requeue
#SBATCH --export=ALL

echo "Running on node: \$(hostname)"
echo "Slurm Task ID: \$SLURM_ARRAY_TASK_ID"

# 在 Python 内部限制线程数，防止 OpenBLAS/MKL 乱抢资源
export OMP_NUM_THREADS=${CPUS_PER_FOV}
export MKL_NUM_THREADS=${CPUS_PER_FOV}
export OPENBLAS_NUM_THREADS=${CPUS_PER_FOV}

# 运行 Worker
# 传 Config 路径 和 Task ID 给 Python
pixi run runpystar --config "$CONFIG_FILE" --task_id "\$SLURM_ARRAY_TASK_ID"

EOF
)

echo "Job submitted! ID: $JOB_ID"
echo "Monitor with: squeue -j $JOB_ID"