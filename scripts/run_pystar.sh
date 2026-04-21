#!/bin/bash

# =================================================================
# 自动读取 Config，计算 Array 长度，提交任务
# 用法: bash run_pystar.sh [config_path]
# =================================================================

set -euo pipefail

CONFIG_FILE="${1:-config/experiment_config.yaml}"
PIXIRUN=(pixi run --manifest-path env/pixi.toml -e pystar)

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found at $CONFIG_FILE"
    exit 1
fi

echo "--- PyStar Launcher ---"
echo "Reading config: $CONFIG_FILE"

NUM_JOBS=$("${PIXIRUN[@]}" python -c "
import yaml
try:
    with open('$CONFIG_FILE', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    fovs = data['dataset']['fov_list']
    if isinstance(fovs, str) and '-' in fovs:
        start, end = map(int, fovs.split('-'))
        print(end - start + 1)
    elif isinstance(fovs, str) and ',' in fovs:
        print(len([x for x in fovs.split(',') if x.strip()]))
    elif isinstance(fovs, list):
        print(len(fovs))
    else:
        print(1)
except Exception:
    print(0)
")

if [ "$NUM_JOBS" -eq "0" ]; then
    echo "Error: Failed to parse fov_list from yaml."
    exit 1
fi

echo "Detected $NUM_JOBS FOVs to process."
mkdir -p logs/pystar

CPUS_PER_FOV=4
Batch_FOV=128

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

export OMP_NUM_THREADS=${CPUS_PER_FOV}
export MKL_NUM_THREADS=${CPUS_PER_FOV}
export OPENBLAS_NUM_THREADS=${CPUS_PER_FOV}

pixi run --manifest-path env/pixi.toml -e pystar   python scripts/batch_pystar.py   --config "$CONFIG_FILE"   --task_id "\$SLURM_ARRAY_TASK_ID"

EOF
)

echo "Job submitted! ID: $JOB_ID"
echo "Monitor with: squeue -j $JOB_ID"
