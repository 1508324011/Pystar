#!/bin/bash

# =================================================================
# 自动读取 Config，计算 Array 长度，提交任务
# 用法: bash run_pystar.sh [config_path]
#
# 注意：这是 SLURM 集群提交模板，不是通用本地运行脚本。
# 使用前请按你的集群修改 partition、qos、time、Batch_FOV、
# CPUS_PER_FOV、MATLAB_LOCAL_WORKERS 等资源设置。
# 本地单 FOV smoke test 请直接调用：
#   pixi run --manifest-path env/pixi.toml -e pystar \
#     python scripts/batch_pystar.py --config my_config.yaml --task_id 1
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

# -----------------------------------------------------------------
# Resource model A: one MATLAB local tile worker consumes one CPU.
#
# These numbers must be treated as one resource contract:
#   CPUS_PER_FOV
#       Slurm CPUs allocated to one Position/FOV array task.
#   MATLAB_LOCAL_WORKERS
#       The intended value for YAML:
#         pipeline.registration.matlab_local_parallel.workers
#       when MATLAB local process_parallel is enabled.
#   Batch_FOV
#       Maximum number of FOV array tasks allowed to run concurrently.
#
# Model A invariant:
#   MATLAB_LOCAL_WORKERS <= CPUS_PER_FOV
#   Normally use MATLAB_LOCAL_WORKERS == CPUS_PER_FOV.
#
# Thread invariant:
#   OMP_NUM_THREADS = MKL_NUM_THREADS = OPENBLAS_NUM_THREADS = 1
#   because each MATLAB local worker is already a process-level worker.
#   Do NOT also give every worker CPUS_PER_FOV BLAS/OpenMP threads; that
#   creates workers * threads oversubscription inside one Slurm allocation.
#
# Cluster pressure estimate when MATLAB local parallel is enabled:
#   max concurrent MATLAB local worker processes/sessions
#     ~= Batch_FOV * MATLAB_LOCAL_WORKERS
#   Example: Batch_FOV=128 and MATLAB_LOCAL_WORKERS=4 can create up to
#   512 concurrent MATLAB worker processes/sessions. Reduce Batch_FOV if
#   licenses, memory, filesystem I/O, or MATLAB startup pressure cannot
#   tolerate that.
#
# If pipeline.registration.matlab_local_parallel.enabled is false, PyStar
# does not start the Stage18D MATLAB local worker pool; MATLAB_LOCAL_WORKERS
# is then only the intended value to keep beside the config when enabling it.
# -----------------------------------------------------------------
CPUS_PER_FOV=4
MATLAB_LOCAL_WORKERS=4
Batch_FOV=128

if [ "$MATLAB_LOCAL_WORKERS" -gt "$CPUS_PER_FOV" ]; then
    echo "Error: MATLAB_LOCAL_WORKERS ($MATLAB_LOCAL_WORKERS) must be <= CPUS_PER_FOV ($CPUS_PER_FOV)."
    echo "Model A is one worker per CPU; this setting would oversubscribe the Slurm allocation."
    exit 1
fi

MATLAB_PARALLEL_INFO=$("${PIXIRUN[@]}" python -c "
import yaml
try:
    with open('$CONFIG_FILE', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    registration = ((data.get('pipeline') or {}).get('registration') or {})
    parallel = registration.get('matlab_local_parallel') or {}

    raw_enabled = parallel.get('enabled', False)
    if isinstance(raw_enabled, str):
        enabled = raw_enabled.strip().lower() in {'1', 'true', 'yes', 'on'}
    else:
        enabled = bool(raw_enabled)

    raw_workers = parallel.get('workers', 2)
    try:
        workers = int(raw_workers)
    except Exception:
        workers = -1
    print(f'{1 if enabled else 0} {workers}')
except Exception:
    print('0 2')
")
read -r CONFIG_MATLAB_PARALLEL_ENABLED CONFIG_MATLAB_LOCAL_WORKERS <<< "$MATLAB_PARALLEL_INFO"

if [ "$CONFIG_MATLAB_PARALLEL_ENABLED" -eq "1" ]; then
    if [ "$CONFIG_MATLAB_LOCAL_WORKERS" -le "0" ]; then
        echo "Error: Invalid pipeline.registration.matlab_local_parallel.workers in $CONFIG_FILE: $CONFIG_MATLAB_LOCAL_WORKERS"
        exit 1
    fi
    if [ "$CONFIG_MATLAB_LOCAL_WORKERS" -ne "$MATLAB_LOCAL_WORKERS" ]; then
        echo "Error: YAML matlab_local_parallel.workers ($CONFIG_MATLAB_LOCAL_WORKERS) does not match MATLAB_LOCAL_WORKERS ($MATLAB_LOCAL_WORKERS)."
        echo "Keep run_pystar.sh resource accounting and config workers identical for Model A."
        exit 1
    fi
fi

MAX_MATLAB_LOCAL_WORKERS=$((Batch_FOV * MATLAB_LOCAL_WORKERS))

echo "Detected $NUM_JOBS FOVs to process."
echo "Resource model: Model A, one MATLAB local worker per CPU."
echo "CPUS_PER_FOV=$CPUS_PER_FOV"
echo "MATLAB_LOCAL_WORKERS=$MATLAB_LOCAL_WORKERS"
echo "Batch_FOV=$Batch_FOV"
echo "Config matlab_local_parallel.enabled=$CONFIG_MATLAB_PARALLEL_ENABLED"
echo "Config matlab_local_parallel.workers=$CONFIG_MATLAB_LOCAL_WORKERS"
echo "Max concurrent MATLAB local workers if enabled: $MAX_MATLAB_LOCAL_WORKERS"
mkdir -p logs/pystar

echo "Preflighting shared codebook debug CSV..."
"${PIXIRUN[@]}" python scripts/preflight_codebook.py --config "$CONFIG_FILE"

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
echo "Resource model: Model A, one MATLAB local worker per CPU"
echo "CPUS_PER_FOV=${CPUS_PER_FOV}"
echo "MATLAB_LOCAL_WORKERS=${MATLAB_LOCAL_WORKERS}"
echo "Batch_FOV=${Batch_FOV}"
echo "Max concurrent MATLAB local workers if enabled: ${MAX_MATLAB_LOCAL_WORKERS}"
echo "Actual PyStar worker count is controlled by pipeline.registration.matlab_local_parallel.workers in ${CONFIG_FILE}."

export PYSTAR_CPUS_PER_FOV=${CPUS_PER_FOV}
export PYSTAR_MATLAB_LOCAL_WORKERS=${MATLAB_LOCAL_WORKERS}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

pixi run --manifest-path env/pixi.toml -e pystar   python scripts/batch_pystar.py   --config "$CONFIG_FILE"   --task_id "\$SLURM_ARRAY_TASK_ID"

EOF
)

echo "Job submitted! ID: $JOB_ID"
echo "Monitor with: squeue -j $JOB_ID"
