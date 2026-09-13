#!/bin/bash
#SBATCH --account=def-dpmeger
#SBATCH --job-name=dmix
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=23:59:00
#SBATCH --array=0-2
#SBATCH --output="/home/chidamv/scratch/logs/smacv2/out/%x-%A_%a.out"
#SBATCH --error="/home/chidamv/scratch/logs/smacv2/log/%x-%A_%a.err"

set -u

start=$(date +%s)

### ============================================================
### Load modules
### ============================================================

module --force purge
module load StdEnv/2023 python/3.10 cuda/12.2 rust/1.85.0

### ============================================================
### Activate virtual environment
### ============================================================

source /home/chidamv/env/mbmarlEnv/bin/activate

### ============================================================
### Repository
### ============================================================

cd /home/chidamv/projects/def-dpmeger/chidamv/marl_baseline


### ============================================================
### Experiment configuration
### ============================================================

# ALGORITHMS="deepset_qmix" # CPU : 8


ENV_CONFIGS=(
    "sc2_v2_protoss_10v10"
    "sc2_v2_terran_10v10"
    "sc2_v2_zerg_10v10"
)

ALG="deepset_qmix"
ENV_CONFIG=${ENV_CONFIGS[$SLURM_ARRAY_TASK_ID]}

### ============================================================
### Training configuration
### ============================================================
SEED=$((RANDOM % 100000))
T_MAX=${2:-5005000}
BUFFER_SIZE=${3:-3500}
SAVE_INTERVAL=${4:-500000}

# Results are stored under this directory.
# Each array task gets its own subdirectory.
RESULTS_ROOT="/home/chidamv/scratch/results/${ENV_CONFIG}"

RESULTS_PATH="${RESULTS_ROOT}/${ALG}/seed_${SEED}"

### ============================================================
### StarCraft II
### ============================================================

export SC2PATH="/home/chidamv/scratch/MARL/StarCraftII/StarCraftII"

### ============================================================
### Threading
### ============================================================

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo
echo "============================================================"
# echo "JOB ID:             $SLURM_JOB_ID"
# echo "ARRAY JOB ID:       $SLURM_ARRAY_JOB_ID"
# echo "ARRAY TASK ID:      $SLURM_ARRAY_TASK_ID"
echo "ALGORITHM:          $ALG"
echo "ENV CONFIG:         $ENV_CONFIG"
echo "T_MAX:              $T_MAX"
echo "BUFFER SIZE:        $BUFFER_SIZE"
echo "SAVE INTERVAL:      $SAVE_INTERVAL"
echo "SC2PATH:            $SC2PATH"
echo "RESULTS:            $RESULTS_PATH"
# echo "CPUS:               $SLURM_CPUS_PER_TASK"
echo "============================================================"
echo

### ============================================================
### Create results directory
### ============================================================

mkdir -p "$RESULTS_PATH"

### ============================================================
### Run EPyMARL
### ============================================================

CMD="python src/main.py \
    --config=${ALG} \
    --env-config=${ENV_CONFIG} \
    with \
    seed=${SEED} \
    t_max=${T_MAX} \
    buffer_size=${BUFFER_SIZE} \
    use_cuda=False \
    use_tensorboard=True \
    save_model=True \
    save_model_interval=${SAVE_INTERVAL} \
    local_results_path=${RESULTS_PATH} \
    thread_num=${SLURM_CPUS_PER_TASK}"

echo "COMMAND:"
echo "$CMD"
echo

eval "$CMD"

EXIT_CODE=$?

### ============================================================
### Completion
### ============================================================

end=$(date +%s)
runtime=$((end - start))

echo
echo "============================================================"

if [ "$EXIT_CODE" -eq 0 ]; then
    echo "Experiment finished successfully."
else
    echo "Experiment FAILED."
    echo "Exit code: $EXIT_CODE"
fi

echo "TASK:       $SLURM_ARRAY_TASK_ID"
echo "ALGORITHM:  $ALG"
echo "ENV:        $ENV_CONFIG"
echo "SEED:       $SEED"
echo "Runtime:    $((runtime / 3600)) hours, $(((runtime % 3600) / 60)) minutes, $((runtime % 60)) seconds"
echo "============================================================"

exit "$EXIT_CODE"