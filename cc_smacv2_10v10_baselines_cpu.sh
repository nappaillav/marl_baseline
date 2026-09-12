#!/bin/bash
#SBATCH --account=def-dpmeger
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=1:00:00
#SBATCH --output=slurm-%j.out

# ===== FIR (written 2026-09-11) ========================================================
# CPU-only job script for the SMACv2 10v10 BASELINE sample runs (HPN-QMIX, HPN-VDN,
# DEEPSET-QMIX). Launcher: cc_smacv2_10v10_baselines_launch.sh.
#
# A study-specific copy, following this repo's convention (cf. cc_smacv2_cpu_ablation_runs.sh):
# the shared cc_smacv2_cpu.sh keeps its 9-argument interface and its 30G/0:59:00 block, so
# nothing else in the repo is disturbed by the settings here.
#
# Positional interface is the 11-argument one from the 10v10 GPU script cc_smacv2_gpu.sh,
# so the two can be swapped without touching the launcher's argument order.
#
# Resources, as specified for these runs: --time=1:00:00, --mem=60G, 8 cpus.
#   cpus  8 = one per SC2 worker (batch_size_run=8), the value the 5v5 studies settled on
#             after measuring 7.5-7.9 cores in use on a 10-core reservation.
#   mem  60G against a replay buffer that at 10v10 preallocates ~12.6 GiB at buffer_size=3500
#             (9653 B/transition x 401 steps x 3500 episodes; the 10v10 GPU script records
#             15.5 GB at 4000 and 19.4 GB at 5000). The buffer is allocated at startup and
#             does NOT grow with t_max. The rest is headroom for the 8 SC2 processes.
# =======================================================================================

module purge
module load StdEnv/2023
module load python/3.11 cuda/12.2 scipy-stack
source /home/zwang182/pip_envs/epymarlEnv/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Fir's StarCraft II is 4.10.0 (build 75689) as of 2026-09-11, matching Nibi and the lab
# machines. A run that reports Base69232 is on the old 4.6.2 tree and is NOT comparable.
export SC2PATH=/home/zwang182/MARL/sc2/3rdparty/StarCraftII

mainPy=$1
alg=$2
sc2_env=$3
seed=$4
t_max=$5
useCuda=False   # CPU-only job; $6 is ignored
usetb=$7
saveModel=$8
ckpt=$9
saveInterval=${10}
bufferSize=${11}

# Baseline outputs kept separate from the method's results/MSEMM_10v10 so the two never mix.
# An ABSOLUTE path redirects all three sinks together (run.py:124 tb_logs, main.py:117
# sacred, run.py:304 models); each is created on demand, so no mkdir is needed.
resultsPath=/home/zwang182/MARL/marl_p3/marl_p3_smacv2/results/MSEMM_10v10_baselines

echo "main.py=${mainPy} algorithm=${alg} smacv2_env=${sc2_env} seed=${seed} t_max=${t_max} use_cuda=${useCuda} use_tensorboard=${usetb} save_model=${saveModel} buffer_size=${bufferSize}"
python ${mainPy} --config=${alg} --env-config=${sc2_env} with seed=${seed} t_max=${t_max} \
  use_cuda=${useCuda} use_tensorboard=${usetb} save_model=${saveModel} checkpoint_path=${ckpt} \
  save_model_interval=${saveInterval} buffer_size=${bufferSize} \
  local_results_path=${resultsPath} thread_num=${SLURM_CPUS_PER_TASK}
