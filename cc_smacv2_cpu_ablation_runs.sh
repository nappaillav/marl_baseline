#!/bin/bash
#SBATCH --account=def-dpmeger
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=0:59:00
#SBATCH --output=slurm-%j.out

# ABLATION-STUDY job script. Copy of cc_smacv2_cpu.sh made 2026-08-26 so that the ablation
# launcher (cc_smacv2_test1.sh) does not depend on a file other sessions may edit.
# Only difference vs the copy source: an optional 10th positional arg `save_model_interval`
# (empty -> 2000000, the default.yaml value); the study passes 250000.
#
# CPU-only SMACv2 job for Nibi. Adapted from cc_mamujoco_cpu.sh (MAMuJoCo, other cluster).
# Same positional-argument interface; see cc_smacv2_test1.sh for the launcher.
#
# Nibi-specific changes vs. the MAMuJoCo original:
#   - StdEnv/2023 + python/3.11 (StdEnv/2020 + python/3.9.6 do not exist here)
#   - epymarlEnv venv; the modules MUST be loaded before `activate` or numpy is missing
#   - SC2PATH instead of the MuJoCo LD_LIBRARY_PATH
#   - `seed=` not `env_args.seed=`  (src/main.py:40 copies seed -> env_args.seed;
#     passing env_args.seed here would be silently overwritten -- see INTEGRATION.md 6)
#   - local_results_path set explicitly to an absolute path under the repo; absolute is
#     what redirects sacred, tb_logs AND models together (INTEGRATION.md 1). Relative
#     would split them: run.py:124 resolves tb_logs against the repo root but run.py:304
#     resolves models against the submit CWD.
#   - thread_num tied to the allocation; the default of 12 oversubscribes a 10-core job
#
# Resource sizing, revised from the 2026-08-25 30-minute sample runs (jobs 20534313-15):
#   cpus  8  = one per SC2 worker (batch_size_run=8). The first round reserved 10 and the
#             jobs time-averaged only 7.5-7.9 cores (seff: 74-79% of 10), because rollout
#             and the gradient update alternate rather than overlap.
#   mem   30G against a measured peak RSS of 16.9G (terran, the heaviest of the three).
#             The replay buffer is preallocated at startup, so this does NOT grow with
#             t_max -- 5M steps needs no more memory than 1M.

module purge
module load StdEnv/2023
module load python/3.11 cuda/12.2 scipy-stack
source /home/zwang182/pip_envs/epymarlEnv/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export SC2PATH=/home/zwang182/MARL/sc2_410/StarCraftII

mainPy=$1
alg=$2
sc2_env=$3
seed=$4
t_max=$5
useCuda=False   # CPU-only job; $6 is ignored
usetb=$7
saveModel=$8
ckpt=$9
saveInterval=${10:-2000000}   # optional 10th arg; empty -> default.yaml value (2000000)

# Results live in the repo. /home is a 50 GB quota (7.8 GB used as of 2026-08-25), which
# is ample for these 30-minute runs -- save_model_interval=2000000 means no checkpoint is
# written before the wall clock stops them. Watch it for full-length runs: INTEGRATION.md 1
# records a zerg run killed by ENOSPC when results sat on a full shared volume.
resultsPath=/home/zwang182/MARL/marl_p3_smacv2/results

echo "main.py=${mainPy} algorithm=${alg} smacv2_env=${sc2_env} seed=${seed} t_max=${t_max} use_cuda=${useCuda} use_tensorboard=${usetb} save_model=${saveModel} save_model_interval=${saveInterval}"
python ${mainPy} --config=${alg} --env-config=${sc2_env} with seed=${seed} t_max=${t_max} \
  use_cuda=${useCuda} use_tensorboard=${usetb} save_model=${saveModel} checkpoint_path=${ckpt} \
  local_results_path=${resultsPath} thread_num=${SLURM_CPUS_PER_TASK} save_model_interval=${saveInterval}
