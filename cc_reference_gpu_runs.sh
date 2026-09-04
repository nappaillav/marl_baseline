#!/bin/bash
#SBATCH --account=def-dpmeger_gpu
#SBATCH --gpus=h100_1g.10gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=0:30:00
#SBATCH --output=slurm-%j.out

# REFERENCE-RUN GPU job script: the FULL METHOD (hpn_saleq_sem_wm_qmix on protoss / zerg / SMACv1,
# hpn_saleq_terran_sem_wm_qmix on terran) run as the reference row of the ablation study and of the
# design-space study, and as the checkpoint source of the world-model / learned-representation
# analyses (notes/learned_world_model_and_reps_experiments/plans-for-proposals.md, I.2-I.4 and I.7).
# Launched by cc_reference_runs.sh.
#
# Copy of cc_smacv2_gpu_ablation_runs.sh made 2026-09-04. Only difference: an optional 11th positional
# arg `map_name`. When non-empty, `env_args.map_name=<map_name>` is appended to the `with` clause --
# that is how SMACv1 maps are selected (--env-config=sc2; sc2.yaml's own default is 3m). For SMACv2
# (--env-config=sc2_v2_<race>) leave it empty and the env config's 10gen_<race> map is used. main.py
# reads the same `env_args.map_name=` token to name the sacred dir (results/sacred/<map>/<name>/).
# Everything else is identical: account, MIG gres, 8 CPUs, 30G, modules, venv, SC2PATH, CUDA
# precheck, `seed=` form, absolute resultsPath, thread_num tied to the allocation.
#
# RULE: a run that hits the wall clock (TIMEOUT / exit 143) is RERUN FROM SCRATCH with a longer
# --time. It is NEVER resumed from its checkpoint: a post-resume performance collapse was observed
# on resumed study runs (the replay buffer, target_mixer and RNG are not checkpointed, so a resumed
# learner trains on a buffer refilled at an already-annealed epsilon). The reference rows must be
# clean single runs -- set the timed-out run's models/ and tb_logs/ dirs aside so the analyses never
# pick them up (an init/full/rerun of the same (name, map, seed) differs only by timestamp).
#
# MIG slice, not a full H100: a 5v5 run needs roughly 2-3 GB VRAM (marl_project3
# SETUP_ENVIRONMENT.md section 9 measures a 5m_vs_6m PAIR at ~5 GB; 24-agent corridor is the
# 22.6 GB outlier, not us). A 1g.10gb slice queues as fast as a full H100 here and
# leaves ample headroom -- VRAM does climb as episodes lengthen toward the 400-step
# limit, so if a run ever OOMs, step up to h100_2g.20gb rather than to a whole card.
# NOTE for the SMACv1 corridor runs (24 enemies, 400-step limit): that 22.6 GB (24 GB with the
# world model, ~25 GB RAM) was measured on a lab 3090 with a 3M budget; whether the learner alone
# (cpu_inference=True here) fits a 10 GB slice is UNVERIFIED. If a corridor job OOMs, resubmit it
# with `--gpus=h100_2g.20gb:1` (or 3g.40gb) on the sbatch line -- command-line flags override the
# #SBATCH block -- and keep this file as is for the other runs.
#
# cpu_inference is deliberately NOT passed, so it keeps default.yaml's `True`. That means
# the GPU accelerates the LEARNER only: parallel_runner.py:77 copies the model back to CPU
# at every reset() and rollout action selection stays on CPU.
#
# Expect a modest gain at best. marl_project3 SETUP_ENVIRONMENT.md section 9, measuring this
# same architecture: "wall-clock is dominated by the 8 parallel StarCraft II processes
# (batch_size_run: 8), not by the learner." The 8 SC2 workers are pure CPU either way.

module purge
module load StdEnv/2023
module load python/3.11 cuda/12.2 scipy-stack
source /home/zwang182/pip_envs/epymarlEnv/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export SC2PATH=/home/zwang182/MARL/sc2_410/StarCraftII

# Fail fast if the GPU is not visible. run.py:331-333 SILENTLY flips use_cuda to False
# when torch.cuda.is_available() is False, so without this check a broken GPU job would
# run to completion on CPU, look successful, and bill GPU queue time for CPU work.
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
if ! python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
  echo "ERROR: torch.cuda.is_available() is False -- aborting instead of silently running on CPU."
  exit 1
fi

mainPy=$1
alg=$2
sc2_env=$3
seed=$4
t_max=$5
useCuda=True    # GPU job; $6 is ignored
usetb=$7
saveModel=$8
ckpt=$9
saveInterval=${10:-2000000}   # optional 10th arg; empty -> default.yaml value (2000000)
mapName=${11:-}               # optional 11th arg; non-empty -> SMACv1 map via env_args.map_name (SMACv2: leave empty)

mapArg=""
if [ -n "${mapName}" ]; then
  mapArg="env_args.map_name=${mapName}"
fi

resultsPath=/home/zwang182/MARL/marl_p3_smacv2/results

echo "main.py=${mainPy} algorithm=${alg} sc2_env=${sc2_env} map_name=${mapName:-<env-config default>} seed=${seed} t_max=${t_max} use_cuda=${useCuda} use_tensorboard=${usetb} save_model=${saveModel} save_model_interval=${saveInterval}"
python ${mainPy} --config=${alg} --env-config=${sc2_env} with seed=${seed} t_max=${t_max} \
  use_cuda=${useCuda} use_tensorboard=${usetb} save_model=${saveModel} checkpoint_path=${ckpt} \
  local_results_path=${resultsPath} thread_num=${SLURM_CPUS_PER_TASK} save_model_interval=${saveInterval} ${mapArg}
