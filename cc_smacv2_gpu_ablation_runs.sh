#!/bin/bash
#SBATCH --account=def-dpmeger_gpu
#SBATCH --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=0:30:00
#SBATCH --output=slurm-%j.out

# ===== FIR VERSION (ported from the `nibi` branch, 2026-09-11) =========================
# Retargeted for Fir: repo root /home/zwang182/MARL/marl_p3/marl_p3_smacv2, SC2PATH
# /home/zwang182/MARL/sc2/3rdparty/StarCraftII, GPU gres nvidia_h100_80gb_hbm3_* (Fir's
# canonical MIG names). --account and the module stack are UNCHANGED: def-dpmeger,
# def-dpmeger_cpu and def-dpmeger_gpu are all valid associations here, and StdEnv/2023 +
# python/3.11 + cuda/12.2 + scipy-stack + epymarlEnv runs this repo unmodified (verified
# 2026-09-11: SMACv1 6h_vs_8z and all three SMACv2 races, CPU and MIG, exit 0).
# CAUTION: every wall-time, steps/s and RSS figure below is a NIBI measurement. A 12k-step
# Fir sample on 2026-09-11 ran ~119 env-steps/s on 8 CPU cores and ~148 on a 1g.10gb MIG
# slice, vs the 18-53 steps/s these headers assume -- re-measure before trusting --time.
# =======================================================================================

# ABLATION-STUDY GPU job script. Copy of cc_smacv2_gpu.sh made 2026-08-26 so that the terran
# ablation launcher (cc_smacv2_ablation_terran.sh) does not depend on a file other sessions may
# edit. Only difference vs the copy source: an optional 10th positional arg `save_model_interval`
# (empty -> 2000000, the default.yaml value); the study passes 400000.
#
# GPU variant of cc_smacv2_cpu.sh. Same positional-argument interface; only the
# resource block, the use_cuda hardcode, and the CUDA precheck differ.
#
# MIG slice, not a full H100: a 5v5 run needs roughly 2-3 GB VRAM (marl_project3
# SETUP_ENVIRONMENT.md section 9 measures a 5m_vs_6m PAIR at ~5 GB; 24-agent corridor is the
# 22.6 GB outlier, not us). A 1g.10gb slice queues as fast as a full H100 here and
# leaves ample headroom -- VRAM does climb as episodes lengthen toward the 400-step
# limit, so if a run ever OOMs, step up to nvidia_h100_80gb_hbm3_2g.20gb rather than to a whole card.
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

export SC2PATH=/home/zwang182/MARL/sc2/3rdparty/StarCraftII

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

resultsPath=/home/zwang182/MARL/marl_p3/marl_p3_smacv2/results

echo "main.py=${mainPy} algorithm=${alg} smacv2_env=${sc2_env} seed=${seed} t_max=${t_max} use_cuda=${useCuda} use_tensorboard=${usetb} save_model=${saveModel} save_model_interval=${saveInterval}"
python ${mainPy} --config=${alg} --env-config=${sc2_env} with seed=${seed} t_max=${t_max} \
  use_cuda=${useCuda} use_tensorboard=${usetb} save_model=${saveModel} checkpoint_path=${ckpt} \
  local_results_path=${resultsPath} thread_num=${SLURM_CPUS_PER_TASK} save_model_interval=${saveInterval}
