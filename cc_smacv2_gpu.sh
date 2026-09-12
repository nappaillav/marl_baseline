#!/bin/bash
#SBATCH --account=def-dpmeger
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=23:59:00
#SBATCH --output=slurm-%j.out

# ===== FIR VERSION (transferred from Nibi 2026-09-11, ported same day) =================
# Source: marl_p3_smacv2_10v10_transfer/MANIFEST.txt, on top of base commit fc853a3.
# Retargeted for Fir: repo root /home/zwang182/MARL/marl_p3/marl_p3_smacv2, SC2PATH
# /home/zwang182/MARL/sc2/3rdparty/StarCraftII, GPU gres nvidia_h100_80gb_hbm3_* (Fir's
# canonical MIG names). --account and the module stack are unchanged: def-dpmeger,
# def-dpmeger_cpu and def-dpmeger_gpu are all valid associations here.
#
# SC2 VERSION: RESOLVED 2026-09-11. Fir originally shipped 4.6.2 (build 69232), which is NOT
# comparable with the Nibi/lab-machine 4.10 runs. SC2PATH was rebuilt with 4.10.0 (build
# 75689) that day -- the same major.minor Nibi uses -- and the 30 SMAC maps reinstalled from
# this repo's vendored copy. Verified: SMACv1 6h_vs_8z and SMACv2 protoss/protoss_10v10/
# zerg_10v10/terran_10v10 all run on Base75689, exit 0. The 4.6.2 tree has been deleted, so
# 4.10 is the only StarCraft II on Fir; reinstall from SC2.4.10.zip if it is ever needed again.
# =======================================================================================

# GPU variant of cc_smacv2_cpu.sh. Same positional-argument interface; only the
# resource block, the use_cuda hardcode, and the CUDA precheck differ.
#
# 40 GB MIG slice (nvidia_h100_80gb_hbm3_3g.40gb), not a full H100: 10v10 doubles the entities per run
# relative to the 5v5 studies, and VRAM climbs further as episodes lengthen toward the
# 400-step limit. 40 GB clears the closest reference point -- marl_project3
# SETUP_ENVIRONMENT.md section 9 has 24-agent corridor at 22.6-24 GB with the world model,
# and 10v10 has a comparable agent-entity pair count (~190 vs ~174). Host --mem=50G covers the replay buffer, which at 10v10 preallocates
# 15.5 GB at buffer_size=4000 (9653 B/transition x 401 x 4000) vs 19.4 GB at 5000 --
# that is why the launcher passes buffer_size=4000. Most of it never becomes resident,
# since real episodes are far shorter than the 401-step worst case.
#
# ZERG IS AN EXCEPTION (2026-09-11): cc_smacv2_gpu_runs.sh overrides zerg onto a 2g.20gb
# slice on the sbatch command line, which beats the #SBATCH line above. Note 20 GB sits
# BELOW the 22.6-24 GB reference point cited above; the argument for it is that zerg ran the
# shortest episodes of the three races in the Fir CPU samples (ep_length_mean ~44 vs protoss
# ~77 and terran ~71, 2026-09-11) and VRAM scales with episode length. That is an inference,
# not a measurement -- those samples were use_cuda=False and measured no VRAM at all. If a
# zerg run dies with a CUDA OOM, move it back to 3g.40gb by deleting the zerg entry in the
# launcher's gpuOf map; per the never-resume rule below that costs the whole run, not the tail.
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
saveInterval=${10}
bufferSize=${11}

# All outputs for this study under one root. An ABSOLUTE local_results_path redirects
# all three sinks together (run.py:124 tb_logs, main.py:117 sacred, run.py:304 models):
#     results/MSEMM_10v10/{tb_logs,sacred,models}/
# Each is created on demand, so no mkdir is needed here.
resultsPath=/home/zwang182/MARL/marl_p3/marl_p3_smacv2/results/MSEMM_10v10

echo "main.py=${mainPy} algorithm=${alg} smacv2_env=${sc2_env} seed=${seed} t_max=${t_max} use_cuda=${useCuda} use_tensorboard=${usetb} save_model=${saveModel}"
python ${mainPy} --config=${alg} --env-config=${sc2_env} with seed=${seed} t_max=${t_max} \
  use_cuda=${useCuda} use_tensorboard=${usetb} save_model=${saveModel} checkpoint_path=${ckpt} \
  save_model_interval=${saveInterval} buffer_size=${bufferSize} \
  local_results_path=${resultsPath} thread_num=${SLURM_CPUS_PER_TASK}
