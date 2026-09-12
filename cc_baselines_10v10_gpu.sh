#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=23:59:00
#SBATCH --output=slurm-%j.out

# GPU job script for the SMACv2 10v10 BASELINE runs (HPN-QMIX, HPN-VDN, QMIX, DEEPSET-QMIX).
# Launched by cc_baselines_10v10_launch.sh, which supplies --account and --gpus on the sbatch
# command line (those override any #SBATCH line here), so this file needs no site edits in its
# resource block. The one thing you MUST edit is the SITE BLOCK below.
#
# Positional arguments (all required, no empty placeholders):
#   $1 mainPy       absolute path to src/main.py
#   $2 alg          --config value      e.g. hpn_qmix
#   $3 sc2_env      --env-config value  e.g. sc2_v2_protoss_10v10
#   $4 seed
#   $5 t_max
#   $6 bufferSize
#   $7 saveInterval
#   $8 resultsPath  absolute path; redirects tb_logs, sacred AND models together
#
# use_cuda / use_tensorboard / save_model are constant for these runs and are hardcoded below.

set -u

# ===== SITE BLOCK -- EDIT THIS FOR YOUR CLUSTER ========================================
module purge
module load StdEnv/2023
module load python/3.11 cuda/12.2 scipy-stack
source /home/zwang182/pip_envs/epymarlEnv/bin/activate

# StarCraft II install. MUST be 4.10.x -- SMAC win rates are NOT comparable across SC2
# versions, so a different major.minor silently makes your numbers incomparable with
# everyone else's. See README1.md.
export SC2PATH=/home/zwang182/MARL/sc2/3rdparty/StarCraftII
# =======================================================================================

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# ---- preflight: fail fast and loudly, before burning a queue slot ----------------------

# 1. StarCraft II present and the right version.
if [ ! -d "${SC2PATH}/Versions" ]; then
  echo "ERROR: SC2PATH=${SC2PATH} has no Versions/ -- set it in the SITE BLOCK above."; exit 1
fi
sc2ver=$(awk -F'|' 'NR==2{print $13}' "${SC2PATH}/.build.info" 2>/dev/null)
echo "StarCraft II: ${sc2ver:-unknown}  ($(ls "${SC2PATH}/Versions" | tr '\n' ' '))"
case "${sc2ver}" in
  4.10*) ;;
  *) echo "ERROR: StarCraft II is '${sc2ver}', expected 4.10.x. Results would not be"
     echo "       comparable with the reference runs. Aborting. See README1.md."; exit 1;;
esac

# 2. The seven maps SMACv2 needs. The 10gen_* generators reference a base terrain map, and a
#    missing one fails LATE inside the env workers -- it reads as a hang, not a missing file.
for m in 10gen_protoss 10gen_terran 10gen_zerg 10gen_empty 32x32_flat 32x32_flat_test 32x32_small; do
  [ -f "${SC2PATH}/Maps/SMAC_Maps/${m}.SC2Map" ] || {
    echo "ERROR: missing ${SC2PATH}/Maps/SMAC_Maps/${m}.SC2Map"
    echo "       install them all with:  cp -n src/envs/smac_v2/official/maps/SMAC_Maps/*.SC2Map \"\$SC2PATH/Maps/SMAC_Maps/\""
    exit 1; }
done

# 3. GPU actually usable. torch.cuda.is_available() is NOT sufficient: it can return True while
#    context creation still fails with cudaErrorDevicesUnavailable (observed on a MIG node where
#    two jobs landed together -- job 59382383, 2026-09-11). Allocating is the real test.
#    run.py silently flips use_cuda to False when CUDA is missing, so without this a broken GPU
#    job would run to completion on CPU, look successful, and bill GPU time for CPU work.
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
if ! python -c "import torch; torch.zeros(1, device='cuda'); torch.cuda.synchronize()" 2>&1; then
  echo "ERROR: could not allocate on the GPU -- aborting instead of silently running on CPU."
  echo "       If this is cudaErrorDevicesUnavailable it is usually a transient MIG collision;"
  echo "       simply resubmit this one job."
  exit 1
fi

# ---- run -------------------------------------------------------------------------------
mainPy=$1; alg=$2; sc2_env=$3; seed=$4; t_max=$5; bufferSize=$6; saveInterval=$7; resultsPath=$8

echo "alg=${alg} env=${sc2_env} seed=${seed} t_max=${t_max} buffer_size=${bufferSize} results=${resultsPath}"

# NOTE: `seed=` not `env_args.seed=`. main.py copies seed -> env_args.seed at run time, so
# passing env_args.seed is decorative and the run silently uses a sacred-generated seed.
python "${mainPy}" --config="${alg}" --env-config="${sc2_env}" with \
  seed="${seed}" t_max="${t_max}" buffer_size="${bufferSize}" \
  use_cuda=True use_tensorboard=True save_model=True save_model_interval="${saveInterval}" \
  local_results_path="${resultsPath}" thread_num="${SLURM_CPUS_PER_TASK}"
