#!/bin/bash
# Near-initialisation checkpoints of the Full method, produced on the LOGIN node (no Slurm).
#
# Purpose: plans 2 and 5 of the world-model / representation analyses need UNTRAINED weights with each
# run's seed (an untrained world model as the "random features" control; the head's embedding at
# initialisation). This script writes one such checkpoint for every (config, map, seed) combination
# of the reference runs launched by cc_reference_runs.sh: 21 in total.
#
# How it works: a run with t_max=1000 saves exactly once, when t_env >= t_max (run.py:301-302), and the
# learner never updates before that (it needs 128 stored episodes, run.py:273) -- so the saved
# agent/mixer/world-model weights are the seed-determined initialisation. Measured on l3 on
# 2026-09-04 with the light settings below: 43-57 s wall, 1.5 GB peak RSS (python + one SC2 process),
# 0 learner updates, all four files saved (agent.th, mixer.th, opt.th, sem_wm.th); two trials with the
# same seed were bit-identical. The light settings (episode runner, buffer_size=8, use_cuda=False) do
# not change the initial weights -- those are drawn right after seeding, before anything runner- or
# buffer-dependent consumes random numbers -- but they DO change the hyper-parameter folder name
# (env_n=1/...bs=8_128 instead of the training runs' env_n=8/...bs=5000_128); probes reference the
# checkpoint directory by absolute path, so this is harmless.
#
# Output goes to its own folder inside the study checkout, next to results/ (which holds only the
# training runs); the folder is not tracked by git (moved here from ~/MARL/init_checkpoints on 2026-09-04):
#   /home/zwang182/MARL/marl_p3_smacv2/init_checkpoints/models/<env>/algo=<name>-agent=<agent>/env_n=1/<hp>/<token>/<step>/
# with <token> = <name>_<map>_seed<N>__<timestamp> and <step> ~ 1000-1100. Logs: <out>/<tag>.log.
#
# Usage (login node; sequential; ~15-20 min for all 21; re-running skips combinations that already
# have a checkpoint, so it can be stopped and resumed):
#   bash /home/zwang182/MARL/marl_p3_smacv2/cc_reference_init_login.sh
# Do NOT run several copies in parallel on the login node.

set -u
repo=/home/zwang182/MARL/marl_p3_smacv2
out=/home/zwang182/MARL/marl_p3_smacv2/init_checkpoints
mkdir -p "${out}"

module purge
module load StdEnv/2023
module load python/3.11 cuda/12.2 scipy-stack
source /home/zwang182/pip_envs/epymarlEnv/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export SC2PATH=/home/zwang182/MARL/sc2_410/StarCraftII

cd "${repo}/src"

common="t_max=1000 runner=episode batch_size_run=1 buffer_size=8 use_cuda=False use_tensorboard=False save_model=True save_model_interval=250000 local_results_path=${out} thread_num=1"

# run_one <tag> <alg config> <env config> <map name or ""> <seed>
run_one() {
  local tag=$1 alg=$2 env=$3 map=$4 seed=$5
  local mapArg=""
  [ -n "${map}" ] && mapArg="env_args.map_name=${map}"
  # token map name: SMACv2 races become 10gen_<race>; SMACv1 uses the map name itself
  local mapname=${map:-10gen_${env#sc2_v2_}}
  local name=${alg##*/}                       # config `name:` equals the file name for these configs
  # depth: models/<env>/algo=.../env_n=1/<hp>/<token>/<step>
  if ls -d "${out}"/models/*/algo=*/*/*/${name}_${mapname}_seed${seed}__*/[0-9]* >/dev/null 2>&1; then
    echo "SKIP ${tag}  (checkpoint exists)"
    return 0
  fi
  echo "RUN  ${tag}"
  timeout 900 python main.py --config="${alg}" --env-config="${env}" with seed="${seed}" ${common} ${mapArg} \
    > "${out}/${tag}.log" 2>&1
  local rc=$?
  echo "     exit=${rc}  saves=$(grep -c 'Saving models' "${out}/${tag}.log")  updates=$(grep -c 'loss_td' "${out}/${tag}.log")"
}

# SMACv2: protoss, zerg (standard agent) and terran (healer-aware twin), seeds 1-5
for race in protoss zerg; do
  for s in 1 2 3 4 5; do run_one "init_${race}_s${s}" hpn_saleq_sem_wm_qmix "sc2_v2_${race}" "" "${s}"; done
done
for s in 1 2 3 4 5; do run_one "init_terran_s${s}" hpn_saleq_terran_sem_wm_qmix sc2_v2_terran "" "${s}"; done

# SMACv1: 6h_vs_8z and corridor, seeds 1-5 (seeds 4-5 added 2026-09-04 with cc_reference_runs_extra.sh)
for map in 6h_vs_8z corridor; do
  for s in 1 2 3 4 5; do run_one "init_v1_${map}_s${s}" hpn_saleq_sem_wm_qmix sc2 "${map}" "${s}"; done
done

echo "done. checkpoints: $(ls -d "${out}"/models/*/algo=*/*/*/*/[0-9]* 2>/dev/null | wc -l) / 25"
