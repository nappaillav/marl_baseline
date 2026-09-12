#!/bin/bash
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
# Resume timed-out SMACv2 study runs from their latest checkpoint.
#
# Usage:  bash cc_smacv2_resume.sh <race> <alg> <seed> [<seed> ...]
#         bash cc_smacv2_resume.sh <race> <alg> all
#         bash cc_smacv2_resume.sh <race> study        <- every arm of the race's ablation study, all seeds
#
#   race  protoss | zerg  -> cc_smacv2_cpu_ablation_runs.sh, 71:55:00, save_model_interval 250000
#         terran          -> cc_smacv2_gpu_ablation_runs.sh, 11:59:00, save_model_interval 400000
#         (the same job script / wall time / interval the study launchers use)
#   alg   the --config value, e.g. ablation_arm_3, terran_ablations/ablation_arm_3,
#         hpn_saleq_sem_wm_qmix, hpn_saleq_terran_sem_wm_qmix
#   seed  one or more seeds, or `all` = every seed that has a checkpoint dir for this (alg, race)
#   study (in place of <alg>) sweeps the race's study arms (the same list as the study launcher:
#         terran_ablations/ablation_arm_* for terran, ablation_arm_* for protoss/zerg) with `all`
#         seeds; finished runs print DONE and only timed-out ones are resubmitted.
#
# Examples:
#   bash cc_smacv2_resume.sh terran terran_ablations/ablation_arm_5a 1 2 3
#   bash cc_smacv2_resume.sh zerg ablation_arm_6 all
#   bash cc_smacv2_resume.sh terran study
# Dry run (prints the sbatch lines, submits nothing):
#   sbatch() { echo "DRY-RUN sbatch $@"; }; export -f sbatch; bash cc_smacv2_resume.sh terran terran_ablations/ablation_arm_1 all
#
# What it does, per (alg, race, seed):
#   1. reads `name:` from src/config/algs/<alg>.yaml and finds every model dir
#        results/models/<...>/<name>_<map>_seed<N>__<timestamp>/
#      (a resumed run writes a NEW dir with a new timestamp, so there may be several per run);
#   2. picks the highest checkpoint step across those dirs whose save is complete (agent.th + opt.th);
#   3. SKIPs if that step >= T_MAX (run.py:302 saves once t_env >= t_max, so a finished run always
#      has such a checkpoint), or if a job with this run's name is already queued/running;
#   4. otherwise submits the same job as the study launcher with checkpoint_path=<that dir>
#      (load_step=0 in default.yaml -> the latest step in the dir) and the same job name.
# Restored: agent (live+target), mixer, Adam state, world model, t_env (so epsilon / test cadence).
# NOT restored: replay buffer (refills from scratch), target_mixer (random until the first target
# update), RNG. The resumed run gets a new sacred id and tb dir; stitch curves by (name, map, seed).
#
# Overrides via environment: T_MAX (5005000), WALLTIME, SAVE_INTERVAL, RESULTS_DIR.

set -u
race=${1:-}; alg=${2:-}; shift 2 2>/dev/null || true

# `study` mode: re-invoke this script once per study arm with `all` seeds.
if [[ "$alg" == "study" ]]; then
  case "$race" in
    terran)       prefix="terran_ablations/" ;;
    protoss|zerg) prefix="" ;;
    *) echo "ERROR: race must be protoss|zerg|terran (got '$race')"; exit 1 ;;
  esac
  for arm in ablation_arm_1 ablation_arm_2 ablation_arm_3 ablation_arm_4 ablation_arm_5a ablation_arm_5b ablation_arm_6; do
    bash "$0" "$race" "${prefix}${arm}" all
  done
  exit 0
fi

if [[ -z "$race" || -z "$alg" || $# -lt 1 ]]; then
  sed -n '13,35p' "$0"; exit 1
fi

repo=/home/zwang182/MARL/marl_p3/marl_p3_smacv2
mainPy=$repo/src/main.py
RESULTS_DIR=${RESULTS_DIR:-$repo/results}
T_MAX=${T_MAX:-5005000}

case "$race" in
  protoss|zerg)
    jobScript=$repo/cc_smacv2_cpu_ablation_runs.sh; WALLTIME=${WALLTIME:-71:55:00}
    SAVE_INTERVAL=${SAVE_INTERVAL:-250000}; useCuda=False ;;
  terran)
    jobScript=$repo/cc_smacv2_gpu_ablation_runs.sh; WALLTIME=${WALLTIME:-11:59:00}
    SAVE_INTERVAL=${SAVE_INTERVAL:-400000}; useCuda=True ;;
  *) echo "ERROR: race must be protoss|zerg|terran (got '$race')"; exit 1 ;;
esac
sc2_env=sc2_v2_${race}; map=10gen_${race}
usetb=True; save_model=True

cfg=$repo/src/config/algs/${alg}.yaml
[[ -f "$cfg" ]] || { echo "ERROR: config not found: $cfg"; exit 1; }
name=$(grep -E '^name:' "$cfg" | head -1 | sed -E 's/^name:[[:space:]]*"?([^"#]*)"?.*/\1/' | tr -d '[:space:]')
[[ -n "$name" ]] || { echo "ERROR: no name: key in $cfg"; exit 1; }
[[ -f "$jobScript" ]] || { echo "ERROR: job script not found: $jobScript"; exit 1; }

# one squeue call (never poll): names of this user's queued/running jobs
queued=$(squeue -u "$USER" -h -o "%j" 2>/dev/null || true)

seeds=("$@")
if [[ "${seeds[0]}" == "all" ]]; then
  mapfile -t seeds < <(find "$RESULTS_DIR/models" -type d -name "${name}_${map}_seed*__*" 2>/dev/null \
                       | sed -E "s/.*_seed([0-9]+)__.*/\1/" | sort -n | uniq)
  [[ ${#seeds[@]} -gt 0 ]] || { echo "no checkpoint dirs for name=$name map=$map under $RESULTS_DIR/models"; exit 0; }
fi

for s in "${seeds[@]}"; do
  jobname="${alg##*/}_${race}_s${s}"
  best_step=-1; best_dir=""
  while IFS= read -r d; do
    for stepdir in "$d"/*/; do
      stepdir=${stepdir%/}; step=$(basename "$stepdir")
      [[ "$step" =~ ^[0-9]+$ ]] || continue
      [[ -s "$stepdir/agent.th" && -s "$stepdir/opt.th" ]] || continue   # skip partial saves
      if (( step > best_step )); then best_step=$step; best_dir=$d; fi
    done
  done < <(find "$RESULTS_DIR/models" -type d -name "${name}_${map}_seed${s}__*" 2>/dev/null)

  if (( best_step < 0 )); then
    echo "SKIP  $jobname: no complete checkpoint for name=$name map=$map seed=$s"; continue
  fi
  if (( best_step >= T_MAX )); then
    echo "DONE  $jobname: latest checkpoint step $best_step >= T_MAX=$T_MAX (run finished)"; continue
  fi
  if grep -qx -- "$jobname" <<< "$queued"; then
    echo "SKIP  $jobname: a job with this name is already queued/running"; continue
  fi
  echo "RESUME $jobname from step $best_step in $best_dir"
  sbatch --time=${WALLTIME} --job-name=${jobname} \
    ${jobScript} ${mainPy} ${alg} ${sc2_env} ${s} ${T_MAX} ${useCuda} ${usetb} ${save_model} "${best_dir}" ${SAVE_INTERVAL}
done
