#!/bin/bash
# Sample runs: our method on SMACv2 protoss_5v5 and zerg_5v5, 3 seeds each,
# 0:59:00 wall, CPU-only. 2 races x 3 seeds = 6 jobs.
#
# terran is NOT here: its 3 sample runs are already done (jobs 20540105-07, 2026-08-25,
# 18.3 env-steps/s steady-state). To rerun it, add terran to `races` AND to `algOf` with
# hpn_saleq_terran_sem_wm_qmix -- it will NOT work with the plain hpn_saleq agent.
#
# Usage:  bash cc_smacv2_sample_runs.sh
# Check:  squeue -u $USER
#
# Both races use the plain hpn_saleq agent -- only terran needs hpn_saleq_terran
# (medivac heal slots, INTEGRATION.md sections 3 and 4). Kept as a race->config map
# so adding terran here later cannot silently pick the wrong agent.
#
# Expected from the 2026-08-25 30-minute round (protoss 46.7, zerg 56.1 steps/s
# steady-state on 10 cores): 59 minutes should reach roughly 150-200k steps.
# Both will hit the wall (TIMEOUT / exit 143) -- normal for a sample run.

mainPy="/home/zwang182/MARL/marl_p3_smacv2/src/main.py"
jobScript="/home/zwang182/MARL/marl_p3_smacv2/cc_smacv2_cpu.sh"

races=(protoss zerg)
seeds=(1 2 3)

declare -A algOf=(
  [protoss]="hpn_saleq_sem_wm_qmix"
  [zerg]="hpn_saleq_sem_wm_qmix"
)

t_max=1000000
useCuda=False       # ignored by cc_smacv2_cpu.sh, which forces CPU
usetb=False         # sample runs: sacred logs are enough
save_model=False    # sample runs: no checkpoints wanted
ckpt=""

for race in "${races[@]}"
do
  alg="${algOf[$race]}"
  sc2_env="sc2_v2_${race}"
  for s in "${seeds[@]}"
  do
    sbatch --time=0:59:00 --job-name=${race}_5v5_s${s} \
      ${jobScript} ${mainPy} ${alg} ${sc2_env} ${s} ${t_max} ${useCuda} ${usetb} ${save_model} "${ckpt}"
  done
done
