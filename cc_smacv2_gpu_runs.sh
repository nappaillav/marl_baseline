#!/bin/bash
# GPU sample runs: one job per SMACv2 map (terran, protoss, zerg), seed 1, 0:30:00 wall.
# 3 jobs total. Purpose: measure whether a GPU is worth its queue wait for this workload.
#
# Usage:  bash cc_smacv2_gpu_runs.sh
# Check:  squeue -u $USER
#
# CPU baselines to compare against (2026-08-25/26, 8 cores, steady-state env-steps/s):
#     terran 18.3 | protoss 39.4 | zerg 31.4
# Read the achieved rate the same way: (last t_env - first t_env) / elapsed between the
# first and last "t_env:" lines in the job's slurm-<jobid>.out.
#
# terran REQUIRES hpn_saleq_terran_sem_wm_qmix (medivac heal slots, INTEGRATION.md 3-4).

mainPy="/home/zwang182/MARL/marl_p3_smacv2/src/main.py"
jobScript="/home/zwang182/MARL/marl_p3_smacv2/cc_smacv2_gpu.sh"

races=(terran protoss zerg)
seeds=(1)

declare -A algOf=(
  [terran]="hpn_saleq_terran_sem_wm_qmix"
  [protoss]="hpn_saleq_sem_wm_qmix"
  [zerg]="hpn_saleq_sem_wm_qmix"
)

t_max=1000000
useCuda=True        # ignored by cc_smacv2_gpu.sh, which forces use_cuda=True
usetb=False         # sample runs: sacred logs are enough
save_model=False    # sample runs: no checkpoints wanted
ckpt=""

for race in "${races[@]}"
do
  alg="${algOf[$race]}"
  sc2_env="sc2_v2_${race}"
  for s in "${seeds[@]}"
  do
    sbatch --time=0:30:00 --job-name=gpu_${race}_5v5_s${s} \
      ${jobScript} ${mainPy} ${alg} ${sc2_env} ${s} ${t_max} ${useCuda} ${usetb} ${save_model} "${ckpt}"
  done
done
