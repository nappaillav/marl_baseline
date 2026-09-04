#!/bin/bash
# Additional Full-method reference runs, decided 2026-09-04 (evening) after the first batch finished. All GPU jobs
# through cc_reference_gpu_runs.sh, same flags as cc_reference_runs.sh:
#
#   1. 10gen_zerg seed 4 -- RERUN FROM SCRATCH of job 21116581, which hit the 11:59 wall at t_env 4,782,302 because it
#      ran at ~110 steps/s (the other zerg seeds ran at ~310 and finished in ~4.5 h). Longer wall per the rule in the
#      job-script header (never resume). The partial run's artefacts (models token
#      hpn_saleq_sem_wm_qmix_10gen_zerg_seed4__2026-09-04_02-19-04, tb dir of the same name, sacred id 8) are to be
#      moved to results_incomplete/ so that no probe globs them by the shared <name>_<map>_seed4__ prefix.
#   2. SMACv1 seeds 4 and 5 on 6h_vs_8z and corridor: the SMACv1 panel goes from 3 to 5 seeds per map.
#      corridor keeps its larger slice (see cc_reference_runs.sh for why).
#
#   1 + 4 = 5 jobs.
#
# Usage:  bash cc_reference_runs_extra.sh
# Dry run (prints the sbatch lines, submits nothing):
#         sbatch() { echo "DRY-RUN sbatch $@"; }; export -f sbatch; bash cc_reference_runs_extra.sh
# Afterwards: run cc_reference_init_login.sh once more on the login node to add the four new SMACv1
# near-initialisation checkpoints (it skips the 21 that exist).

mainPy="/home/zwang182/MARL/marl_p3_smacv2/src/main.py"
jobScript="/home/zwang182/MARL/marl_p3_smacv2/cc_reference_gpu_runs.sh"

useCuda=True
usetb=True
save_model=True
ckpt=""

# ---- 1. zerg seed 4, from scratch, longer wall ----
sbatch --time=23:55:00 --job-name=full_zerg_s4 \
  ${jobScript} ${mainPy} hpn_saleq_sem_wm_qmix sc2_v2_zerg 4 5005000 ${useCuda} ${usetb} ${save_model} "${ckpt}" 250000

# ---- 2. SMACv1 seeds 4-5 ----
v1_alg=hpn_saleq_sem_wm_qmix
v1_env=sc2
v1_t_max=2050000
v1_walltime=11:59:00
v1_save_interval=250000
v1_resources_corridor="--gpus=h100_3g.40gb:1 --mem=40G"

for map in 6h_vs_8z corridor; do
  extra=""
  if [ "${map}" = "corridor" ]; then extra="${v1_resources_corridor}"; fi
  for s in 4 5; do
    sbatch --time=${v1_walltime} --job-name=full_v1_${map}_s${s} ${extra} \
      ${jobScript} ${mainPy} ${v1_alg} ${v1_env} ${s} ${v1_t_max} ${useCuda} ${usetb} ${save_model} "${ckpt}" ${v1_save_interval} ${map}
  done
done
