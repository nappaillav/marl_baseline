#!/bin/bash
# ===== FIR (written 2026-09-11) ========================================================
# SAMPLE runs of the three BASELINES on SMACv2 10v10, CPU-only, 1 hour each.
#   HPN-QMIX      protoss_10v10, terran_10v10, zerg_10v10    3 jobs
#   HPN-VDN       protoss_10v10, terran_10v10, zerg_10v10    3 jobs
#   DEEPSET-QMIX  terran_10v10 ONLY (as specified)           1 job
#                                                      total 7 jobs
#
# Usage:  bash cc_smacv2_10v10_baselines_launch.sh
# Check:  squeue -u $USER          (wait >= 60 s between checks)
# Dry run (prints the sbatch lines, submits nothing):
#         sbatch() { echo "DRY-RUN sbatch $@"; }; export -f sbatch; bash cc_smacv2_10v10_baselines_launch.sh
#
# THESE ARE SAMPLE RUNS AND ARE EXPECTED TO TIME OUT (exit 143 / TIMEOUT).
# t_max=5005000 at a 1:00:00 wall cannot finish: the measured Fir rate is ~119 env-steps/s
# for 5v5 protoss on 8 cores (2026-09-11, 12k-step sample), and 10v10 roughly doubles the
# entities per step, so expect on the order of 10^5 steps in the hour -- enough to confirm
# the algorithms build, roll out, learn and log, which is the point. Nothing here produces
# a comparable result row. For real runs, raise --time and re-measure from a run that has
# reached representative episode length.
#
# TERRAN CAVEAT -- none of these three baselines is heal-aware.
#   hpns_rnn (HPN-QMIX, HPN-VDN) gates its rescue head on `map_type == "MMM"`, and SMACv2
#   terran reports `terran_gen` (hpns_rnn_agent.py:72,150,195). deepset_rnn has no map_type
#   handling at all. So none of them asserts -- they run -- but a medivac's trailing action
#   slots are scored as "attack enemy j" rather than "heal ally j". The method's own terran
#   runs avoid this by using hpn_saleq_terran_sem_wm_qmix. Treat terran baseline numbers as
#   not-heal-aware, or state the caveat wherever they are reported.
#
# Seed 1 on every job. No collision with the method's 10v10 block (Nibi 1-5, Fir 31-35):
# those are `hpn_saleq*` runs and results are keyed by (name, map, seed), so a different
# `name` is a different row.
#
# Outputs: results/MSEMM_10v10_baselines/{tb_logs,sacred,models}/ -- deliberately separate
# from the method's results/MSEMM_10v10.
#   tensorboard --logdir results/MSEMM_10v10_baselines/tb_logs
# =======================================================================================

mainPy="/home/zwang182/MARL/marl_p3/marl_p3_smacv2/src/main.py"
jobScript="/home/zwang182/MARL/marl_p3/marl_p3_smacv2/cc_smacv2_10v10_baselines_cpu.sh"

seed=1
t_max=5005000
bufferSize=3500
walltime=1:00:00
memory=60G
useCuda=False       # ignored by the job script, which forces CPU
usetb=True
save_model=False    # sample runs: nothing worth checkpointing in an hour
saveInterval=2000000
ckpt=""

# alg -> the maps it runs on
declare -A mapsOf=(
  [hpn_qmix]="protoss terran zerg"
  [hpn_vdn]="protoss terran zerg"
  [deepset_qmix]="terran"
)

for alg in hpn_qmix hpn_vdn deepset_qmix
do
  for race in ${mapsOf[$alg]}
  do
    sc2_env="sc2_v2_${race}_10v10"
    sbatch --time=${walltime} --mem=${memory} --job-name=${alg}_${race}_10v10_s${seed} \
      ${jobScript} ${mainPy} ${alg} ${sc2_env} ${seed} ${t_max} ${useCuda} ${usetb} ${save_model} "${ckpt}" \
      ${saveInterval} ${bufferSize}
  done
done
