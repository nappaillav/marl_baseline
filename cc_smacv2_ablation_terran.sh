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
# Terran ablation launcher: the 7 ablation arms on SMACv2 terran_5v5 (10gen_terran), 3 seeds each,
# GPU jobs (MIG 1g.10gb via cc_smacv2_gpu_ablation_runs.sh), t_max=5005000, 11:59:00 wall per job.
#   7 algs x 1 race x 3 seeds = 21 jobs.
#
# Usage:  bash cc_smacv2_ablation_terran.sh
# Check:  squeue -u $USER          (wait >= 60 s between checks)
# Dry run (prints the sbatch lines, submits nothing):
#         sbatch() { echo "DRY-RUN sbatch $@"; }; export -f sbatch; bash cc_smacv2_ablation_terran.sh
#
# Configs live in src/config/algs/terran_ablations/ and are passed as
# --config=terran_ablations/<arm> (main.py joins the path under config/algs). Each carries
# name: <arm>_terran. Arms whose protoss/zerg version deploys hpn_saleq use hpn_saleq_terran here
# (medivac heal slots -- INTEGRATION.md sections 3-4):
#   ablation_arm_1   1B   PLACEHOLDER: deepset_vdn (flat_saleq asserts on terran_gen)   agent deepset_rnn
#   ablation_arm_2   2a   vanilla HPN read-out + WM  (runs; heals scored via enemy read-out) agent hpns_rnn
#   ablation_arm_3   3a   head SEM -> AvgL1Norm, WM kept                                 agent hpn_saleq_terran
#   ablation_arm_4   4a   no world model                                                 agent hpn_saleq_terran
#   ablation_arm_5a  5a   mixer sees traj only                                           agent hpn_saleq_terran
#   ablation_arm_5b  5d-i mixer sees a constant                                          agent hpn_saleq_terran
#   ablation_arm_6   6a   KL balancing off (wm_alpha=0)                                  agent hpn_saleq_terran
#
# Wall time: a GPU terran_5v5 run needs ~14 h for 5.005M steps (measured by another session,
# 2026-08-26), and GPU jobs <= 12 h schedule on gpubase_bygpu_b2 and start within hours, whereas
# >= 48 h jobs were estimated weeks out (sbatch --test-only, 2026-08-26). So each run is split:
# this launcher submits the first 11:59:00 chunk; every run WILL hit the wall (TIMEOUT / exit 143)
# and must be resumed once from its latest checkpoint (models are saved every 400000 steps ->
# at most ~1 h of progress lost). To resume, pass the run's models/<...>/<unique_token> dir as
# `ckpt` (load_step=0 picks the latest step); the replay buffer and target_mixer are not saved.
#
# Seeds: `seed=N` (NOT env_args.seed -- INTEGRATION.md section 6); read `seed` from each run's
# config.json for analysis (sacred ids are arrival-ordered, not seeds).

mainPy="/home/zwang182/MARL/marl_p3/marl_p3_smacv2/src/main.py"
jobScript="/home/zwang182/MARL/marl_p3/marl_p3_smacv2/cc_smacv2_gpu_ablation_runs.sh"   # study-specific copy of cc_smacv2_gpu.sh

race=terran
sc2_env="sc2_v2_${race}"
seeds=(1 2 3)
algs=(
  terran_ablations/ablation_arm_1
  terran_ablations/ablation_arm_2
  terran_ablations/ablation_arm_3
  terran_ablations/ablation_arm_4
  terran_ablations/ablation_arm_5a
  terran_ablations/ablation_arm_5b
  terran_ablations/ablation_arm_6
)

t_max=5005000
walltime=11:59:00
useCuda=True            # ignored by cc_smacv2_gpu.sh, which forces use_cuda=True
usetb=True              # tensorboard curves per run
save_model=True         # required for the resume after the wall-clock kill
save_interval=400000    # checkpoint every 400k steps (~1 h at the measured GPU rate)
ckpt=""                 # fresh runs; set per run to resume

for alg in "${algs[@]}"
do
  arm="${alg##*/}"
  for s in "${seeds[@]}"
  do
    sbatch --time=${walltime} --job-name=${arm}_${race}_s${s} \
      ${jobScript} ${mainPy} ${alg} ${sc2_env} ${s} ${t_max} ${useCuda} ${usetb} ${save_model} "${ckpt}" ${save_interval}
  done
done
