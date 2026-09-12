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
# DESIGN-SPACE STUDY -- TERRAN launcher: config-only terran twins of the design variants on SMACv2
# terran_5v5 (10gen_terran), 3 seeds each, GPU jobs (MIG 1g.10gb via cc_smacv2_gpu_ablation_runs.sh),
# t_max=4005000, 11:59:00 wall per job. Submitted in BATCHES (the study is large); edit `algs` per batch.
#
#   BATCH 1 (submitted on Nibi 2026-08-27): axes A1 + A4  ->  5 algs x 1 race x 3 seeds = 15 jobs.
#   BATCHES 3 + 4 (SUBMITTED ON RORQUAL 2026-08-27): B2 + B5 + C1 terran twins -> 30 GPU jobs run there, NOT on Nibi.
#   BATCH 5 (SUBMITTED ON NARVAL 2026-08-27): axis C4 terran twins -> 9 GPU jobs run there, NOT on Nibi.
#   BATCH 6 (SUBMITTED ON NARVAL 2026-08-28): D1 + D2 + E3 terran twins -> 18 GPU jobs run there, NOT on Nibi.
#   The repo is replicated on Rorqual and Narval; batches are split across clusters -- check the claim
#   markers below (and in cc_smacv2_design_runs.sh) before moving anything into `algs`.
#
# Usage:  bash cc_smacv2_design_terran.sh
# Check:  squeue -u $USER          (wait >= 60 s between checks)
# Dry run (prints the sbatch lines, submits nothing):
#         sbatch() { echo "DRY-RUN sbatch $@"; }; export -f sbatch; bash cc_smacv2_design_terran.sh
#
# Configs live in src/config/algs/terran_design_space/ and are passed as
# --config=terran_design_space/<name> (main.py joins the path under config/algs). Each is its
# design_space/ source with `agent: hpn_saleq_terran` (healer-aware head, required on terran_gen) and
# name: terran_<source>; the variant's key is honoured unchanged on the terran path (verified 2026-08-27:
# static + startup checks on all 24, SC2 end-to-end smokes on the most extreme value of each axis).
# The protoss/zerg counterpart of this launcher is cc_smacv2_design_runs.sh (CPU, 5 seeds).
#
# 24 twins exist (B1 and B3 have NO terran twin: hpn_saleq_terran hard-codes a two-way sem/AvgL1Norm
# branch and never reads saleq_sem_temp, so config-only twins would silently duplicate ablation_arm_3
# / the full terran method). Axes and values ('p' = decimal point):
#   A1  terran_design_a1_wmk_{1,2,10}              wm_k            (full: 5)      <- batch 1
#   A4  terran_design_a4_traj_{16,64}              wm_traj_dim     (full: 32)     <- batch 1
#   B2  terran_design_b2_groups_{2,4,16,32}        saleq_sem_groups (full: 8)
#   B5  terran_design_b5_unimix_{0p001,0p005,0p5}  wm_unimix       (full: 0.01)
#   C1  terran_design_c1_lr_{1e-4,5e-4,1e-2}       lr              (full: 1e-3)
#   C4  terran_design_c4_tdlambda_{0p1,0p3,0p9}    td_lambda       (full: 0.6)
#   D1  terran_design_d1_cwm_{0p25,4}              c_wm            (full: 1)
#   D2  terran_design_d2_alpha_{0p5,1p0}           wm_alpha        (full: 0.8)
#   E3  terran_design_e3_mix_{s_zs_traj,zs_traj}   mixer input [learner design_wm_learner] (full: [s;traj])
#
# Budget and metric: t_max=4005000; every row is scored on the mean test win rate over 3.9M-4.0M env
# steps (plus AUC over 0-4.0M). Reference row = the full terran method's lab-machine results (no Nibi
# re-run). Reused terran ablation rows (5.005M budget, read in the same window):
# ablation_arm_6_terran (wm_alpha=0; axis D2), ablation_arm_5a/_5b_terran (mixer input; axis E3).
#
# Wall time: a GPU terran_5v5 run needs ~14 h for 5.005M steps (measured 2026-08-26), i.e. ~11.2 h for
# 4.005M -- borderline against 11:59:00, so expect a fraction of runs to TIMEOUT (exit 143) and need ONE
# resume from the latest 800k checkpoint (~0.8M, 1.6M, 2.4M, 3.2M, final 4.005M; up to ~2 h redone). GPU jobs
# <= 12 h schedule on gpubase_bygpu_b2 and start within
# hours; >= 48 h GPU jobs were estimated weeks out (sbatch --test-only, 2026-08-26). Resume with
#     T_MAX=4005000 SAVE_INTERVAL=800000 bash cc_smacv2_resume.sh terran terran_design_space/<name> <seed> [<seed> ...]
# (both must be given -- the resume script defaults to the ablation budget and a 400k interval; race=terran selects this
# GPU job script). A resumed run gets a NEW sacred id and a NEW tb dir with the same
# <name>_<map>_seed<N>__ prefix; the replay buffer and target_mixer are not checkpointed. Analysis: stitch
# by (name, map, seed), later log wins from its start step; read `seed` from config.json (run id != seed).
#
# Job names strip the subdir (arm="${alg##*/}"): cc_smacv2_resume.sh matches queued jobs by that
# stripped name, so an unstripped name would defeat its double-submit guard.
#
# Seeds: `seed=N` (NOT env_args.seed -- INTEGRATION.md section 6).

mainPy="/home/zwang182/MARL/marl_p3/marl_p3_smacv2/src/main.py"
jobScript="/home/zwang182/MARL/marl_p3/marl_p3_smacv2/cc_smacv2_gpu_ablation_runs.sh"   # study-specific copy of cc_smacv2_gpu.sh (10th arg = save_model_interval)

race=terran
sc2_env="sc2_v2_${race}"
seeds=(1 2 3)

# ---- BATCH 1: A1 + A4 ----
algs=(
  terran_design_space/terran_design_a1_wmk_1
  terran_design_space/terran_design_a1_wmk_2
  terran_design_space/terran_design_a1_wmk_10
  terran_design_space/terran_design_a4_traj_16
  terran_design_space/terran_design_a4_traj_64
)
# ---- BATCH 5: C4     (SUBMITTED ON NARVAL 2026-08-27, 9 terran GPU jobs) -- do not resubmit ----
#  terran_design_space/terran_design_c4_tdlambda_{0p1,0p3,0p9}
# ---- BATCHES 3 + 4: B2 + B5 + C1 terran twins (SUBMITTED ON RORQUAL 2026-08-27, 30 GPU jobs) -- do not resubmit ----
#  terran_design_space/terran_design_b2_groups_{2,4,16,32}
#  terran_design_space/terran_design_b5_unimix_{0p001,0p005,0p5}
#  terran_design_space/terran_design_c1_lr_{1e-4,5e-4,1e-2}
# ---- BATCH 6: D1 + D2 + E3 terran twins (SUBMITTED ON NARVAL 2026-08-28, 18 GPU jobs) -- do not resubmit ----
#  terran_design_space/terran_design_d1_cwm_{0p25,4}
#  terran_design_space/terran_design_d2_alpha_{0p5,1p0}
#  terran_design_space/terran_design_e3_mix_{s_zs_traj,zs_traj}
# ---- ALL 24 TERRAN TWINS ARE NOW SUBMITTED (Nibi: A1/A4; Rorqual: B2/B5/C1; Narval: C4/D1/D2/E3). Nothing left to launch. ----

t_max=4005000
walltime=11:59:00
useCuda=True            # ignored by the GPU job script, which forces use_cuda=True
usetb=True              # tensorboard curves per run (tb_logs/<name>_<map>_seed<N>__<ts>)
save_model=True         # required for the resume after a wall-clock kill
save_interval=800000    # checkpoint every 800k steps (user setting, 2026-08-27; ~2 h at the measured GPU rate)
ckpt=""                 # fresh runs; resume via cc_smacv2_resume.sh (see header)

for alg in "${algs[@]}"
do
  arm="${alg##*/}"   # job name without the terran_design_space/ subdir
  for s in "${seeds[@]}"
  do
    sbatch --time=${walltime} --job-name=${arm}_${race}_s${s} \
      ${jobScript} ${mainPy} ${alg} ${sc2_env} ${s} ${t_max} ${useCuda} ${usetb} ${save_model} "${ckpt}" ${save_interval}
  done
done
