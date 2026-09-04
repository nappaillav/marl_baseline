#!/bin/bash
# DESIGN-SPACE STUDY launcher for hpn_saleq_sem_wm_qmix: single-variable variants on SMACv2
# protoss_5v5 and zerg_5v5, 5 seeds each, CPU-only, t_max=4005000, 71:55:00 wall per job.
# Submitted in BATCHES (the study is large: 30 variants = 300 jobs in total); edit `algs` per batch.
#   BATCH 1 (submitted 2026-08-27): axes A1 + A4  ->  5 algs x 2 races x 5 seeds = 50 jobs.
#   BATCH 2 (this file as committed): axes B1 + B3  ->  6 algs x 2 races x 5 seeds = 60 jobs (protoss + zerg only;
#            B1/B3 have no terran twin -- hpn_saleq_terran ignores saleq_norm variants and saleq_sem_temp).
#   BATCH 3 (SUBMITTED ON RORQUAL 2026-08-27): axes B2 + B5 -> 70 CPU jobs + 21 terran GPU jobs run there, NOT on Nibi.
#   BATCH 4 (SUBMITTED ON RORQUAL 2026-08-27): axis C1 -> 30 CPU jobs + 9 terran GPU jobs run there, NOT on Nibi.
#   BATCH 5 (SUBMITTED ON NARVAL 2026-08-27): axis C4 -> 30 CPU jobs + 9 terran GPU jobs run there, NOT on Nibi.
#   BATCH 6 (SUBMITTED ON NARVAL 2026-08-28): axes D1 + D2 + E3 -> 60 CPU jobs + 18 terran GPU jobs run there, NOT on Nibi.
#   => all 300 CPU + 72 terran GPU jobs of the study are submitted across the three clusters.
#   The repo is replicated on Rorqual and Narval; batches are split across clusters -- check the claim
#   markers below before moving anything into `algs`.
#   The terran counterpart (GPU, 3 seeds, config-only twins) is cc_smacv2_design_terran.sh.
#
# Usage:  bash cc_smacv2_design_runs.sh
# Check:  squeue -u $USER          (wait >= 60 s between checks)
# Dry run (prints the sbatch lines, submits nothing):
#         sbatch() { echo "DRY-RUN sbatch $@"; }; export -f sbatch; bash cc_smacv2_design_runs.sh
#
# What this study is: each variant changes ONE design choice / hyperparameter of the full method
# (configs in src/config/algs/design_space/, each YAML header states the diff + hypothesis). It
# complements the ablation study (cc_smacv2_test1.sh), which REMOVES whole components.
#
# Budget and metric: t_max=4005000 (vs 5005000 for the ablations, to cut wall time). Every row --
# these runs, the reused ablation rows and the reference -- is scored on the mean test win rate
# over 3.9M-4.0M env steps (the last ~10 test points at test_interval=10000), plus AUC over 0-4.0M.
# Reference row = the full method's protoss/zerg results from the LAB MACHINES (no Nibi re-run,
# per the user's decision). Reused ablation rows (already on Nibi, 5.005M budget, read in the
# same 3.9-4.0M window): ablation_arm_3 (AvgL1Norm; axis B1), ablation_arm_6 (wm_alpha=0; axis D2),
# ablation_arm_5a / _5b ([traj] only / constant mixer input; axis E3).
#
# Variants (value in the name; 'p' = decimal point):
#   A1  design_a1_wmk_{1,2,10}            imagination horizon wm_k            (full: 5)
#   A4  design_a4_traj_{16,64}            wm_traj_dim                         (full: 32)
#   B1  design_b1_norm_{ln,none,sem_noln} head normalization  [agent design_saleq]  (full: sem)
#   B2  design_b2_groups_{2,4,16,32}      saleq_sem_groups                    (full: 8)
#   B3  design_b3_temp_{0p1,0p5,2}        head SEM temperature [agent design_saleq] (full: 1)
#   B5  design_b5_unimix_{0p001,0p005,0p5} wm_unimix                          (full: 0.01)
#   C1  design_c1_lr_{1e-4,5e-4,1e-2}     lr                                  (full: 1e-3)
#   C4  design_c4_tdlambda_{0p1,0p3,0p9}  td_lambda                           (full: 0.6)
#   D1  design_d1_cwm_{0p25,4}            c_wm                                (full: 1)
#   D2  design_d2_alpha_{0p5,1p0}         wm_alpha                            (full: 0.8)
#   E3  design_e3_mix_{s_zs_traj,zs_traj} mixer input [learner design_wm_learner] (full: [s;traj])
# Likely degenerate by construction (kept as requested): design_c1_lr_1e-2, design_b5_unimix_0p5.
#
# terran is NOT here (every variant deploys hpn_saleq / design_saleq, which assert on terran_gen).
#
# Wall time / checkpoints: measured CPU rate is ~18-53 env-steps/s on 8 cores, so 4.005M steps is
# ~21-62 h; most runs fit in 71:55:00, the slowest may TIMEOUT (exit 143) once. Models are
# checkpointed every 1M steps (save_model=True + save_interval below; ~1M, 2M, 3M and the final 4.005M), so
# up to 1M steps (~5-15 h at the measured rate) are redone after a kill. Resume a timed-out run with
#     T_MAX=4005000 SAVE_INTERVAL=1000000 bash cc_smacv2_resume.sh <protoss|zerg> design_space/<name> <seed> [<seed> ...]
# (both must be given -- the resume script defaults to the ablation budget and a 250k interval). A resumed
# run gets a NEW sacred id and a NEW tb dir with the same <name>_<map>_seed<N>__ prefix; the replay
# buffer and target_mixer are not checkpointed. Analysis: stitch by (name, map, seed), later log wins
# from its start step, interpolate onto a 10k grid; read `seed` from config.json (run id != seed).
#
# Job names strip the `design_space/` subdir (arm="${alg##*/}"): cc_smacv2_resume.sh matches
# queued jobs by that stripped name, so an unstripped name would defeat its double-submit guard.
#
# Seeds: this repo takes `seed=N` (NOT env_args.seed -- INTEGRATION.md section 6).

mainPy="/home/zwang182/MARL/marl_p3_smacv2/src/main.py"
jobScript="/home/zwang182/MARL/marl_p3_smacv2/cc_smacv2_cpu_ablation_runs.sh"   # study-specific copy of cc_smacv2_cpu.sh (10th arg = save_model_interval)

races=(protoss zerg)
seeds=(1 2 3 4 5)
# ---- BATCH 1: A1 + A4 (SUBMITTED 2026-08-27 -- do not resubmit) ----
#  design_space/design_a1_wmk_1
#  design_space/design_a1_wmk_2
#  design_space/design_a1_wmk_10
#  design_space/design_a4_traj_16
#  design_space/design_a4_traj_64
# ---- BATCH 2: B1 + B3 ----
algs=(
  design_space/design_b1_norm_ln
  design_space/design_b1_norm_none
  design_space/design_b1_norm_sem_noln
  design_space/design_b3_temp_0p1
  design_space/design_b3_temp_0p5
  design_space/design_b3_temp_2
)
# ---- BATCH 3: B2 + B5  <-- SUBMITTED ON RORQUAL 2026-08-27 (70 CPU jobs + 21 terran GPU jobs), do not resubmit ----
#  design_space/design_b2_groups_{2,4,16,32}
#  design_space/design_b5_unimix_{0p001,0p005,0p5}
#  (all three races on Rorqual: protoss + zerg here, the B2/B5 terran twins via its cc_smacv2_design_terran.sh)
# ---- BATCH 4: C1     (SUBMITTED ON RORQUAL 2026-08-27, 30 CPU jobs + 9 terran GPU jobs) -- do not resubmit ----
#  design_space/design_c1_lr_{1e-4,5e-4,1e-2}
#  (all three races on Rorqual: protoss + zerg here, the C1 terran twins via its cc_smacv2_design_terran.sh)
# ---- BATCH 5: C4     (SUBMITTED ON NARVAL 2026-08-27, all three races) ----
#  design_space/design_c4_tdlambda_{0p1,0p3,0p9}
#  (30 CPU jobs protoss + zerg AND the 9 terran twin jobs -- all on Narval; do not resubmit anywhere)
# ---- BATCH 6: D1 + D2 + E3  (SUBMITTED ON NARVAL 2026-08-28, all three races) -- do not resubmit ----
#  design_space/design_d1_cwm_{0p25,4}
#  design_space/design_d2_alpha_{0p5,1p0}
#  design_space/design_e3_mix_{s_zs_traj,zs_traj}
#  (60 CPU jobs protoss + zerg AND the 18 terran twin jobs -- all on Narval)
# ---- ALL 30 VARIANTS ARE NOW SUBMITTED (Nibi: batches 1-2; Rorqual: 3-4; Narval: 5-6). Nothing left to launch. ----

t_max=4005000
walltime=71:55:00
useCuda=False           # ignored by the CPU job script, which forces CPU
usetb=True              # tensorboard curves per run (tb_logs/<name>_<map>_seed<N>__<ts>)
save_model=True         # required for resume after a wall-clock kill
save_interval=1000000   # checkpoint every 1M steps (user setting, 2026-08-27); <= 1M steps (~5-15 h) redone per kill
ckpt=""                 # fresh runs; resume via cc_smacv2_resume.sh (see header)

for race in "${races[@]}"
do
  sc2_env="sc2_v2_${race}"
  for alg in "${algs[@]}"
  do
    arm="${alg##*/}"   # job name without the design_space/ subdir
    for s in "${seeds[@]}"
    do
      sbatch --time=${walltime} --job-name=${arm}_${race}_s${s} \
        ${jobScript} ${mainPy} ${alg} ${sc2_env} ${s} ${t_max} ${useCuda} ${usetb} ${save_model} "${ckpt}" ${save_interval}
    done
  done
done
