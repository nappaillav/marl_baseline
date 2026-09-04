#!/bin/bash
# Reference-run launcher: the FULL METHOD only, all GPU jobs (MIG h100_1g.10gb via cc_reference_gpu_runs.sh).
#   SMACv2  protoss_5v5 + zerg_5v5 (hpn_saleq_sem_wm_qmix) and terran_5v5 (hpn_saleq_terran_sem_wm_qmix),
#           seeds 1-5, t_max=5005000, 11:59:00 wall                      3 races x 5 seeds = 15 jobs
#   SMACv1  6h_vs_8z + corridor (hpn_saleq_sem_wm_qmix, --env-config=sc2 + env_args.map_name),
#           seeds 1-3, t_max=2050000, 5:59:00 wall                        2 maps x 3 seeds =  6 jobs
#   INIT    near-initialisation checkpoints: the same 21 (config, env, map, seed) combinations with
#           t_max=1000, 0:20:00 wall (run_init=true below; false -> skip)                     21 jobs
#                                                                                      total 42 jobs
#
# Usage:  bash cc_reference_runs.sh
# Check:  squeue -u $USER          (wait >= 60 s between checks)
# Dry run (prints the sbatch lines, submits nothing):
#         sbatch() { echo "DRY-RUN sbatch $@"; }; export -f sbatch; bash cc_reference_runs.sh
#
# Why: these are the Full-method reference rows of the ablation study (cc_smacv2_test1.sh,
# cc_smacv2_ablation_terran.sh) and of the design-space study (cc_smacv2_design_runs.sh,
# cc_smacv2_design_terran.sh) -- neither launcher runs the full method -- and the Full-method
# checkpoints of the six world-model / representation analyses (notes/learned_world_model_and_reps_
# experiments/plans-for-proposals.md, I.2-I.4 and I.7; provenance rule I.6: the Full method must come
# from Nibi, never from lab-machine checkpoints). Flags per I.4: save_model=True, use_tensorboard=True,
# save_model_interval=250000 on protoss/zerg (20 checkpoints per run) and 200000 on terran (25, which
# lines up with the 400k grid of the terran arms); ~2 MB per checkpoint, ~50 MB per run.
#
# Wall time, SMACv2 (11:59:00): GPU jobs <= 12 h schedule on gpubase_bygpu_b2 and start within hours,
# whereas >= 48 h GPU jobs were estimated weeks out (sbatch --test-only, 2026-08-26). On this MIG slice
# the terran design runs took ~5.1 h (most; a few 7-8.4 h) for 4.005M steps (batch 1, 2026-08-27), so
# 5.005M should fit at ~6.5 h; but the 5.005M terran ablation arms took 4-11.3 h and 3 of 21 hit the
# wall (2026-08-26), so expect a small fraction of TIMEOUTs (exit 143). protoss/zerg have no GPU
# timing on Nibi yet [unverified]; check the first finished seed. RULE (job-script header): a
# timed-out run is RERUN FROM SCRATCH with a longer --time, never resumed -- resumed runs collapsed.
#
# SMACv1 (5:59:00): t_max=2050000 MUST be passed explicitly -- the alg config carries t_max 1.05M and
# is applied after the env config's 2.05M (main.py: env then alg update), so the default would be the
# validation budget. 2.05M at the terran GPU rate above is ~2.6 h; corridor (24 enemies, 400-step
# limit) is slower per step and its VRAM/RAM need on a 10 GB slice is UNVERIFIED -- see the corridor
# note in the job-script header (OOM -> resubmit with --gpus=h100_2g.20gb:1 on the sbatch line).
# Both maps have no healer units (map_type hydralisks / zealots), so the standard hpn_saleq agent
# applies unchanged. MMM2 is deliberately NOT included: hpn_saleq asserts on map_type MMM
# (src/modules/agents/hpn_saleq_agent.py:62 -- rescue/heal slots are unsupported by this agent).
# Each run saves 8 checkpoints (250k ... 2M) plus the final one at 2.05M.
#
# Near-initialisation checkpoints (init_* jobs, t_max=1000): plans 2 and 5 need untrained weights
# with the run's seed. run.py:301-302 saves when t_env >= t_max (or every save_model_interval), so a
# t_max=1000 run saves exactly once, at the end. The learner's first update needs 128 stored episodes
# (buffer.can_sample(batch_size), run.py:273), i.e. 16 parallel rollouts of 8, which 1000 env steps
# never reach (episodes are tens of steps long), so the saved weights are the seeded initialisation,
# untouched by training. These runs write sacred / models dirs with the SAME <name>_<map>_seed<N>__
# prefix as the full runs -- tell them apart by t_max=1000 in config.json and the single ~1000-step
# checkpoint dir. use_tensorboard is off for them (nothing to plot; no stub curves next to the real ones).
#
# Job names: full_<race>_s<N>, full_v1_<map>_s<N>, init_<race>_s<N>, init_v1_<map>_s<N>.
# Seeds: `seed=N` (NOT env_args.seed -- INTEGRATION.md section 6); sacred ids are arrival-ordered,
# so read `seed` from each run's config.json for analysis.

mainPy="/home/zwang182/MARL/marl_p3_smacv2/src/main.py"
jobScript="/home/zwang182/MARL/marl_p3_smacv2/cc_reference_gpu_runs.sh"   # copy of cc_smacv2_gpu_ablation_runs.sh + 11th arg map_name

useCuda=True            # ignored by the job script, which forces use_cuda=True
usetb=True              # tensorboard curves per full run (tb_logs/<name>_<map>_seed<N>__<ts>)
save_model=True         # checkpoints are the raw material of every probe (plans I.4)
ckpt=""                 # fresh runs only -- never resume (rule in the job-script header)

run_init=true           # also submit the 21 near-initialisation runs; false -> full runs only
init_t_max=1000
init_walltime=0:20:00
init_usetb=False

# ---- SMACv2: 3 races x 5 seeds, t_max 5.005M --------------------------------------------------
races=(protoss zerg terran)
v2_seeds=(1 2 3 4 5)
v2_t_max=5005000
v2_walltime=11:59:00

for race in "${races[@]}"
do
  sc2_env="sc2_v2_${race}"
  if [ "${race}" = terran ]; then
    alg=hpn_saleq_terran_sem_wm_qmix   # healer-aware head (medivacs) -- INTEGRATION.md sections 3-4
    save_interval=200000               # 25 checkpoints, aligned with the terran arms' 400k grid
  else
    alg=hpn_saleq_sem_wm_qmix
    save_interval=250000               # 20 checkpoints
  fi
  for s in "${v2_seeds[@]}"
  do
    sbatch --time=${v2_walltime} --job-name=full_${race}_s${s} \
      ${jobScript} ${mainPy} ${alg} ${sc2_env} ${s} ${v2_t_max} ${useCuda} ${usetb} ${save_model} "${ckpt}" ${save_interval}
    if [ "${run_init}" = true ]; then
      sbatch --time=${init_walltime} --job-name=init_${race}_s${s} \
        ${jobScript} ${mainPy} ${alg} ${sc2_env} ${s} ${init_t_max} ${useCuda} ${init_usetb} ${save_model} "${ckpt}" ${save_interval}
    fi
  done
done

# ---- SMACv1: 2 maps x 3 seeds, t_max 2.05M ----------------------------------------------------
v1_alg=hpn_saleq_sem_wm_qmix
v1_env=sc2
v1_maps=(6h_vs_8z corridor)
v1_seeds=(1 2 3)
v1_t_max=2050000        # MUST be passed (alg config carries 1.05M; see header)
v1_walltime=11:59:00    # raised from 5:59 on 2026-09-04 (user decision); still inside the <=12 h fast-start bucket
v1_save_interval=250000 # 8 checkpoints (250k ... 2M) + the final one at 2.05M

# Per-map resource override. corridor (6 zealots vs 24 zerglings, obs 156 / state 282, 400-step
# episodes) is the memory outlier: marl_project3 SETUP_ENVIRONMENT.md section 9 measured 22.6-24 GB
# VRAM and ~25 GB RAM for the full model on it, which does not fit the job script's 10 GB MIG slice
# and is tight on its 30G. Options given on the sbatch command line override the #SBATCH lines in
# the job script, so corridor gets a 3g.40gb slice and 40G RAM here without touching the script.
# 6h_vs_8z (6 vs 8, obs 78 / state 140) is 5v5-sized and keeps the defaults.
v1_resources_corridor="--gpus=h100_3g.40gb:1 --mem=40G"

for map in "${v1_maps[@]}"
do
  extra=""
  if [ "${map}" = "corridor" ]; then extra="${v1_resources_corridor}"; fi
  for s in "${v1_seeds[@]}"
  do
    sbatch --time=${v1_walltime} --job-name=full_v1_${map}_s${s} ${extra} \
      ${jobScript} ${mainPy} ${v1_alg} ${v1_env} ${s} ${v1_t_max} ${useCuda} ${usetb} ${save_model} "${ckpt}" ${v1_save_interval} ${map}
    if [ "${run_init}" = true ]; then
      sbatch --time=${init_walltime} --job-name=init_v1_${map}_s${s} ${extra} \
        ${jobScript} ${mainPy} ${v1_alg} ${v1_env} ${s} ${init_t_max} ${useCuda} ${init_usetb} ${save_model} "${ckpt}" ${v1_save_interval} ${map}
    fi
  done
done
