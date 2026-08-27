#!/bin/bash
# Ablation study launcher: the 7 ablation arms on SMACv2 protoss_5v5 and zerg_5v5,
# 5 seeds each, CPU-only, t_max=5005000, 71:55:00 wall per job.
#   7 algs x 2 races x 5 seeds = 70 jobs.
# The full method (hpn_saleq_sem_wm_qmix) is NOT run here: its 5-seed protoss/zerg results
# already exist from the lab machines. Add it back to `algs` if a Nibi reference row is needed.
#
# Usage:  bash cc_smacv2_test1.sh
# Check:  squeue -u $USER          (wait >= 60 s between checks)
# Dry run (prints the sbatch lines, submits nothing):
#         sbatch() { echo "DRY-RUN sbatch $@"; }; export -f sbatch; bash cc_smacv2_test1.sh
#
# Arms (single-variable diffs vs hpn_saleq_sem_wm_qmix; details in each YAML header):
#   ablation_arm_1         1B   flat (non-HPN) trunk + action-in head       agent flat_saleq
#   ablation_arm_2         2a   vanilla HPN read-out, no action-in head      agent hpns_rnn
#   ablation_arm_3         3a   head SEM -> AvgL1Norm, WM kept
#   ablation_arm_4         4a   no world model (= hpn_saleq_sem_qmix)        learner nq_learner
#   ablation_arm_5a        5a   mixer sees traj only (no global state)       learner ablation_mixstate_learner
#   ablation_arm_5b        5d-i mixer sees a constant (no state, no traj)    learner ablation_mixstate_learner
#   ablation_arm_6         6a   KL balancing off (wm_alpha=0)
#
# terran is NOT here: every arm above deploys an hpn_saleq-derived agent that asserts on
# terran_gen (arm 2's hpns_rnn runs there but silently mis-scores medivac heals). The full
# method on terran needs hpn_saleq_terran_sem_wm_qmix -- see INTEGRATION.md sections 3-4.
#
# Wall time / checkpoints: measured CPU rate on 10 cores is ~18-53 env-steps/s (2026-08-26
# 59-minute samples), so 5.005M steps is ~1.7-3.2 days and slow seeds may hit the 71:55:00
# wall (TIMEOUT / exit 143). Models are therefore checkpointed every 250k steps
# (save_model=True + save_model_interval below); a timed-out run is resumed by passing its
# models/<...>/<unique_token> dir as checkpoint_path (load_step=0 picks the latest step).
# NOTE: the replay buffer and target_mixer are not checkpointed -- a resumed run refills the
# buffer from scratch at the restored t_env (epsilon already annealed).
#
# Seeds: this repo takes `seed=N` (NOT env_args.seed -- INTEGRATION.md section 6). Sacred run
# ids are handed out in arrival order, so for analysis read `seed` from each run's config.json.
# Concurrent same-second starts are safe: tb dirs carry name+map+seed, sacred retries ids.

mainPy="/home/zwang182/MARL/marl_p3_smacv2/src/main.py"
jobScript="/home/zwang182/MARL/marl_p3_smacv2/cc_smacv2_cpu_ablation_runs.sh"   # study-specific copy of cc_smacv2_cpu.sh

races=(protoss zerg)
seeds=(1 2 3 4 5)
algs=(
  ablation_arm_1
  ablation_arm_2
  ablation_arm_3
  ablation_arm_4
  ablation_arm_5a
  ablation_arm_5b
  ablation_arm_6
)

t_max=5005000
walltime=71:55:00
useCuda=False           # ignored by cc_smacv2_cpu.sh, which forces CPU
usetb=True              # tensorboard curves per run (tb_logs/<name>_<map>_seed<N>__<ts>)
save_model=True         # required for resume after a wall-clock kill
save_interval=1000000    # <= ~2 h of progress lost per kill at the measured rates
ckpt=""                 # fresh runs; set per run to resume (see header)

for race in "${races[@]}"
do
  sc2_env="sc2_v2_${race}"
  for alg in "${algs[@]}"
  do
    for s in "${seeds[@]}"
    do
      sbatch --time=${walltime} --job-name=${alg}_${race}_s${s} \
        ${jobScript} ${mainPy} ${alg} ${sc2_env} ${s} ${t_max} ${useCuda} ${usetb} ${save_model} "${ckpt}" ${save_interval}
    done
  done
done
