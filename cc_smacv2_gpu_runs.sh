#!/bin/bash
# ===== FIR VERSION (transferred from Nibi 2026-09-11, ported same day) =================
# Source: marl_p3_smacv2_10v10_transfer/MANIFEST.txt, on top of base commit fc853a3.
# Retargeted for Fir: repo root /home/zwang182/MARL/marl_p3/marl_p3_smacv2, SC2PATH
# /home/zwang182/MARL/sc2/3rdparty/StarCraftII, GPU gres nvidia_h100_80gb_hbm3_* (Fir's
# canonical MIG names). --account and the module stack are unchanged: def-dpmeger,
# def-dpmeger_cpu and def-dpmeger_gpu are all valid associations here.
#
# SC2 VERSION: RESOLVED 2026-09-11. Fir originally shipped 4.6.2 (build 69232), which is NOT
# comparable with the Nibi/lab-machine 4.10 runs. SC2PATH was rebuilt with 4.10.0 (build
# 75689) that day -- the same major.minor Nibi uses -- and the 30 SMAC maps reinstalled from
# this repo's vendored copy. Verified: SMACv1 6h_vs_8z and SMACv2 protoss/protoss_10v10/
# zerg_10v10/terran_10v10 all run on Base75689, exit 0. The 4.6.2 tree has been deleted, so
# 4.10 is the only StarCraft II on Fir; reinstall from SC2.4.10.zip if it is ever needed again.
# =======================================================================================
# Our method on the three SMACv2 10v10 scenarios, 5 seeds each, GPU.
#   3 maps x 5 seeds = 15 jobs, nvidia_h100_80gb_hbm3_3g.40gb MIG slice, 23:59:00 wall each.
#
# Usage:  bash cc_smacv2_gpu_runs.sh
# Check:  squeue -u $USER
#
# Baselines (HPN-QMIX / HPN-VDN) are deliberately NOT launched here. They were smoke-tested
# on all three 10v10 maps and work, but note that on terran neither is heal-aware: agent
# `hpns_rnn` gates its rescue head on `map_type == "MMM"` and SMACv2 reports `terran_gen`,
# so a medivac's trailing slots are scored as "attack enemy j" (verified: 98,631 params on
# terran_gen vs 102,856 under MMM). To add them, extend `algs` below -- and keep terran on
# a heal-aware agent or state the caveat.
#
# terran uses a DIFFERENT config: 10gen_terran generates medivacs, and plain `hpn_saleq`
# asserts on terran_gen at any team size (hpn_saleq_agent.py:68). See INTEGRATION.md 3-4.
#
# buffer_size=4000 is passed here rather than edited into hpn_saleq_sem_wm_qmix.yaml, so the
# 5v5 ablation/design/reference runs that share that config keep their 5000 and stay
# comparable. At 10v10 the buffer preallocates 15.5 GB at 4000 vs 19.4 GB at 5000.
#
# WALL TIME -- 23:59:00 is Nibi's default and is KEPT here by decision (2026-09-11), but it is
# unverified at 10v10 on either cluster: Nibi's own 10v10 jobs (21391721-21391750) were still
# PENDING when this was transferred, so no 10v10 GPU rate exists anywhere yet.
# What is measured: Fir CPU 10v10 BASELINES on 2026-09-11 (7 one-hour jobs, 8 cores) ran
# 11-15 env-steps/s steady-state for the hpn_* agents -- 5.005M steps would be 94-124 h there.
# For scale, Nibi 5v5 GPU reference runs ran ~110-310 steps/s (cc_reference_runs_extra.sh).
# The method on a 3g.40gb slice at 10v10 sits somewhere between and is UNMEASURED.
# This matters because of the never-resume rule in cc_smacv2_gpu.sh / cc_reference_gpu_runs.sh:
# a TIMEOUT is RERUN FROM SCRATCH with a longer --time, so a short wall costs the whole 24 h
# x 15 jobs, not just the tail. save_model here is for the probes, NOT for resuming.
# Cheap insurance: run ONE seed on one map first and read its steady rate before the other 14.
#
# SEED BOOKKEEPING (MANIFEST.txt asks that each cluster record its block):
#   Nibi  seeds 1-5    all three maps (jobs 21391721-21391750, submitted 2026-09-08).
#   Fir   seeds 31-35  all three maps (this launcher).
# Disjoint on purpose: both clusters now run StarCraft II 4.10, so results merge by
# (name, map, seed) and a shared seed would mean two different runs with one identity.
# Do not change these to 1-5 without first confirming the Nibi jobs are gone.
#
# All outputs go to results/MSEMM_10v10/{tb_logs,sacred,models}/.
#   tensorboard --logdir results/MSEMM_10v10/tb_logs

mainPy="/home/zwang182/MARL/marl_p3/marl_p3_smacv2/src/main.py"
jobScript="/home/zwang182/MARL/marl_p3/marl_p3_smacv2/cc_smacv2_gpu.sh"

races=(protoss zerg terran)
seeds=(31 32 33 34 35)          # Fir's block -- see SEED BOOKKEEPING above

declare -A algOf=(
  [protoss]="hpn_saleq_sem_wm_qmix"
  [zerg]="hpn_saleq_sem_wm_qmix"
  [terran]="hpn_saleq_terran_sem_wm_qmix"
)

t_max=5005000
saveInterval=800000
bufferSize=4000
useCuda=True        # ignored by cc_smacv2_gpu.sh, which forces use_cuda=True
usetb=True
save_model=True
ckpt=""

# Per-race GPU override (decision 2026-09-11): zerg runs on a 2g.20gb slice, protoss and
# terran keep the job script's 3g.40gb. Options on the sbatch command line override the
# #SBATCH lines in the job script, so this needs no second copy of cc_smacv2_gpu.sh --
# the same idiom cc_reference_runs.sh uses for corridor.
declare -A gpuOf=(
  [protoss]=""                                              # inherit 3g.40gb from the job script
  [terran]=""                                               # inherit 3g.40gb from the job script
  [zerg]="--gpus=nvidia_h100_80gb_hbm3_2g.20gb:1"
)

for race in "${races[@]}"
do
  alg="${algOf[$race]}"
  sc2_env="sc2_v2_${race}_10v10"
  gpu="${gpuOf[$race]}"
  for s in "${seeds[@]}"
  do
    sbatch --time=23:59:00 ${gpu} --job-name=${race}_10v10_s${s} \
      ${jobScript} ${mainPy} ${alg} ${sc2_env} ${s} ${t_max} ${useCuda} ${usetb} ${save_model} "${ckpt}" \
      ${saveInterval} ${bufferSize}
  done
done
