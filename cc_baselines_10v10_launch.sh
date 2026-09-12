#!/bin/bash
# Launcher: four SMACv2 10v10 BASELINES x three maps x four seeds = 48 GPU jobs.
#
#   HPN-QMIX      hpn_qmix       protoss_10v10, terran_10v10, zerg_10v10
#   HPN-VDN       hpn_vdn        protoss_10v10, terran_10v10, zerg_10v10
#   QMIX          qmix           protoss_10v10, terran_10v10, zerg_10v10
#   DEEPSET-QMIX  deepset_qmix   protoss_10v10, terran_10v10, zerg_10v10
#
# Usage:  bash cc_baselines_10v10_launch.sh
# Dry run (prints the sbatch lines, submits nothing) -- DO THIS FIRST:
#         sbatch() { echo "DRY-RUN sbatch $@"; }; export -f sbatch; bash cc_baselines_10v10_launch.sh
#
# Paths are derived from this script's own location, so the repo can live anywhere.
#
# TERRAN CAVEAT: none of these four baselines is heal-aware. hpns_rnn (HPN-QMIX, HPN-VDN)
# gates its rescue head on `map_type == "MMM"` and SMACv2 terran reports `terran_gen`;
# n_rnn (QMIX) and deepset_rnn (DEEPSET-QMIX) have no map_type handling at all. Nothing
# asserts -- they run -- but a medivac's trailing action slots are scored as "attack enemy j"
# rather than "heal ally j". Report terran baseline numbers with that caveat.

set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ===== EDIT THIS BLOCK FOR YOUR CLUSTER ================================================
ACCOUNT="def-dpmeger"          # your Slurm allocation
WALLTIME="23:59:00"
MEMORY="50G"
CPUS=8                         # one per SC2 worker (batch_size_run=8)

# GPU request per map. What matters is the VRAM figure in the comment -- translate it to
# whatever gres names your cluster uses (these are Fir's MIG names).
#   zerg   20 GB: directly evidenced. The FULL METHOD (hpn_saleq + world model, strictly
#          heavier than any baseline here) ran zerg_10v10 on a 20 GB slice for hours with
#          no OOM (jobs 59382384-7, 2026-09-11), at ep_length_mean ~33-42.
#   others 40 GB: NOT measured. protoss and terran ran ~1.8x longer episodes than zerg in
#          the CPU samples (~77 and ~71 vs ~44) and VRAM scales with episode length, so the
#          margin is kept. If queue time hurts, try ONE seed at 20 GB before moving them all.
declare -A gpuOf=(
  [protoss]="--gpus=nvidia_h100_80gb_hbm3_3g.40gb:1"   # 40 GB
  [terran]="--gpus=nvidia_h100_80gb_hbm3_3g.40gb:1"    # 40 GB
  [zerg]="--gpus=nvidia_h100_80gb_hbm3_2g.20gb:1"      # 20 GB
)
# =======================================================================================

algs=(hpn_qmix hpn_vdn qmix deepset_qmix)
races=(protoss terran zerg)
seeds=(1 2 3 4)

t_max=5005000
bufferSize=4000                # matches the reference 10v10 runs, so the comparison is clean.
                               # Host-RAM cost: the replay buffer preallocates ~14.4 GiB at
                               # 4000 (9653 B/transition x 401 steps). Measured peak RSS for
                               # these agents at 10v10 was 24.6-42.5 GB, so 50G fits with
                               # room; drop to 3500 (~12.6 GiB) if anything is OOM-killed.
saveInterval=800000            # 6 checkpoints over the run

resultsPath="${REPO}/results/baselines_10v10"
mainPy="${REPO}/src/main.py"
jobScript="${REPO}/cc_baselines_10v10_gpu.sh"

# WALL TIME, where 23:59:00 comes from: the full method on a 20 GB slice at zerg_10v10 ran
# 68.6-151.5 env-steps/s across four seeds (2026-09-11), i.e. 9-20 h for 5.005M steps. These
# baselines are lighter than that method (no world model), and on CPU the light agents were
# far faster than the HPN ones (deepset ~155 steps/s vs hpn ~11-15), so the HPN baselines set
# the pace here. 23:59:00 covers the slowest observed case with ~4 h of margin. QMIX and
# DEEPSET-QMIX will very likely finish in well under half of it -- if your scheduler favours
# short jobs, giving those two a 11:59:00 wall will start them sooner.

for alg in "${algs[@]}"; do
  for race in "${races[@]}"; do
    for s in "${seeds[@]}"; do
      sbatch --account="${ACCOUNT}" --time="${WALLTIME}" --mem="${MEMORY}" \
        --cpus-per-task="${CPUS}" ${gpuOf[$race]} \
        --job-name="${alg}_${race}_10v10_s${s}" \
        "${jobScript}" "${mainPy}" "${alg}" "sc2_v2_${race}_10v10" "${s}" \
        "${t_max}" "${bufferSize}" "${saveInterval}" "${resultsPath}"
    done
  done
done
