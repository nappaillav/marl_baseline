#!/bin/bash
# SMACv2 10gen_terran — 10 independent seeds for our method, run TWO AT A TIME.
#
#     source /usr/local/data/zwang182/envs/marl_project3/bin/activate
#     cd /usr/local/data/zwang182/MARL/pymarl3_related_code
#     bash testrun2.sh 2>&1 | tee results/sweep_terran_$(date +%F_%H-%M-%S).log
#
# TERRAN NEEDS ITS OWN AGENT. `hpn_saleq_terran_sem_wm_qmix` deploys `hpn_saleq_terran`, which handles
# medivac heal actions; the plain `hpn_saleq_sem_wm_qmix` refuses terran_gen with a loud assert
# (INTEGRATION.md §3/§4). Do not "simplify" this to the other config.
#
# WHY TWO IN PARALLEL — measured on Hawaii 2026-08-11, not assumed:
#     solo                869 env-steps/s
#     2 concurrent        685 + 628 = 1314 steps/s aggregate = 1.51x solo (~34 % wall-clock saving)
#     per run             12.74 GB CPU PSS, 3.27 GB VRAM  ->  x2 = 25.5 GB / 6.5 GB
# That fits comfortably on the 62 GB / 32-core / 24 GB-VRAM machines (Hawaii, bikini, hermes, hormoz,
# hokkaido): ~46 GB RAM free and 3.6x GPU headroom. SMACv2 parallelises better than SMACv1 (1.51x vs
# 1.36x) because 5v5 SC2 workers contend less for the 32 cores.
#
# >>> DO NOT run this on barbados. <<< 31.9 GB RAM with ~15.5 GB held by the desktop leaves ~13-16 GB;
# two runs need 25.5 GB and would thrash to swap (measured 36-47x slower there on SMACv1). barbados also
# lacks the 10gen_* maps entirely — see INTEGRATION.md §1.5.
#
# GPU CAVEAT: episode_limit is 400 and the numbers above were measured at ep_length_mean ~62. VRAM grows
# with max_t_filled, so it can climb as agents survive longer. Two runs would need a 3.6x increase to
# exhaust a 24 GB card, which is unlikely (max_t_filled saturates early, being a max over 128 sampled
# episodes) -- but watch `nvidia-smi` on the first pair. If per-run VRAM approaches ~10 GB, drop to
# sequential by setting PARALLEL=1 below.

CONFIG=hpn_saleq_terran_sem_wm_qmix   # terran-aware agent — see the note above
ENVCFG=sc2_v2_terran                  # 10gen_terran, 5v5 (NOT 10v10; INTEGRATION.md §6), limit 400
T_MAX=5005000                         # as used by the original testrun scripts. NOTE our alg configs
                                      # default to the SMACv1 value 1050000 and pymarl3's own use
                                      # 10050000, so this MUST be passed explicitly (INTEGRATION.md §1).
BUFFER_SIZE=4000                      # 4.26 GB preallocated at these dims (5000 would be 5.33 GB)
seeds=(10 11 12 13 14 15 16 17 18 19) # 10 independent seeds; no SMACv2 terran data exists yet
PARALLEL=2                            # runs at a time — set to 1 for sequential (see GPU CAVEAT)
# SEED_TIMEOUT=4h                     # DISABLED for now. ~2x the ~2.1 h/run expected at the concurrent
                                      # rate. It guards the deadlock seen on hormoz, where a lost env
                                      # worker left the runner blocked forever: 10 h 44 m burned for
                                      # zero evaluations, and the sweep never advanced.
                                      # TO RE-ENABLE: uncomment this line AND the `timeout` line in
                                      # run_seed() below. Both must be changed together — a bare
                                      # `timeout --kill-after=2m` with no duration is a usage error.
                                      # While disabled, a hung seed blocks its group indefinitely; the
                                      # 124/137 branch in run_seed() is inert.

# ---- preflight: results must not land on the shared NFS home (this repo already lost a zerg run to
# ---- OSError: [Errno 28] there, and that volume is under 1 GB free)
RESULTS_REAL=$(readlink -e results 2>/dev/null || echo "$PWD/results")
case "$RESULTS_REAL" in
    /home/*) echo "*** ABORT: results/ resolves to $RESULTS_REAL, on the shared NFS home."
             echo "    Run from the local copy (/usr/local/data/zwang182/MARL/pymarl3_related_code),"
             echo "    or pass local_results_path=<absolute local path> (INTEGRATION.md §1)."
             exit 1 ;;
esac

echo "=== $CONFIG on $ENVCFG | seeds: ${seeds[*]} | t_max=$T_MAX ==="
echo "=== $PARALLEL run(s) in parallel | buffer_size=$BUFFER_SIZE | started $(date) ==="
echo "=== results -> $RESULTS_REAL | RAM avail $(free -g | awk '/Mem:/{print $7}') GB (needs ~$((13*PARALLEL)) GB) ==="

run_seed () {
    local seed=$1
    echo "--- seed $seed : start $(date '+%F %T') ---"
    # timeout --kill-after=2m $SEED_TIMEOUT \   # DISABLED — re-enable together with SEED_TIMEOUT above
    # PASS `seed=`, NOT `env_args.seed=`. In THIS repo main.py:40 does
    #     config['env_args']['seed'] = config["seed"]
    # i.e. env_args.seed is OVERWRITTEN by sacred's auto-generated seed, so `env_args.seed=N` is
    # decorative — it lands in config.json and never reaches the RNG. (marl_project3 assigns the other
    # way round, `config["seed"] = config['env_args']['seed']`, which is why the SMACv1 sweeps are
    # correctly seeded with the same syntax. Do not copy that syntax into this repo.)
    python3 src/main.py --config=$CONFIG --env-config=$ENVCFG \
        with seed=$seed \
        obs_agent_id=True obs_last_action=False runner=parallel \
        buffer_size=$BUFFER_SIZE t_max=$T_MAX \
        save_model=False use_cuda=True use_tensorboard=True
    local rc=$?
    if [ $rc -eq 124 ] || [ $rc -eq 137 ]; then
        echo "--- seed $seed : *** TIMED OUT after $SEED_TIMEOUT *** (exit $rc) ---"
    else
        echo "--- seed $seed : finished $(date '+%F %T') with exit code $rc ---"
    fi
}

# Launch in groups of $PARALLEL and wait for each group, so at most $PARALLEL runs are ever live.
for ((i = 0; i < ${#seeds[@]}; i += PARALLEL)); do
    group=("${seeds[@]:i:PARALLEL}")
    echo ""
    echo "================ group $((i / PARALLEL + 1)): seeds ${group[*]} : $(date '+%F %T') ================"
    for s in "${group[@]}"; do run_seed "$s" & done
    wait     # no 'set -e': one failed seed must not abandon the rest

    # Sweep up orphans between groups. A run that dies abnormally leaves its env workers and their
    # SC2_x64 processes reparented to init, holding tens of GB — seen on hormoz, helen and barbados.
    # Safe here because nothing of ours is live between groups. The [4] bracket stops the pattern
    # matching this script itself.
    ORPHANS=$(pgrep -u "$USER" -f 'src/main\.py'; pgrep -u "$USER" -f 'SC2_x6[4]')
    if [ -n "$ORPHANS" ]; then
        echo "--- cleaning up $(echo $ORPHANS | wc -w) orphaned process(es) ---"
        for p in $ORPHANS; do kill -TERM $p 2>/dev/null; done
        sleep 10
        for p in $(pgrep -u "$USER" -f 'src/main\.py'; pgrep -u "$USER" -f 'SC2_x6[4]'); do
            kill -KILL $p 2>/dev/null
        done
        sleep 5
    fi
    rm -rf /tmp/sc-* 2>/dev/null
done

echo ""
echo "=== sweep finished $(date) ==="
echo "per-seed exit codes: grep -E '^--- seed .* exit code' <this log>   (every one must read 0)"
echo "timeouts           : grep -i 'TIMED OUT' <this log>                (expected none)"
echo "curves             : results/tb_logs/   |   sacred: results/sacred/"
echo "NOTE: runs were PARALLEL x$PARALLEL, so logged steps/s is ~72-79 % of the solo rate and is not"
echo "      comparable to the sequential SMACv1 sweeps. Results themselves are unaffected."
