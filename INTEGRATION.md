# Integrating our method into pymarl3 for SMACv2

**What this is:** our method — the HPN + `φ^{oa}` action-in Q-head + SEM + centralized world model, developed in
`marl_project3` on SMACv1 — registered inside this pymarl3 checkout so it can run on **SMACv2** (`10gen_protoss`,
`10gen_terran`, `10gen_zerg`). Integrated 2026-08-10/11. All changes here are **additive**; pymarl3's own
algorithms are untouched.

---

## 1. Run it

```bash
# helen (main dev machine):
conda activate maserEnv
# every other lab machine — the marl_project3 venv works; no extra packages needed (see 1.5):
source /usr/local/data/zwang182/envs/marl_project3/bin/activate

export SC2PATH=<this machine's StarCraftII>   # NOT portable — five different layouts across six
                                              # machines; the venv's bin/activate already exports it
cd <repo>/          # run from the repo root

python3 src/main.py --config=<ALG> --env-config=<ENV> with env_args.seed=1 \
    t_max=<STEPS> use_cuda=True use_tensorboard=True save_model=False
```

**Pick the agent by race — this matters:**

| Race | `--env-config` | `--config` (alg) | Why |
|---|---|---|---|
| protoss | `sc2_v2_protoss` | `hpn_saleq_sem_wm_qmix` | no healer units |
| zerg | `sc2_v2_zerg` | `hpn_saleq_sem_wm_qmix` | no healer units |
| **terran** | `sc2_v2_terran` | **`hpn_saleq_terran_sem_wm_qmix`** | medivacs → *heal* actions (§3) |

**Measured throughput differs ~3× across machines** — plan wall-clock per machine, not fleet-wide.
40k-step samples: Hawaii ~870 steps/s, hermes ~670, **barbados 288** (RTX 3090, 24 cores, already
swapping). And short samples are **~1.8× optimistic**: the sustained paired rate on Hawaii is ~380
steps/s per run, because episodes lengthen toward the 400-step limit and both step cost and VRAM climb
with them. Estimate from a run that has reached a representative episode length, never from 40k steps.
At 5.005M steps that is ~18 h for 10 seeds on Hawaii/hermes (2 in parallel) but ~8.7 h **per run** on
barbados, which can only run one at a time (31.9 GB RAM, ~14 GB free, one run needs ~12.7 GB).

Ablation arms (same agents, fewer components): `hpn_saleq_qmix` (no SEM, no WM) and `hpn_saleq_sem_qmix` (no WM).
Baselines already in pymarl3: `hpn_qmix`, `hpn_vdn`, `hpn_qplex`.

**Two overrides you almost certainly need:**
- **`t_max`** — our configs carry the SMACv1 value **1050000**; pymarl3's own configs use 10050000. Set it per run.
- **`local_results_path`** — defaults to `results` *inside the repo*. If the repo sits on the shared NFS home,
  runs write there and **will fail when it fills** (this already killed a zerg run with `OSError: [Errno 28]`;
  as of 2026-08-11 that volume has **under 1 GB free**, so this is not hypothetical).
  Point it at local disk: `local_results_path=/usr/local/data/zwang182/<...>/results`.
  An **absolute** path here redirects *both* outputs — `run.py:118` builds tb_logs from
  `args.local_results_path` and `main.py:115` builds the sacred path from it, and `os.path.join` discards the
  repo-root prefix when given an absolute path.
  > ⚠️ **This differs from `marl_project3`,** where `run.py:51` hardcodes tb_logs to `<repo>/results/tb_logs`
  > and ignores `local_results_path` entirely — there it takes a *symlink*. Do not carry that assumption over
  > in either direction.

---

## 1.5 Environment — the `marl_project3` venv runs this repo (no new packages; two source fixes were needed)

**No new packages are required**, but the repo did need two version-compatibility fixes — see below. Verified 2026-08-11 by import analysis:

- **SMACv2 is vendored** at `src/envs/smac_v2/`; the `smacv2` name in `envs/__init__.py` is just a local
  success flag. The pip package `smacv2` is **not** needed and is not installed anywhere.
- Full third-party import set: `torch, numpy, yaml, sacred, pysc2, s2clientprotocol, pygame, absl, smac,
  tensorboard_logger, cloudpickle`. The venv has all of them.
- `cloudpickle` is the only one absent from `requirements-venv.txt`, but it is present: `marl_project3`'s
  own `parallel_runner.py` imports it the same way and every sweep runs with `batch_size_run=8`.
- The pip `smac` package (oxwhirl) is imported only by `official/sc2_official.py`, which is **not** on the
  import path — `smac_v2/__init__.py` pulls in only `StarCraft2Env2Wrapper`. It is installed regardless.

### Two version fixes were needed (applied 2026-08-11)

No new *packages*, but the repo was written against **older versions** than the venv ships, and failed twice
before running. Both are now fixed in this source tree, so machines copied from here get working code.
They never surfaced under conda `maserEnv`, which pins older versions.

| Fix | Why | Where |
|---|---|---|
| `yaml.load(f)` → `yaml.load(f, Loader=yaml.FullLoader)` | **PyYAML ≥ 6** made `Loader` required; venv has 6.0.3. Failed instantly at startup. Matches how `marl_project3/src/main.py` already does it. | `src/main.py:58,97` |
| `np.bool` → `bool` (11 occurrences, 4 files) | **NumPy ≥ 1.24** removed the alias; venv has 1.26.4. Nastier failure: config parsed, env built, `Beginning training` printed, then it died *inside the env workers*. | `src/envs/smac_v1/official/starcraft2.py`, `src/envs/smac_v2/official/{starcraft2,starcraft2_hxt,sc2_official}.py` |

`.bak-npbool-*` backups sit alongside the patched env files. If you ever run this repo under an older
interpreter/stack again, the fixes are backward-compatible — `Loader=` and builtin `bool` work on the old
versions too.

**The real per-machine prerequisite is MAPS, not packages** — and **this repo ships them**, so no download
is needed. They live at `src/envs/smac_v2/official/maps/SMAC_Maps/` (30 files) and arrive with the rsync.

⚠️ **The four `10gen_*` maps are NOT sufficient on their own.** They are generators and reference a base
terrain map. A machine with `10gen_*` but no `32x32_flat` fails with:

```
ValueError: Map 'SMAC_Maps/32x32_flat.SC2Map' not found.
```

and — this is the nasty part — it fails **late, inside the env workers, after `Beginning training` has
printed and 8 SC2 processes have spawned**. It reads as a hang, not as a missing map, and it leaves
orphaned SC2 processes behind. Check for all seven before running:

```bash
ls $SC2PATH/Maps/SMAC_Maps/{10gen_protoss,10gen_terran,10gen_zerg,10gen_empty,32x32_flat,32x32_flat_test,32x32_small}.SC2Map
```

If any are missing, install every map the repo ships (harmless — `cp -n` skips what is already there):

```bash
cp -n src/envs/smac_v2/official/maps/SMAC_Maps/*.SC2Map "$SC2PATH/Maps/SMAC_Maps/"
```

Machines whose StarCraft II came from a pymarl3 tree already have all of them. **barbados did not** — its
SC2 was installed from the oxwhirl `SMAC_Maps.zip`, which ships only the 23 SMACv1 maps; it was missing
all four `10gen_*` and all three `32x32_*`. **Fixed 2026-08-11** by the `cp -n` above (23 → 30 maps), and
a protoss sample then ran clean.

---

## 2. What was added

**New files**
```
src/modules/agents/hpn_saleq_agent.py        # φ^{oa} action-in Q-head (+ sem(), AvgL1Norm)
src/modules/agents/hpn_saleq_terran.py       # terran variant — subclasses the above (§3)
src/modules/model/{__init__,sem_wm}.py       # centralized SEM world model
src/learners/saleq_wm_learner.py             # SaleqWMLearner(NQLearner)
src/config/algs/hpn_saleq_qmix.yaml
src/config/algs/hpn_saleq_sem_qmix.yaml
src/config/algs/hpn_saleq_sem_wm_qmix.yaml
src/config/algs/hpn_saleq_terran_sem_wm_qmix.yaml
src/tests/{__init__,test_hpn_saleq,test_saleq_wm,test_hpn_saleq_terran}.py
```

**Edited (additive only; `.bak-*` backups alongside)**
```
src/modules/agents/__init__.py   # + hpn_saleq, hpn_saleq_terran  (all 9 originals intact → 11 keys)
src/learners/__init__.py         # + saleq_wm_learner
src/modules/agents/hpn_saleq_agent.py   # terran_gen capability guard (§4)

# version-compatibility fixes, 2026-08-11 — not part of the method (§1.5):
src/main.py                              # yaml.load(..., Loader=yaml.FullLoader)  x2
src/envs/smac_v1/official/starcraft2.py  # np.bool -> bool
src/envs/smac_v2/official/starcraft2.py        # np.bool -> bool
src/envs/smac_v2/official/starcraft2_hxt.py    # np.bool -> bool
src/envs/smac_v2/official/sc2_official.py      # np.bool -> bool

# parallel-run correctness fix, 2026-08-11 (§6) — REQUIRED before running anything in parallel:
src/run/run.py                           # unique_token now includes map + seed
```

This repo is **not** a git repo — hence the `.bak-*` files.

### Copying it to another machine

**Destination on every local machine: `/usr/local/data/zwang182/MARL/pymarl3_related_code/`** — alongside
the `marl_project3` clone already there. **Copy every file except `results/`**, which is per-machine output
and must not travel.

```bash
rsync -av --exclude 'results/' \
    /home/adaptation/zwang182/Projects/project3_related_codebases/pymarl3_related_code/ \
    /usr/local/data/zwang182/MARL/pymarl3_related_code/
```

`$HOME` is mounted on every lab machine, so the source path above works from anywhere. The destination is on
**local disk**, not the shared home — see the `local_results_path` note in §1 for why that matters.

---

## 3. Terran needs its own agent

SMACv2's `10gen_terran` generates **medivacs**, whose trailing action slots mean *heal ally j*, not *attack
enemy j*. The env knows this (`starcraft2.py:790` branches on `map_type in ["MMM","terran_gen"]`), but **no HPN
agent in pymarl3 does** — `hpns_rnn_agent.py` tests only `== "MMM"`, so HPN-QMIX/VDN/QPLEX all score heal actions
through the enemy read-out on this map. Silently: they run to completion without error. (Verified empirically:
both variants exit 0, and the deployed agent's parameter count matches the no-rescue build.)

`hpn_saleq_terran` fixes this for our method:
- a per-agent **healer gate read from the observed unit type** (`own_context[..., -1]`, medivac = bit **2**),
  so it is resolved per episode and does **not** rely on agent ordering;
- ally embeddings for a healer's trailing slots, enemy embeddings otherwise;
- slot-free action encoding `[onehot_{A_inv}; is_attack; is_rescue]` — the rescue flag never carries the ally
  slot index, preserving exact permutation-equivariance.

Handles **any** number of healers per episode (k varies; observed {0: 13, 1: 8, 2: 4} over 25 resets) at **any**
position. Tested at k=0/1/2 with tail, non-tail and non-contiguous placements.

---

## 4. Guards, and one gap

`hpn_saleq_agent.py` carries a capability-gated assert: it **refuses `terran_gen`** (`_supports_terran_gen = False`)
and points at `hpn_saleq_terran`, which sets the flag `True`. This converts a silent-wrong-results path into a
loud startup failure.

⚠️ **The reverse guard is missing.** `hpn_saleq_terran` does **not** assert `map_type == "terran_gen"`. Run it on
protoss or zerg and it would gate on the last unit-type bit — **colossus** and **baneling** respectively — treating
those as healers and scoring their attacks through the ally encoder. Silently. **Use the table in §1**; adding the
symmetric assert is a two-line fix worth doing.

---

## 5. Validation status

| Check | Status |
|---|---|
| CPU test suite (agent, WM, learner, terran incl. non-tail ordering) | ✅ all pass |
| Registry / configs resolve; pymarl3's own algs unaffected | ✅ |
| `10gen_protoss` real run | ✅ trained, losses finite |
| `10gen_terran` real run (`hpn_saleq_terran`) | ✅ trained, losses finite, no asserts |
| `10gen_zerg` real run | ✅ **confirmed 2026-08-11** (was previously unconfirmed — an earlier attempt died on `ENOSPC`, not on any code fault) |
| Runs under the `marl_project3` venv, all three races | ✅ **Hawaii, 2026-08-11** — see below |

**Re-validated on Hawaii 2026-08-11** from the local copy at `/usr/local/data/zwang182/MARL/pymarl3_related_code`,
under the `marl_project3` venv (Python 3.10 / torch 2.6.0+cu124 / numpy 1.26.4 / PyYAML 6.0.3), 40k steps each:

| Race | `--config` | agent deployed | Result |
|---|---|---|---|
| protoss | `hpn_saleq_sem_wm_qmix` | `hpn_saleq` | exit 0, 0 errors |
| zerg | `hpn_saleq_sem_wm_qmix` | `hpn_saleq` | exit 0, 0 errors |
| terran | `hpn_saleq_terran_sem_wm_qmix` | `hpn_saleq_terran`, `map_type: terran_gen` | exit 0, 0 errors |

All WM losses finite and updating (`loss_kl`, `loss_rec`, `loss_pred`, `loss_act`, `traj_grad_norm`, `loss_td`).
This required the two version fixes in §1.5 — without them protoss failed at startup (PyYAML) and then inside
the env workers (NumPy).

Dimension arithmetic confirms the head derives its shapes correctly from SMACv2's entity contract: `loss_act` at
init ≈ `n_agents × ln(|A|)` — protoss 12.048 vs 11.99, terran 11.975 vs 11.990 — and `loss_rec` ≈ `8 × ln(8)` = 16.64.

---

## 6. Facts worth knowing before you run

- **`10gen_*` maps run 5v5, not 10v10.** The map registry says `n_agents: 10`, but `sc2_v2_*.yaml` sets
  `team_gen.n_units: 5`, which wins (`starcraft2.py:272`). Affects budgeting and any expected-dimension check.
- **Medivac is unit-type bit 2** — hardcoded `marine→0, marauder→1, medivac→2` in `get_unit_type_id`. Not
  alphabetical, not unit-type-id order. Do not infer it; it is an arbitrary table.
- **Enemy teams contain medivacs too**, carrying the *raw* id `Terran.Medivac = 54`, not the generated ally id
  1975 — a `== medivac_id` check silently misses them. Harmless (enemies are only attacked), but don't be misled.
- **`ally_encoder` only trains on episodes containing a healer** (~48% of terran episodes). Slower convergence
  for that submodule is expected, not a bug.

- 🔴 **Pass `seed=N`, NOT `env_args.seed=N`.** `main.py:40` does `config['env_args']['seed'] = config["seed"]`,
  so `env_args.seed` is *overwritten* by sacred's auto-generated seed. Passing `env_args.seed=10` is
  **decorative** — it lands in `config.json` and never reaches the RNG. Observed 2026-08-11: a sweep labelled
  seeds 10/11 actually ran with sacred's 20276472 / 119885084. Runs are still independent, but unlabelled and
  unreproducible.
  **For analysis, read `config['seed']`** — `env_args.seed` shows `null` once you pass `seed=` correctly,
  because sacred snapshots the config before `main.py:40` fills it in at run time.
  ⚠️ This is the **opposite** of `marl_project3` (§7.3), where `env_args.seed=N` is the correct syntax.

- 🔴 **`unique_token` had no seed — parallel runs collided in `tb_logs`.** It was `{name}__{timestamp}` to the
  second, so *any* pair launched together resolved to the **same** tb directory and interleaved their events
  into one file. Sacred was unaffected (auto-incrementing run ids), so only the tensorboard curves were
  corrupted — silently. **Fixed 2026-08-11** in `src/run/run.py`: the token is now
  `{name}_{map}_seed{N}__{timestamp}`, matching `marl_project3`.
  Verified: two runs launched in the same second now produce
  `..._10gen_terran_seed10__2026-08-11_20-12-30` and `..._seed11__2026-08-11_20-12-30`.
  **If you copy this repo from an older source, apply this before running anything in parallel.**

## 7. Divergences from `marl_project3`

Differences to keep in mind when syncing the two copies. 1–2 are deliberate and marked in-file; 3 is inherited
from pymarl3 itself and is the one most likely to cost you results.

1. `saleq_wm_learner.py` calls the **6-arg** `build_td_lambda_targets`; ours takes a 7th unused `n_agents`.
   Passing the 7-arg form here is a `TypeError`.
2. The `terran_gen` capability guard in `hpn_saleq_agent.py` does not exist in our copy.
3. 🔴 **Seed assignment runs in the OPPOSITE direction** (§6):

   | repo | `main.py` | correct syntax |
   |---|---|---|
   | `marl_project3` | `config["seed"] = config['env_args']['seed']` | `env_args.seed=N` |
   | **this repo** | `config['env_args']['seed'] = config["seed"]` | **`seed=N`** |

   Copying a run command between the two silently produces unseeded runs in one direction. The SMACv1
   sweep scripts all use `env_args.seed=`, which is correct **there** and wrong **here**.

`hpn_MMM2_agent.py` (SMACv1 MMM2) was deliberately **not** ported — it assumes a single medivac at a fixed index,
which SMACv2 violates. `hpn_saleq_terran` supersedes it.
