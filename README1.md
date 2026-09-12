# marl_baseline — SMACv2 10v10 baselines

A [pymarl3](https://github.com/tjuHaoXiaotian/pymarl3)-derived codebase for StarCraft II
multi-agent RL. This copy exists to run **four value-factorisation baselines on the three
SMACv2 10v10 scenarios**:

| algorithm | `--config` | agent |
|---|---|---|
| HPN-QMIX | `hpn_qmix` | `hpns_rnn` |
| HPN-VDN | `hpn_vdn` | `hpns_rnn` |
| QMIX | `qmix` | `n_rnn` |
| DEEPSET-QMIX | `deepset_qmix` | `deepset_rnn` |

against `--env-config` of `sc2_v2_protoss_10v10`, `sc2_v2_terran_10v10`, `sc2_v2_zerg_10v10`.

Four seeds per (algorithm, map) = **48 GPU jobs**, driven by
`cc_baselines_10v10_launch.sh` → `cc_baselines_10v10_gpu.sh`.

**SMACv2 is vendored** at `src/envs/smac_v2/`, including all 30 `.SC2Map` files under
`src/envs/smac_v2/official/maps/SMAC_Maps/`. You do **not** need the `smacv2` pip package, and
you do not need to download maps separately — but you do need to copy them into your
StarCraft II install (see below).

---

## 1. Requirements

**Python 3.11.** Wheels for the scientific stack are only built for recent interpreters; 3.10
and 3.12 also work, 3.9 and older will fight you over dependencies.

Third-party imports the code actually uses, with the versions this was last verified against:

```
torch               2.10.0        sacred              0.8.7
numpy               2.4.2         PySC2               4.0.0
PyYAML              5.3.1         s2clientprotocol    5.0.15.95299.0
pygame              2.5.2         SMAC                1.0.0
absl-py             2.4.0         tensorboard_logger  0.1.0
cloudpickle         3.1.2         protobuf            3.20.3
scipy / matplotlib  (for plotting only)
```

Two version notes that have bitten this codebase before, both already fixed in this tree:

- **`protobuf` must stay at 3.20.x.** PySC2 4.0.0 requires it. If you install something that
  pulls protobuf 4+ or 6+, PySC2 breaks at import. This is why this environment must be kept
  separate from environments that want a modern protobuf.
- The source is clean under **NumPy ≥ 2** and **PyYAML ≥ 6** (`np.bool` and the implicit
  `yaml.load` loader were removed years ago and are already patched out here).

**StarCraft II 4.10.x** — see §3. Not 4.6, not 5.x.

---

## 2. Setting up the virtual environment

### On a Digital Research Alliance of Canada cluster (Fir / Narval / Nibi / Rorqual)

Use `virtualenv` with the **local wheelhouse**, not conda. Wheels tagged `+computecanada` are
compiled for the cluster hardware and are conflict-free.

```bash
module load StdEnv/2023 python/3.11 cuda/12.2 scipy-stack
virtualenv --no-download ~/venvs/marl_baseline
source ~/venvs/marl_baseline/bin/activate
pip install --no-index --upgrade pip

# one call, so pip resolves everything together
pip install --no-index torch sacred pyyaml pygame absl-py cloudpickle tensorboard_logger
pip install pysc2==4.0.0 smac protobuf==3.20.3     # not all are in the wheelhouse
```

Three things that will waste an afternoon if you skip them:

1. **`numpy`, `scipy`, `matplotlib` and `pandas` come from the `scipy-stack` module**, not
   from pip, and they arrive via `PYTHONPATH`. The venv is created with
   `include-system-site-packages = false`, so **you must load the same modules every time
   before `source .../activate`** — otherwise you get an instant `ImportError: no numpy`.
   Put the `module load` line in your job script, never in `~/.bashrc`.
2. **Never create the venv under `$SCRATCH`.** Scratch is purged on a schedule and will
   delete it out from under a running job. `$HOME` or `/project` only.
3. Check what the wheelhouse has with `avail_wheels <name>` before assuming pip must go to
   PyPI.

### Anywhere else

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch numpy 'pyyaml' sacred pysc2==4.0.0 smac 'protobuf==3.20.3' \
            pygame absl-py cloudpickle tensorboard_logger scipy matplotlib
```

### Verify before submitting anything

```bash
cd src
python -c "import envs; print(envs.REGISTRY.keys())"
# expect: dict_keys(['sc2', 'sc2_v2'])
python -m pytest tests/ -q        # 27 tests, ~25 s, no GPU or StarCraft II needed
```

`envs/__init__.py` swallows import errors and prints them instead of raising, so an empty or
partial `REGISTRY` means a dependency is missing — read the printed traceback.

---

## 3. StarCraft II

Download the Linux package (Blizzard's AI/ML build; the zip password is
`iagreetotheeula`) and unpack it anywhere:

```bash
wget https://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip
unzip -P iagreetotheeula SC2.4.10.zip -d /path/to/sc2
export SC2PATH=/path/to/sc2/StarCraftII
```

Then **install the SMAC maps**, which the Blizzard zip does not contain:

```bash
mkdir -p "$SC2PATH/Maps/SMAC_Maps"
cp -n src/envs/smac_v2/official/maps/SMAC_Maps/*.SC2Map "$SC2PATH/Maps/SMAC_Maps/"
```

> **The version matters.** SMAC win rates are **not comparable across StarCraft II
> versions**, so everyone contributing runs to the same table must be on the same one. This
> project uses **4.10.0 (build 75689)**. Check yours with
> `awk -F'|' 'NR==2{print $13}' "$SC2PATH/.build.info"`.
> `cc_baselines_10v10_gpu.sh` refuses to start on anything that is not 4.10.x.

> The four `10gen_*` maps are **not sufficient on their own** — they are generators that
> reference a base terrain map (`32x32_flat`). A missing base map fails *late*, inside the
> env workers after "Beginning training" has printed, and reads like a hang. The job script
> checks for all seven up front.

---

## 4. Running

Edit the two marked blocks first:

- `cc_baselines_10v10_launch.sh` → `ACCOUNT`, and `gpuOf` (GPU names are cluster-specific;
  the comment gives the VRAM each map needs, translate from that).
- `cc_baselines_10v10_gpu.sh` → the `SITE BLOCK`: module loads, venv path, `SC2PATH`.

Then dry-run before committing 48 jobs:

```bash
sbatch() { echo "DRY-RUN sbatch $@"; }; export -f sbatch; bash cc_baselines_10v10_launch.sh
bash cc_baselines_10v10_launch.sh      # for real
```

Outputs land in `results/baselines_10v10/{tb_logs,sacred,models}/`, which is `.gitignore`d.
`tensorboard --logdir results/baselines_10v10/tb_logs`.

---

## 5. Things worth knowing before you trust a number

- **Pass `seed=N`, never `env_args.seed=N`.** `main.py` copies `seed` into `env_args.seed` at
  run time, so `env_args.seed` is overwritten by a sacred-generated value. A sweep passed the
  wrong way is unlabelled and unreproducible. For analysis, read `config['seed']`.
- **`10gen_*` scenarios run at whatever `team_gen.n_units` says**, not the `n_agents: 10` in
  the map registry. The `_10v10` env configs set it to 10; the stock ones set 5.
- **None of these four baselines is heal-aware on terran.** `hpns_rnn` gates its rescue head
  on `map_type == "MMM"` and SMACv2 reports `terran_gen`; `n_rnn` and `deepset_rnn` have no
  `map_type` handling. Nothing asserts — the runs are valid — but a medivac's trailing action
  slots are scored as "attack enemy j" rather than "heal ally j". State the caveat wherever
  terran numbers are reported.
- **A timed-out run should be rerun from scratch with a longer wall, not resumed.** The replay
  buffer, target mixer and RNG are not checkpointed, and resumed runs in this codebase showed
  a performance collapse. `save_model=True` here is for analysis, not recovery.
- Expect throughput to **fall** as training progresses: episodes lengthen toward the 400-step
  cap, so a rate measured in the first few minutes is optimistic by roughly 2x. Size wall
  times from a run that has reached a representative `ep_length_mean`.

`INTEGRATION.md` has the fuller version of these, plus the architecture notes.
