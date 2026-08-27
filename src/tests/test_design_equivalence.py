# =====================================================================================================
# DESIGN-SPACE TEST -- the GATE for using the design-space copies at all.
# Proves that with the design flags absent (and again with them present at their defaults):
#   (a) DesignSaleQAgent == HPNSaleQAgent : same-seed construction gives identical state_dict keys and
#       bit-identical tensors; identical inputs give th.equal Q and v_tilde -- for saleq_norm in (sem, avgl1).
#   (b) DesignWMLearner == SaleqWMLearner : same-seeded hpn_macs + identical synthetic batches, th.manual_seed
#       immediately before each of 2 train() calls (the only RNG consumers on the learner path are the
#       straight-through multinomial samples, and the copy preserves their order) -> every parameter of agent,
#       mixer, target_mixer and WM is th.equal afterwards, and every logged stat is identical.
# Exact equality (th.equal / ==) is asserted; th.set_num_threads(1) keeps BLAS reductions deterministic.
# (If exact equality ever fails only by BLAS non-determinism, relax to allclose(rtol=1e-7, atol=0) and
# say so here -- as of writing, exact equality holds.)
#
# Run from src with:  python -m tests.test_design_equivalence   (or pytest)
# =====================================================================================================
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src

import torch as th
from components.episode_buffer import EpisodeBatch
from components.transforms import OneHot
from controllers import REGISTRY as mac_REGISTRY
from learners import REGISTRY as le_REGISTRY
from modules.agents.hpn_saleq_agent import HPNSaleQAgent
from modules.agents.design_saleq_agent import DesignSaleQAgent

th.set_num_threads(1)


def build_args(norm="sem", agent="hpn_saleq", learner="saleq_wm_learner", design_flags=False):
    n_agents, n_enemies, n_normal = 3, 3, 6
    n_actions = n_normal + n_enemies
    ns = SimpleNamespace(
        n_agents=n_agents, n_actions=n_actions, n_enemies=n_enemies, n_allies=2,
        output_normal_actions=n_normal, obs_component=[4, (n_enemies, 5), (2, 5), 1],
        state_shape=48, rnn_hidden_dim=64, hpn_head_num=1, hpn_hyper_dim=64,
        saleq_emb_dim=32, saleq_zsa_dim=64, saleq_hidden_dim=64,
        saleq_norm=norm, saleq_sem_groups=8,
        obs_agent_id=True, obs_last_action=False, map_type="marines",
        agent=agent, mac="hpn_mac", agent_output_type="q",
        learner=learner,
        wm_groups=8, wm_vertices=8, wm_state_dim=64, wm_sa_dim=64, wm_traj_dim=32,
        wm_k=5, wm_k_train=1, wm_alpha=0.8, wm_unimix=0.01,
        c_wm=1.0, c_kl=1.0, c_rec=1.0, c_pred=1.0, c_act=1.0,
        action_selector="epsilon_greedy", epsilon_start=1.0, epsilon_finish=0.05,
        epsilon_anneal_time=100000, test_greedy=True,
        mixer="qmix", mixing_embed_dim=32, hypernet_embed=64,
        optimizer="adam", lr=1e-3, td_lambda=0.6, q_lambda=False, gamma=0.99,
        grad_norm_clip=10, target_update_interval=1, learner_log_interval=2000,
        use_cuda=False, device="cpu",
    )
    if design_flags:  # explicit defaults -- must change nothing
        ns.saleq_sem_temp = 1.0
        ns.design_mixer_input = "s_traj"
        ns.design_mixer_zs_detach = True
    return ns


class _NullConsole:
    def info(self, *a, **k):
        pass


class _RecLogger:
    console_logger = _NullConsole()

    def __init__(self):
        self.stats = {}

    def log_stat(self, k, v, t):
        self.stats[k] = float(v)


def make_batch(args, bs=2, T=6, seed=123):
    th.manual_seed(seed)
    n_agents, n_actions, state_shape, obs_shape = args.n_agents, args.n_actions, args.state_shape, 30
    scheme = {
        "state": {"vshape": state_shape},
        "obs": {"vshape": obs_shape, "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (n_actions,), "group": "agents", "dtype": th.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {"agents": n_agents}
    preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=n_actions)])}
    batch = EpisodeBatch(scheme, groups, bs, T, preprocess=preprocess, device="cpu")
    data = {
        "state": th.randn(bs, T, state_shape),
        "obs": th.randn(bs, T, n_agents, obs_shape),
        "avail_actions": th.ones(bs, T, n_agents, n_actions, dtype=th.int),
        "actions": th.randint(0, n_actions, (bs, T, n_agents, 1)),
        "reward": th.randn(bs, T, 1),
        "terminated": th.zeros(bs, T, 1, dtype=th.uint8),
    }
    data["terminated"][:, T - 2] = 1
    batch.update(data, ts=slice(0, T), mark_filled=True)
    return batch, scheme, groups


def _assert_state_dicts_equal(a, b, tag):
    ka, kb = list(a.keys()), list(b.keys())
    assert ka == kb, f"[{tag}] state_dict keys differ:\n {ka}\n {kb}"
    for k in ka:
        assert th.equal(a[k], b[k]), f"[{tag}] tensor {k} differs"


# ---------------------------------------------------------------- (a) agent
def _agent_inputs(args, bs=2):
    th.manual_seed(5)
    n, ne, na = args.n_agents, args.n_enemies, args.n_allies
    own = th.randn(bs * n, 4 + 1)                  # own_context + agent-id slot is handled by indices
    enemy = th.randn(bs * n * ne, 5)
    ally = th.randn(bs * n * na, 5)
    idx = [th.arange(n).unsqueeze(0).expand(bs, -1)]
    h = th.zeros(bs, n, args.rnn_hidden_dim)
    return (bs, own, enemy, ally, idx), h


def test_agent_equivalence():
    for norm in ("sem", "avgl1"):
        for flags in (False, True):
            args = build_args(norm=norm, design_flags=flags)
            input_shape = (5, (args.n_enemies, 5), (args.n_allies, 5))
            th.manual_seed(0)
            ref = HPNSaleQAgent(input_shape, args)
            th.manual_seed(0)
            dut = DesignSaleQAgent(input_shape, args)
            tag = f"agent norm={norm} flags={flags}"
            _assert_state_dicts_equal(ref.state_dict(), dut.state_dict(), tag)
            inp, h = _agent_inputs(args)
            with th.no_grad():
                q_r, v_r = ref(inp, h)
                q_d, v_d = dut(inp, h)
            assert th.equal(q_r, q_d), f"[{tag}] Q differs"
            assert th.equal(v_r, v_d), f"[{tag}] v_tilde differs"
            assert th.equal(ref.last_phi, dut.last_phi), f"[{tag}] last_phi differs"


# ---------------------------------------------------------------- (b) learner
def _run_learner(learner_key, args, seeds=(11, 12)):
    batch, scheme, groups = make_batch(args)
    th.manual_seed(0)
    mac = mac_REGISTRY["hpn_mac"](batch.scheme, groups, args)
    th.manual_seed(1)
    logger = _RecLogger()
    learner = le_REGISTRY[learner_key](mac, batch.scheme, logger, args)
    stats = []
    for i, s in enumerate(seeds):
        th.manual_seed(s)
        learner.train(batch, t_env=i * 3000, episode_num=i)
        stats.append(dict(logger.stats))
    return learner, stats


def test_learner_equivalence():
    for flags in (False, True):
        ref, st_r = _run_learner("saleq_wm_learner", build_args(design_flags=flags))
        dut, st_d = _run_learner("design_wm_learner", build_args(design_flags=flags))
        tag = f"learner flags={flags}"
        assert ref.mixer.input_dim == dut.mixer.input_dim == 48 + 32, tag
        _assert_state_dicts_equal(ref.mac.agent.state_dict(), dut.mac.agent.state_dict(), tag + " agent")
        _assert_state_dicts_equal(ref.target_mac.agent.state_dict(), dut.target_mac.agent.state_dict(), tag + " target agent")
        _assert_state_dicts_equal(ref.mixer.state_dict(), dut.mixer.state_dict(), tag + " mixer")
        _assert_state_dicts_equal(ref.target_mixer.state_dict(), dut.target_mixer.state_dict(), tag + " target mixer")
        _assert_state_dicts_equal(ref.wm.state_dict(), dut.wm.state_dict(), tag + " wm")
        assert len(st_r) == len(st_d) == 2
        for i, (a, b) in enumerate(zip(st_r, st_d)):
            assert set(a) == set(b), f"[{tag}] logged keys differ at step {i}: {set(a) ^ set(b)}"
            for k in a:
                assert a[k] == b[k], f"[{tag}] stat {k} differs at step {i}: {a[k]} vs {b[k]}"
            assert "traj_grad_norm" in a and "loss_kl" in a, f"[{tag}] expected WM stats logged at step {i}"


if __name__ == "__main__":
    test_agent_equivalence()
    print("DesignSaleQAgent == HPNSaleQAgent (sem, avgl1; flags absent and at defaults) PASSED")
    test_learner_equivalence()
    print("DesignWMLearner == SaleqWMLearner (2 train steps: params + stats bit-identical) PASSED")
