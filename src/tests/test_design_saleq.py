# =====================================================================================================
# DESIGN-SPACE TEST -- serves design_b1_norm_{ln,none,sem_noln} and design_b3_temp_{0p1,0p5,2}.
# Copied from: src/tests/test_hpn_saleq.py (that file is NOT modified).
# What changed: build_args() takes (norm, tau) and uses agent="design_saleq"; the train-step test is
# parametrized over norm in (ln, none, sem_noln) and (sem, tau in {0.1, 2}); asserts finite loss/params, agent +
# head + mixer params update, target update fires, no unavailable action selected; plus a PI/PE check of the
# built agent under each setting (enemy-permutation invariance of normal Q / v_tilde, exact equivariance of
# attack Q, ally-permutation invariance) and a check that an unknown saleq_norm is rejected.
#
# Run from src with:  python -m tests.test_design_saleq   (or pytest)
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
from modules.agents import REGISTRY as agent_REGISTRY

SETTINGS = [("ln", 1.0), ("none", 1.0), ("sem_noln", 1.0), ("sem", 0.1), ("sem", 2.0)]


def build_args(norm="sem", tau=1.0):
    n_agents, n_enemies, n_normal = 3, 3, 6
    n_actions = n_normal + n_enemies
    return SimpleNamespace(
        n_agents=n_agents, n_actions=n_actions, n_enemies=n_enemies, n_allies=2,
        output_normal_actions=n_normal, obs_component=[4, (n_enemies, 5), (2, 5), 1],
        state_shape=48, rnn_hidden_dim=64, hpn_head_num=1, hpn_hyper_dim=64,
        saleq_emb_dim=32, saleq_zsa_dim=64, saleq_hidden_dim=64,
        saleq_norm=norm, saleq_sem_groups=8, saleq_sem_temp=tau,
        obs_agent_id=True, obs_last_action=False, map_type="marines",
        agent="design_saleq", mac="hpn_mac", agent_output_type="q",
        action_selector="epsilon_greedy", epsilon_start=1.0, epsilon_finish=0.05,
        epsilon_anneal_time=100000, test_greedy=True,
        mixer="qmix", mixing_embed_dim=32, hypernet_embed=64,
        optimizer="adam", lr=1e-3, td_lambda=0.6, q_lambda=False, gamma=0.99,
        grad_norm_clip=10, target_update_interval=1, learner_log_interval=2000,
        use_cuda=False, device="cpu",
    )


class _NullConsole:
    def info(self, *a, **k):
        pass


class _NullLogger:
    console_logger = _NullConsole()

    def log_stat(self, k, v, t):
        pass


def make_batch(args, bs=2, T=6, avail_upto=None):
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
    avail = th.ones(bs, T, n_agents, n_actions, dtype=th.int)
    if avail_upto is not None:
        avail[..., avail_upto:] = 0
    hi = avail_upto if avail_upto is not None else n_actions
    data = {
        "state": th.randn(bs, T, state_shape),
        "obs": th.randn(bs, T, n_agents, obs_shape),
        "avail_actions": avail,
        "actions": th.randint(0, hi, (bs, T, n_agents, 1)),
        "reward": th.randn(bs, T, 1),
        "terminated": th.zeros(bs, T, 1, dtype=th.uint8),
    }
    data["terminated"][:, T - 2] = 1
    batch.update(data, ts=slice(0, T), mark_filled=True)
    return batch, scheme, groups


def _pi_pe_check(args):
    """Enemy/ally permutation properties of the built agent (mirrors the source smoke), any norm / tau."""
    th.manual_seed(0)
    bs, n, ne, na, n_normal = 2, args.n_agents, args.n_enemies, args.n_allies, args.output_normal_actions
    agent = agent_REGISTRY["design_saleq"]((5, (ne, 5), (na, 5)), args).eval()
    own, enemy, ally = th.randn(bs * n, 5), th.randn(bs * n * ne, 5), th.randn(bs * n * na, 5)
    idx = [th.arange(n).unsqueeze(0).expand(bs, -1)]
    h = th.zeros(bs, n, args.rnn_hidden_dim)
    perm, aperm = th.tensor([2, 0, 1]), th.tensor([1, 0])
    enemy_p = enemy.view(bs * n, ne, 5)[:, perm, :].reshape(bs * n * ne, 5)
    ally_p = ally.view(bs * n, na, 5)[:, aperm, :].reshape(bs * n * na, 5)
    with th.no_grad():
        q, v = agent((bs, own, enemy, ally, idx), h)
        q_p, v_p = agent((bs, own, enemy_p, ally, idx), h)
        q_ap, _ = agent((bs, own, enemy, ally_p, idx), h)
    tag = f"{args.saleq_norm}, tau={args.saleq_sem_temp}"
    assert th.isfinite(q).all(), f"[{tag}] non-finite Q"
    assert th.allclose(q_p[..., :n_normal], q[..., :n_normal], atol=1e-5), f"[{tag}] normal Q not enemy-invariant"
    assert th.allclose(v_p, v, atol=1e-5), f"[{tag}] v_tilde not enemy-invariant"
    assert th.allclose(q_p[..., n_normal:], q[..., n_normal:][..., perm], atol=1e-5), f"[{tag}] attack Q not PE"
    assert th.allclose(q_ap, q, atol=1e-5), f"[{tag}] Q not ally-invariant"
    has_ln = any(k.startswith("sem_ln.") for k in agent.state_dict())
    assert has_ln == (args.saleq_norm in ("sem", "ln")), f"[{tag}] sem_ln presence {has_ln}"


def _train_and_check(args):
    th.manual_seed(0)
    batch, scheme, groups = make_batch(args)
    mac = mac_REGISTRY["hpn_mac"](batch.scheme, groups, args)
    learner = le_REGISTRY["nq_learner"](mac, batch.scheme, _NullLogger(), args)

    agent_before = [p.detach().clone() for p in mac.parameters()]
    mixer_before = [p.detach().clone() for p in learner.mixer.parameters()]
    head_keys = ("enemy_encoder.0.weight", "no_target", "phi.0.weight", "head.0.weight")
    named = dict(mac.agent.named_parameters())
    for key in head_keys:
        assert key in named, f"missing head param {key}"
    head_before = {k: named[k].detach().clone() for k in head_keys}

    learner.train(batch, t_env=0, episode_num=0)
    learner.train(batch, t_env=3000, episode_num=1)  # 2nd step: triggers the target update (interval=1)

    tag = f"{args.saleq_norm}, tau={args.saleq_sem_temp}"
    assert any(not th.equal(a, b) for a, b in zip(agent_before, mac.parameters())), f"[{tag}] agent did not update"
    assert any(not th.equal(a, b) for a, b in zip(mixer_before, learner.mixer.parameters())), f"[{tag}] mixer did not update"
    assert all(th.isfinite(p).all() for p in mac.parameters()), f"[{tag}] non-finite agent params"
    assert learner.last_target_update_episode > 0, f"[{tag}] target update did not fire"
    named_after = dict(mac.agent.named_parameters())
    for key in head_keys:
        assert not th.equal(head_before[key], named_after[key]), f"[{tag}] head param {key} did not update"

    masked, _, _ = make_batch(args, avail_upto=3)
    mac.init_hidden(masked.batch_size)
    picked = mac.select_actions(masked, t_ep=0, t_env=0, test_mode=True)
    assert (picked < 3).all(), f"[{tag}] select_actions chose an unavailable action"


def test_design_saleq_pi_pe():
    for norm, tau in SETTINGS:
        _pi_pe_check(build_args(norm, tau))


def test_design_saleq_train_step():
    for norm, tau in SETTINGS:
        _train_and_check(build_args(norm, tau))


def test_design_saleq_rejects_unknown_norm():
    args = build_args("batchnorm")
    try:
        agent_REGISTRY["design_saleq"]((5, (3, 5), (2, 5)), args)
    except AssertionError:
        return
    raise AssertionError("unknown saleq_norm was accepted")


if __name__ == "__main__":
    test_design_saleq_pi_pe()
    print("DesignSaleQAgent PI/PE under (ln, none, sem_noln, sem tau=0.1, sem tau=2) PASSED")
    test_design_saleq_train_step()
    print("DesignSaleQAgent NQLearner train-step under all 5 settings PASSED")
    test_design_saleq_rejects_unknown_norm()
    print("DesignSaleQAgent rejects unknown saleq_norm PASSED")
