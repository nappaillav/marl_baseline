# =====================================================================================================
# ABLATION TEST -- serves ablation_arm_1 (arm 1B: flat trunk, no HPN; agent "flat_saleq").
# Copied from: src/tests/test_hpn_saleq.py (that file is NOT modified).
# What changed: build_args() uses agent="flat_saleq"; adds (a) a direct-forward test that the enemy-permutation
# invariance/equivariance of the HPN trunk does NOT hold for the flat trunk (documents the ablation),
# (b) a train step through the real saleq_wm_learner (the learner arm 1 deploys), (c) parameter counts printed
# for flat_saleq vs hpn_saleq. The NQLearner train-step checks are kept verbatim.
#
# Run from src with:  python -m tests.test_ablation_flat_saleq   (or pytest)
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


def build_args(norm="sem", agent="flat_saleq", learner="nq_learner"):
    n_agents, n_enemies, n_normal = 3, 3, 6
    n_actions = n_normal + n_enemies
    return SimpleNamespace(
        n_agents=n_agents, n_actions=n_actions, n_enemies=n_enemies, n_allies=2,
        output_normal_actions=n_normal, obs_component=[4, (n_enemies, 5), (2, 5), 1],
        state_shape=48, rnn_hidden_dim=64, hpn_head_num=1, hpn_hyper_dim=64,
        saleq_emb_dim=32, saleq_zsa_dim=64, saleq_hidden_dim=64,
        saleq_norm=norm, saleq_sem_groups=8,
        obs_agent_id=True, obs_last_action=False, map_type="marines",
        agent=agent, mac="hpn_mac", agent_output_type="q",
        learner=learner,
        wm_groups=8, wm_vertices=8, wm_state_dim=64, wm_sa_dim=64, wm_traj_dim=32,
        wm_k=3, wm_k_train=1, wm_alpha=0.8, wm_unimix=0.01,
        c_wm=1.0, c_kl=1.0, c_rec=1.0, c_pred=1.0, c_act=1.0,
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


def _train_and_check(args):
    th.manual_seed(0)
    batch, scheme, groups = make_batch(args)
    mac = mac_REGISTRY["hpn_mac"](batch.scheme, groups, args)
    learner = le_REGISTRY[args.learner](mac, batch.scheme, _NullLogger(), args)

    agent_before = [p.detach().clone() for p in mac.parameters()]
    mixer_before = [p.detach().clone() for p in learner.mixer.parameters()]
    head_keys = ("enemy_encoder.0.weight", "no_target", "phi.0.weight", "head.0.weight", "fc1.weight")
    named = dict(mac.agent.named_parameters())
    for key in head_keys:
        assert key in named, f"missing param {key}"
    for gone in ("hyper_enemy.0.weight", "hyper_ally.0.weight", "fc1_own.weight", "fc2_normal_actions.weight"):
        assert gone not in named, f"HPN module {gone} survived in flat_saleq"
    head_before = {k: named[k].detach().clone() for k in head_keys}

    learner.train(batch, t_env=0, episode_num=0)
    learner.train(batch, t_env=3000, episode_num=1)  # 2nd step: triggers the target update (interval=1)

    assert any(not th.equal(a, b) for a, b in zip(agent_before, mac.parameters())), "agent did not update"
    assert any(not th.equal(a, b) for a, b in zip(mixer_before, learner.mixer.parameters())), "mixer did not update"
    assert all(th.isfinite(p).all() for p in mac.parameters()), "non-finite agent params"
    assert learner.last_target_update_episode > 0, "target update did not fire"

    named_after = dict(mac.agent.named_parameters())
    for key in head_keys:
        assert not th.equal(head_before[key], named_after[key]), f"param {key} did not update"

    masked, _, _ = make_batch(args, avail_upto=3)
    mac.init_hidden(masked.batch_size)
    picked = mac.select_actions(masked, t_ep=0, t_env=0, test_mode=True)
    assert (picked < 3).all(), "select_actions chose an unavailable action"


def test_flat_saleq_train_step_nq():
    _train_and_check(build_args("avgl1", learner="nq_learner"))
    _train_and_check(build_args("sem", learner="nq_learner"))


def test_flat_saleq_train_step_wm():  # the learner ablation_arm_1 actually deploys
    _train_and_check(build_args("sem", learner="saleq_wm_learner"))


def test_flat_saleq_loses_permutation_invariance():
    """The ablation's defining property: with the flat trunk, permuting enemy slots changes normal-action Q
    and v_tilde, and the attack-Q is no longer exactly permutation-equivariant. hpn_saleq (the full method)
    is checked side by side to prove the test would catch a PI trunk."""
    th.manual_seed(0)
    bs, n_agents, n_enemies, n_allies = 2, 3, 3, 2
    own_dim, en_dim, al_dim, n_normal = 5, 5, 5, 6
    input_shape = (own_dim, (n_enemies, en_dim), (n_allies, al_dim))
    own = th.randn(bs * n_agents, own_dim)
    enemy = th.randn(bs * n_agents * n_enemies, en_dim)
    ally = th.randn(bs * n_agents * n_allies, al_dim)
    emb_idx = [th.arange(n_agents).unsqueeze(0).expand(bs, -1)]
    h = th.zeros(bs, n_agents, 64)
    perm = th.tensor([2, 0, 1])
    enemy_p = enemy.view(bs * n_agents, n_enemies, en_dim)[:, perm, :].reshape(-1, en_dim)

    def q_of(agent_key):
        args = build_args("sem", agent=agent_key)
        agent = agent_REGISTRY[agent_key](input_shape, args).eval()
        n_params = sum(p.numel() for p in agent.parameters())
        with th.no_grad():
            q, v = agent((bs, own, enemy, ally, emb_idx), h)
            q_p, v_p = agent((bs, own, enemy_p, ally, emb_idx), h)
        return q, v, q_p, v_p, n_params

    q, v, q_p, v_p, n_flat = q_of("flat_saleq")
    assert q.shape == (bs, n_agents, n_normal + n_enemies) and th.isfinite(q).all()
    assert not th.allclose(q_p[..., :n_normal], q[..., :n_normal], atol=1e-5), "flat trunk: normal Q unexpectedly PI"
    assert not th.allclose(v_p, v, atol=1e-5), "flat trunk: v_tilde unexpectedly PI"
    assert not th.allclose(q_p[..., n_normal:], q[..., n_normal:][..., perm], atol=1e-5), \
        "flat trunk: attack Q unexpectedly exactly permutation-equivariant"

    hq, hv, hq_p, hv_p, n_hpn = q_of("hpn_saleq")   # control: the full method's trunk IS PI/PE
    assert th.allclose(hq_p[..., :n_normal], hq[..., :n_normal], atol=1e-5)
    assert th.allclose(hq_p[..., n_normal:], hq[..., n_normal:][..., perm], atol=1e-5)
    print(f"  param count: flat_saleq = {n_flat}, hpn_saleq = {n_hpn}")


if __name__ == "__main__":
    test_flat_saleq_loses_permutation_invariance()
    print("flat_saleq: PI/PE LOST as intended (hpn_saleq control still PI/PE) PASSED")
    test_flat_saleq_train_step_nq()
    print("flat_saleq: NQLearner train step (avgl1 + sem) PASSED")
    test_flat_saleq_train_step_wm()
    print("flat_saleq: SaleqWMLearner train step (ablation_arm_1 stack) PASSED")
