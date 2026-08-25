"""Unit test for Phase-2 thread C (SaleqWMLearner + SemWorldModel), phase2-plan_v2 §3.4 / gate GC.0.
2 CPU train steps through the real learner + hpn_mac on a synthetic 3m-shaped batch (no SC2 launch).

Run from src with:  python -m tests.test_saleq_wm
Checks: model — shapes, latent simplicial property, prior causality, straight-through gradient, and the
rollout-leakage guard (the posterior/traj at t do not see future actions); learner — 2 train steps with all
losses (incl. L_kl/L_rec/L_pred/L_act) finite, WM + agent + augmented-mixer params update, target update fires,
no unavailable action selected, and the mixer reads state+traj of the augmented dim.
"""
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src

import torch as th
import torch.nn.functional as F
from components.episode_buffer import EpisodeBatch
from components.transforms import OneHot
from controllers import REGISTRY as mac_REGISTRY
from learners import REGISTRY as le_REGISTRY
from modules.model.sem_wm import SemWorldModel


def build_args():
    n_agents, n_enemies, n_normal = 3, 3, 6
    n_actions = n_normal + n_enemies
    return SimpleNamespace(
        n_agents=n_agents, n_actions=n_actions, n_enemies=n_enemies, n_allies=2,
        output_normal_actions=n_normal, obs_component=[4, (n_enemies, 5), (2, 5), 1],
        state_shape=48, rnn_hidden_dim=64, hpn_head_num=1, hpn_hyper_dim=64,
        saleq_emb_dim=32, saleq_zsa_dim=64, saleq_hidden_dim=64,
        saleq_norm="sem", saleq_sem_groups=8,
        obs_agent_id=True, obs_last_action=False, map_type="marines",
        agent="hpn_saleq", mac="hpn_mac", agent_output_type="q",
        learner="saleq_wm_learner",
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


def test_wm_model_properties():
    th.manual_seed(0)
    args = build_args()
    wm = SemWorldModel(args)
    bs, T, n, nA = 2, 5, args.n_agents, args.n_actions
    s = th.randn(bs, T, args.state_shape)
    a = F.one_hot(th.randint(0, nA, (bs, T, n)), nA).float()

    z_s = wm.state_embed(s)
    assert z_s.shape == (bs, T, 64)
    assert th.allclose(z_s.reshape(bs, T, 8, 8).sum(-1), th.ones(bs, T, 8), atol=1e-5), "z^s not simplicial"

    # prior causality + posterior/traj no future-action leak (seed the ST sampling to isolate the perturbation)
    def zq(state, act):
        th.manual_seed(3)
        return wm.st_sample(wm.probs(wm.posterior_logits(wm.state_embed(state), wm.sa_embed(state, act))))
    z_base = zq(s, a)
    s2 = s.clone(); s2[:, 3] += 10.0
    p_base = wm.prior_logits(z_base[:, :-1], a[:, :-1])
    p_pert = wm.prior_logits(zq(s2, a)[:, :-1], a[:, :-1])
    assert th.allclose(p_base[:, 2], p_pert[:, 2], atol=1e-5), "prior for t leaked step-t state (causality)"
    # posterior at t must not see FUTURE actions -> traj_t (= f(z_q[t])) cannot leak future behavior
    a2 = a.clone(); a2[:, 4] = F.one_hot((a[:, 4].argmax(-1) + 1) % nA, nA).float()  # perturb last action
    z_fut = zq(s, a2)
    assert th.allclose(z_base[:, :4], z_fut[:, :4], atol=1e-5), "posterior at t leaked a future action (rollout guard)"
    # and the mixer-facing quantity itself: traj_t = A(imagine(z_q[t])) must not move when a FUTURE action changes
    def traj_of(z):
        th.manual_seed(11)
        return wm.aggregate(wm.imagine(z, wm.k))
    assert th.allclose(traj_of(z_base)[:, :4], traj_of(z_fut)[:, :4], atol=1e-5), \
        "traj_t leaked a future action (W7 imagination-rollout guard)"

    # straight-through gradient reaches posterior + prior nets
    z_g = wm.st_sample(wm.probs(wm.posterior_logits(wm.state_embed(s), wm.sa_embed(s, a))))
    wm.aggregate(wm.imagine(z_g, wm.k)).sum().backward()
    assert wm.posterior[0].weight.grad is not None, "no ST gradient into posterior"
    assert wm.prior[0].weight.grad is not None, "no ST gradient into prior (via imagination)"


def test_saleq_wm_train_step():
    th.manual_seed(0)
    args = build_args()
    batch, scheme, groups = make_batch(args)
    mac = mac_REGISTRY["hpn_mac"](batch.scheme, groups, args)
    learner = le_REGISTRY["saleq_wm_learner"](mac, batch.scheme, _NullLogger(), args)

    # mixer must read the augmented state dim (state_shape + wm_traj_dim)
    assert learner.mixer.input_dim == args.state_shape + args.wm_traj_dim, learner.mixer.input_dim

    agent_before = [p.detach().clone() for p in mac.parameters()]
    mixer_before = [p.detach().clone() for p in learner.mixer.parameters()]
    wm_before = [p.detach().clone() for p in learner.wm.parameters()]

    learner.train(batch, t_env=0, episode_num=0)
    learner.train(batch, t_env=3000, episode_num=1)

    assert any(not th.equal(a, b) for a, b in zip(agent_before, mac.parameters())), "agent did not update"
    assert any(not th.equal(a, b) for a, b in zip(mixer_before, learner.mixer.parameters())), "mixer did not update"
    assert any(not th.equal(a, b) for a, b in zip(wm_before, learner.wm.parameters())), "world model did not update"
    assert all(th.isfinite(p).all() for p in mac.parameters()), "non-finite agent params"
    assert all(th.isfinite(p).all() for p in learner.wm.parameters()), "non-finite WM params"
    assert learner.last_target_update_episode > 0, "target update did not fire"

    masked, _, _ = make_batch(args, avail_upto=3)
    mac.init_hidden(masked.batch_size)
    picked = mac.select_actions(masked, t_ep=0, t_env=0, test_mode=True)
    assert (picked < 3).all(), "select_actions chose an unavailable action"


if __name__ == "__main__":
    test_wm_model_properties()
    print("SEM world-model properties (simplicial, causality, ST-grad, rollout guard) PASSED")
    test_saleq_wm_train_step()
    print("SaleqWMLearner train-step (losses finite, WM+agent+mixer update, augmented mixer) PASSED")
