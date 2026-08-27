# =====================================================================================================
# ABLATION TEST -- serves ablation_arm_5a (mixer_state="traj") and ablation_arm_5b (mixer_state="none").
# Copied from: src/tests/test_saleq_wm.py (that file is NOT modified).
# What changed: build_args() takes mixer_state and uses learner="ablation_mixstate_learner"; the train-step
# test is parametrized over ("traj", "none") and asserts the mixer input dim is wm_traj_dim / 1, losses finite,
# agent + mixer + WM update, target update fires; adds a save/load round-trip (incl. sem_wm.th) and a check
# that a missing / invalid mixer_state raises ValueError. The model-property test is dropped (unchanged model).
#
# Run from src with:  python -m tests.test_ablation_mixstate   (or pytest)
# =====================================================================================================
import os
import sys
import tempfile
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src

import torch as th
from components.episode_buffer import EpisodeBatch
from components.transforms import OneHot
from controllers import REGISTRY as mac_REGISTRY
from learners import REGISTRY as le_REGISTRY

LEARNER = "ablation_mixstate_learner"


def build_args(mixer_state):
    n_agents, n_enemies, n_normal = 3, 3, 6
    n_actions = n_normal + n_enemies
    ns = SimpleNamespace(
        n_agents=n_agents, n_actions=n_actions, n_enemies=n_enemies, n_allies=2,
        output_normal_actions=n_normal, obs_component=[4, (n_enemies, 5), (2, 5), 1],
        state_shape=48, rnn_hidden_dim=64, hpn_head_num=1, hpn_hyper_dim=64,
        saleq_emb_dim=32, saleq_zsa_dim=64, saleq_hidden_dim=64,
        saleq_norm="sem", saleq_sem_groups=8,
        obs_agent_id=True, obs_last_action=False, map_type="marines",
        agent="hpn_saleq", mac="hpn_mac", agent_output_type="q",
        learner=LEARNER,
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
    if mixer_state is not None:
        ns.mixer_state = mixer_state
    return ns


class _NullConsole:
    def info(self, *a, **k):
        pass


class _RecLogger:
    console_logger = _NullConsole()

    def __init__(self):
        self.stats = {}

    def log_stat(self, k, v, t):
        self.stats[k] = v


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


def _run_arm(mixer_state):
    th.manual_seed(0)
    args = build_args(mixer_state)
    batch, scheme, groups = make_batch(args)
    mac = mac_REGISTRY["hpn_mac"](batch.scheme, groups, args)
    logger = _RecLogger()
    learner = le_REGISTRY[LEARNER](mac, batch.scheme, logger, args)

    expected_dim = args.wm_traj_dim if mixer_state == "traj" else 1
    assert learner.mixer.input_dim == expected_dim, (mixer_state, learner.mixer.input_dim)
    assert learner.target_mixer.input_dim == expected_dim

    agent_before = [p.detach().clone() for p in mac.parameters()]
    mixer_before = [p.detach().clone() for p in learner.mixer.parameters()]
    wm_before = [p.detach().clone() for p in learner.wm.parameters()]

    learner.train(batch, t_env=0, episode_num=0)       # log_now=True on the first call -> stats recorded
    learner.train(batch, t_env=3000, episode_num=1)

    assert any(not th.equal(a, b) for a, b in zip(agent_before, mac.parameters())), "agent did not update"
    assert any(not th.equal(a, b) for a, b in zip(mixer_before, learner.mixer.parameters())), "mixer did not update"
    assert any(not th.equal(a, b) for a, b in zip(wm_before, learner.wm.parameters())), "world model did not update"
    assert all(th.isfinite(p).all() for p in mac.parameters()), "non-finite agent params"
    assert all(th.isfinite(p).all() for p in learner.wm.parameters()), "non-finite WM params"
    assert learner.last_target_update_episode > 0, "target update did not fire"
    for k in ("loss_td", "loss_kl", "loss_rec", "loss_pred", "loss_act", "kl_drift_h1"):
        assert k in logger.stats and th.isfinite(th.tensor(float(logger.stats[k]))), f"{k} missing / non-finite"
    if mixer_state == "traj":
        assert "traj_grad_norm" in logger.stats, "traj_grad_norm must still be logged in the traj arm"
    else:
        assert "traj_grad_norm" not in logger.stats, "no traj exists in the none arm"

    masked, _, _ = make_batch(args, avail_upto=3)
    mac.init_hidden(masked.batch_size)
    picked = mac.select_actions(masked, t_ep=0, t_env=0, test_mode=True)
    assert (picked < 3).all(), "select_actions chose an unavailable action"

    # save / load round-trip incl. sem_wm.th
    with tempfile.TemporaryDirectory() as d:
        learner.save_models(d)
        assert os.path.exists(os.path.join(d, "sem_wm.th"))
        args2 = build_args(mixer_state)
        mac2 = mac_REGISTRY["hpn_mac"](batch.scheme, groups, args2)
        learner2 = le_REGISTRY[LEARNER](mac2, batch.scheme, _RecLogger(), args2)
        learner2.load_models(d)
        for a, b in zip(learner.wm.parameters(), learner2.wm.parameters()):
            assert th.equal(a, b), "WM state did not round-trip"
        for a, b in zip(learner.mixer.parameters(), learner2.mixer.parameters()):
            assert th.equal(a, b), "mixer state did not round-trip"


def test_mixstate_traj():
    _run_arm("traj")


def test_mixstate_none():
    _run_arm("none")


def test_mixstate_requires_arg():
    for bad in (None, "s_traj", "s"):
        args = build_args(bad)
        batch, scheme, groups = make_batch(args)
        mac = mac_REGISTRY["hpn_mac"](batch.scheme, groups, args)
        try:
            le_REGISTRY[LEARNER](mac, batch.scheme, _RecLogger(), args)
        except ValueError:
            continue
        raise AssertionError(f"mixer_state={bad!r} should have raised ValueError")


if __name__ == "__main__":
    test_mixstate_requires_arg()
    print("ablation_mixstate_learner: missing/invalid mixer_state raises ValueError PASSED")
    test_mixstate_traj()
    print("ablation_mixstate_learner[traj] (arm 5a): mixer dim 32, losses finite, updates, save/load PASSED")
    test_mixstate_none()
    print("ablation_mixstate_learner[none] (arm 5b): mixer dim 1, losses finite, updates, save/load PASSED")
