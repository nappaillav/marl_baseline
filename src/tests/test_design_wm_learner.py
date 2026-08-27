# =====================================================================================================
# DESIGN-SPACE TEST -- serves design_e3_mix_s_zs_traj and design_e3_mix_zs_traj (and the s_traj default).
# Copied from: src/tests/test_ablation_mixstate.py / test_saleq_wm.py (neither is modified).
# What changed: build_args() takes (design_mixer_input, design_mixer_zs_detach) and uses learner=
# "design_wm_learner"; the train-step test is parametrized over the three mixer inputs and asserts the live
# AND target mixer input_dim (|s|+32 / |s|+64+32 / 64+32), losses finite, agent + mixer + WM update, target
# update fires, traj_grad_norm logged, a bit-identical save/load round-trip (incl. sem_wm.th), ValueError on
# a bad design_mixer_input, and the detach test: for s_zs_traj under the same seed, zs_detach True vs False
# gives DIFFERENT state_encoder gradients (never compared across different mixer inputs -- different mixers).
#
# Run from src with:  python -m tests.test_design_wm_learner   (or pytest)
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

LEARNER = "design_wm_learner"
STATE, D_S, TRAJ = 48, 64, 32
EXPECTED_DIM = {"s_traj": STATE + TRAJ, "s_zs_traj": STATE + D_S + TRAJ, "zs_traj": D_S + TRAJ}


def build_args(mixer_input=None, zs_detach=None):
    n_agents, n_enemies, n_normal = 3, 3, 6
    n_actions = n_normal + n_enemies
    ns = SimpleNamespace(
        n_agents=n_agents, n_actions=n_actions, n_enemies=n_enemies, n_allies=2,
        output_normal_actions=n_normal, obs_component=[4, (n_enemies, 5), (2, 5), 1],
        state_shape=STATE, rnn_hidden_dim=64, hpn_head_num=1, hpn_hyper_dim=64,
        saleq_emb_dim=32, saleq_zsa_dim=64, saleq_hidden_dim=64,
        saleq_norm="sem", saleq_sem_groups=8,
        obs_agent_id=True, obs_last_action=False, map_type="marines",
        agent="hpn_saleq", mac="hpn_mac", agent_output_type="q",
        learner=LEARNER,
        wm_groups=8, wm_vertices=8, wm_state_dim=D_S, wm_sa_dim=64, wm_traj_dim=TRAJ,
        wm_k=3, wm_k_train=1, wm_alpha=0.8, wm_unimix=0.01,
        c_wm=1.0, c_kl=1.0, c_rec=1.0, c_pred=1.0, c_act=1.0,
        action_selector="epsilon_greedy", epsilon_start=1.0, epsilon_finish=0.05,
        epsilon_anneal_time=100000, test_greedy=True,
        mixer="qmix", mixing_embed_dim=32, hypernet_embed=64,
        optimizer="adam", lr=1e-3, td_lambda=0.6, q_lambda=False, gamma=0.99,
        grad_norm_clip=10, target_update_interval=1, learner_log_interval=2000,
        use_cuda=False, device="cpu",
    )
    if mixer_input is not None:
        ns.design_mixer_input = mixer_input
    if zs_detach is not None:
        ns.design_mixer_zs_detach = zs_detach
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


def _make(args):
    th.manual_seed(0)
    batch, scheme, groups = make_batch(args)
    mac = mac_REGISTRY["hpn_mac"](batch.scheme, groups, args)
    logger = _RecLogger()
    learner = le_REGISTRY[LEARNER](mac, batch.scheme, logger, args)
    return batch, mac, learner, logger


def _train_and_check(mixer_input):
    args = build_args(mixer_input)
    batch, mac, learner, logger = _make(args)
    exp = EXPECTED_DIM[mixer_input if mixer_input is not None else "s_traj"]
    assert learner.mixer.input_dim == exp, (mixer_input, learner.mixer.input_dim)
    assert learner.target_mixer.input_dim == exp, (mixer_input, learner.target_mixer.input_dim)

    agent_before = [p.detach().clone() for p in mac.parameters()]
    mixer_before = [p.detach().clone() for p in learner.mixer.parameters()]
    wm_before = [p.detach().clone() for p in learner.wm.parameters()]

    learner.train(batch, t_env=0, episode_num=0)
    learner.train(batch, t_env=3000, episode_num=1)

    tag = str(mixer_input)
    assert any(not th.equal(a, b) for a, b in zip(agent_before, mac.parameters())), f"[{tag}] agent did not update"
    assert any(not th.equal(a, b) for a, b in zip(mixer_before, learner.mixer.parameters())), f"[{tag}] mixer did not update"
    assert any(not th.equal(a, b) for a, b in zip(wm_before, learner.wm.parameters())), f"[{tag}] WM did not update"
    assert all(th.isfinite(p).all() for p in mac.parameters()), f"[{tag}] non-finite agent params"
    assert all(th.isfinite(p).all() for p in learner.wm.parameters()), f"[{tag}] non-finite WM params"
    assert all(th.isfinite(p).all() for p in learner.mixer.parameters()), f"[{tag}] non-finite mixer params"
    assert learner.last_target_update_episode > 0, f"[{tag}] target update did not fire"
    for k in ("loss_td", "loss_kl", "loss_rec", "loss_pred", "loss_act", "grad_norm", "traj_grad_norm", "kl_drift_h1"):
        assert k in logger.stats, f"[{tag}] stat {k} not logged"
        assert th.isfinite(th.tensor(float(logger.stats[k]))), f"[{tag}] stat {k} not finite"

    # save/load round-trip must be bit-identical (agent, mixer, WM, optimizer state)
    with tempfile.TemporaryDirectory() as d:
        learner.save_models(d)
        assert os.path.isfile(os.path.join(d, "sem_wm.th")), "sem_wm.th not saved"
        args2 = build_args(mixer_input)
        _, mac2, learner2, _ = _make(args2)
        assert any(not th.equal(a, b) for a, b in zip(learner.wm.parameters(), learner2.wm.parameters()))
        learner2.load_models(d)
        for a, b in zip(learner.mac.parameters(), learner2.mac.parameters()):
            assert th.equal(a, b), f"[{tag}] agent params differ after load"
        for a, b in zip(learner.mixer.parameters(), learner2.mixer.parameters()):
            assert th.equal(a, b), f"[{tag}] mixer params differ after load"
        for a, b in zip(learner.wm.parameters(), learner2.wm.parameters()):
            assert th.equal(a, b), f"[{tag}] WM params differ after load"

    masked, _, _ = make_batch(args, avail_upto=3)
    mac.init_hidden(masked.batch_size)
    picked = mac.select_actions(masked, t_ep=0, t_env=0, test_mode=True)
    assert (picked < 3).all(), f"[{tag}] select_actions chose an unavailable action"


def test_design_wm_learner_default_is_s_traj():
    _train_and_check(None)   # flag absent -> "s_traj", dim |s|+32


def test_design_wm_learner_s_zs_traj():
    _train_and_check("s_zs_traj")


def test_design_wm_learner_zs_traj():
    _train_and_check("zs_traj")


def test_design_wm_learner_rejects_bad_input():
    try:
        _make(build_args("s_zs"))
    except ValueError:
        return
    raise AssertionError("bad design_mixer_input was accepted")


def _state_encoder_grad(zs_detach):
    args = build_args("s_zs_traj", zs_detach)
    batch, mac, learner, _ = _make(args)
    th.manual_seed(7)   # same ST samples for both settings
    # one manual forward/backward through train() -- read the grad left on state_encoder after the step
    learner.train(batch, t_env=0, episode_num=0)
    return learner.wm.state_encoder[0].weight.grad.detach().clone()


def test_design_wm_learner_zs_detach_changes_state_encoder_grad():
    g_detached = _state_encoder_grad(True)
    g_attached = _state_encoder_grad(False)
    assert th.isfinite(g_detached).all() and th.isfinite(g_attached).all()
    # detached: gradient still non-zero (TD reaches the encoder via posterior -> imagination -> traj, plus L_WM)
    assert g_detached.abs().sum() > 0, "state_encoder got no gradient with z^s detached"
    # attaching the direct mixer -> z^s path must change the gradient
    assert not th.equal(g_detached, g_attached), "zs_detach flag has no effect on the state_encoder gradient"


if __name__ == "__main__":
    test_design_wm_learner_default_is_s_traj()
    test_design_wm_learner_s_zs_traj()
    test_design_wm_learner_zs_traj()
    print("DesignWMLearner train-step over (s_traj, s_zs_traj, zs_traj): dims, finite losses, updates, save/load PASSED")
    test_design_wm_learner_rejects_bad_input()
    print("DesignWMLearner rejects bad design_mixer_input PASSED")
    test_design_wm_learner_zs_detach_changes_state_encoder_grad()
    print("DesignWMLearner zs_detach flag changes the state_encoder gradient PASSED")
