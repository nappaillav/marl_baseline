"""Unit test for HPNSaleQTerranAgent (SMACv2 10gen_terran) on the HPN-QMIX / world-model backbone.

The agent's PI/PE properties are covered by its smoke block
(`python -m modules.agents.hpn_saleq_terran`). What this adds is the *integration* path the smoke
block cannot reach: registry lookup, hpn_mac's entity split, TD(lambda), the monotone mixer,
epsilon-greedy, and the world-model learner -- on 10gen_terran-shaped data.

Run from src with:  python -m tests.test_hpn_saleq_terran
"""
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src

import torch as th
from components.episode_buffer import EpisodeBatch
from components.transforms import OneHot
from controllers import REGISTRY as mac_REGISTRY
from learners import REGISTRY as le_REGISTRY


def build_args(norm="sem", learner="saleq_wm_learner"):
    # 10gen_terran: 5 agents, 5 enemies, 4 allies, |A| = 6 + 5 = 11
    # obs_component = [move 4, (5, 8) enemy, (4, 8) ally, own 6]; own ends with the 3 unit-type bits.
    n_agents, n_enemies, n_normal = 5, 5, 6
    n_allies = n_agents - 1
    n_actions = n_normal + n_enemies
    return SimpleNamespace(
        n_agents=n_agents, n_actions=n_actions, n_enemies=n_enemies, n_allies=n_allies,
        output_normal_actions=n_normal, obs_component=[4, (n_enemies, 8), (n_allies, 8), 6],
        state_shape=100, rnn_hidden_dim=64, hpn_head_num=1, hpn_hyper_dim=64,
        saleq_emb_dim=32, saleq_zsa_dim=64, saleq_hidden_dim=64,
        saleq_norm=norm, saleq_sem_groups=8,
        obs_agent_id=True, obs_last_action=False, map_type="terran_gen",
        agent="hpn_saleq_terran", mac="hpn_mac", agent_output_type="q", learner=learner,
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


class _NullConsole:
    def info(self, *a, **k):
        pass


class _NullLogger:
    console_logger = _NullConsole()

    def log_stat(self, k, v, t):
        pass


def make_batch(args, bs=2, T=6, avail_upto=None, healers=(4,)):
    n_agents, n_actions, state_shape = args.n_agents, args.n_actions, args.state_shape
    obs_shape = 4 + args.n_enemies * 8 + args.n_allies * 8 + 6
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

    obs = th.randn(bs, T, n_agents, obs_shape)
    # own_feats are the final 6 entries; the last 3 are the unit-type one-hot.
    obs[..., -3:] = 0.0
    obs[..., -3] = 1.0                       # marine (type_id 0) by default
    for i in healers:
        obs[:, :, i, -3:] = th.tensor([0., 0., 1.])   # medivac (type_id 2)

    data = {
        "state": th.randn(bs, T, state_shape),
        "obs": obs,
        "avail_actions": avail,
        "actions": th.randint(0, hi, (bs, T, n_agents, 1)),
        "reward": th.randn(bs, T, 1),
        "terminated": th.zeros(bs, T, 1, dtype=th.uint8),
    }
    data["terminated"][:, T - 2] = 1
    batch.update(data, ts=slice(0, T), mark_filled=True)
    return batch, scheme, groups


def _train_and_check(args, healers):
    th.manual_seed(0)
    batch, scheme, groups = make_batch(args, healers=healers)
    mac = mac_REGISTRY["hpn_mac"](batch.scheme, groups, args)
    learner = le_REGISTRY[args.learner](mac, batch.scheme, _NullLogger(), args)

    agent_before = [p.detach().clone() for p in mac.parameters()]
    named = dict(mac.agent.named_parameters())
    # both entity encoders must train -- a silent detach on the heal branch would leave
    # ally_encoder frozen forever, which no shape check would reveal
    head_keys = ("enemy_encoder.0.weight", "ally_encoder.0.weight", "no_target",
                 "phi.0.weight", "head.0.weight")
    for k in head_keys:
        assert k in named, f"missing head param {k}"
    head_before = {k: named[k].detach().clone() for k in head_keys}

    learner.train(batch, t_env=0, episode_num=0)
    learner.train(batch, t_env=3000, episode_num=1)

    assert any(not th.equal(a, b) for a, b in zip(agent_before, mac.parameters())), "agent did not update"
    assert all(th.isfinite(p).all() for p in mac.parameters()), "non-finite agent params"
    named_after = dict(mac.agent.named_parameters())
    for k in head_keys:
        if k == "ally_encoder.0.weight" and not healers:
            # Correct and expected: with k=0 healers, th.where selects enemy embeddings everywhere, so
            # ally_encoder is not on the computation graph and receives no gradient. It must therefore
            # stay UNCHANGED -- asserting the opposite would be asserting an impossibility.
            assert th.equal(head_before[k], named_after[k]), \
                "ally_encoder moved with no healers present -- it should be off the graph entirely"
            continue
        assert not th.equal(head_before[k], named_after[k]), f"head param {k} did not update"

    masked, _, _ = make_batch(args, avail_upto=3, healers=healers)
    mac.init_hidden(masked.batch_size)
    picked = mac.select_actions(masked, t_ep=0, t_env=0, test_mode=True)
    assert (picked < 3).all(), "select_actions chose an unavailable action"


def test_wm_learner_sem_tail_healer():
    _train_and_check(build_args("sem"), healers=(4,))


def test_wm_learner_two_healers():
    _train_and_check(build_args("sem"), healers=(3, 4))


def test_wm_learner_no_healer():
    # k=0 is the most common episode type; the head must degrade to pure-attack cleanly
    _train_and_check(build_args("sem"), healers=())


def test_wm_learner_non_tail_healer():
    # the ordering-independence claim, exercised through the full learner
    _train_and_check(build_args("sem"), healers=(1,))


def test_nq_learner_avgl1():
    _train_and_check(build_args("avgl1", learner="nq_learner"), healers=(4,))


def test_registry_key_matches_config():
    import yaml
    from modules.agents import REGISTRY as agent_REGISTRY
    cfg = yaml.safe_load(open(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config", "algs", "hpn_saleq_terran_sem_wm_qmix.yaml")))
    assert cfg["agent"] in agent_REGISTRY, "config's agent key is not registered"
    assert cfg["wm_k"] == 5, "wm_k should stay at the validated imagination depth"


def test_parent_rejects_terran_gen():
    # the defensive guard: HPNSaleQAgent must refuse terran_gen loudly rather than mis-score heals
    from modules.agents.hpn_saleq_agent import HPNSaleQAgent
    args = build_args("sem")
    try:
        HPNSaleQAgent((10, (5, 8), (4, 8)), args)
    except AssertionError as e:
        assert "hpn_saleq_terran" in str(e), "guard should point at the correct agent"
        return
    raise AssertionError("HPNSaleQAgent accepted terran_gen -- the defensive guard is missing!")


if __name__ == "__main__":
    test_registry_key_matches_config()
    print("registry/config consistency PASSED")
    test_parent_rejects_terran_gen()
    print("defensive guard on HPNSaleQAgent(terran_gen) PASSED")
    test_wm_learner_no_healer()
    print("WM learner, k=0 healers PASSED")
    test_wm_learner_sem_tail_healer()
    print("WM learner, k=1 tail healer (SEM) PASSED")
    test_wm_learner_two_healers()
    print("WM learner, k=2 healers PASSED")
    test_wm_learner_non_tail_healer()
    print("WM learner, NON-TAIL healer PASSED")
    test_nq_learner_avgl1()
    print("NQLearner, avgl1 norm PASSED")
