"""Unit test for the value-based adaptation agent (HPNSaleQAgent) on the HPN-QMIX backbone.
Master-plan sec.5.2: the test_nq_learner scaffold with agent="hpn_saleq" -- 2 CPU train steps through
the real NQLearner + hpn_mac + monotone mixer on a synthetic 3m-shaped batch (no SC2 launch).

Run from src with:  python -m tests.test_hpn_saleq
Checks: 2 NQLearner.train() steps run end-to-end (loss finite via the ported assert; agent params
update -- including the head's Enc/no_target/phi/head; mixer updates; target update fires), and
epsilon_greedy select_actions never picks an unavailable action.
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


def build_args(norm="avgl1"):
    n_agents, n_enemies, n_normal = 3, 3, 6
    n_actions = n_normal + n_enemies
    return SimpleNamespace(
        n_agents=n_agents, n_actions=n_actions, n_enemies=n_enemies, n_allies=2,
        output_normal_actions=n_normal, obs_component=[4, (n_enemies, 5), (2, 5), 1],
        state_shape=48, rnn_hidden_dim=64, hpn_head_num=1, hpn_hyper_dim=64,
        saleq_emb_dim=32, saleq_zsa_dim=64, saleq_hidden_dim=64,
        saleq_norm=norm, saleq_sem_groups=8,
        obs_agent_id=True, obs_last_action=False, map_type="marines",
        agent="hpn_saleq", mac="hpn_mac", agent_output_type="q",
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
    console_logger = _NullConsole()   # nq_learner._update_targets calls logger.console_logger.info(...)

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
    learner = le_REGISTRY["nq_learner"](mac, batch.scheme, _NullLogger(), args)

    agent_before = [p.detach().clone() for p in mac.parameters()]
    mixer_before = [p.detach().clone() for p in learner.mixer.parameters()]
    # snapshot the head params (Enc / e_hat_null / phi / head) to confirm the TD loss trains them too
    head_keys = ("enemy_encoder.0.weight", "no_target", "phi.0.weight", "head.0.weight")
    named = dict(mac.agent.named_parameters())
    for key in head_keys:
        assert key in named, f"missing head param {key}"
    head_before = {k: named[k].detach().clone() for k in head_keys}

    learner.train(batch, t_env=0, episode_num=0)
    learner.train(batch, t_env=3000, episode_num=1)  # 2nd step: triggers the target update (interval=1)

    assert any(not th.equal(a, b) for a, b in zip(agent_before, mac.parameters())), "agent did not update"
    assert any(not th.equal(a, b) for a, b in zip(mixer_before, learner.mixer.parameters())), "mixer did not update"
    assert all(th.isfinite(p).all() for p in mac.parameters()), "non-finite agent params"
    assert learner.last_target_update_episode > 0, "target update did not fire"

    named_after = dict(mac.agent.named_parameters())
    for key in head_keys:  # each head component must be moved by the TD loss (grad flowed + optimizer stepped)
        assert not th.equal(head_before[key], named_after[key]), f"head param {key} did not update"

    # epsilon_greedy (greedy at test) must never pick an unavailable action
    masked, _, _ = make_batch(args, avail_upto=3)
    mac.init_hidden(masked.batch_size)
    picked = mac.select_actions(masked, t_ep=0, t_env=0, test_mode=True)
    assert (picked < 3).all(), "select_actions chose an unavailable action"


def test_hpn_saleq_train_step():
    _train_and_check(build_args("avgl1"))


def test_hpn_saleq_sem_train_step():  # phase2 thread B: SEM norm through the real NQLearner
    _train_and_check(build_args("sem"))


if __name__ == "__main__":
    test_hpn_saleq_train_step()
    print("HPN SALE-Q agent unit test (avgl1, HPN-QMIX backbone) PASSED")
    test_hpn_saleq_sem_train_step()
    print("HPN SALE-Q agent unit test (SEM, HPN-QMIX backbone) PASSED")
