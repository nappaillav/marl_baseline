import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from modules.agents.hpn_saleq_agent import sem  # W5: share ONLY the low-level grouped-softmax helper


class SemWorldModel(nn.Module):
    """Centralized, training-time-only SEM world model (Phase-2 thread C, `phase2-plan_v2.md` §1).

    Built clean from the `state_space_model` template (W3): same structural roles (encoders, posterior/prior,
    shared decoder, aggregator) but the Gaussian latent is replaced by v2's **SEM categorical latent** — a product
    of `L_w` categoricals over `V` vertices, class-probs via unimix, sampled straight-through (§1.2).

    All components are centralized and training-only: they live in the learner, never in `controllers/`, and
    (posterior encodes the *global state* `s`, per the thread-A lesson) never touch the decentralized agent path.

    Shapes (leading dims `...` are `[bs, T]` in the learner):
      state_embed(s):     s [.., state_dim]          -> z^s   [.., d_s]   (simplicial, grouped-softmax)
      sa_embed(s, a):     a [.., n, |A|]             -> phi^sa [.., d_sa]
      posterior_logits:   [z^s ; phi^sa]             -> [.., d_wm]
      prior_logits:       [z_prev ; a_prev]          -> [.., d_wm]        (§1.3 causal: no step-t input)
      decode(z):          z [.., d_wm]               -> (state_logits [.., d_s], phi_hat [.., d_sa],
                                                          act_logits [.., n, |A|])
      imagine(z0, k):     z0 [.., d_wm]              -> imagined future [.., k, d_wm]
      aggregate(z_seq):   [.., k, d_wm]              -> traj [.., traj_dim]
    """

    def __init__(self, args):
        super(SemWorldModel, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.state_dim = int(np.prod(args.state_shape))
        self.act_dim = self.n_agents * self.n_actions  # joint one-hot dim

        self.groups = getattr(args, "wm_groups", 8)
        self.vertices = getattr(args, "wm_vertices", 8)
        self.d_wm = self.groups * self.vertices
        self.d_s = getattr(args, "wm_state_dim", 64)
        self.d_sa = getattr(args, "wm_sa_dim", 64)
        self.traj_dim = getattr(args, "wm_traj_dim", 32)
        self.unimix = getattr(args, "wm_unimix", 0.01)
        self.k = getattr(args, "wm_k", 3)              # imagination depth (W6/W13)
        h = 128
        assert self.d_s % self.groups == 0, "wm_state_dim must be divisible by wm_groups"

        # 1.1 encoders
        self.state_encoder = nn.Sequential(nn.Linear(self.state_dim, h), nn.ReLU(), nn.Linear(h, self.d_s))
        self.state_ln = nn.LayerNorm(self.d_s)
        self.sa_encoder = nn.Linear(self.state_dim + self.act_dim, self.d_sa)

        # 1.3 posterior q(z | z^s, phi^sa) ; prior p(z | z_{t-1}, a_{t-1})
        self.posterior = nn.Sequential(nn.Linear(self.d_s + self.d_sa, h), nn.ReLU(), nn.Linear(h, self.d_wm))
        self.prior = nn.Sequential(nn.Linear(self.d_wm + self.act_dim, h), nn.ReLU(), nn.Linear(h, self.d_wm))

        # 1.4 shared decoder trunk + three heads (state-CE, phi-MSE, joint-action)
        self.dec_trunk = nn.Sequential(nn.Linear(self.d_wm, h), nn.ReLU())
        self.head_state = nn.Linear(h, self.d_s)
        self.head_phi = nn.Linear(h, self.d_sa)
        self.head_action = nn.Linear(h, self.act_dim)

        # 1.6 aggregator A_w: mean-pool over k latents + linear (W9)
        self.aggregator = nn.Linear(self.d_wm, self.traj_dim)

    # ---------- encoders ----------
    def state_embed(self, s):
        # z^s = SEM_det(LN(E_w(s))) : deterministic simplicial state embedding (§1.1)
        return sem(self.state_ln(self.state_encoder(s)), self.groups)

    def sa_embed(self, s, a_onehot):
        a_flat = a_onehot.reshape(*a_onehot.shape[:-2], self.act_dim)
        return self.sa_encoder(th.cat([s, a_flat], dim=-1))

    # ---------- categorical latent (§1.2) ----------
    def probs(self, logits):
        # logits [.., d_wm] -> per-group softmax with unimix -> [.., groups, vertices]
        p = F.softmax(logits.reshape(*logits.shape[:-1], self.groups, self.vertices), dim=-1)
        return (1.0 - self.unimix) * p + self.unimix * (1.0 / self.vertices)

    def st_sample(self, probs):
        # straight-through one-hot sample; forward = one-hot, backward = probs gradient -> [.., d_wm]
        G, V = self.groups, self.vertices
        lead = probs.shape[:-2]
        idx = th.multinomial(probs.detach().reshape(-1, V), 1).squeeze(-1)
        onehot = F.one_hot(idx, V).float().reshape(*lead, G, V)
        st = onehot + probs - probs.detach()
        return st.reshape(*lead, G * V)

    def posterior_logits(self, z_s, phi_sa):
        return self.posterior(th.cat([z_s, phi_sa], dim=-1))

    def prior_logits(self, z_prev, a_prev_onehot):
        a_flat = a_prev_onehot.reshape(*a_prev_onehot.shape[:-2], self.act_dim)
        return self.prior(th.cat([z_prev, a_flat], dim=-1))

    # ---------- decoder ----------
    def decode(self, z):
        h = self.dec_trunk(z)
        act_logits = self.head_action(h).reshape(*z.shape[:-1], self.n_agents, self.n_actions)
        return self.head_state(h), self.head_phi(h), act_logits

    def _act_logits(self, z):
        # action head only (used inside the imagination rollout)
        return self.head_action(self.dec_trunk(z)).reshape(*z.shape[:-1], self.n_agents, self.n_actions)

    # ---------- imagination (§1.6) ----------
    def imagine(self, z0, k):
        # Roll the prior forward k steps from z0, each step's action = per-agent argmax of the action head
        # (discrete -> no gradient through the action; latent sampling is straight-through -> gradient flows).
        z = z0
        outs = []
        for _ in range(k):
            with th.no_grad():   # discrete argmax carries no gradient; no_grad also frees the decoder activations
                a_idx = self._act_logits(z).argmax(dim=-1)                  # [.., n]
            a_onehot = F.one_hot(a_idx, self.n_actions).float()            # [.., n, |A|]
            z = self.st_sample(self.probs(self.prior_logits(z, a_onehot)))
            outs.append(z)
        return th.stack(outs, dim=-2)                                      # [.., k, d_wm]

    def aggregate(self, z_seq):
        return self.aggregator(z_seq.mean(dim=-2))                          # [.., traj_dim]


if __name__ == "__main__":
    # Smoke test (CLAUDE.md sec.7 / phase2-plan_v2 sec.4 GC.0): shapes, simplicial latent, prior causality,
    # straight-through gradient, imagination-rollout leakage guard.  Run:  python -m modules.model.sem_wm
    from types import SimpleNamespace

    th.manual_seed(0)
    bs, T, n, nA, sdim = 2, 5, 3, 9, 48
    args = SimpleNamespace(n_agents=n, n_actions=nA, state_shape=sdim,
                           wm_groups=8, wm_vertices=8, wm_state_dim=64, wm_sa_dim=64,
                           wm_traj_dim=32, wm_unimix=0.01, wm_k=3)
    wm = SemWorldModel(args)

    s = th.randn(bs, T, sdim)
    a = F.one_hot(th.randint(0, nA, (bs, T, n)), nA).float()

    z_s = wm.state_embed(s); phi = wm.sa_embed(s, a)
    assert z_s.shape == (bs, T, 64) and phi.shape == (bs, T, 64)
    # simplicial: each of the 8 groups of z^s sums to 1
    assert th.allclose(z_s.reshape(bs, T, 8, 8).sum(-1), th.ones(bs, T, 8), atol=1e-5), "z^s not simplicial"
    q_probs = wm.probs(wm.posterior_logits(z_s, phi))
    assert th.allclose(q_probs.sum(-1), th.ones(bs, T, 8), atol=1e-5), "posterior probs not per-group simplex"
    z_q = wm.st_sample(q_probs)
    assert z_q.shape == (bs, T, 64)
    print("shapes + simplicial latent OK")

    # prior causality (§1.3): perturbing the step-t global state must not change the prior for t.
    # Seed the ST sampling identically so the check isolates the perturbation from the sampling RNG.
    def _zq(state):
        th.manual_seed(7)
        return wm.st_sample(wm.probs(wm.posterior_logits(wm.state_embed(state), wm.sa_embed(state, a))))
    s2 = s.clone(); s2[:, 3] += 10.0                          # perturb step 3
    pA = wm.prior_logits(_zq(s)[:, :-1], a[:, :-1])           # index i -> prior for t=i+1
    pB = wm.prior_logits(_zq(s2)[:, :-1], a[:, :-1])
    assert th.allclose(pA[:, 2], pB[:, 2], atol=1e-5), "prior for t=3 leaked step-3 state (causality)!"
    assert not th.allclose(pA[:, 3], pB[:, 3], atol=1e-5), "prior for t=4 ignored its own step-3 input"
    print("prior causality OK (prior for t sees only step-(t-1) inputs)")

    # straight-through gradient reaches the posterior/prior nets
    z_q.sum().backward()
    assert wm.posterior[0].weight.grad is not None, "no ST gradient into posterior"
    print("straight-through gradient OK")

    # imagination: shapes + rollout consumes only (z, action-head), never future batch actions
    with th.no_grad():
        traj = wm.aggregate(wm.imagine(z_q, wm.k))
    assert traj.shape == (bs, T, 32), traj.shape
    print("imagination + aggregate OK")
    print("SEM WORLD MODEL SMOKE TEST PASSED")
