# =====================================================================================================
# ABLATION FILE -- serves ablation_arm_1 (arm 1B: "flat trunk, no HPN").
#
# Copied from: src/modules/agents/hpn_saleq_agent.py (HPNSaleQAgent). That file is NOT modified.
#
# What changed relative to the source:
#   * The HPN permutation-invariant trunk (modules A-B: hypernet-generated per-entity input weights
#     `hyper_enemy` / `hyper_ally`, summed entity embeddings, `unify_input_heads`, `fc1_own`) is replaced
#     by the vanilla pymarl RNNAgent trunk: ONE linear layer `fc1` over the flat concatenation
#     [own_context ; enemy_feats (all slots) ; ally_feats (all slots)] -> ReLU -> the same GRUCell.
#     The flat input dim is derived from `input_shape`, never hardcoded.
#   * The agent-id / last-action embedding tables are kept (added to the fc1 output, as in HPN), so the
#     only variable vs. the full method is the trunk.
#   * All parent trunk/read-out modules that the flat trunk does not use are deleted in __init__ so they
#     are not dead parameters: hyper_enemy, hyper_ally, unify_input_heads, fc1_own, fc2_normal_actions,
#     unify_output_heads.  `init_hidden` is overridden because the parent's uses `fc1_own`.
#   * The action-in Q-head (Enc / no_target / slot-free a_tilde / phi / SEM-or-AvgL1Norm / head), the
#     `last_phi` / `last_v` exports, both map_type asserts and `_supports_terran_gen = False` are kept
#     verbatim.  `AvgL1Norm` and `sem` are imported from the untouched source file.
#   * The __main__ smoke test is rewritten: the enemy/ally permutation-invariance and attack-Q
#     permutation-equivariance checks of the source are EXPECTED TO FAIL here (that is the ablation),
#     so they are replaced by shape checks, a NEGATIVE check that permuting enemy slots changes the
#     normal-action Q (documents the loss of PI), the e_hat_null isolation, attack valuation and
#     SEM-simplex checks, plus a parameter-count print.
#
# Consequence: Q is no longer invariant to enemy/ally slot order and attack-Q is no longer exactly
# permutation-equivariant. Race scope: protoss + zerg only (fails at the terran_gen assert, as the source).
# =====================================================================================================
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .hpns_rnn_agent import HPNS_RNNAgent
from .hpn_saleq_agent import AvgL1Norm, sem


class FlatSaleQAgent(HPNS_RNNAgent):
    """ABLATION arm 1B: flat (non-HPN) trunk + the unchanged unified action-in Q-head.

    Trunk: v_tilde = GRU( ReLU( fc1([own ; e_1..e_n_en ; a_1..a_n_al]) + id_embeddings ) ), i.e. the vanilla
    pymarl RNNAgent encoder over the slot-ordered flat observation. No hypernetworks, no entity pooling.

    Head (identical to HPNSaleQAgent):
        phi^{oa}(a) = norm( phi([ v_tilde ; e_hat(a) ; a_tilde(a) ]) ),  Q^i(a) = head( phi^{oa}(a) )
    with e_hat(a) = Enc(e^i_j) for attack(j) else no_target, and slot-free a_tilde(a).

    ``forward(inputs, hidden) -> (Q [bs, n, |A|], v_tilde [bs, n, d])`` -- same signature as the source, so
    ``hpn_mac`` (which still delivers the split own/enemy/ally inputs), the WM learner, the mixer and
    epsilon-greedy all work unchanged.
    """

    # Subclasses that correctly handle terran_gen healers set this True. Kept False, as in the source.
    _supports_terran_gen = False

    def __init__(self, input_shape, args):
        super(FlatSaleQAgent, self).__init__(input_shape, args)
        # D9: trailing action slots index enemies (attack); on MMM they index allies (rescue),
        # which e_hat(a) would embed as the wrong entity. Fail loudly until MMM is handled.
        assert getattr(args, "map_type", None) != "MMM", \
            "FlatSaleQAgent does not support MMM maps (rescue actions); v1 targets non-MMM maps."
        # Same terran_gen guard as the source (heal slots would be scored through the enemy read-out).
        assert getattr(args, "map_type", None) != "terran_gen" or self._supports_terran_gen, \
            "FlatSaleQAgent mis-scores medivac heal actions on terran_gen (SMACv2 10gen_terran): its " \
            "trailing action slots assume enemies. This ablation arm is protoss/zerg only."

        self.n_normal = args.output_normal_actions

        # --- ABLATION: delete the whole HPN trunk + the vanilla read-outs (not dead params) ---
        del self.hyper_enemy
        del self.hyper_ally
        del self.unify_input_heads
        del self.fc1_own
        del self.fc2_normal_actions
        del self.unify_output_heads

        # --- ABLATION: flat trunk. Input dim derived from input_shape = (own_dim, (n_en, en_dim), (n_al, al_dim)).
        own_dim, enemy_shape, ally_shape = input_shape
        self.flat_input_dim = int(own_dim) + int(np.prod(enemy_shape)) + int(np.prod(ally_shape))
        self.fc1 = nn.Linear(self.flat_input_dim, self.rnn_hidden_dim, bias=True)
        # self.rnn (GRUCell) and the id-embedding tables are inherited unchanged.

        # --- Head (verbatim from HPNSaleQAgent). ---
        self.emb_dim = getattr(args, "saleq_emb_dim", 32)
        self.zsa_dim = getattr(args, "saleq_zsa_dim", 64)
        self.hdim = getattr(args, "saleq_hidden_dim", 64)

        # Enc: per-enemy encoder -> attacked-enemy embedding e_hat(attack(j)).
        self.enemy_encoder = nn.Sequential(
            nn.Linear(self.enemy_feats_dim, self.hdim), nn.ReLU(),
            nn.Linear(self.hdim, self.emb_dim),
        )
        # e_hat_null: learned, observation-independent no-target vector for normal actions (D6).
        self.no_target = nn.Parameter(th.zeros(self.emb_dim))

        # a_tilde(a): slot-free action encoding (D11), constant [|A|, n_normal + 1] table.
        a_tilde = th.zeros(self.n_actions, self.n_normal + 1)
        a_tilde[:self.n_normal, :self.n_normal] = th.eye(self.n_normal)
        a_tilde[self.n_normal:, self.n_normal] = 1.0
        self.register_buffer("a_tilde", a_tilde)

        # phi^{oa} MLP: [v_tilde ; e_hat ; a_tilde] -> zsa; head: zsa -> scalar Q.
        phi_in = self.rnn_hidden_dim + self.emb_dim + (self.n_normal + 1)
        self.phi = nn.Sequential(
            nn.Linear(phi_in, self.hdim), nn.ReLU(),
            nn.Linear(self.hdim, self.zsa_dim),
        )
        self.head = nn.Sequential(nn.Linear(self.zsa_dim, self.hdim), nn.ReLU(), nn.Linear(self.hdim, 1))

        # phi^{oa} normalization: "avgl1" or deterministic SEM (same switch as the source).
        self.saleq_norm = getattr(args, "saleq_norm", "avgl1")
        if self.saleq_norm == "sem":
            self.sem_groups = getattr(args, "saleq_sem_groups", 8)
            assert self.zsa_dim % self.sem_groups == 0, "saleq_sem_groups must divide saleq_zsa_dim"
            self.sem_ln = nn.LayerNorm(self.zsa_dim)

    def init_hidden(self):
        # parent's init_hidden references fc1_own, which the flat trunk deleted
        return self.fc1.weight.new(1, self.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        # inputs: (bs, own_feats [bs*n, own], enemy_feats [bs*n*n_en, en], ally_feats [bs*n*n_al, al], idx)
        bs, own_feats_t, enemy_feats_t, ally_feats_t, embedding_indices = inputs
        BN = bs * self.n_agents

        # ===== ABLATION trunk: flat concatenation over slots (order-dependent) -> fc1 -> ReLU -> GRU =====
        flat = th.cat([own_feats_t, enemy_feats_t.reshape(BN, -1), ally_feats_t.reshape(BN, -1)], dim=-1)
        embedding = self.fc1(flat)  # [bs * n_agents, rnn_hidden_dim]

        # ID embeddings (same tables / same placement as the HPN trunk)
        if self.args.obs_agent_id:
            agent_indices = embedding_indices[0]
            embedding = embedding + self.agent_id_embedding(agent_indices).view(-1, self.rnn_hidden_dim)
        if self.args.obs_last_action:
            last_action_indices = embedding_indices[-1]
            if last_action_indices is not None:  # t != 0
                embedding = embedding + self.action_id_embedding(last_action_indices).view(-1, self.rnn_hidden_dim)

        x = F.relu(embedding, inplace=True)
        h_in = hidden_state.reshape(-1, self.rnn_hidden_dim)
        hh = self.rnn(x, h_in)  # v_tilde: [bs * n_agents, rnn_hidden_dim]

        # ===== Head: enumerate |A| candidate actions (verbatim from HPNSaleQAgent) =====
        enemy_feats = enemy_feats_t.view(BN, self.n_enemies, self.enemy_feats_dim)
        enemy_emb = self.enemy_encoder(enemy_feats)  # [BN, n_enemies, emb_dim]
        no_target = self.no_target.view(1, 1, self.emb_dim).expand(BN, self.n_normal, self.emb_dim)
        e_hat = th.cat([no_target, enemy_emb], dim=1)  # [BN, |A|, emb_dim]

        v_tilde = hh.unsqueeze(1).expand(BN, self.n_actions, self.rnn_hidden_dim)  # [BN, |A|, d]
        a_tilde = self.a_tilde.unsqueeze(0).expand(BN, self.n_actions, self.n_normal + 1)  # [BN, |A|, n_normal+1]

        z = self.phi(th.cat([v_tilde, e_hat, a_tilde], dim=-1))  # [BN, |A|, zsa]
        phi = sem(self.sem_ln(z), self.sem_groups) if self.saleq_norm == "sem" else AvgL1Norm(z)  # phi^{oa}(a)
        q = self.head(phi).view(bs, self.n_agents, self.n_actions)  # [bs, n_agents, |A|]
        self.last_phi = phi.view(bs, self.n_agents, self.n_actions, self.zsa_dim)  # [bs, n, |A|, zsa]
        self.last_v = hh.view(bs, self.n_agents, -1)                               # [bs, n, rnn_hidden_dim]
        return q, hh.view(bs, self.n_agents, -1)


if __name__ == "__main__":
    # Smoke test for the ABLATION trunk. Run from src with:  python -m modules.agents.flat_saleq_agent
    # Unlike the source's smoke, enemy/ally permutation invariance and attack-Q equivariance are NOT expected
    # to hold (the flat trunk is slot-order dependent); we assert the loss of PI explicitly.
    from types import SimpleNamespace

    bs, n_agents, n_enemies, n_allies = 2, 3, 3, 2
    own_dim, en_dim, al_dim, n_normal = 5, 5, 5, 6
    n_actions = n_normal + n_enemies
    input_shape = (own_dim, (n_enemies, en_dim), (n_allies, al_dim))

    def make_agent(norm):
        args = SimpleNamespace(
            n_agents=n_agents, n_allies=n_allies, n_enemies=n_enemies, n_actions=n_actions,
            hpn_head_num=1, hpn_hyper_dim=64, rnn_hidden_dim=64, obs_agent_id=True,
            obs_last_action=False, map_type="marines", output_normal_actions=n_normal,
            saleq_emb_dim=32, saleq_zsa_dim=64, saleq_hidden_dim=64,
            saleq_norm=norm, saleq_sem_groups=8,
        )
        return FlatSaleQAgent(input_shape, args).eval()

    own = th.randn(bs * n_agents, own_dim)
    enemy = th.randn(bs * n_agents * n_enemies, en_dim)
    ally = th.randn(bs * n_agents * n_allies, al_dim)
    emb_idx = [th.arange(n_agents).unsqueeze(0).expand(bs, -1)]
    h = th.zeros(bs, n_agents, 64)
    perm = th.tensor([2, 0, 1])
    enemy_p = enemy.view(bs * n_agents, n_enemies, en_dim)[:, perm, :].reshape(bs * n_agents * n_enemies, en_dim)

    def checks(norm):
        th.manual_seed(0)
        agent = make_agent(norm)
        n_params = sum(p.numel() for p in agent.parameters())
        assert not any(hasattr(agent, m) for m in ("hyper_enemy", "hyper_ally", "unify_input_heads", "fc1_own",
                                                     "fc2_normal_actions", "unify_output_heads")), "HPN module survived"
        assert agent.flat_input_dim == own_dim + n_enemies * en_dim + n_allies * al_dim
        with th.no_grad():
            q, v_tilde = agent((bs, own, enemy, ally, emb_idx), h)
        assert q.shape == (bs, n_agents, n_actions), q.shape
        assert v_tilde.shape == (bs, n_agents, 64), v_tilde.shape
        assert th.isfinite(q).all()
        normal, attack = q[..., :n_normal], q[..., n_normal:]
        # NEGATIVE check (the ablation): permuting enemy slots changes normal Q / v_tilde -> PI is lost.
        with th.no_grad():
            q_p, v_p = agent((bs, own, enemy_p, ally, emb_idx), h)
        assert not th.allclose(q_p[..., :n_normal], normal, atol=1e-5), \
            f"[{norm}] normal Q was enemy-invariant -- flat trunk unexpectedly PI"
        assert not th.allclose(v_p, v_tilde, atol=1e-5), f"[{norm}] v_tilde unexpectedly permutation-invariant"
        # e_hat_null isolation: zeroing Enc leaves normal Q unchanged, changes attack Q (head unchanged).
        with th.no_grad():
            agent.enemy_encoder[-1].weight.zero_(); agent.enemy_encoder[-1].bias.zero_()
            q_z, _ = agent((bs, own, enemy, ally, emb_idx), h)
        assert th.allclose(q_z[..., :n_normal], normal, atol=1e-5), f"[{norm}] normal Q leaks Enc's e_hat slot"
        assert not th.allclose(q_z[..., n_normal:], attack, atol=1e-5), f"[{norm}] attack Q ignores Enc"
        # attack-valuation: distinct enemy features -> distinct attack Q
        assert not th.allclose(attack[..., 0], attack[..., 1], atol=1e-4), f"[{norm}] cannot distinguish enemies"
        print(f"  [{norm}] flat-trunk checks OK: shapes, PI LOST (as intended), e_hat_null, attack valuation; "
              f"params = {n_params}")

    for norm in ("avgl1", "sem"):
        checks(norm)

    s = sem(th.nn.LayerNorm(64)(th.randn(4, n_actions, 64)), 8).reshape(4, n_actions, 8, 8)
    assert th.all(s >= 0) and th.allclose(s.sum(-1), th.ones(4, n_actions, 8), atol=1e-5)
    print("  [sem] simplicial property OK")
    print("FLAT SALE-Q AGENT (ablation arm 1B) SMOKE TEST PASSED (both norms)")
