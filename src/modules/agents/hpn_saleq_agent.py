import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .hpns_rnn_agent import HPNS_RNNAgent


def AvgL1Norm(x, eps=1e-8):
    # SALE/TD7 normalization: keeps the embedding on a bounded L1 shell (ported from
    # sale_factored_critic.py). D4: the placeholder SEM replaces exactly this in Phase 2.
    return x / x.abs().mean(-1, keepdim=True).clamp(min=eps)


def sem(x, groups):
    """Simplicial Embedding (MRS.Q): softmax within each of `groups` equal chunks of the last dim, turning
    the embedding into a concatenation of `groups` simplices (each chunk sums to 1, entries >= 0). Deterministic
    -- no sampling / KL / straight-through (that machinery belongs to the stochastic thread-C latent). Phase-2
    thread B: replaces AvgL1Norm at the phi^{oa} site, applied to LayerNorm(x) as in MRS.Q's sem(ln(x), groups).
    Elementwise-per-group on the last dim, so it does not affect the D11 PI/PE properties."""
    *lead, d = x.shape
    x = x.reshape(*lead, groups, d // groups)
    x = F.softmax(x, dim=-1)
    return x.reshape(*lead, d)


class HPNSaleQAgent(HPNS_RNNAgent):
    """HPN value agent with a unified action-in Q-head (value-based adaptation, v1).

    Trunk (modules A-B) is structurally the HPN backbone from ``HPNS_RNNAgent`` -- the
    permutation-invariant input hypernetworks + GRU producing the post-GRU embedding
    ``v_tilde`` (the code's ``hh``). The read-out (modules C/D) is **replaced** (D1) by a
    single action-conditioned head evaluated by enumeration over the |A| candidate actions:

        phi^{oa}(a) = AvgL1Norm( phi([ v_tilde ; e_hat(a) ; a_tilde(a) ]) )
        Q^i(a)      = head( phi^{oa}(a) )

    where (master-plan sec.2):
      * ``e_hat(a)``   -- attacked-enemy embedding: ``Enc(e^i_j)`` if a = attack(j), else the
                         learned no-target vector ``no_target`` (e_hat_null, D6, zero-init).
      * ``a_tilde(a)`` -- **slot-free** action encoding ``[onehot_{A_inv}(a); is_attack(a)]`` (D11):
                         normal-action identity, or a bare is-attack flag; **never** the attack
                         slot index. All attack actions share ``a_tilde = [0...0; 1]`` -- the
                         attacked enemy's identity is carried solely by ``e_hat(a)``. Feeding the
                         raw ``onehot(a)`` would leak the order-dependent slot index and break
                         exact permutation-equivariance (that is ``SALEFactoredCritic``'s defect).

    ``forward(inputs, hidden) -> (Q [bs, n, |A|], v_tilde [bs, n, d])`` -- identical signature to
    ``HPNS_RNNAgent.forward``, so ``hpn_mac``, ``nq_learner`` (Double-Q, TD(lambda)), the monotone
    mixer and epsilon-greedy all work unchanged; enumeration is internal to the head (D8).

    Per D1a the ``hyper_enemy`` output layer is shrunk to the module-A portion only (module D's
    read-out slice is gone) so the vanilla-vs-adaptation parameter count stays honest.
    """

    # Subclasses that correctly handle terran_gen healers set this True (see hpn_saleq_terran.py).
    _supports_terran_gen = False

    def __init__(self, input_shape, args):
        super(HPNSaleQAgent, self).__init__(input_shape, args)
        # D9: trailing action slots index enemies (attack); on MMM they index allies (rescue),
        # which e_hat(a) would embed as the wrong entity. Fail loudly until MMM is handled.
        assert getattr(args, "map_type", None) != "MMM", \
            "HPNSaleQAgent does not support MMM maps (rescue actions); v1 targets non-MMM maps."
        # PYMARL3-ONLY DIVERGENCE (not present in marl_project3's copy). SMACv2's terran_gen also has
        # healers: for a medivac the trailing slots are "heal ally j", which this head would score
        # through the enemy read-out -- silently, since map_type is "terran_gen", not "MMM". Fail loudly
        # and point at the agent that handles it correctly.
        assert getattr(args, "map_type", None) != "terran_gen" or self._supports_terran_gen, \
            "HPNSaleQAgent mis-scores medivac heal actions on terran_gen (SMACv2 10gen_terran): its " \
            "trailing action slots assume enemies. Use agent 'hpn_saleq_terran' instead."

        self.n_normal = args.output_normal_actions

        # --- D1a: shrink hyper_enemy to the module-A input portion only (drop module-D read-out
        # slice). Same output shape as the non-MMM hyper_ally: enemy_feats_dim * rnn_hidden_dim * heads.
        self.hyper_enemy = nn.Sequential(
            nn.Linear(self.enemy_feats_dim, args.hpn_hyper_dim),
            nn.ReLU(inplace=True),
            nn.Linear(args.hpn_hyper_dim, self.enemy_feats_dim * self.rnn_hidden_dim * self.n_heads),
        )
        # D1: modules C/D read-outs are replaced by the head -- delete them so they are not dead params.
        del self.fc2_normal_actions
        del self.unify_output_heads

        # --- Head (ported from sale_factored_critic.py; single q head per D3; fresh Enc per D5). ---
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
        #   rows [0, n_normal):     [onehot_{A_inv}(a) ; is_attack=0]
        #   rows [n_normal, |A|):   [0...0             ; is_attack=1]   (shared across attacks)
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

        # phi^{oa} normalization (P3/P13): "avgl1" (G2 default, config bit-unchanged) or deterministic SEM.
        self.saleq_norm = getattr(args, "saleq_norm", "avgl1")
        if self.saleq_norm == "sem":
            self.sem_groups = getattr(args, "saleq_sem_groups", 8)
            assert self.zsa_dim % self.sem_groups == 0, "saleq_sem_groups must divide saleq_zsa_dim"
            self.sem_ln = nn.LayerNorm(self.zsa_dim)  # created only when SEM is on (MRS.Q: sem(ln(x)))

    def forward(self, inputs, hidden_state):
        # inputs: (bs, own_feats [bs*n, own], enemy_feats [bs*n*n_en, en], ally_feats [bs*n*n_al, al], idx)
        bs, own_feats_t, enemy_feats_t, ally_feats_t, embedding_indices = inputs

        # ===== Trunk (modules A-B) -- mirrors HPNS_RNNAgent.forward, with the shrunk hyper_enemy =====
        # (1) Own feature
        embedding_own = self.fc1_own(own_feats_t)  # [bs * n_agents, rnn_hidden_dim]

        # (2) ID embeddings
        if self.args.obs_agent_id:
            agent_indices = embedding_indices[0]
            embedding_own = embedding_own + self.agent_id_embedding(agent_indices).view(-1, self.rnn_hidden_dim)
        if self.args.obs_last_action:
            last_action_indices = embedding_indices[-1]
            if last_action_indices is not None:  # t != 0
                embedding_own = embedding_own + self.action_id_embedding(last_action_indices).view(
                    -1, self.rnn_hidden_dim)

        # (3) Enemy feature (module A input hypernet; shrunk to module-A weights only per D1a)
        fc1_w_enemy = self.hyper_enemy(enemy_feats_t).view(
            -1, self.enemy_feats_dim, self.rnn_hidden_dim * self.n_heads
        )  # [bs * n_agents * n_enemies, enemy_fea_dim, rnn_hidden_dim * heads]
        embedding_enemies = th.matmul(enemy_feats_t.unsqueeze(1), fc1_w_enemy).view(
            bs * self.n_agents, self.n_enemies, self.n_heads, self.rnn_hidden_dim
        ).sum(dim=1, keepdim=False)  # [bs * n_agents, n_heads, rnn_hidden_dim]

        # (4) Ally feature (non-MMM; MMM excluded by the D9 assert)
        fc1_w_ally = self.hyper_ally(ally_feats_t).view(-1, self.ally_feats_dim, self.rnn_hidden_dim * self.n_heads)
        embedding_allies = th.matmul(ally_feats_t.unsqueeze(1), fc1_w_ally).view(
            bs * self.n_agents, self.n_allies, self.n_heads, self.rnn_hidden_dim
        ).sum(dim=1, keepdim=False)  # [bs * n_agents, n_heads, rnn_hidden_dim]

        embedding = embedding_own + self.unify_input_heads(embedding_enemies + embedding_allies)
        x = F.relu(embedding, inplace=True)
        h_in = hidden_state.reshape(-1, self.rnn_hidden_dim)
        hh = self.rnn(x, h_in)  # v_tilde: [bs * n_agents, rnn_hidden_dim]

        # ===== Head: enumerate |A| candidate actions (D8: one batched sweep) =====
        BN = bs * self.n_agents
        # e_hat(a): no_target for normal actions, Enc(e^i_j) for attack(j) -- concat in action order.
        enemy_feats = enemy_feats_t.view(BN, self.n_enemies, self.enemy_feats_dim)
        enemy_emb = self.enemy_encoder(enemy_feats)  # [BN, n_enemies, emb_dim]
        no_target = self.no_target.view(1, 1, self.emb_dim).expand(BN, self.n_normal, self.emb_dim)
        e_hat = th.cat([no_target, enemy_emb], dim=1)  # [BN, |A|, emb_dim]

        v_tilde = hh.unsqueeze(1).expand(BN, self.n_actions, self.rnn_hidden_dim)  # [BN, |A|, d]
        a_tilde = self.a_tilde.unsqueeze(0).expand(BN, self.n_actions, self.n_normal + 1)  # [BN, |A|, n_normal+1]

        z = self.phi(th.cat([v_tilde, e_hat, a_tilde], dim=-1))  # [BN, |A|, zsa]
        phi = sem(self.sem_ln(z), self.sem_groups) if self.saleq_norm == "sem" else AvgL1Norm(z)  # phi^{oa}(a)
        q = self.head(phi).view(bs, self.n_agents, self.n_actions)  # [bs, n_agents, |A|]
        # Phase-2 thread A: expose phi^{oa}(a) and the trunk v_tilde for the predictive-loss learner
        # (SaleqReprLearner reads these off the live/target MAC). No signature change; ~free at execution.
        self.last_phi = phi.view(bs, self.n_agents, self.n_actions, self.zsa_dim)  # [bs, n, |A|, zsa]
        self.last_v = hh.view(bs, self.n_agents, -1)                               # [bs, n, rnn_hidden_dim]
        return q, hh.view(bs, self.n_agents, -1)


if __name__ == "__main__":
    # Smoke test (CLAUDE.md sec.7 / master-plan sec.5.1 + phase2-plan sec.4 thread B): shapes, enemy-permutation
    # PI (normal Q, v_tilde) + exact PE (attack Q, the D11 guard), ally-permutation PI, e_hat_null path isolation,
    # attack valuation -- run under BOTH norms ("avgl1", "sem"); plus the SEM simplicial property.
    # Run from src with:  python -m modules.agents.hpn_saleq_agent
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
        return HPNSaleQAgent(input_shape, args).eval()

    own = th.randn(bs * n_agents, own_dim)
    enemy = th.randn(bs * n_agents * n_enemies, en_dim)
    ally = th.randn(bs * n_agents * n_allies, al_dim)
    emb_idx = [th.arange(n_agents).unsqueeze(0).expand(bs, -1)]  # agent-id indices (obs_last_action=False)
    h = th.zeros(bs, n_agents, 64)
    perm = th.tensor([2, 0, 1])   # new enemy position i receives old enemy perm[i]
    aperm = th.tensor([1, 0])
    enemy_p = enemy.view(bs * n_agents, n_enemies, en_dim)[:, perm, :].reshape(bs * n_agents * n_enemies, en_dim)
    ally_p = ally.view(bs * n_agents, n_allies, al_dim)[:, aperm, :].reshape(bs * n_agents * n_allies, al_dim)

    def pi_pe_checks(norm):
        th.manual_seed(0)
        agent = make_agent(norm)
        with th.no_grad():
            q, v_tilde = agent((bs, own, enemy, ally, emb_idx), h)
        assert q.shape == (bs, n_agents, n_actions), q.shape
        assert v_tilde.shape == (bs, n_agents, 64), v_tilde.shape
        normal, attack = q[..., :n_normal], q[..., n_normal:]
        # enemy-permutation: normal Q & v_tilde invariant; attack Q equivariant (D11 guard)
        with th.no_grad():
            q_p, v_tilde_p = agent((bs, own, enemy_p, ally, emb_idx), h)
        assert th.allclose(q_p[..., :n_normal], normal, atol=1e-5), f"[{norm}] normal Q not enemy-invariant"
        assert th.allclose(v_tilde_p, v_tilde, atol=1e-5), f"[{norm}] v_tilde not permutation-invariant"
        assert th.allclose(q_p[..., n_normal:], attack[..., perm], atol=1e-5), \
            f"[{norm}] attack Q not permutation-equivariant (D11: slot index leaked into the head)"
        # ally-permutation: entire Q-vector invariant
        with th.no_grad():
            q_ap, _ = agent((bs, own, enemy, ally_p, emb_idx), h)
        assert th.allclose(q_ap, q, atol=1e-5), f"[{norm}] Q not ally-invariant"
        # e_hat_null isolation: zeroing Enc leaves normal Q unchanged, changes attack Q
        with th.no_grad():
            agent.enemy_encoder[-1].weight.zero_(); agent.enemy_encoder[-1].bias.zero_()
            q_z, _ = agent((bs, own, enemy, ally, emb_idx), h)
        assert th.allclose(q_z[..., :n_normal], normal, atol=1e-5), f"[{norm}] normal Q leaks Enc's e_hat slot"
        assert not th.allclose(q_z[..., n_normal:], attack, atol=1e-5), f"[{norm}] attack Q ignores Enc"
        # attack-valuation: distinct enemy features -> distinct attack Q
        assert not th.allclose(attack[..., 0], attack[..., 1], atol=1e-4), f"[{norm}] cannot distinguish enemies"
        print(f"  [{norm}] PI/PE checks OK: shapes, enemy PI + PE (D11), ally PI, e_hat_null, attack valuation")

    for norm in ("avgl1", "sem"):
        pi_pe_checks(norm)

    # SEM simplicial property: each of the 8 groups softmaxes to a simplex (sums to 1, entries >= 0).
    s = sem(th.nn.LayerNorm(64)(th.randn(4, n_actions, 64)), 8).reshape(4, n_actions, 8, 8)
    assert th.all(s >= 0), "SEM entries must be >= 0"
    assert th.allclose(s.sum(-1), th.ones(4, n_actions, 8), atol=1e-5), "each SEM group must sum to 1"
    print("  [sem] simplicial property OK (8 groups x 8 vertices; each sums to 1, entries >= 0)")

    print("HPN SALE-Q AGENT SMOKE TEST PASSED (both norms)")
