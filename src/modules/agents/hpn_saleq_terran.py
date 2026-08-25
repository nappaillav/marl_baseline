import torch as th
import torch.nn as nn

from .hpn_saleq_agent import HPNSaleQAgent, AvgL1Norm, sem


class HPNSaleQTerranAgent(HPNSaleQAgent):
    """HPN + phi^{oa} action-in Q-head for SMACv2's `10gen_terran` (map_type == "terran_gen").

    Why this exists
    ---------------
    On `terran_gen` the generated ally team may contain **medivacs**, and for a medivac the trailing
    action slots are *heal ally j*, not *attack enemy j* (the env agrees: `starcraft2.py:790` branches
    heal on `map_type in ["MMM", "terran_gen"]`). `HPNSaleQAgent` assumes every trailing slot indexes an
    enemy, so it would score heals through the enemy read-out -- silently, since its guard only tests
    `!= "MMM"`. This subclass fixes that.

    How it differs from `hpn_MMM2_agent.py` (the SMACv1 sibling in marl_project3)
    ---------------------------------------------------------------------------
    That agent hardcodes the medivac at index ``n_agents - 1``, which is right for MMM2 (exactly one
    medivac, fixed position). SMACv2 **regenerates the team every episode**: the healer count k varies
    (observed {0: 13, 1: 8, 2: 4} over 25 resets) and is not known at construction. So the healer set is
    resolved **per forward pass from each agent's own observed unit type**, not from its index:

        is_healer[i] = own_feats[i, -1] > 0.5

    `own_context` is ``[move_feats ; own_feats]`` (hpn_controller._build_inputs) and `own_feats` ends with
    the unit-type one-hot, so the last element is the medivac bit -- `get_unit_type_id` hardcodes
    ``terran_gen: marine->0, marauder->1, medivac->2`` (verified empirically: medivac own_feats ends
    ``[0, 0, 1]``, marauder ``[0, 1, 0]``).

    This is strictly more robust than the positional approach: it handles arbitrary k for free, and does
    **not** rely on the `attrgetter("unit_type", ...)` sort that happens to place medivacs in a contiguous
    tail. Nothing breaks if that ordering ever changes.

    Enemy medivacs (raw unit id 54) do occur and are *not* tail-ordered, but need no handling: enemies are
    only ever attacked, never healed. Their unit type reaches `enemy_encoder` through the entity features
    like any other attribute.

    Note on a dead agent: `get_obs_agent` returns all-zero own_feats, so `is_healer` reads False. Its
    Q-values are masked by `avail_actions` and never used, so the mis-classification is inert.

    Implementation note: `forward`'s trunk (modules A-B) is duplicated from `HPNSaleQAgent.forward`
    because that method computes trunk and head together and the head must change here. The same
    trunk-duplication caveat already applies between `hpn_saleq_agent.py` and `hpn_MMM2_agent.py`;
    keep the three in sync if the trunk is ever modified.
    """

    # Permits the parent's terran_gen guard: this subclass is exactly the correct alternative it names.
    _supports_terran_gen = True

    def __init__(self, input_shape, args):
        super(HPNSaleQTerranAgent, self).__init__(input_shape, args)
        # A healer's heal actions occupy trailing slots [0, n_allies); the remaining trailing slots do
        # not exist for it. That requires at least as many trailing slots as allies.
        assert self.n_enemies >= self.n_allies, (
            "terran_gen head needs n_enemies >= n_allies so a healer's heal targets fit the trailing "
            f"action slots (got n_enemies={self.n_enemies}, n_allies={self.n_allies})."
        )

        # Enc_ally (D5: fresh encoder, not shared with the input hypernets) -- a healer's heal targets.
        self.ally_encoder = nn.Sequential(
            nn.Linear(self.ally_feats_dim, self.hdim), nn.ReLU(),
            nn.Linear(self.hdim, self.emb_dim),
        )

        # Widen the action encoding to carry is_rescue, and replace the parent's phi to match.
        # (Reassigned rather than kept alongside, so no dead parameters remain.)
        del self.a_tilde
        self.register_buffer("a_tilde_table", self._build_a_tilde_table())
        phi_in = self.rnn_hidden_dim + self.emb_dim + (self.n_normal + 2)
        self.phi = nn.Sequential(
            nn.Linear(phi_in, self.hdim), nn.ReLU(),
            nn.Linear(self.hdim, self.zsa_dim),
        )

    def _build_a_tilde_table(self):
        """Slot-free action encodings (D11), one row-block per agent kind -> [2, |A|, n_normal + 2].

        Last dim layout: ``[ onehot_{A_inv}(a) (n_normal) , is_attack , is_rescue ]``.

          kind 0 (combat: marine / marauder)
            rows [0, n_normal)              : onehot(a), flags 0     -- invariant normal actions
            rows [n_normal, |A|)            : zeros, is_attack = 1   -- shared by ALL attacks
          kind 1 (medivac)
            rows [0, n_normal)              : onehot(a), flags 0
            rows [n_normal, n_normal+n_al)  : zeros, is_rescue = 1   -- shared by ALL heals
            rows [n_normal+n_al, |A|)       : all zeros              -- slot does not exist for a healer;
                                                                       constant and distinct from every
                                                                       real encoding, so it cannot disturb
                                                                       equivariance. avail_actions masks it.

        No row encodes *which* enemy or *which* ally. That is what keeps permutation-equivariance exact:
        the target's identity is carried solely by e_hat.
        """
        n_normal, n_actions, n_allies = self.n_normal, self.n_actions, self.n_allies
        table = th.zeros(2, n_actions, n_normal + 2)
        eye = th.eye(n_normal)
        table[0, :n_normal, :n_normal] = eye
        table[1, :n_normal, :n_normal] = eye
        table[0, n_normal:, n_normal] = 1.0                          # is_attack (combat)
        table[1, n_normal:n_normal + n_allies, n_normal + 1] = 1.0   # is_rescue (medivac)
        return table

    def forward(self, inputs, hidden_state):
        bs, own_feats_t, enemy_feats_t, ally_feats_t, embedding_indices = inputs
        n, n_en, n_al = self.n_agents, self.n_enemies, self.n_allies
        BN = bs * n

        # ===== Trunk (modules A-B) -- mirrors HPNSaleQAgent.forward =====
        embedding_own = self.fc1_own(own_feats_t)
        if self.args.obs_agent_id:
            embedding_own = embedding_own + self.agent_id_embedding(
                embedding_indices[0]).view(-1, self.rnn_hidden_dim)
        if self.args.obs_last_action:
            last_action_indices = embedding_indices[-1]
            if last_action_indices is not None:
                embedding_own = embedding_own + self.action_id_embedding(
                    last_action_indices).view(-1, self.rnn_hidden_dim)

        fc1_w_enemy = self.hyper_enemy(enemy_feats_t).view(
            -1, self.enemy_feats_dim, self.rnn_hidden_dim * self.n_heads)
        embedding_enemies = th.matmul(enemy_feats_t.unsqueeze(1), fc1_w_enemy).view(
            BN, n_en, self.n_heads, self.rnn_hidden_dim).sum(dim=1, keepdim=False)

        fc1_w_ally = self.hyper_ally(ally_feats_t).view(
            -1, self.ally_feats_dim, self.rnn_hidden_dim * self.n_heads)
        embedding_allies = th.matmul(ally_feats_t.unsqueeze(1), fc1_w_ally).view(
            BN, n_al, self.n_heads, self.rnn_hidden_dim).sum(dim=1, keepdim=False)

        embedding = embedding_own + self.unify_input_heads(embedding_enemies + embedding_allies)
        x = th.nn.functional.relu(embedding, inplace=True)
        hh = self.rnn(x, hidden_state.reshape(-1, self.rnn_hidden_dim))   # v_tilde: [BN, rnn_hidden_dim]

        # ===== Who is a healer, this episode? Read it from the observed unit type. =====
        # own_context = [move_feats ; own_feats]; own_feats ends with the unit-type one-hot and
        # terran_gen assigns medivac -> type_id 2 of 3, i.e. the final element.
        is_healer = own_feats_t[:, -1] > 0.5                              # [BN] bool

        # ===== Head: enumerate |A| candidates with an agent-conditional target embedding (D8) =====
        enemy_emb = self.enemy_encoder(enemy_feats_t.view(BN, n_en, self.enemy_feats_dim))
        ally_emb = self.ally_encoder(ally_feats_t.view(BN, n_al, self.ally_feats_dim))
        no_tgt = self.no_target.view(1, 1, self.emb_dim)

        # A healer's trailing block: heal targets first, then no_target for the slots it does not have.
        healer_trailing = th.cat(
            [ally_emb, no_tgt.expand(BN, n_en - n_al, self.emb_dim)], dim=1)      # [BN, n_en, emb]
        trailing = th.where(is_healer.view(BN, 1, 1), healer_trailing, enemy_emb)  # [BN, n_en, emb]
        e_hat = th.cat([no_tgt.expand(BN, self.n_normal, self.emb_dim), trailing], dim=1)  # [BN,|A|,emb]

        v_tilde = hh.unsqueeze(1).expand(BN, self.n_actions, self.rnn_hidden_dim)
        a_tilde = self.a_tilde_table[is_healer.long()]                    # [BN, |A|, n_normal+2]

        z = self.phi(th.cat([v_tilde, e_hat, a_tilde], dim=-1))
        phi = sem(self.sem_ln(z), self.sem_groups) if self.saleq_norm == "sem" else AvgL1Norm(z)
        q = self.head(phi).view(bs, n, self.n_actions)

        # Exposed for the Phase-2 predictive-loss / world-model learner (same contract as the parent).
        self.last_phi = phi.view(bs, n, self.n_actions, self.zsa_dim)
        self.last_v = hh.view(bs, n, -1)
        return q, hh.view(bs, n, -1)


if __name__ == "__main__":
    # Smoke test: PI/PE under both norms, with healers placed by OBSERVED TYPE at arbitrary positions.
    # Run from src with:  python -m modules.agents.hpn_saleq_terran
    from types import SimpleNamespace
    import torch.nn.functional as F

    # 10gen_terran shape: 5 agents, 5 enemies, 4 allies, |A| = 6 + 5 = 11, own_context = move(4)+own(6)
    bs, n, n_en, n_al = 2, 5, 5, 4
    own_dim, en_dim, al_dim, n_normal = 10, 8, 8, 6
    n_actions = n_normal + n_en

    def make(norm):
        args = SimpleNamespace(
            n_agents=n, n_allies=n_al, n_enemies=n_en, n_actions=n_actions,
            hpn_head_num=1, hpn_hyper_dim=64, rnn_hidden_dim=64, obs_agent_id=True,
            obs_last_action=False, map_type="terran_gen", output_normal_actions=n_normal,
            saleq_emb_dim=32, saleq_zsa_dim=64, saleq_hidden_dim=64,
            saleq_norm=norm, saleq_sem_groups=8,
        )
        return HPNSaleQTerranAgent((own_dim, (n_en, en_dim), (n_al, al_dim)), args).eval()

    def obs(healer_idx):
        """own_feats with the medivac bit (last element) set for the given agent indices."""
        own = th.randn(bs * n, own_dim)
        own[:, -3:] = 0.0
        own[:, -3] = 1.0                                  # default: marine (type_id 0)
        for i in healer_idx:
            for b in range(bs):
                own[b * n + i, -3:] = th.tensor([0., 0., 1.])   # medivac (type_id 2)
        return own

    enemy = th.randn(bs * n * n_en, en_dim)
    ally = th.randn(bs * n * n_al, al_dim)
    idx = [th.arange(n).unsqueeze(0).expand(bs, -1)]
    h = th.zeros(bs, n, 64)
    perm_en = th.tensor([2, 0, 1, 4, 3])
    perm_al = th.tensor([1, 0, 3, 2])
    en_p = enemy.view(bs * n, n_en, en_dim)[:, perm_en, :].reshape(-1, en_dim)
    al_p = ally.view(bs * n, n_al, al_dim)[:, perm_al, :].reshape(-1, al_dim)

    def run(agent, own, e, a):
        th.manual_seed(7)                      # ST sampling is stochastic -- seed for comparability
        with th.no_grad():
            return agent((bs, own, e, a, idx), h)[0]

    for norm in ("avgl1", "sem"):
        th.manual_seed(0)
        ag = make(norm)
        # --- healers at arbitrary positions, including NON-TAIL ---
        for label, hidx in [("k=0 (none)", []), ("k=1 tail", [4]),
                            ("k=2 tail", [3, 4]), ("k=1 NON-TAIL idx1", [1]),
                            ("k=2 non-contiguous", [0, 3])]:
            own = obs(hidx)
            q = run(ag, own, enemy, ally)
            assert q.shape == (bs, n, n_actions)
            comb = [i for i in range(n) if i not in hidx]
            # enemy permutation: combat attack Q equivariant; healer Q enemy-invariant
            qp = run(ag, own, en_p, ally)
            for i in comb:
                assert th.allclose(qp[:, i, n_normal:], q[:, i, n_normal:][:, perm_en], atol=1e-5), \
                    f"[{norm}/{label}] combat agent {i} attack Q not enemy-equivariant"
            for i in hidx:
                assert th.allclose(qp[:, i], q[:, i], atol=1e-5), \
                    f"[{norm}/{label}] healer {i} Q moved under enemy permutation"
            # ally permutation: healer heal Q equivariant; combat Q ally-invariant
            qa = run(ag, own, enemy, al_p)
            for i in hidx:
                heal = q[:, i, n_normal:n_normal + n_al]
                assert th.allclose(qa[:, i, n_normal:n_normal + n_al], heal[:, perm_al], atol=1e-5), \
                    f"[{norm}/{label}] healer {i} heal Q not ally-equivariant"
            for i in comb:
                assert th.allclose(qa[:, i], q[:, i], atol=1e-5), \
                    f"[{norm}/{label}] combat agent {i} Q moved under ally permutation"
        print(f"  [{norm}] OK: k=0/1/2, tail AND non-tail placements; enemy PE + ally PE, cross-invariance")

    # --- negative control: leak the ally slot index into a_tilde -> ally-equivariance MUST break ---
    th.manual_seed(0)
    bad = make("avgl1")
    with th.no_grad():
        tbl = bad.a_tilde_table.clone()
        for j in range(n_al):                                  # slot index leaks in
            tbl[1, n_normal + j, n_normal + 1] = 1.0 + 0.1 * j
        bad.a_tilde_table.copy_(tbl)
    own = obs([4])
    q0, q1 = run(bad, own, enemy, ally), run(bad, own, enemy, al_p)
    heal0 = q0[:, 4, n_normal:n_normal + n_al][:, perm_al]
    assert not th.allclose(q1[:, 4, n_normal:n_normal + n_al], heal0, atol=1e-5), \
        "negative control FAILED to break -- the ally-equivariance test is vacuous!"
    print("  negative control OK: leaking the ally slot index does break ally-equivariance (D11 guard live)")

    # --- gradients must reach ally_encoder (a silent detach would leave Enc_ally untrained forever) ---
    th.manual_seed(0)
    g = make("sem")
    own = obs([3, 4])
    g((bs, own, enemy, ally, idx), h)[0].sum().backward()
    assert g.ally_encoder[0].weight.grad is not None and g.ally_encoder[0].weight.grad.abs().sum() > 0, \
        "no gradient reaches ally_encoder"
    assert g.enemy_encoder[0].weight.grad is not None, "no gradient reaches enemy_encoder"
    print("  gradient flow OK: both ally_encoder and enemy_encoder receive gradient")

    print("HPN SALEQ TERRAN AGENT SMOKE TEST PASSED (both norms + non-tail + negative control)")
