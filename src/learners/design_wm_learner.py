# =====================================================================================================
# DESIGN-SPACE STUDY FILE -- serves design_e3_mix_s_zs_traj and design_e3_mix_zs_traj
# ("what does the mixing network read as its state input").
#
# Copied from: src/learners/saleq_wm_learner.py (SaleqWMLearner). That file is NOT modified.
# Instantiates the ORIGINAL world model (src/modules/model/sem_wm.py, unchanged).
#
# What changed relative to the source (everything else -- WM auxiliary losses, one-step prior training,
# imagination rollout, TD(lambda) targets, the single Adam over {agent, mixer, WM}, target updates,
# logging incl. traj_grad_norm / kl_drift_h*, cuda(), save/load incl. the `sem_wm.th` filename --
# is identical to the source):
#   * New flag `design_mixer_input` (default "s_traj" = the full method), one of
#       "s_traj"     mixer reads [ s_t ; traj_t ]          dim |s| + wm_traj_dim         (source)
#       "s_zs_traj"  mixer reads [ s_t ; z^s_t ; traj_t ]  dim |s| + wm_state_dim + wm_traj_dim
#       "zs_traj"    mixer reads [ z^s_t ; traj_t ]        dim wm_state_dim + wm_traj_dim
#     where z^s_t = SEM(LN(E_w(s_t))) is the WM's deterministic simplicial state embedding (the same
#     tensor that serves, detached, as the target of L_rec / L_pred). Any other value -> ValueError.
#     The live/target mixers are built with the matching input dim (checkpoint shapes differ from the
#     full method for the two non-default values).
#   * New flag `design_mixer_zs_detach` (default True): whether z^s is detached before entering the
#     mixer. NOTE: detaching removes only the DIRECT mixer -> z^s path. The TD loss still reaches the
#     WM state encoder through posterior(z^s, phi^sa) -> z_q -> imagine -> traj -> mixer (verified
#     empirically in this session with c_wm=0: state_encoder receives non-zero TD gradient in the full
#     method). Detach is the minimal-change default so that the only new gradient route into the WM
#     targets is not introduced; False is kept for a follow-up.
#   * Construction order in __init__ is preserved exactly (super().__init__ -> SemWorldModel(args) ->
#     Mixer(aug_args) -> deepcopy -> optimizer), so same-seed RNG consumption matches the source.
#   * The flag name is deliberately NOT `mixer_state` -- that key already has different semantics in
#     src/learners/ablation_mixstate_learner.py (ablation arms 5a/5b).
#
# Defaults (design_mixer_input="s_traj") reproduce the full method bit-identically (verified by
# src/tests/test_design_equivalence.py). The world model always encodes the full global state s_t;
# only what the MIXER sees changes, so each variant is a single-variable diff vs. the full method.
# Race scope: protoss + zerg only (the deployed agent `hpn_saleq` asserts on terran_gen).
# =====================================================================================================
import copy
import time

import numpy as np
import torch as th
import torch.nn.functional as F
from torch.optim import RMSprop, Adam

from components.episode_buffer import EpisodeBatch
from modules.mixers.nmix import Mixer
from modules.model.sem_wm import SemWorldModel
from utils.rl_utils import build_td_lambda_targets
from .nq_learner import NQLearner, calculate_target_q


class DesignWMLearner(NQLearner):
    """DESIGN-SPACE copy of SaleqWMLearner with a configurable mixer state input
    (``design_mixer_input`` in {"s_traj", "s_zs_traj", "zs_traj"}, ``design_mixer_zs_detach``).
    See the file header for the full diff vs. ``SaleqWMLearner``.
    """

    VALID_MIXER_INPUT = ("s_traj", "s_zs_traj", "zs_traj")

    def __init__(self, mac, scheme, logger, args):
        super().__init__(mac, scheme, logger, args)
        self.wm = SemWorldModel(args)
        self.k_train = getattr(args, "wm_k_train", 1)
        if self.k_train != 1:
            raise NotImplementedError("wm_k_train>1 is the W12 escalation (multi-step overshoot); not in v1.")
        self.alpha = getattr(args, "wm_alpha", 0.8)
        self.c_wm = getattr(args, "c_wm", 1.0)
        self.c_kl = getattr(args, "c_kl", 1.0)
        self.c_rec = getattr(args, "c_rec", 1.0)
        self.c_pred = getattr(args, "c_pred", 1.0)
        self.c_act = getattr(args, "c_act", 1.0)

        # DESIGN-SPACE: what the mixer reads.
        self.mixer_input = getattr(args, "design_mixer_input", "s_traj")
        if self.mixer_input not in self.VALID_MIXER_INPUT:
            raise ValueError("design_mixer_input must be one of {} (got {!r})".format(
                self.VALID_MIXER_INPUT, self.mixer_input))
        self.zs_detach = bool(getattr(args, "design_mixer_zs_detach", True))

        # Rebuild the mixers with the chosen input dim (nmix reads args.state_shape at construction; pass a copy).
        state_dim = int(np.prod(args.state_shape))
        mix_dim = {
            "s_traj": state_dim + self.wm.traj_dim,
            "s_zs_traj": state_dim + self.wm.d_s + self.wm.traj_dim,
            "zs_traj": self.wm.d_s + self.wm.traj_dim,
        }[self.mixer_input]
        aug_args = copy.copy(args)
        aug_args.state_shape = mix_dim
        self.mixer = Mixer(aug_args)
        self.target_mixer = copy.deepcopy(self.mixer)

        # Single optimizer over {agent, mixer, WM} (W4: TD grads flow into the WM through the mixer input).
        self.params = list(self.mac.parameters()) + list(self.mixer.parameters()) + list(self.wm.parameters())
        if self.args.optimizer == 'adam':
            self.optimiser = Adam(params=self.params, lr=args.lr, weight_decay=getattr(args, "weight_decay", 0))
        else:
            self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

    # ---------- WM loss helpers ----------
    def _kl(self, qp, pp):  # per-group KL(qp || pp) summed over groups -> [bs, L]
        return (qp * (th.log(qp + 1e-8) - th.log(pp + 1e-8))).sum(-1).sum(-1)

    def _ce_state(self, logits, target):  # per-group CE vs simplicial target -> [bs, L]
        G, V = self.wm.groups, self.wm.vertices
        lp = F.log_softmax(logits.reshape(*logits.shape[:-1], G, V), dim=-1)
        return -(target.reshape(*target.shape[:-1], G, V) * lp).sum(-1).sum(-1)

    def _ce_action(self, logits, target_idx):  # per-agent CE, SUMMED over agents (§1.4: L_act = Sum_i CE) -> [bs, L]
        lp = F.log_softmax(logits, dim=-1)
        return (-lp.gather(-1, target_idx.unsqueeze(-1)).squeeze(-1)).sum(-1)

    @staticmethod
    def _mmean(x, m):  # masked mean; x [bs, L], m [bs, L, 1]
        m = m.squeeze(-1)
        return (x * m).sum() / m.sum().clamp(min=1.0)

    def _mixer_state(self, states, z_s, traj):
        # DESIGN-SPACE: assemble the mixer's state input per `design_mixer_input`.
        if self.mixer_input == "s_traj":
            return th.cat([states, traj], dim=-1)
        zs_in = z_s.detach() if self.zs_detach else z_s
        if self.mixer_input == "s_zs_traj":
            return th.cat([states, zs_in, traj], dim=-1)
        return th.cat([zs_in, traj], dim=-1)  # "zs_traj"

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        start_time = time.time()
        if self.args.use_cuda and str(self.mac.get_device()) == "cpu":
            self.mac.cuda()

        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        states = batch["state"]                        # [bs, T, state_dim]
        actions_onehot = batch["actions_onehot"].float()  # [bs, T, n, |A|]
        filled = batch["filled"].float()               # [bs, T, 1]

        # ===== Live Q rollout (as NQLearner) =====
        self.mac.set_train_mode()
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            mac_out.append(self.mac.forward(batch, t=t))
        mac_out = th.stack(mac_out, dim=1)
        mac_out[avail_actions == 0] = -9999999
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)

        with th.no_grad():
            target_mac_out = calculate_target_q(self.target_mac, batch)
            cur_max_actions = mac_out.max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)

        # ===== World model (§1) =====
        z_s = self.wm.state_embed(states)                      # [bs, T, d_s]  simplicial (targets are detached below)
        phi_sa = self.wm.sa_embed(states, actions_onehot)      # [bs, T, d_sa]
        q_probs = self.wm.probs(self.wm.posterior_logits(z_s, phi_sa))   # [bs, T, G, V]
        z_q = self.wm.st_sample(q_probs)                       # [bs, T, d_wm]

        # one-step prior (W13 k_train=1): p_t from (z_q[t-1], a[t-1]) for t=1..T-1
        p_probs = self.wm.probs(self.wm.prior_logits(z_q[:, :-1], actions_onehot[:, :-1]))  # [bs, T-1, G, V]
        z_p = self.wm.st_sample(p_probs)                       # [bs, T-1, d_wm]

        s_tgt = z_s.detach()                                   # sg(SEM(LN(E(s))))
        phi_tgt = phi_sa.detach()

        # recon (posterior side, all t) ; pred (prior side, t=1..T-1)
        q_sl, q_ph, _ = self.wm.decode(z_q)
        L_rec = self._ce_state(q_sl, s_tgt) + F.mse_loss(q_ph, phi_tgt, reduction='none').mean(-1)       # [bs, T]
        p_sl, p_ph, p_al = self.wm.decode(z_p)
        L_pred = self._ce_state(p_sl, s_tgt[:, 1:]) + F.mse_loss(p_ph, phi_tgt[:, 1:], reduction='none').mean(-1)  # [bs,T-1]
        # action head (prior side): predict replayed a_t from the action-uninformed prior latent z_p (§1.4)
        L_act = self._ce_action(p_al, actions_onehot[:, 1:].argmax(-1))                                   # [bs, T-1]
        # KL (per-group, balanced) between posterior q_t and prior p_t, t=1..T-1
        q_t = q_probs[:, 1:]
        L_KL = self.alpha * self._kl(q_t.detach(), p_probs) + (1 - self.alpha) * self._kl(q_t, p_probs.detach())

        m_all, m_step = filled, filled[:, 1:]
        L_WM = (self.c_kl * self._mmean(L_KL, m_step) + self.c_rec * self._mmean(L_rec, m_all)
                + self.c_pred * self._mmean(L_pred, m_step) + self.c_act * self._mmean(L_act, m_step))

        # ===== Imagination -> traj (one live-WM rollout, sliced for both mixers) =====
        traj = self.wm.aggregate(self.wm.imagine(z_q, self.wm.k))   # [bs, T, traj_dim]
        log_now = t_env - self.log_stats_t >= self.args.learner_log_interval
        if log_now:
            traj.retain_grad()   # W4 monitor: how much TD gradient reaches the WM through the mixer input
        aug_state = self._mixer_state(states, z_s, traj)            # DESIGN-SPACE: [bs, T, mix_dim]

        # ===== TD(lambda) with the augmented mixer state =====
        self.mixer.train()
        chosen_action_qvals = self.mixer(chosen_action_qvals, aug_state[:, :-1])
        with th.no_grad():
            self.target_mixer.eval()
            target_qtot = self.target_mixer(target_max_qvals, aug_state)   # over all T
            # PORT NOTE (pymarl3): this repo's build_td_lambda_targets is the 6-arg form
            # (rewards, terminated, mask, target_qs, gamma, td_lambda). marl_project3's variant takes an
            # extra, unused `n_agents` and is called there as (..., target_qtot, None, gamma, td_lambda).
            # Keep these two copies in sync manually -- passing the 7-arg form here is a TypeError.
            targets = build_td_lambda_targets(rewards, terminated, mask, target_qtot,
                                              self.args.gamma, self.args.td_lambda)

        td_error = (chosen_action_qvals - targets)
        mask_td = mask.expand_as(td_error)
        masked_td_error = 0.5 * td_error.pow(2) * mask_td
        loss_td = masked_td_error.sum() / mask_td.sum()

        loss = loss_td + self.c_wm * L_WM

        # ===== Optimise =====
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        assert th.isfinite(loss) and th.isfinite(grad_norm), "NaN/Inf in loss/grad (TD + WM)"
        self.optimiser.step()

        self.train_t += 1
        self.avg_time += (time.time() - start_time - self.avg_time) / self.train_t
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            print("Avg cost {} seconds".format(self.avg_time))

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if log_now:
            self.logger.log_stat("loss_td", loss_td.item(), t_env)
            self.logger.log_stat("loss_kl", self._mmean(L_KL, m_step).item(), t_env)
            self.logger.log_stat("loss_rec", self._mmean(L_rec, m_all).item(), t_env)
            self.logger.log_stat("loss_pred", self._mmean(L_pred, m_step).item(), t_env)
            self.logger.log_stat("loss_act", self._mmean(L_act, m_step).item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            if traj.grad is not None:  # W4: TD-gradient magnitude arriving at the WM via traj
                self.logger.log_stat("traj_grad_norm", traj.grad.norm().item(), t_env)
            for h, d in self._open_loop_drift(z_q.detach(), q_probs.detach(), actions_onehot, filled).items():
                self.logger.log_stat("kl_drift_h{}".format(h), d, t_env)
            self.log_stats_t = t_env

    def _open_loop_drift(self, z_q, q_probs, actions_onehot, filled):
        # W13 diagnostic (no_grad): KL(posterior_{t+h} || prior rolled h steps open-loop, teacher-forced actions).
        # Masked to real transitions -- this drives the W12 escalation decision, so padded steps must not bias it.
        out = {}
        with th.no_grad():
            z_roll = z_q
            for h in range(1, self.wm.k + 1):
                a_h = actions_onehot[:, h - 1:]
                L = min(z_roll.size(1), a_h.size(1))
                if L <= 0:
                    break
                p_h = self.wm.probs(self.wm.prior_logits(z_roll[:, :L], a_h[:, :L]))
                z_roll = self.wm.st_sample(p_h)
                q_h = q_probs[:, h:h + p_h.size(1)]
                Lc = min(q_h.size(1), p_h.size(1))
                if Lc <= 0:
                    break
                out[h] = self._mmean(self._kl(q_h[:, :Lc], p_h[:, :Lc]), filled[:, h:h + Lc]).item()
        return out

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        self.mixer.cuda()
        self.target_mixer.cuda()
        self.wm.cuda()

    def save_models(self, path):
        super().save_models(path)
        th.save(self.wm.state_dict(), "{}/sem_wm.th".format(path))

    def load_models(self, path):
        super().load_models(path)
        self.wm.load_state_dict(th.load("{}/sem_wm.th".format(path), map_location=lambda storage, loc: storage))
