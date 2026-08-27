# =====================================================================================================
# ABLATION FILE -- serves ablation_arm_5a (mixer_state="traj") and ablation_arm_5b (mixer_state="none"):
# "global state passed into the mixing network".
#
# Copied from: src/learners/saleq_wm_learner.py (SaleqWMLearner). That file is NOT modified.
#
# What changed relative to the source:
#   * New REQUIRED arg `mixer_state` in {"traj", "none"} (no default: this learner exists only for the
#     ablation; a missing/unknown value raises ValueError so a mis-configured run fails at startup).
#   * __init__: the live/target mixers are rebuilt with the ablated input dim instead of
#     state_shape + wm_traj_dim:   "traj" -> wm.traj_dim ;  "none" -> 1.
#   * train():
#       "traj": aug_state = traj  (imagined-trajectory summary ONLY; raw global state s_t dropped).
#               traj.retain_grad() is kept so `traj_grad_norm` still logs.
#       "none": aug_state = ones[bs, T, 1]  (constant). The imagination rollout has no consumer, so it is
#               skipped entirely (saves the k-step prior rollout); the `traj_grad_norm` log is skipped.
#               With a constant input every hypernetwork output is a learned constant, so the mixer
#               degenerates to state-INdependent, non-negative (abs -> monotone) mixing weights and biases:
#               a nonlinear-VDN control ("no global state at all").
#   * Everything else -- WM auxiliary losses (L_kl/L_rec/L_pred/L_act, still trained), TD(lambda) targets,
#     the single Adam over {agent, mixer, WM}, target updates, logging incl. kl_drift_h*, save/load incl.
#     sem_wm.th -- is identical to the source.
#
# The world model still encodes the FULL global state s_t in both arms; only what the MIXER sees changes,
# so each arm is a single-variable diff vs. the full method.
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


class AblationMixStateLearner(NQLearner):
    """ABLATION arms 5a/5b: SaleqWMLearner with the mixer's global-state input ablated.

    ``mixer_state="traj"`` -> mixer reads traj_t only (no s_t).
    ``mixer_state="none"`` -> mixer reads a constant (no s_t, no traj_t): state-independent monotone mixing.
    See the file header for the full diff vs. ``SaleqWMLearner``.
    """

    VALID_MIXER_STATE = ("traj", "none")

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

        # ABLATION: which input the mixer sees. Required; no silent fallback to the full method.
        self.mixer_state = getattr(args, "mixer_state", None)
        if self.mixer_state not in self.VALID_MIXER_STATE:
            raise ValueError(
                "ablation_mixstate_learner requires mixer_state in {} (got {!r}). Use the full method's "
                "saleq_wm_learner for the [s ; traj] mixer input.".format(self.VALID_MIXER_STATE, self.mixer_state))
        mix_dim = self.wm.traj_dim if self.mixer_state == "traj" else 1

        # Rebuild the mixers with the ABLATED input dim (nmix reads args.state_shape at construction; pass a copy).
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

    # ---------- WM loss helpers (verbatim) ----------
    def _kl(self, qp, pp):  # per-group KL(qp || pp) summed over groups -> [bs, L]
        return (qp * (th.log(qp + 1e-8) - th.log(pp + 1e-8))).sum(-1).sum(-1)

    def _ce_state(self, logits, target):  # per-group CE vs simplicial target -> [bs, L]
        G, V = self.wm.groups, self.wm.vertices
        lp = F.log_softmax(logits.reshape(*logits.shape[:-1], G, V), dim=-1)
        return -(target.reshape(*target.shape[:-1], G, V) * lp).sum(-1).sum(-1)

    def _ce_action(self, logits, target_idx):  # per-agent CE, SUMMED over agents -> [bs, L]
        lp = F.log_softmax(logits, dim=-1)
        return (-lp.gather(-1, target_idx.unsqueeze(-1)).squeeze(-1)).sum(-1)

    @staticmethod
    def _mmean(x, m):  # masked mean; x [bs, L], m [bs, L, 1]
        m = m.squeeze(-1)
        return (x * m).sum() / m.sum().clamp(min=1.0)

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

        # ===== World model (unchanged: encodes the FULL global state) =====
        z_s = self.wm.state_embed(states)                      # [bs, T, d_s]
        phi_sa = self.wm.sa_embed(states, actions_onehot)      # [bs, T, d_sa]
        q_probs = self.wm.probs(self.wm.posterior_logits(z_s, phi_sa))   # [bs, T, G, V]
        z_q = self.wm.st_sample(q_probs)                       # [bs, T, d_wm]

        p_probs = self.wm.probs(self.wm.prior_logits(z_q[:, :-1], actions_onehot[:, :-1]))  # [bs, T-1, G, V]
        z_p = self.wm.st_sample(p_probs)                       # [bs, T-1, d_wm]

        s_tgt = z_s.detach()
        phi_tgt = phi_sa.detach()

        q_sl, q_ph, _ = self.wm.decode(z_q)
        L_rec = self._ce_state(q_sl, s_tgt) + F.mse_loss(q_ph, phi_tgt, reduction='none').mean(-1)       # [bs, T]
        p_sl, p_ph, p_al = self.wm.decode(z_p)
        L_pred = self._ce_state(p_sl, s_tgt[:, 1:]) + F.mse_loss(p_ph, phi_tgt[:, 1:], reduction='none').mean(-1)  # [bs,T-1]
        L_act = self._ce_action(p_al, actions_onehot[:, 1:].argmax(-1))                                   # [bs, T-1]
        q_t = q_probs[:, 1:]
        L_KL = self.alpha * self._kl(q_t.detach(), p_probs) + (1 - self.alpha) * self._kl(q_t, p_probs.detach())

        m_all, m_step = filled, filled[:, 1:]
        L_WM = (self.c_kl * self._mmean(L_KL, m_step) + self.c_rec * self._mmean(L_rec, m_all)
                + self.c_pred * self._mmean(L_pred, m_step) + self.c_act * self._mmean(L_act, m_step))

        # ===== ABLATION: mixer input =====
        log_now = t_env - self.log_stats_t >= self.args.learner_log_interval
        traj = None
        if self.mixer_state == "traj":
            # imagined-trajectory summary ONLY (raw s_t dropped)
            traj = self.wm.aggregate(self.wm.imagine(z_q, self.wm.k))   # [bs, T, traj_dim]
            if log_now:
                traj.retain_grad()   # W4 monitor: TD gradient reaching the WM through the mixer input
            aug_state = traj
        else:  # "none": constant input -> state-independent (but still monotone) learned mixing weights
            aug_state = states.new_ones(states.shape[0], states.shape[1], 1)   # [bs, T, 1]

        # ===== TD(lambda) with the ablated mixer input =====
        self.mixer.train()
        chosen_action_qvals = self.mixer(chosen_action_qvals, aug_state[:, :-1])
        with th.no_grad():
            self.target_mixer.eval()
            target_qtot = self.target_mixer(target_max_qvals, aug_state)   # over all T
            # 6-arg build_td_lambda_targets (pymarl3 form), as in the source.
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
            if traj is not None and traj.grad is not None:  # only in the "traj" arm
                self.logger.log_stat("traj_grad_norm", traj.grad.norm().item(), t_env)
            for h, d in self._open_loop_drift(z_q.detach(), q_probs.detach(), actions_onehot, filled).items():
                self.logger.log_stat("kl_drift_h{}".format(h), d, t_env)
            self.log_stats_t = t_env

    def _open_loop_drift(self, z_q, q_probs, actions_onehot, filled):
        # W13 diagnostic (no_grad, verbatim): KL(posterior_{t+h} || prior rolled h steps open-loop).
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
