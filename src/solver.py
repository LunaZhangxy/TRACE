"""
TRACE Solver: Trajectory-Conditioned Delayed Feedback Model.

Implements the training and inference pipeline described in the paper,
including trajectory-conditioned estimation (Sec. 3.1), retrospective
trajectory completion (Sec. 3.2), and unified optimisation (Sec. 3.3).

Supports both datasets:
  - Criteo:  K=1 (purchase only)
  - Taobao:  K=3 (purchase, cart, fav)
"""

import os
import math
import numpy as np
import torch
import torch.nn.functional as F

from metric import MetricAccumulator, cal_metric
from utils import ProgressInfo, get_optimizer
from data import get_train_data_from_batch
from model import CriteoMLP, CriteoDXY, TaobaoMLP, TaobaoDXY
from completer import CriteoCompleter, TaobaoCompleter


def binary_entropy_bits(p, eps=1e-8):
    """Binary entropy H(p) in bits, element-wise."""
    p = p.clamp(eps, 1 - eps)
    return -(p * torch.log2(p) + (1 - p) * torch.log2(1 - p))

def confidence_score(p, eps=1e-6):
    """Completer confidence: v = 1 - H(p) (bits), clamped to [0, 1]."""
    return (1.0 - binary_entropy_bits(p, eps)).clamp(0.0, 1.0)


class Solver:

    def __init__(self, pretrain_dataset, stream_dataset, params):
        self.params = params
        self.device = params["device"]
        self.dataset = params["dataset"]
        self.pretrain_dataset = pretrain_dataset
        self.stream_dataset = stream_dataset
        self.method = params["method"]

        self.hidden_size = params["hidden_size"]
        self.y_class_num = params["y_class_num"]

        # ---- Trajectory estimator g_psi (Sec. 3.1) ----
        if self.method == "trace":
            self.d_type = params["d_type"]
            self.d_size = params["d_size"]       # K * 2 (total logit dims per window)
            self.d_nt = params["d_nt"]           # H: number of observation windows
            self.K = self.d_size // 2            # K: number of action types
            self.nt = params["nt"]
            self.beta = params["beta"]
            assert self.d_type == "category"

            if self.dataset == "criteo":
                self.g_psi = CriteoDXY(
                    y_size=self.y_class_num,
                    d_size=self.d_size * self.d_nt,
                ).to(self.device)
            else:
                self.g_psi = TaobaoDXY(
                    y_size=self.y_class_num,
                    d_size=self.d_size * self.d_nt,
                ).to(self.device)

            self.optimizer_psi = get_optimizer(
                params["optimizer"], params)(self.g_psi.parameters())

        # ---- Static intent estimator f_theta (Sec. 3.1) ----
        if self.dataset == "criteo":
            self.f_theta = CriteoMLP(output_size=self.y_class_num).to(self.device)
        else:
            self.f_theta = TaobaoMLP(output_size=self.y_class_num).to(self.device)

        self.optimizer_theta = get_optimizer(
            params["optimizer"], params)(self.f_theta.parameters())

        # ---- Retrospective trajectory completer q_phi (Sec. 3.2) ----
        self.completer = None
        self.lambda_con = float(params.get("lambda_con", 0.1))
        completer_path = params.get("completer_ckpt_path", None)

        if completer_path is not None and self.lambda_con > 0:
            if self.dataset == "criteo":
                self.completer = CriteoCompleter(
                    hidden_size=self.hidden_size,
                    horizon_embed_dim=16,
                    H=self.d_nt if self.method == "trace" else params.get("d_nt", 6),
                ).to(self.device)
            else:
                self.completer = TaobaoCompleter(
                    hidden_size=self.hidden_size,
                    horizon_embed_dim=16,
                    H=self.d_nt if self.method == "trace" else params.get("d_nt", 5),
                ).to(self.device)

            state = torch.load(completer_path, map_location=self.device)
            self.completer.load_state_dict(state)
            self.completer.eval()
            for p in self.completer.parameters():
                p.requires_grad = False
            print(f"[TRACE] Loaded retrospective completer from {completer_path}")

        self.global_stream_step = 0
        if self.method == "trace":
            self._h_arange = torch.arange(self.d_nt, device=self.device).view(1, -1)

    # Reliability gate  w_i  (Sec. 3.2, Eq. 11)

    def _compute_reliability_gate(self, p_online, q_completer, m):
        # v in [0,1]
        v = confidence_score(q_completer)              # 1 - H(q)
        # H(p) in [0,1]
        H_p = binary_entropy_bits(p_online)            # bits entropy
        # kappa in [0,1], sparsity = 1-kappa
        kappa = m.float().sum(dim=1) / float(self.d_nt)
        sparsity = (1.0 - kappa).clamp(0.0, 1.0)
        w = torch.sigmoid(H_p) * torch.sigmoid(v) * torch.sigmoid(sparsity)
        w = w.clamp(0.0, 1.0)
        w = torch.nan_to_num(w, nan=0.0, posinf=1.0, neginf=0.0)
        return w.detach(), v.detach(), H_p.detach(), kappa.detach()

    # Horizon weight computation  eta_h  (Sec. 3.1, Eq. 6)

    def compute_horizon_weights(self, y, d):
        assert self.y_class_num == 2
        cond_entropy = []

        for i in range(self.d_nt):
            if d.dim() == 3:
                d_i = d[:, i, 0].long()
            else:
                d_i = d[:, i].long()

            n_cls = 2
            pdy = torch.zeros((n_cls, 2), device=d.device, dtype=torch.float32)
            for vy in range(2):
                y_mask = (y == vy)
                for vd in range(n_cls):
                    pdy[vd, vy] = (y_mask & (d_i == vd)).float().mean()

            pd = F.one_hot(d_i, n_cls).float().mean(dim=0).view(-1, 1)
            eps = 1e-12
            ce = torch.sum(pdy * torch.log(pd.clamp_min(eps))
                        - pdy * torch.log(pdy.clamp_min(eps)))
            cond_entropy.append(ce)

        cond_entropy = torch.stack(cond_entropy)
        max_ce = cond_entropy.max()
        if torch.isfinite(max_ce) and max_ce > 0:
            cond_entropy = cond_entropy / max_ce
        else:
            cond_entropy = torch.zeros_like(cond_entropy)

        h_indices = torch.arange(1, self.d_nt + 1, device=d.device, dtype=torch.float32)
        w_time = torch.exp(-h_indices / self.d_nt)                    # exp(-h/H)
        w_info = torch.exp(-self.beta * cond_entropy)                  # exp(-beta * C_tilde_h)
        correction = 1.0 / torch.arange(                               # (H-h+1)^{-1}
            self.d_nt, 0, -1, device=d.device, dtype=torch.float32)

        eta = w_time * w_info * correction                             

        mean_eta = eta.mean()                                          
        if torch.isfinite(mean_eta) and mean_eta > 0:
            eta = eta / mean_eta
        eta = torch.nan_to_num(eta, nan=1.0, posinf=1.0, neginf=1.0)

        self.eta = eta.to(self.device)
        assert torch.isfinite(self.eta).all(), f"eta contains NaN/Inf: {self.eta}"

  
    def _get_label_from_d(self, d, h_idx):
        """Extract binary label at horizon h_idx (pay action for multi-action)."""
        if d.dim() == 3:
            return d[:, h_idx, 0]    # pay action
        return d[:, h_idx]


    def update_step(self, x, y, d, y_mask, d_mask, streaming):
        B = x.shape[0]
        if streaming:
            self.global_stream_step += 1
        if B <= 1:
            return None

        last_h = d_mask.shape[1] - 1   # last window index

        if self.method != "trace":
            if self.method == "ce":
                if not streaming:
                    raise ValueError("CE requires streaming mode")
                _mask = d_mask[:, last_h]
                _label = self._get_label_from_d(d, last_h)
                self.f_theta.train()
                lx = x[_mask]
                ly = _label[_mask]
                if lx.shape[0] <= 1:
                    return None
                logits = self.f_theta(lx)
                prob = F.softmax(logits, dim=1)
                loss = F.nll_loss(F.log_softmax(logits, dim=1), ly.view(-1))

            elif self.method == "pretrain":
                if streaming:
                    raise ValueError("Pretrain does not run in streaming")
                mask_last = d_mask[:, last_h]
                label_last = self._get_label_from_d(d, last_h)
                lx = x[mask_last]
                ly = label_last[mask_last]
                if lx.shape[0] <= 1:
                    return None
                logits = self.f_theta(lx)
                prob = F.softmax(logits, dim=1)
                loss = F.nll_loss(F.log_softmax(logits, dim=1), ly.view(-1))

            elif self.method == "oracle":
                if not streaming:
                    raise ValueError("Oracle requires streaming mode")
                self.f_theta.train()
                h0_mask = d_mask[:, 0] & ~d_mask[:, 1]
                lx = x[h0_mask]
                ly = y[h0_mask]
                if lx.shape[0] <= 1:
                    return None
                logits = self.f_theta(lx)
                prob = F.softmax(logits, dim=1)
                loss = F.nll_loss(F.log_softmax(logits, dim=1), ly.view(-1))
            else:
                raise ValueError(f"Unknown method: {self.method}")

            self.optimizer_theta.zero_grad()
            loss.backward()
            self.optimizer_theta.step()
            return {
                "loss": loss.item(),
                "target": ly.detach().cpu().numpy(),
                "prob": prob.detach().cpu().numpy(),
                "cvr": (ly == 1).float().mean().item(),
                "real_batch_size": ly.shape[0],
            }


        if not streaming:
            y_onehot = F.one_hot(y, self.y_class_num).float()

            # g_psi raw output: (B, d_nt * K * 2)
            d_pred_raw = self.g_psi(x, y_onehot)

            if self.K == 1:
                # Criteo: d is (B, H, 1) or (B, H)
                d_label = d.squeeze(-1) if d.dim() == 3 else d
                d_label = d_label.long()                                # (B, H)
                d_pred = d_pred_raw.view(B, self.d_nt, 2)              # (B, H, 2)
                _pred = d_pred.view(B * self.d_nt, 2)
                _label = d_label.contiguous().view(B * self.d_nt)
            else:
                # Taobao: d is (B, H, K)
                d_label = d.long()                                      # (B, H, K)
                d_pred = d_pred_raw.view(B, self.d_nt, self.K, 2)      # (B, H, K, 2)
                _pred = d_pred.view(B * self.d_nt * self.K, 2)
                _label = d_label.contiguous().view(B * self.d_nt * self.K)

            d_loss = F.nll_loss(F.log_softmax(_pred, dim=1), _label, reduction="mean")

            self.optimizer_psi.zero_grad()
            d_loss.backward()
            self.optimizer_psi.step()
            return {"d_loss": d_loss.item()}


        # TRACE: Online streaming update of f_theta
        self.g_psi.eval()
        self.f_theta.train()

        #  Static intent logits 
        y_logits = self.f_theta(x)                                         # (B, 2)

        #  Trajectory likelihood  log g(xi | x, y)  (Eq. 6-7) 
        sample_labels = torch.arange(self.y_class_num, device=x.device)
        sample_labels = sample_labels.unsqueeze(0).expand(B, -1)           # (B, C)
        sample_oh = F.one_hot(sample_labels, self.y_class_num).float()     # (B, C, C)
        sample_x = x.unsqueeze(1).expand(-1, self.y_class_num, -1)         # (B, C, dim)

        if self.K == 1:
            # Criteo: d is (B, H, 1) or (B, H) -> (B, H)
            d_label = d.squeeze(-1) if d.dim() == 3 else d
            d_label = d_label.long()                                       # (B, H)
            sample_d = d_label.unsqueeze(1).expand(-1, self.y_class_num, -1)  # (B, C, H)

            pred_d = self.g_psi(
                sample_x.reshape(B * self.y_class_num, -1),
                sample_oh.reshape(B * self.y_class_num, -1),
            ).view(B, self.y_class_num, self.d_nt, 2)                     # (B, C, H, 2)

            d_onehot = F.one_hot(sample_d, 2).float()                     # (B, C, H, 2)
            log_prob_d = (F.log_softmax(pred_d, dim=3) * d_onehot
                          ).sum(dim=3).detach().float()                    # (B, C, H)
        else:
            # Taobao: d is (B, H, K)
            d_label = d.long()                                             # (B, H, K)
            sample_d = d_label.unsqueeze(1).expand(
                -1, self.y_class_num, -1, -1)                              # (B, C, H, K)

            pred_d = self.g_psi(
                sample_x.reshape(B * self.y_class_num, -1),
                sample_oh.reshape(B * self.y_class_num, -1),
            ).view(B, self.y_class_num, self.d_nt, self.K, 2)             # (B, C, H, K, 2)

            d_onehot = F.one_hot(sample_d, 2).float()                     # (B, C, H, K, 2)
            log_prob_d_per_action = (
                F.log_softmax(pred_d, dim=4) * d_onehot
            ).sum(dim=4).detach().float()                                  # (B, C, H, K)
            # Sum over K actions (independence assumption)
            log_prob_d = log_prob_d_per_action.sum(dim=3)                  # (B, C, H)

        log_prob_y = F.log_softmax(y_logits, dim=1
                     ).unsqueeze(2).expand(-1, -1, self.d_nt).float()      # (B, C, H)

        # Per-sample per-horizon marginal NLL  (Eq. 7)
        per_ih_loss = -torch.logsumexp(log_prob_d + log_prob_y, dim=1)     # (B, H)

        m = d_mask.float()                                                 # (B, H)
        eta_h = self.eta.view(1, -1).float()                               # (1, H)

        # Normalised horizon weights  alpha_{i,h}  (Eq. 6)
        W_ih = m * eta_h
        denom = W_ih.sum(dim=1, keepdim=True).clamp_min(1e-6)
        alpha_ih = W_ih / denom                                            # (B, H)

        # Trajectory log-likelihood score per class  (Eq. 6)
        log_e_y = (alpha_ih.unsqueeze(1) * log_prob_d).sum(dim=2)          # (B, 2)

        # Posterior  p(y | x, xi)  (Eq. 4)
        log_prior_y = F.log_softmax(y_logits, dim=1).float()              # (B, 2)
        post_logits = log_prior_y + log_e_y
        post_prob = torch.softmax(post_logits, dim=1)                      # (B, 2)
        post_prob_pos = post_prob[:, 1]                                    # (B,)

        # Trajectory loss  L_trj  (Eq. 8)
        W = m * eta_h
        trj_loss = (W * per_ih_loss).sum() / W.sum().clamp_min(1e-6)

        # Supervised loss  L_sup  for revealed set  R_tau  (Eq. 9) 
        y_mask_bool = y_mask.view(-1).bool()
        y_true = y.view(-1)

        sup_loss = torch.tensor(0.0, device=self.device)
        if y_mask_bool.any():
            idx_rev = torch.where(y_mask_bool)[0]
            sup_loss = F.nll_loss(
                torch.log(post_prob[idx_rev] + 1e-12),
                y_true[idx_rev], reduction="mean")

        #  Consistency loss  L_con  for unrevealed set  U_tau  (Eq. 11) 
        con_loss = torch.tensor(0.0, device=self.device)
        idx_unrev = torch.where(~y_mask_bool)[0]

        if idx_unrev.numel() > 0 and self.lambda_con > 0 and self.completer is not None:
            with torch.no_grad():
                o_input = d if d.dim() == 3 else d.unsqueeze(-1)
                q_prob = self.completer(x, o_input, d_mask)                # (B,)

            p_unrev = post_prob_pos[idx_unrev]
            q_unrev = q_prob[idx_unrev]
            p_c = p_unrev.clamp(1e-6, 1 - 1e-6)
            q_c = q_unrev.clamp(1e-6, 1 - 1e-6)
            bce_vec = F.binary_cross_entropy(p_c, q_c, reduction="none")

            w_all = self._compute_reliability_gate(
                post_prob_pos, q_prob, d_mask)[0]
            w = w_all[idx_unrev]

            con_loss = (w * bce_vec).sum() / w.sum().clamp_min(1e-12)

        # Total loss  (Eq. 12) 
        total_loss = trj_loss + sup_loss + self.lambda_con * con_loss

        self.optimizer_theta.zero_grad()
        if not torch.isfinite(total_loss):
            print("[WARN] Loss is NaN/Inf, skipping step.")
            return None
        total_loss.backward()
        self.optimizer_theta.step()

        return {
            "trj_loss": trj_loss.item(),
            "sup_loss": sup_loss.item(),
            "con_loss": con_loss.item(),
        }

    def pretrain(self):

        ckpt_path = f"./pretrain_model/{self.dataset}_pretrain.pt"

        if os.path.isfile(ckpt_path):
            print(f"Loading pretrained from {ckpt_path}")
            self.f_theta.load_state_dict(torch.load(ckpt_path))
            self.f_theta.train()
        else:
            print("Pretrained model not found, training ")
            assert self.method == "pretrain", \
                "Pretrain checkpoint missing; run with method=pretrain first"
            x, y, d, y_mask, d_mask, test_mask = (
                self.pretrain_dataset["x"], self.pretrain_dataset["y"],
                self.pretrain_dataset["d"], self.pretrain_dataset["y_mask"],
                self.pretrain_dataset["d_mask"], self.pretrain_dataset["test_mask"])
            x, y, d, y_mask, d_mask = get_train_data_from_batch(
                x, y, d, y_mask, d_mask, test_mask)
            for epoch in range(self.params["pretrain_epochs"]):
                print(f"Pretrain epoch {epoch}")
                self.update_batch(x, y, d, y_mask, d_mask, streaming=False)
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            torch.save(self.f_theta.state_dict(), ckpt_path)
            return

        if self.method in ["pretrain", "ce", "oracle"]:
            return

        x, y, d, y_mask, d_mask, test_mask = (
            self.pretrain_dataset["x"], self.pretrain_dataset["y"],
            self.pretrain_dataset["d"], self.pretrain_dataset["y_mask"],
            self.pretrain_dataset["d_mask"], self.pretrain_dataset["test_mask"])
        x, y, d, y_mask, d_mask = get_train_data_from_batch(
            x, y, d, y_mask, d_mask, test_mask)

        self.compute_horizon_weights(y, d)

        for epoch in range(self.params["pretrain_epochs"]):
            print(f"Pretrain trajectory estimator epoch {epoch}")
            self.update_batch(x, y, d, y_mask, d_mask, streaming=False)


    def update_batch(self, x, y, d, y_mask, d_mask, streaming):
        if streaming and self.method == "pretrain":
            return

        perm = np.random.permutation(x.shape[0])
        x, y, d = x[perm], y[perm], d[perm]
        y_mask, d_mask = y_mask[perm], d_mask[perm]

        bs = self.params["batch_size"]
        x_b, y_b, d_b = x.split(bs), y.split(bs), d.split(bs)
        ym_b, dm_b = y_mask.split(bs), d_mask.split(bs)

        update_steps = self.params["update_steps"] if streaming else 1

        if self.method == "trace":
            self.g_psi.eval() if streaming else self.g_psi.train()
        self.f_theta.train()

        for _ in range(update_steps):
            progress = ProgressInfo(
                total_step=len(x_b), prefix="update", log_steps=self.params["log_steps"])
            for bx, by, bd, bym, bdm in zip(x_b, y_b, d_b, ym_b, dm_b):
                progress.step()
                bx = bx.to(self.device)
                by = by.to(self.device)
                bd = bd.to(self.device)
                bym = bym.to(self.device)
                bdm = bdm.to(self.device)
                info = self.update_step(bx, by, bd, bym, bdm, streaming)


    def predict(self, x, d=None, d_mask=None):
        """
        Compute p(y|x, xi) for evaluation.

        """
        self.f_theta.eval()
        if self.method == "trace":
            self.g_psi.eval()

        probs = []
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        chunk_x = torch.split(x, self.params["test_batch_size"])
        chunk_d = torch.split(d, self.params["test_batch_size"]) if d is not None else None
        chunk_m = torch.split(d_mask, self.params["test_batch_size"]) if d_mask is not None else None

        for j, _x in enumerate(chunk_x):
            _x = _x.to(self.device)
            _d = chunk_d[j].to(self.device) if chunk_d is not None else None
            _m = chunk_m[j].to(self.device) if chunk_m is not None else None

            with torch.no_grad():
                if self.method != "trace":
                    prob = F.softmax(self.f_theta(_x), dim=1)
                else:
                    y_logits = self.f_theta(_x)
                    if _d is None or _m is None:
                        prob = F.softmax(y_logits, dim=1)
                    else:
                        B = _x.size(0)
                        sample_labels = torch.arange(
                            self.y_class_num, device=_x.device
                        ).unsqueeze(0).expand(B, -1)
                        sample_oh = F.one_hot(
                            sample_labels, self.y_class_num).float()
                        sample_x = _x.unsqueeze(1).expand(
                            -1, self.y_class_num, -1)

                        if self.K == 1:
                            dl = _d.squeeze(-1).long() if _d.dim() == 3 else _d.long()
                            sample_dl = dl.unsqueeze(1).expand(
                                -1, self.y_class_num, -1)              # (B, C, H)

                            pred_d = self.g_psi(
                                sample_x.reshape(B * self.y_class_num, -1),
                                sample_oh.reshape(B * self.y_class_num, -1),
                            ).view(B, self.y_class_num, self.d_nt, 2)

                            d_oh = F.one_hot(sample_dl, 2).float()
                            log_prob_d = (F.log_softmax(pred_d, dim=3) * d_oh
                                          ).sum(dim=3).float()         # (B, C, H)
                        else:
                            dl = _d.long()                              # (B, H, K)
                            sample_dl = dl.unsqueeze(1).expand(
                                -1, self.y_class_num, -1, -1)          # (B, C, H, K)

                            pred_d = self.g_psi(
                                sample_x.reshape(B * self.y_class_num, -1),
                                sample_oh.reshape(B * self.y_class_num, -1),
                            ).view(B, self.y_class_num, self.d_nt, self.K, 2)

                            d_oh = F.one_hot(sample_dl, 2).float()
                            log_prob_d = (
                                F.log_softmax(pred_d, dim=4) * d_oh
                            ).sum(dim=4).sum(dim=3).float()            # (B, C, H)

                        m = _m.float()
                        eta_h = self.eta.view(1, -1).float()
                        W_ih = m * eta_h
                        denom = W_ih.sum(dim=1, keepdim=True).clamp_min(1e-6)
                        alpha_ih = W_ih / denom

                        log_e_y = (alpha_ih.unsqueeze(1) * log_prob_d).sum(dim=2)
                        log_prior = F.log_softmax(y_logits, dim=1).float()
                        post_logits = log_prior + log_e_y
                        prob = torch.softmax(post_logits, dim=1)


                if not torch.isfinite(prob).all():
                    prob = torch.nan_to_num(prob, nan=0.0, posinf=1.0, neginf=0.0)
                    prob = prob / prob.sum(dim=1, keepdim=True).clamp_min(1e-12)

            probs.append(prob.detach().cpu())

        return {"prob": torch.cat(probs, dim=0)}


    def stream_train_and_predict(self):
        stream_metric = MetricAccumulator()
        stream_pred = []
        progress = ProgressInfo(
            total_step=len(self.stream_dataset),
            prefix="stream", log_steps=self.params["log_steps"])

        for batch in self.stream_dataset:
            train_batch, test_batch = batch
            progress.step()
            test_x, test_y, test_d, test_y_mask, test_d_mask = test_batch

            if test_x.shape[0] > 0:
                pred = self.predict(test_x, test_d, test_d_mask)
                pred["test_y"] = test_y
                stream_pred.append(pred)
                batch_metric = cal_metric(pred, test_y)
                stream_metric.add(batch_metric, test_x.shape[0])

            if progress.on_log_steps():
                print(stream_metric)

            train_x, train_y, train_d, train_y_mask, train_d_mask = train_batch
            if train_x.shape[0] > 0:
                self.update_batch(
                    train_x, train_y, train_d,
                    train_y_mask, train_d_mask, streaming=True)

        return {"metric": stream_metric, "pred": stream_pred}
