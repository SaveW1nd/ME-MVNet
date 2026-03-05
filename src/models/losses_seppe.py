"""Losses for ME-MVSepPE."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .pit_perm import align_true_by_perm, best_perm_from_pairwise, pairwise_sep_cost
from .sisdr import si_sdr


def compute_sep_loss(
    batch: dict[str, torch.Tensor],
    sep_out: dict[str, torch.Tensor],
    loss_cfg: dict | None = None,
) -> dict[str, torch.Tensor]:
    """Compute separation loss with PIT-jam + behavior-guided regularizers."""
    loss_cfg = loss_cfg or {}
    w_bg = float(loss_cfg.get("w_bg", 0.01))
    w_sil = float(loss_cfg.get("w_sil", 0.2))
    w_orth = float(loss_cfg.get("w_orth", 0.05))
    w_div = float(loss_cfg.get("w_div", 0.0))
    w_bgtrue = float(loss_cfg.get("w_bgtrue", 0.0))
    w_bgenv = float(loss_cfg.get("w_bgenv", 0.0))

    j_hat = sep_out["j_hat"]  # (B,3,2,N)
    b_hat = sep_out["b_hat"]  # (B,2,N)
    j_true = batch["J"]  # (B,3,2,N)
    nf_true = batch["NF"]  # (B,3)

    pair_cost = pairwise_sep_cost(j_hat=j_hat, j_true=j_true, nf_true=nf_true)
    perm, jam_cost = best_perm_from_pairwise(pair_cost)
    l_jam = jam_cost.mean()

    j_true_aligned = align_true_by_perm(j_true, perm)
    nf_true_aligned = align_true_by_perm(nf_true, perm)
    inactive_mask = nf_true_aligned == 0
    slot_energy = torch.mean(j_hat * j_hat, dim=(2, 3))  # (B,3)
    mix_energy = torch.mean(batch["X"] * batch["X"], dim=(1, 2)).unsqueeze(1)  # (B,1)
    slot_ratio = slot_energy / (mix_energy + 1e-8)
    if torch.any(inactive_mask):
        l_sil = slot_ratio[inactive_mask].mean()
    else:
        l_sil = torch.zeros((), device=j_hat.device, dtype=j_hat.dtype)

    b_true = batch["X"] - torch.sum(j_true, dim=1)
    l_bg_raw = F.mse_loss(b_hat, b_true)
    l_bg = w_bg * l_bg_raw

    b_norm = torch.sqrt(torch.sum(b_hat * b_hat, dim=(1, 2)) + 1e-8)  # (B,)
    orth_terms = []
    for k in range(3):
        jk = j_hat[:, k, :, :]
        dot_k = torch.sum(jk * b_hat, dim=(1, 2))
        jk_norm = torch.sqrt(torch.sum(jk * jk, dim=(1, 2)) + 1e-8)
        orth_terms.append(torch.abs(dot_k / (jk_norm * b_norm + 1e-8)))
    l_orth = torch.stack(orth_terms, dim=1).mean()

    bgtrue_terms = []
    for k in range(3):
        active_k = nf_true_aligned[:, k] > 0
        if not torch.any(active_k):
            continue
        jt = j_true_aligned[:, k, :, :]
        dot_t = torch.sum(jt * b_hat, dim=(1, 2))
        jt_norm = torch.sqrt(torch.sum(jt * jt, dim=(1, 2)) + 1e-8)
        corr_t = torch.abs(dot_t / (jt_norm * b_norm + 1e-8))
        bgtrue_terms.append(corr_t[active_k])
    if bgtrue_terms:
        l_bgtrue = torch.cat(bgtrue_terms, dim=0).mean()
    else:
        l_bgtrue = torch.zeros((), device=j_hat.device, dtype=j_hat.dtype)

    b_amp = torch.sqrt(torch.clamp(b_hat[:, 0] * b_hat[:, 0] + b_hat[:, 1] * b_hat[:, 1], min=1e-8))  # (B,N)
    j_sum_true = torch.sum(j_true_aligned, dim=1)  # (B,2,N)
    j_amp = torch.sqrt(torch.clamp(j_sum_true[:, 0] * j_sum_true[:, 0] + j_sum_true[:, 1] * j_sum_true[:, 1], min=1e-8))
    b0 = b_amp - b_amp.mean(dim=1, keepdim=True)
    j0 = j_amp - j_amp.mean(dim=1, keepdim=True)
    num = torch.sum(b0 * j0, dim=1)
    den = torch.sqrt(torch.sum(b0 * b0, dim=1) * torch.sum(j0 * j0, dim=1) + 1e-8)
    l_bgenv = torch.abs(num / (den + 1e-8)).mean()

    env = torch.sqrt(torch.clamp(j_hat[:, :, 0] * j_hat[:, :, 0] + j_hat[:, :, 1] * j_hat[:, :, 1], min=1e-8))
    env_norm = env / torch.sqrt(torch.sum(env * env, dim=2, keepdim=True) + 1e-8)
    c12 = torch.abs(torch.sum(env_norm[:, 0] * env_norm[:, 1], dim=1))
    c13 = torch.abs(torch.sum(env_norm[:, 0] * env_norm[:, 2], dim=1))
    c23 = torch.abs(torch.sum(env_norm[:, 1] * env_norm[:, 2], dim=1))
    l_div = torch.mean((c12 + c13 + c23) / 3.0)

    l_sep = l_jam + l_bg + w_sil * l_sil + w_orth * l_orth + w_div * l_div + w_bgtrue * l_bgtrue + w_bgenv * l_bgenv
    return {
        "L_sep": l_sep,
        "L_sep_jam": l_jam,
        "L_sep_bg": l_bg,
        "L_sep_bg_raw_mse": l_bg_raw,
        "L_sep_sil": l_sil,
        "L_sep_orth": l_orth,
        "L_sep_bgtrue": l_bgtrue,
        "L_sep_bgenv": l_bgenv,
        "L_sep_div": l_div,
        "w_bg": torch.tensor(w_bg, device=j_hat.device, dtype=j_hat.dtype),
        "w_sil": torch.tensor(w_sil, device=j_hat.device, dtype=j_hat.dtype),
        "w_orth": torch.tensor(w_orth, device=j_hat.device, dtype=j_hat.dtype),
        "w_bgtrue": torch.tensor(w_bgtrue, device=j_hat.device, dtype=j_hat.dtype),
        "w_bgenv": torch.tensor(w_bgenv, device=j_hat.device, dtype=j_hat.dtype),
        "w_div": torch.tensor(w_div, device=j_hat.device, dtype=j_hat.dtype),
        "perm": perm,
    }


def compute_joint_loss(
    batch: dict[str, torch.Tensor],
    out: dict[str, torch.Tensor],
    loss_cfg: dict,
) -> dict[str, torch.Tensor]:
    """Compute full ME-MVSepPE loss.

    Total:
        L = w_sep*L_sep + w_param*(L_Tl + L_NF + w_ts*L_Ts) + w_gate*L_gate
    """
    sep_loss = compute_sep_loss(batch=batch, sep_out=out)
    perm = sep_loss["perm"]

    g_true = align_true_by_perm(batch["G"], perm).float()  # (B,3,N)
    tl_true = align_true_by_perm(batch["Tl_us"], perm).float()  # (B,3)
    ts_true = align_true_by_perm(batch["Ts_us"], perm).float()  # (B,3)
    nf_true = align_true_by_perm(batch["NF"], perm).long()  # (B,3)

    g_logit = out["g_logit"]  # (B,3,N)
    tl_hat = out["Tl_hat_us"]  # (B,3)
    nf_logits = out["NF_logits"]  # (B,3,4)

    l_gate = F.binary_cross_entropy_with_logits(g_logit, g_true)

    active_mask = nf_true > 0
    if torch.any(active_mask):
        tl_err = F.smooth_l1_loss(tl_hat[active_mask], tl_true[active_mask], reduction="none")
        if bool(loss_cfg.get("tl_nf_weighted", False)):
            tl_w = nf_true[active_mask].to(dtype=tl_err.dtype) + 1.0
            l_tl = torch.mean(tl_w * tl_err)
        else:
            l_tl = torch.mean(tl_err)
    else:
        l_tl = torch.zeros((), device=tl_hat.device, dtype=tl_hat.dtype)

    nf_class_weights = loss_cfg.get("nf_class_weights", None)
    ce_weight = None
    if nf_class_weights is not None:
        ce_weight = torch.as_tensor(nf_class_weights, dtype=nf_logits.dtype, device=nf_logits.device)
        if ce_weight.numel() != nf_logits.shape[-1]:
            raise ValueError(
                f"nf_class_weights length {int(ce_weight.numel())} "
                f"!= num_classes {int(nf_logits.shape[-1])}"
            )
    l_nf = F.cross_entropy(
        nf_logits.reshape(-1, nf_logits.shape[-1]),
        nf_true.reshape(-1),
        weight=ce_weight,
    )

    if torch.any(active_mask):
        nf_probs = torch.softmax(nf_logits, dim=-1)
        nf_values = torch.arange(nf_logits.shape[-1], device=nf_logits.device, dtype=nf_logits.dtype)
        e_nf = torch.sum(nf_probs * nf_values.view(1, 1, -1), dim=-1)
        ts_hat = (e_nf + 1.0) * tl_hat
        l_ts = F.smooth_l1_loss(ts_hat[active_mask], ts_true[active_mask])
    else:
        l_ts = torch.zeros((), device=tl_hat.device, dtype=tl_hat.dtype)

    w_sep = float(loss_cfg["w_sep"])
    w_param = float(loss_cfg["w_param"])
    w_gate = float(loss_cfg["w_gate"])
    w_ts = float(loss_cfg.get("w_ts", 0.0))
    l_param = l_tl + l_nf + w_ts * l_ts
    total = w_sep * sep_loss["L_sep"] + w_param * l_param + w_gate * l_gate

    return {
        "L_total": total,
        "L_sep": sep_loss["L_sep"],
        "L_sep_jam": sep_loss["L_sep_jam"],
        "L_sep_bg": sep_loss["L_sep_bg"],
        "L_gate": l_gate,
        "L_param": l_param,
        "L_Tl": l_tl,
        "L_NF": l_nf,
        "L_Ts": l_ts,
        "perm": perm,
        "aligned_G": g_true,
        "aligned_Tl_us": tl_true,
        "aligned_Ts_us": ts_true,
        "aligned_NF": nf_true,
    }
