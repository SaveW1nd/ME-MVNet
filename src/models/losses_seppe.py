"""Losses for ME-MVSepPE."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .pit_perm import align_true_by_perm, best_perm_from_pairwise, pairwise_sep_cost
from .sisdr import si_sdr


def compute_sep_loss(
    batch: dict[str, torch.Tensor],
    sep_out: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Compute separation loss with PIT-jam + weighted background MSE."""
    j_hat = sep_out["j_hat"]  # (B,3,2,N)
    b_hat = sep_out["b_hat"]  # (B,2,N)
    j_true = batch["J"]  # (B,3,2,N)
    nf_true = batch["NF"]  # (B,3)

    pair_cost = pairwise_sep_cost(j_hat=j_hat, j_true=j_true, nf_true=nf_true)
    perm, jam_cost = best_perm_from_pairwise(pair_cost)
    l_jam = jam_cost.mean()

    b_true = batch["X"] - torch.sum(j_true, dim=1)
    l_bg_raw = F.mse_loss(b_hat, b_true)
    l_bg = 0.1 * l_bg_raw

    l_sep = l_jam + l_bg
    return {
        "L_sep": l_sep,
        "L_sep_jam": l_jam,
        "L_sep_bg": l_bg,
        "L_sep_bg_raw_mse": l_bg_raw,
        "perm": perm,
    }


def compute_joint_loss(
    batch: dict[str, torch.Tensor],
    out: dict[str, torch.Tensor],
    loss_cfg: dict,
) -> dict[str, torch.Tensor]:
    """Compute full ME-MVSepPE loss.

    Total:
        L = L_sep + 0.5*(L_Tl + L_NF) + 0.2*L_gate
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
        l_tl = F.smooth_l1_loss(tl_hat[active_mask], tl_true[active_mask])
    else:
        l_tl = torch.zeros((), device=tl_hat.device, dtype=tl_hat.dtype)

    l_nf = F.cross_entropy(nf_logits.reshape(-1, nf_logits.shape[-1]), nf_true.reshape(-1))

    w_sep = float(loss_cfg["w_sep"])
    w_param = float(loss_cfg["w_param"])
    w_gate = float(loss_cfg["w_gate"])
    total = w_sep * sep_loss["L_sep"] + w_param * (l_tl + l_nf) + w_gate * l_gate

    return {
        "L_total": total,
        "L_sep": sep_loss["L_sep"],
        "L_sep_jam": sep_loss["L_sep_jam"],
        "L_sep_bg": sep_loss["L_sep_bg"],
        "L_gate": l_gate,
        "L_Tl": l_tl,
        "L_NF": l_nf,
        "perm": perm,
        "aligned_G": g_true,
        "aligned_Tl_us": tl_true,
        "aligned_Ts_us": ts_true,
        "aligned_NF": nf_true,
    }
