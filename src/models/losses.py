"""Loss functions for ME-MVNet."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_total_loss(
    pred: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    loss_cfg: dict,
    nf_values: list[int],
    mask_weight_scale: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Compute total loss and components.

    Args:
        pred: model outputs.
        batch: batch dictionary from dataset.
        loss_cfg: loss config section.
        nf_values: class values in logits order, fixed [1,2,4].
        mask_weight_scale: warmup scaling factor for mask loss.

    Returns:
        Dictionary with component losses and total loss.
    """
    tl_hat = pred["Tl_hat"]
    tf_hat = pred["Tf_hat"]
    nf_logits = pred["NF_logits"]
    mask_hat = pred["mask_hat"]

    tl_gt = batch["Tl_s"]
    tf_gt = batch["Tf_s"]
    nf_gt_index = batch["NF_index"]
    mask_gt = batch["mask"]

    l_tl = F.smooth_l1_loss(tl_hat, tl_gt)
    l_tf = F.smooth_l1_loss(tf_hat, tf_gt)
    l_nf = F.cross_entropy(nf_logits, nf_gt_index)
    if "mask_logits" in pred:
        l_mask = F.binary_cross_entropy_with_logits(pred["mask_logits"], mask_gt)
    else:
        # Fallback for compatibility if only probabilities are available.
        l_mask = F.binary_cross_entropy(torch.clamp(mask_hat, 1e-6, 1.0 - 1e-6), mask_gt)

    nf_values_t = torch.tensor(nf_values, device=nf_logits.device, dtype=nf_logits.dtype)
    nf_prob = torch.softmax(nf_logits, dim=1)
    expected_nf = torch.sum(nf_prob * nf_values_t.unsqueeze(0), dim=1)
    l_phy = torch.mean(torch.abs(tf_hat - expected_nf * tl_hat))

    w_tl = float(loss_cfg["w_tl"])
    w_tf = float(loss_cfg["w_tf"])
    w_nf = float(loss_cfg["w_nf"])
    w_mask = float(loss_cfg["w_mask"]) * float(mask_weight_scale)
    w_phy = float(loss_cfg["w_phy"])

    total = w_tl * l_tl + w_tf * l_tf + w_nf * l_nf + w_mask * l_mask + w_phy * l_phy
    return {
        "L_Tl": l_tl,
        "L_Tf": l_tf,
        "L_NF": l_nf,
        "L_mask": l_mask,
        "L_phy": l_phy,
        "L_total": total,
    }
