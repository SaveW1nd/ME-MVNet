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
    w_pulse = float(loss_cfg.get("w_pulse", 0.0))
    w_div = float(loss_cfg.get("w_div", 0.0))
    w_div_active = float(loss_cfg.get("w_div_active", 0.0))
    w_bgtrue = float(loss_cfg.get("w_bgtrue", 0.0))
    w_bgenv = float(loss_cfg.get("w_bgenv", 0.0))
    w_sep_occ = float(loss_cfg.get("w_sep_occ", 0.0))
    w_sep_edge = float(loss_cfg.get("w_sep_edge", 0.0))
    sep_edge_pos_weight = float(loss_cfg.get("sep_edge_pos_weight", 4.0))
    hard_mining_enable = bool(loss_cfg.get("hard_mining_enable", False))
    hard_sisdri_thr_db = float(loss_cfg.get("hard_sisdri_thr_db", 0.0))
    hard_weight = float(loss_cfg.get("hard_weight", 1.0))

    j_hat = sep_out["j_hat"]  # (B,3,2,N)
    b_hat = sep_out["b_hat"]  # (B,2,N)
    j_true = batch["J"]  # (B,3,2,N)
    nf_true = batch["NF"]  # (B,3)

    pair_cost = pairwise_sep_cost(j_hat=j_hat, j_true=j_true, nf_true=nf_true)
    perm, jam_cost = best_perm_from_pairwise(pair_cost)
    j_true_aligned = align_true_by_perm(j_true, perm)
    nf_true_aligned = align_true_by_perm(nf_true, perm)
    g_true_aligned = align_true_by_perm(batch["G"], perm).float()
    active = nf_true_aligned > 0
    active_f = active.to(dtype=j_hat.dtype)
    active_cnt = active_f.sum(dim=1).clamp_min(1.0)

    edge_true = torch.zeros_like(g_true_aligned)
    edge_true[..., 1:] = torch.abs(g_true_aligned[..., 1:] - g_true_aligned[..., :-1])

    sisdri_parts = []
    for k in range(3):
        s_hat = si_sdr(j_hat[:, k], j_true_aligned[:, k]).to(dtype=j_hat.dtype)
        s_mix = si_sdr(batch["X"], j_true_aligned[:, k]).to(dtype=j_hat.dtype)
        sisdri_parts.append((s_hat - s_mix).unsqueeze(1))
    sisdri = torch.cat(sisdri_parts, dim=1)  # (B,3)
    case_sisdri = torch.sum(sisdri * active_f, dim=1) / active_cnt

    if hard_mining_enable and hard_weight > 1.0:
        hard_mask = case_sisdri < hard_sisdri_thr_db
        sample_w = 1.0 + (hard_weight - 1.0) * hard_mask.to(dtype=jam_cost.dtype)
        l_jam = torch.mean(jam_cost * sample_w)
        hard_ratio = hard_mask.to(dtype=j_hat.dtype).mean()
    else:
        l_jam = jam_cost.mean()
        hard_ratio = torch.zeros((), device=j_hat.device, dtype=j_hat.dtype)

    pulse_mask = g_true_aligned.unsqueeze(2)  # (B,3,1,N)
    pulse_num = torch.sum(((j_hat - j_true_aligned) ** 2) * pulse_mask)
    pulse_den = torch.sum(pulse_mask) * float(j_hat.shape[2])
    if float(pulse_den.item()) > 0.0:
        l_pulse = pulse_num / (pulse_den + 1e-8)
    else:
        l_pulse = torch.zeros((), device=j_hat.device, dtype=j_hat.dtype)

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
    div_terms = []
    a12 = active[:, 0] & active[:, 1]
    a13 = active[:, 0] & active[:, 2]
    a23 = active[:, 1] & active[:, 2]
    if torch.any(a12):
        div_terms.append(c12[a12])
    if torch.any(a13):
        div_terms.append(c13[a13])
    if torch.any(a23):
        div_terms.append(c23[a23])
    if div_terms:
        l_div_active = torch.cat(div_terms, dim=0).mean()
    else:
        l_div_active = torch.zeros((), device=j_hat.device, dtype=j_hat.dtype)

    if "sep_gate_logit" in sep_out and sep_out["sep_gate_logit"] is not None:
        l_sep_occ = F.binary_cross_entropy_with_logits(sep_out["sep_gate_logit"], g_true_aligned)
    else:
        l_sep_occ = torch.zeros((), device=j_hat.device, dtype=j_hat.dtype)

    if "sep_edge_logit" in sep_out and sep_out["sep_edge_logit"] is not None:
        edge_pos_w = torch.tensor(sep_edge_pos_weight, device=j_hat.device, dtype=j_hat.dtype)
        l_sep_edge = F.binary_cross_entropy_with_logits(
            sep_out["sep_edge_logit"],
            edge_true,
            pos_weight=edge_pos_w,
        )
    else:
        l_sep_edge = torch.zeros((), device=j_hat.device, dtype=j_hat.dtype)

    l_sep = (
        l_jam
        + l_bg
        + w_sil * l_sil
        + w_orth * l_orth
        + w_div * l_div
        + w_div_active * l_div_active
        + w_bgtrue * l_bgtrue
        + w_bgenv * l_bgenv
        + w_pulse * l_pulse
        + w_sep_occ * l_sep_occ
        + w_sep_edge * l_sep_edge
    )
    return {
        "L_sep": l_sep,
        "L_sep_jam": l_jam,
        "L_sep_bg": l_bg,
        "L_sep_pulse": l_pulse,
        "L_sep_bg_raw_mse": l_bg_raw,
        "L_sep_sil": l_sil,
        "L_sep_orth": l_orth,
        "L_sep_bgtrue": l_bgtrue,
        "L_sep_bgenv": l_bgenv,
        "L_sep_div": l_div,
        "L_sep_div_active": l_div_active,
        "L_sep_occ": l_sep_occ,
        "L_sep_edge": l_sep_edge,
        "L_sep_jam_unweighted": jam_cost.mean(),
        "sep_case_sisdri_db_mean": torch.mean(case_sisdri),
        "sep_case_sisdri_db_p10": torch.quantile(case_sisdri, q=0.10),
        "hard_ratio": hard_ratio,
        "w_bg": torch.tensor(w_bg, device=j_hat.device, dtype=j_hat.dtype),
        "w_sil": torch.tensor(w_sil, device=j_hat.device, dtype=j_hat.dtype),
        "w_orth": torch.tensor(w_orth, device=j_hat.device, dtype=j_hat.dtype),
        "w_div_active": torch.tensor(w_div_active, device=j_hat.device, dtype=j_hat.dtype),
        "w_bgtrue": torch.tensor(w_bgtrue, device=j_hat.device, dtype=j_hat.dtype),
        "w_bgenv": torch.tensor(w_bgenv, device=j_hat.device, dtype=j_hat.dtype),
        "w_pulse": torch.tensor(w_pulse, device=j_hat.device, dtype=j_hat.dtype),
        "w_div": torch.tensor(w_div, device=j_hat.device, dtype=j_hat.dtype),
        "w_sep_occ": torch.tensor(w_sep_occ, device=j_hat.device, dtype=j_hat.dtype),
        "w_sep_edge": torch.tensor(w_sep_edge, device=j_hat.device, dtype=j_hat.dtype),
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
    sep_loss = compute_sep_loss(batch=batch, sep_out=out, loss_cfg=loss_cfg)
    perm_sep = sep_loss["perm"]

    def pairwise_param_cost() -> torch.Tensor:
        g_true_raw = batch["G"].float()
        tl_true_raw = batch["Tl_us"].float()
        ts_true_raw = batch["Ts_us"].float()
        nf_true_raw = batch["NF"].long()

        g_logit_pred = out["g_logit"]
        tl_hat_pred = out["Tl_hat_us"]
        ts_hat_pred = out["Ts_hat_us"]
        nf_logits_pred = out["NF_logits"]

        w_gate_cost = float(loss_cfg.get("perm_gate_weight", 1.0))
        w_tl_cost = float(loss_cfg.get("perm_tl_weight", 1.0))
        w_ts_cost = float(loss_cfg.get("perm_ts_weight", 1.0))
        w_nf_cost = float(loss_cfg.get("perm_nf_weight", 1.0))

        cost = torch.zeros(
            (nf_logits_pred.shape[0], 3, 3),
            device=nf_logits_pred.device,
            dtype=nf_logits_pred.dtype,
        )
        for k in range(3):
            gk = g_logit_pred[:, k, :]
            tlk = tl_hat_pred[:, k]
            tsk = ts_hat_pred[:, k]
            nfk = nf_logits_pred[:, k, :]
            for j in range(3):
                nfj = nf_true_raw[:, j]
                active_j = (nfj > 0).to(dtype=nf_logits_pred.dtype)
                gate_cost = F.binary_cross_entropy_with_logits(
                    gk,
                    g_true_raw[:, j, :],
                    reduction="none",
                ).mean(dim=1)
                nf_cost = F.cross_entropy(nfk, nfj, reduction="none")
                tl_cost = F.smooth_l1_loss(tlk, tl_true_raw[:, j], reduction="none") * active_j
                ts_cost = F.smooth_l1_loss(tsk, ts_true_raw[:, j], reduction="none") * active_j
                cost[:, k, j] = (
                    w_gate_cost * gate_cost
                    + w_nf_cost * nf_cost
                    + w_tl_cost * tl_cost
                    + w_ts_cost * ts_cost
                )
        return cost

    param_perm_mode = str(loss_cfg.get("param_perm_mode", "sep")).lower()
    if param_perm_mode == "sep":
        perm_param = perm_sep
    else:
        param_pair_cost = pairwise_param_cost()
        if param_perm_mode == "param":
            perm_param, _ = best_perm_from_pairwise(param_pair_cost)
        elif param_perm_mode == "hybrid":
            sep_pair_cost = pairwise_sep_cost(j_hat=out["j_hat"], j_true=batch["J"], nf_true=batch["NF"])
            sep_scale = sep_pair_cost.detach().abs().mean(dim=(1, 2), keepdim=True).clamp_min(1e-6)
            param_scale = param_pair_cost.detach().abs().mean(dim=(1, 2), keepdim=True).clamp_min(1e-6)
            perm_sep_weight = float(loss_cfg.get("perm_sep_weight", 1.0))
            perm_param_weight = float(loss_cfg.get("perm_param_weight", 1.0))
            hybrid_cost = (
                perm_sep_weight * (sep_pair_cost / sep_scale)
                + perm_param_weight * (param_pair_cost / param_scale)
            )
            perm_param, _ = best_perm_from_pairwise(hybrid_cost)
        else:
            raise ValueError(f"Unsupported param_perm_mode: {param_perm_mode}")

    g_true = align_true_by_perm(batch["G"], perm_param).float()  # (B,3,N)
    tl_true = align_true_by_perm(batch["Tl_us"], perm_param).float()  # (B,3)
    ts_true = align_true_by_perm(batch["Ts_us"], perm_param).float()  # (B,3)
    nf_true = align_true_by_perm(batch["NF"], perm_param).long()  # (B,3)

    g_logit = out["g_logit"]  # (B,3,N)
    tl_hat = out["Tl_hat_us"]  # (B,3)
    ts_hat = out["Ts_hat_us"]  # (B,3)
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

    l_nf = F.cross_entropy(
        nf_logits.reshape(-1, nf_logits.shape[-1]),
        nf_true.reshape(-1),
    )
    if torch.any(active_mask):
        zero_logit = nf_logits[..., 0]
        nonzero_logit = torch.max(nf_logits[..., 1:], dim=-1).values
        margin = nonzero_logit - zero_logit
        active_zero_margin = float(loss_cfg.get("active_zero_margin", 0.5))
        l_nozero_margin = F.relu(active_zero_margin - margin[active_mask]).mean()
    else:
        l_nozero_margin = torch.zeros((), device=tl_hat.device, dtype=tl_hat.dtype)
    active_true = active_mask.to(dtype=nf_logits.dtype)
    if "NF_active_logit" in out:
        active_logit = out["NF_active_logit"]
    else:
        active_logit = torch.logsumexp(nf_logits[..., 1:], dim=-1) - nf_logits[..., 0]
    active_pos_weight = float(loss_cfg.get("active_pos_weight", 1.0))
    l_active_nf = F.binary_cross_entropy_with_logits(
        active_logit,
        active_true,
        pos_weight=torch.tensor(active_pos_weight, device=nf_logits.device, dtype=nf_logits.dtype),
    )

    if torch.any(active_mask):
        l_ts = F.smooth_l1_loss(ts_hat[active_mask], ts_true[active_mask])
    else:
        l_ts = torch.zeros((), device=tl_hat.device, dtype=tl_hat.dtype)

    if "Ts_hat_direct_us" in out and torch.any(active_mask):
        l_ts_direct = F.smooth_l1_loss(out["Ts_hat_direct_us"][active_mask], ts_true[active_mask])
    else:
        l_ts_direct = torch.zeros((), device=tl_hat.device, dtype=tl_hat.dtype)

    if "K_active_logit" in out and "K_active" in batch:
        k_true = batch["K_active"].to(device=tl_hat.device)
        multi_true = (k_true == 3).to(dtype=tl_hat.dtype)
        l_kactive = F.binary_cross_entropy_with_logits(out["K_active_logit"], multi_true)
    else:
        l_kactive = torch.zeros((), device=tl_hat.device, dtype=tl_hat.dtype)

    if "K_active" in batch:
        k_true = batch["K_active"].to(device=tl_hat.device)
        target_zero_count = (3 - k_true).to(dtype=tl_hat.dtype)  # dual:1, multi:0
        zero_prob = torch.softmax(nf_logits, dim=-1)[..., 0]
        l_zero_count = F.smooth_l1_loss(zero_prob.sum(dim=1), target_zero_count)
    else:
        l_zero_count = torch.zeros((), device=tl_hat.device, dtype=tl_hat.dtype)

    if "Tl_periodic_us" in out and torch.any(active_mask):
        l_tl_aux = F.smooth_l1_loss(out["Tl_periodic_us"][active_mask], tl_true[active_mask])
    else:
        l_tl_aux = torch.zeros((), device=tl_hat.device, dtype=tl_hat.dtype)

    if "Ts_hat_direct_us" in out and "Ts_struct_us" in out and torch.any(active_mask):
        l_phys = F.smooth_l1_loss(
            out["Ts_hat_direct_us"][active_mask],
            out["Ts_struct_us"][active_mask],
        )
    else:
        l_phys = torch.zeros((), device=tl_hat.device, dtype=tl_hat.dtype)

    if "Tl_anchor_us" in out and "Ts_anchor_us" in out and torch.any(active_mask):
        l_anchor_tl = F.smooth_l1_loss(out["Tl_anchor_us"][active_mask], tl_true[active_mask])
        l_anchor_ts = F.smooth_l1_loss(out["Ts_anchor_us"][active_mask], ts_true[active_mask])
        l_anchor = l_anchor_tl + l_anchor_ts
    else:
        l_anchor = torch.zeros((), device=tl_hat.device, dtype=tl_hat.dtype)

    w_sep = float(loss_cfg["w_sep"])
    w_param = float(loss_cfg["w_param"])
    w_gate = float(loss_cfg["w_gate"])
    w_ts = float(loss_cfg.get("w_ts", 0.0))
    w_ts_direct = float(loss_cfg.get("w_ts_direct", 0.0))
    w_active_nf = float(loss_cfg.get("w_active_nf", 0.0))
    w_nozero_margin = float(loss_cfg.get("w_nozero_margin", 0.0))
    w_kactive = float(loss_cfg.get("w_kactive", 0.0))
    w_zero_count = float(loss_cfg.get("w_zero_count", 0.0))
    w_tl_aux = float(loss_cfg.get("w_tl_aux", 0.0))
    w_phys = float(loss_cfg.get("w_phys", 0.0))
    w_anchor = float(loss_cfg.get("w_anchor", 0.0))
    l_param = (
        l_tl
        + l_nf
        + w_ts * l_ts
        + w_ts_direct * l_ts_direct
        + w_active_nf * l_active_nf
        + w_nozero_margin * l_nozero_margin
        + w_kactive * l_kactive
        + w_zero_count * l_zero_count
        + w_tl_aux * l_tl_aux
        + w_phys * l_phys
        + w_anchor * l_anchor
    )
    total = w_sep * sep_loss["L_sep"] + w_param * l_param + w_gate * l_gate

    return {
        "L_total": total,
        "L_sep": sep_loss["L_sep"],
        "L_sep_jam": sep_loss["L_sep_jam"],
        "L_sep_bg": sep_loss["L_sep_bg"],
        "L_sep_pulse": sep_loss["L_sep_pulse"],
        "L_sep_occ": sep_loss["L_sep_occ"],
        "L_sep_edge": sep_loss["L_sep_edge"],
        "L_gate": l_gate,
        "L_param": l_param,
        "L_Tl": l_tl,
        "L_NF": l_nf,
        "L_activeNF": l_active_nf,
        "L_NoZeroMargin": l_nozero_margin,
        "L_Ts": l_ts,
        "L_TsDirect": l_ts_direct,
        "L_KActive": l_kactive,
        "L_ZeroCount": l_zero_count,
        "L_TlAux": l_tl_aux,
        "L_phys": l_phys,
        "L_anchor": l_anchor,
        "perm": perm_param,
        "perm_sep": perm_sep,
        "perm_param": perm_param,
        "aligned_G": g_true,
        "aligned_Tl_us": tl_true,
        "aligned_Ts_us": ts_true,
        "aligned_NF": nf_true,
    }
