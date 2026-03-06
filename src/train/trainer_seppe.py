"""Joint trainer for ME-MVSepPE stage-2."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.models.losses_seppe import compute_joint_loss
from src.utils.io import ensure_dir
from src.utils.logging import append_jsonl, build_logger
from src.utils.meters import AverageMeter


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
    return out


def _batch_scores(
    out: dict[str, torch.Tensor],
    losses: dict[str, torch.Tensor],
    tl_tol_us: float = 0.15,
    ts_tol_us: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor]:
    nf_pred = torch.argmax(out["NF_logits"], dim=-1)  # (B,3)
    nf_true = losses["aligned_NF"]  # (B,3)
    nf_acc = (nf_pred == nf_true).float().mean()

    active = nf_true > 0
    if torch.any(active):
        ok_tl = torch.abs(out["Tl_hat_us"] - losses["aligned_Tl_us"]) <= tl_tol_us
        ok_ts = torch.abs(out["Ts_hat_us"] - losses["aligned_Ts_us"]) <= ts_tol_us
        ok_nf = nf_pred == nf_true
        ok_total = ok_tl & ok_ts & ok_nf & active
        a_total = ok_total.float().sum() / active.float().sum().clamp_min(1.0)
    else:
        a_total = torch.zeros((), device=nf_true.device, dtype=torch.float32)
    return nf_acc, a_total


def _run_epoch(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    amp_enabled: bool,
    grad_clip: float,
    loss_cfg: dict,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    meters = {
        "L_total": AverageMeter(),
        "L_sep": AverageMeter(),
        "L_sep_pulse": AverageMeter(),
        "L_sep_occ": AverageMeter(),
        "L_sep_edge": AverageMeter(),
        "L_gate": AverageMeter(),
        "L_param": AverageMeter(),
        "L_Tl": AverageMeter(),
        "L_NF": AverageMeter(),
        "L_activeNF": AverageMeter(),
        "L_NoZeroMargin": AverageMeter(),
        "L_Ts": AverageMeter(),
        "L_TsDirect": AverageMeter(),
        "L_KActive": AverageMeter(),
        "L_ZeroCount": AverageMeter(),
        "L_TlAux": AverageMeter(),
        "L_phys": AverageMeter(),
        "L_anchor": AverageMeter(),
        "NF_acc": AverageMeter(),
        "A_total": AverageMeter(),
    }
    skipped = 0

    for batch in loader:
        batch = _to_device(batch, device)
        bsz = batch["X"].shape[0]

        with torch.set_grad_enabled(is_train):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
                enabled=amp_enabled,
            ):
                out = model(batch["X"])
                losses = compute_joint_loss(batch=batch, out=out, loss_cfg=loss_cfg)
                loss = losses["L_total"]

            if not torch.isfinite(loss):
                skipped += int(bsz)
                if is_train:
                    optimizer.zero_grad(set_to_none=True)
                continue

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()

        with torch.no_grad():
            nf_acc, a_total = _batch_scores(out=out, losses=losses)
            nf_acc = torch.nan_to_num(nf_acc, nan=0.0, posinf=0.0, neginf=0.0)
            a_total = torch.nan_to_num(a_total, nan=0.0, posinf=0.0, neginf=0.0)
            meters["NF_acc"].update(float(nf_acc.item()), n=bsz)
            meters["A_total"].update(float(a_total.item()), n=bsz)
            meters["L_total"].update(
                float(torch.nan_to_num(losses["L_total"], nan=0.0, posinf=0.0, neginf=0.0).item()),
                n=bsz,
            )
            meters["L_sep"].update(float(torch.nan_to_num(losses["L_sep"], nan=0.0, posinf=0.0, neginf=0.0).item()), n=bsz)
            if "L_sep_pulse" in losses:
                meters["L_sep_pulse"].update(
                    float(torch.nan_to_num(losses["L_sep_pulse"], nan=0.0, posinf=0.0, neginf=0.0).item()),
                    n=bsz,
                )
            if "L_sep_occ" in losses:
                meters["L_sep_occ"].update(
                    float(torch.nan_to_num(losses["L_sep_occ"], nan=0.0, posinf=0.0, neginf=0.0).item()),
                    n=bsz,
                )
            if "L_sep_edge" in losses:
                meters["L_sep_edge"].update(
                    float(torch.nan_to_num(losses["L_sep_edge"], nan=0.0, posinf=0.0, neginf=0.0).item()),
                    n=bsz,
                )
            meters["L_gate"].update(
                float(torch.nan_to_num(losses["L_gate"], nan=0.0, posinf=0.0, neginf=0.0).item()),
                n=bsz,
            )
            meters["L_param"].update(
                float(torch.nan_to_num(losses["L_param"], nan=0.0, posinf=0.0, neginf=0.0).item()),
                n=bsz,
            )
            meters["L_Tl"].update(float(torch.nan_to_num(losses["L_Tl"], nan=0.0, posinf=0.0, neginf=0.0).item()), n=bsz)
            meters["L_NF"].update(float(torch.nan_to_num(losses["L_NF"], nan=0.0, posinf=0.0, neginf=0.0).item()), n=bsz)
            meters["L_activeNF"].update(
                float(torch.nan_to_num(losses["L_activeNF"], nan=0.0, posinf=0.0, neginf=0.0).item()),
                n=bsz,
            )
            if "L_NoZeroMargin" in losses:
                meters["L_NoZeroMargin"].update(
                    float(torch.nan_to_num(losses["L_NoZeroMargin"], nan=0.0, posinf=0.0, neginf=0.0).item()),
                    n=bsz,
                )
            meters["L_Ts"].update(float(torch.nan_to_num(losses["L_Ts"], nan=0.0, posinf=0.0, neginf=0.0).item()), n=bsz)
            if "L_TsDirect" in losses:
                meters["L_TsDirect"].update(
                    float(torch.nan_to_num(losses["L_TsDirect"], nan=0.0, posinf=0.0, neginf=0.0).item()),
                    n=bsz,
                )
            if "L_KActive" in losses:
                meters["L_KActive"].update(
                    float(torch.nan_to_num(losses["L_KActive"], nan=0.0, posinf=0.0, neginf=0.0).item()),
                    n=bsz,
                )
            if "L_ZeroCount" in losses:
                meters["L_ZeroCount"].update(
                    float(torch.nan_to_num(losses["L_ZeroCount"], nan=0.0, posinf=0.0, neginf=0.0).item()),
                    n=bsz,
                )
            if "L_TlAux" in losses:
                meters["L_TlAux"].update(
                    float(torch.nan_to_num(losses["L_TlAux"], nan=0.0, posinf=0.0, neginf=0.0).item()),
                    n=bsz,
                )
            if "L_phys" in losses:
                meters["L_phys"].update(
                    float(torch.nan_to_num(losses["L_phys"], nan=0.0, posinf=0.0, neginf=0.0).item()),
                    n=bsz,
                )
            if "L_anchor" in losses:
                meters["L_anchor"].update(
                    float(torch.nan_to_num(losses["L_anchor"], nan=0.0, posinf=0.0, neginf=0.0).item()),
                    n=bsz,
                )

    out = {k: float(v.avg) for k, v in meters.items()}
    out["N_skip"] = float(skipped)
    return out


def _resolve_epoch_loss_cfg(loss_cfg: dict, epoch: int) -> dict:
    """Resolve epoch-wise loss weights (optional ramp for nozero-margin)."""
    cfg = dict(loss_cfg)
    ramp_epochs = int(loss_cfg.get("w_nozero_ramp_epochs", 0))
    if ramp_epochs <= 0:
        return cfg

    base_w = float(loss_cfg.get("w_nozero_margin", 0.0))
    start_w = float(loss_cfg.get("w_nozero_start", 0.0))
    if ramp_epochs == 1:
        cfg["w_nozero_margin"] = base_w
        return cfg

    t = min(max((float(epoch) - 1.0) / float(ramp_epochs - 1), 0.0), 1.0)
    cfg["w_nozero_margin"] = start_w + (base_w - start_w) * t
    return cfg


def _save_ckpt(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_a_total: float,
    save_train_state: bool,
) -> None:
    payload = {
        "model_state": model.state_dict(),
        "epoch": int(epoch),
        "best_a_total": float(best_a_total),
    }
    if save_train_state:
        payload["optimizer_state"] = optimizer.state_dict()
        payload["scheduler_state"] = scheduler.state_dict()
        payload["scaler_state"] = scaler.state_dict()
    torch.save(payload, path)


def fit_seppe_joint(
    *,
    model: torch.nn.Module,
    train_loader: DataLoader,
    train_eval_loader: DataLoader | None,
    val_loader: DataLoader,
    device: torch.device,
    train_cfg: dict,
    loss_cfg: dict,
    run_dir: Path,
    epochs: int,
) -> dict[str, Any]:
    """Train joint SepNet + PENet stage-2."""
    run_dir = ensure_dir(run_dir)
    ckpt_dir = ensure_dir(run_dir / "checkpoints")
    log_dir = ensure_dir(run_dir / "logs")
    logger = build_logger(log_dir / "train.log")
    metrics_path = log_dir / "metrics.jsonl"

    lr_sep = float(train_cfg["lr_sep"])
    lr_pe = float(train_cfg["lr_pe"])
    wd = float(train_cfg["weight_decay"])
    optimizer = torch.optim.AdamW(
        [
            {"params": model.sepnet.parameters(), "lr": lr_sep},
            {"params": model.penet.parameters(), "lr": lr_pe},
        ],
        weight_decay=wd,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=1e-6)

    amp_enabled = bool(train_cfg["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device=device.type, enabled=amp_enabled)
    grad_clip = float(train_cfg["grad_clip"])
    patience = int(train_cfg["early_stop_patience"])
    save_train_state = bool(train_cfg.get("save_train_state", True))

    best_a_total = -1.0
    best_epoch = 0
    stale = 0
    start = time.time()

    for epoch in range(1, epochs + 1):
        epoch_loss_cfg = _resolve_epoch_loss_cfg(loss_cfg=loss_cfg, epoch=epoch)
        train_m = _run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            amp_enabled=amp_enabled,
            grad_clip=grad_clip,
            loss_cfg=epoch_loss_cfg,
        )
        with torch.no_grad():
            train_eval_m = None
            if train_eval_loader is not None:
                train_eval_m = _run_epoch(
                    model=model,
                    loader=train_eval_loader,
                    device=device,
                    optimizer=None,
                    scaler=scaler,
                    amp_enabled=amp_enabled,
                    grad_clip=0.0,
                    loss_cfg=epoch_loss_cfg,
                )
            val_m = _run_epoch(
                model=model,
                loader=val_loader,
                device=device,
                optimizer=None,
                scaler=scaler,
                amp_enabled=amp_enabled,
                grad_clip=0.0,
                loss_cfg=epoch_loss_cfg,
            )
        scheduler.step()

        rec = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "w_nozero_margin": float(epoch_loss_cfg.get("w_nozero_margin", 0.0)),
            "train": train_m,
            "train_eval": train_eval_m,
            "val": val_m,
        }
        append_jsonl(metrics_path, rec)
        if train_eval_m is not None:
            logger.info(
                "Epoch %03d | w_nozero=%.4f | train L=%.5f A_total=%.4f NF_acc=%.4f skip=%d | "
                "train_eval L=%.5f A_total=%.4f NF_acc=%.4f skip=%d | "
                "val L=%.5f A_total=%.4f NF_acc=%.4f skip=%d",
                epoch,
                float(epoch_loss_cfg.get("w_nozero_margin", 0.0)),
                train_m["L_total"],
                train_m["A_total"],
                train_m["NF_acc"],
                int(train_m["N_skip"]),
                train_eval_m["L_total"],
                train_eval_m["A_total"],
                train_eval_m["NF_acc"],
                int(train_eval_m["N_skip"]),
                val_m["L_total"],
                val_m["A_total"],
                val_m["NF_acc"],
                int(val_m["N_skip"]),
            )
        else:
            logger.info(
                "Epoch %03d | w_nozero=%.4f | train L=%.5f A_total=%.4f NF_acc=%.4f skip=%d | "
                "val L=%.5f A_total=%.4f NF_acc=%.4f skip=%d",
                epoch,
                float(epoch_loss_cfg.get("w_nozero_margin", 0.0)),
                train_m["L_total"],
                train_m["A_total"],
                train_m["NF_acc"],
                int(train_m["N_skip"]),
                val_m["L_total"],
                val_m["A_total"],
                val_m["NF_acc"],
                int(val_m["N_skip"]),
            )

        _save_ckpt(
            ckpt_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            best_a_total=best_a_total,
            save_train_state=save_train_state,
        )

        improved = val_m["A_total"] > best_a_total + 1e-8
        if improved:
            best_a_total = val_m["A_total"]
            best_epoch = epoch
            stale = 0
            _save_ckpt(
                ckpt_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_a_total=best_a_total,
                save_train_state=save_train_state,
            )
        else:
            stale += 1
            if stale >= patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

    elapsed = time.time() - start
    logger.info("Stage2 done. best_epoch=%d best_A_total=%.6f", best_epoch, best_a_total)
    return {
        "best_epoch": best_epoch,
        "best_A_total": best_a_total,
        "elapsed_seconds": elapsed,
        "best_checkpoint": str(ckpt_dir / "best.pt"),
        "last_checkpoint": str(ckpt_dir / "last.pt"),
    }
