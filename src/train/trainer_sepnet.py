"""Trainer for SepNet stage-1."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.models.losses_seppe import compute_sep_loss
from src.models.pit_perm import align_true_by_perm
from src.models.sisdr import si_sdr
from src.utils.io import ensure_dir
from src.utils.logging import append_jsonl, build_logger
from src.utils.meters import AverageMeter


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
    return out


def _compute_sep_quality(
    batch: dict[str, torch.Tensor],
    sep_out: dict[str, torch.Tensor],
    perm: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    j_true_aligned = align_true_by_perm(batch["J"], perm)  # (B,3,2,N)
    nf_true_aligned = align_true_by_perm(batch["NF"], perm)  # (B,3)
    j_hat = sep_out["j_hat"]

    active = nf_true_aligned > 0
    sisdr_vals = []
    for k in range(3):
        s = si_sdr(j_hat[:, k], j_true_aligned[:, k])
        if torch.any(active[:, k]):
            sisdr_vals.append(s[active[:, k]])
    if sisdr_vals:
        jam_sisdr = torch.cat(sisdr_vals, dim=0).mean()
    else:
        jam_sisdr = torch.zeros((), device=j_hat.device, dtype=j_hat.dtype)

    b_true = batch["X"] - torch.sum(batch["J"], dim=1)
    bg_sisdr = si_sdr(sep_out["b_hat"], b_true).mean()
    return jam_sisdr, bg_sisdr


def _run_epoch(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    amp_enabled: bool,
    grad_clip: float,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    meters = {
        "L_sep": AverageMeter(),
        "L_sep_jam": AverageMeter(),
        "L_sep_bg": AverageMeter(),
        "SI_SDR_jam": AverageMeter(),
        "SI_SDR_bg": AverageMeter(),
    }

    for batch in loader:
        batch = _to_device(batch, device)
        bsz = batch["X"].shape[0]

        with torch.set_grad_enabled(is_train):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
                enabled=amp_enabled,
            ):
                sep_out = model(batch["X"])
                losses = compute_sep_loss(batch=batch, sep_out=sep_out)
                loss = losses["L_sep"]

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()

        with torch.no_grad():
            jam_sisdr, bg_sisdr = _compute_sep_quality(batch=batch, sep_out=sep_out, perm=losses["perm"])
            meters["SI_SDR_jam"].update(float(jam_sisdr.item()), n=bsz)
            meters["SI_SDR_bg"].update(float(bg_sisdr.item()), n=bsz)
            meters["L_sep"].update(float(losses["L_sep"].item()), n=bsz)
            meters["L_sep_jam"].update(float(losses["L_sep_jam"].item()), n=bsz)
            meters["L_sep_bg"].update(float(losses["L_sep_bg"].item()), n=bsz)

    return {k: float(v.avg) for k, v in meters.items()}


def _save_ckpt(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_val: float,
) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "epoch": int(epoch),
            "best_val": float(best_val),
        },
        path,
    )


def fit_sepnet(
    *,
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    cfg: dict,
    run_dir: Path,
    epochs: int,
) -> dict[str, Any]:
    """Train SepNet stage1."""
    run_dir = ensure_dir(run_dir)
    ckpt_dir = ensure_dir(run_dir / "checkpoints")
    log_dir = ensure_dir(run_dir / "logs")
    logger = build_logger(log_dir / "train.log")
    metrics_path = log_dir / "metrics.jsonl"

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=1e-6)
    amp_enabled = bool(cfg["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device=device.type, enabled=amp_enabled)
    grad_clip = float(cfg["grad_clip"])
    patience = int(cfg["early_stop_patience"])

    best_val = float("inf")
    best_epoch = 0
    stale = 0
    start = time.time()

    for epoch in range(1, epochs + 1):
        train_m = _run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            amp_enabled=amp_enabled,
            grad_clip=grad_clip,
        )
        with torch.no_grad():
            val_m = _run_epoch(
                model=model,
                loader=val_loader,
                device=device,
                optimizer=None,
                scaler=scaler,
                amp_enabled=amp_enabled,
                grad_clip=0.0,
            )
        scheduler.step()

        rec = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], "train": train_m, "val": val_m}
        append_jsonl(metrics_path, rec)
        logger.info(
            "Epoch %03d | train L_sep=%.5f SIjam=%.3f | val L_sep=%.5f SIjam=%.3f",
            epoch,
            train_m["L_sep"],
            train_m["SI_SDR_jam"],
            val_m["L_sep"],
            val_m["SI_SDR_jam"],
        )

        _save_ckpt(
            ckpt_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            best_val=best_val,
        )

        if val_m["L_sep"] < best_val:
            best_val = val_m["L_sep"]
            best_epoch = epoch
            stale = 0
            _save_ckpt(
                ckpt_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_val=best_val,
            )
        else:
            stale += 1
            if stale >= patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

    elapsed = time.time() - start
    logger.info("Stage1 done. best_epoch=%d best_val=%.6f", best_epoch, best_val)
    return {
        "best_epoch": best_epoch,
        "best_val": best_val,
        "elapsed_seconds": elapsed,
        "best_checkpoint": str(ckpt_dir / "best.pt"),
        "last_checkpoint": str(ckpt_dir / "last.pt"),
    }
