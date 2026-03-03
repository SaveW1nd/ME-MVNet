"""Training loop implementation."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.models.losses import compute_total_loss
from src.utils.io import ensure_dir
from src.utils.logging import append_jsonl, build_logger
from src.utils.meters import AverageMeter
from .optim import build_optimizer
from .scheduler import build_scheduler


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def _save_checkpoint(
    ckpt_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    best_val: float,
    extra: dict[str, Any] | None = None,
) -> None:
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "epoch": int(epoch),
        "best_val": float(best_val),
        "extra": extra or {},
    }
    torch.save(payload, ckpt_path)


def _run_one_epoch(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.cuda.amp.GradScaler,
    loss_cfg: dict,
    nf_values: list[int],
    mask_weight_scale: float,
    amp_enabled: bool,
    grad_clip: float,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    meters = {
        "L_total": AverageMeter(),
        "L_Tl": AverageMeter(),
        "L_Tf": AverageMeter(),
        "L_NF": AverageMeter(),
        "L_mask": AverageMeter(),
        "L_phy": AverageMeter(),
        "NF_acc": AverageMeter(),
    }

    for batch in loader:
        batch = _to_device(batch, device)
        x = batch["X"]
        bsz = x.shape[0]

        with torch.set_grad_enabled(is_train):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
                enabled=amp_enabled,
            ):
                pred = model(x)
                losses = compute_total_loss(
                    pred=pred,
                    batch=batch,
                    loss_cfg=loss_cfg,
                    nf_values=nf_values,
                    mask_weight_scale=mask_weight_scale,
                )
                total = losses["L_total"]

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(total).backward()
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()

        nf_pred = torch.argmax(pred["NF_logits"], dim=1)
        nf_acc = (nf_pred == batch["NF_index"]).float().mean()

        for name, meter in meters.items():
            if name == "NF_acc":
                meter.update(float(nf_acc.item()), n=bsz)
            else:
                meter.update(float(losses[name].item()), n=bsz)

    return {k: float(v.avg) for k, v in meters.items()}


def fit(
    *,
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    train_cfg: dict,
    loss_cfg: dict,
    nf_values: list[int],
    run_dir: Path,
    epochs: int,
) -> dict[str, Any]:
    """Train and validate model, saving best/last checkpoints."""
    run_dir = ensure_dir(run_dir)
    checkpoints_dir = ensure_dir(run_dir / "checkpoints")
    logs_dir = ensure_dir(run_dir / "logs")
    logger = build_logger(logs_dir / "train.log")
    metrics_jsonl = logs_dir / "metrics.jsonl"

    optimizer = build_optimizer(model, train_cfg)
    scheduler = build_scheduler(optimizer, total_epochs=epochs)
    amp_enabled = bool(train_cfg["amp"]) and device.type == "cuda"
    grad_clip = float(train_cfg["grad_clip"])
    early_stop_patience = int(train_cfg["early_stop_patience"])
    warmup_epochs = max(int(loss_cfg.get("mask_warmup_epochs", 0)), 0)

    scaler = torch.amp.GradScaler(device=device.type, enabled=amp_enabled)
    best_val = float("inf")
    best_epoch = 0
    stale_epochs = 0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        if warmup_epochs > 0:
            mask_weight_scale = min(1.0, epoch / warmup_epochs)
        else:
            mask_weight_scale = 1.0

        train_metrics = _run_one_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            loss_cfg=loss_cfg,
            nf_values=nf_values,
            mask_weight_scale=mask_weight_scale,
            amp_enabled=amp_enabled,
            grad_clip=grad_clip,
        )
        with torch.no_grad():
            val_metrics = _run_one_epoch(
                model=model,
                loader=val_loader,
                device=device,
                optimizer=None,
                scaler=scaler,
                loss_cfg=loss_cfg,
                nf_values=nf_values,
                mask_weight_scale=1.0,
                amp_enabled=amp_enabled,
                grad_clip=0.0,
            )
        scheduler.step()

        record = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "val": val_metrics,
        }
        append_jsonl(metrics_jsonl, record)
        logger.info(
            "Epoch %03d | train L=%.6f acc=%.4f | val L=%.6f acc=%.4f",
            epoch,
            train_metrics["L_total"],
            train_metrics["NF_acc"],
            val_metrics["L_total"],
            val_metrics["NF_acc"],
        )

        _save_checkpoint(
            checkpoints_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            best_val=best_val,
            extra={"train_metrics": train_metrics, "val_metrics": val_metrics},
        )

        if val_metrics["L_total"] < best_val:
            best_val = val_metrics["L_total"]
            best_epoch = epoch
            stale_epochs = 0
            _save_checkpoint(
                checkpoints_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_val=best_val,
                extra={"train_metrics": train_metrics, "val_metrics": val_metrics},
            )
        else:
            stale_epochs += 1
            if stale_epochs >= early_stop_patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

    elapsed = time.time() - start_time
    logger.info("Training finished. Best epoch=%d best_val=%.6f", best_epoch, best_val)
    return {
        "best_epoch": best_epoch,
        "best_val": best_val,
        "elapsed_seconds": elapsed,
        "best_checkpoint": str(checkpoints_dir / "best.pt"),
        "last_checkpoint": str(checkpoints_dir / "last.pt"),
    }
