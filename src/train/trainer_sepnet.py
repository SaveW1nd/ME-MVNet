"""Trainer for SepNet stage-1."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.eval.sep_worst_cases import export_worst_cases
from src.models.losses_seppe import compute_sep_loss
from src.models.pit_perm import align_true_by_perm
from src.models.sisdr import si_sdr
from src.utils.io import ensure_dir
from src.utils.logging import append_jsonl, build_logger
from src.utils.meters import AverageMeter


def _build_optimizer(
    model: torch.nn.Module,
    cfg: dict,
) -> tuple[torch.optim.Optimizer, str]:
    opt_cfg = dict(cfg.get("optimizer", {}))
    opt_type = str(opt_cfg.get("type", "adamw")).lower()
    lr = float(cfg["lr"])
    weight_decay = float(cfg["weight_decay"])

    if opt_type == "adam":
        return (
            torch.optim.Adam(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            ),
            opt_type,
        )
    if opt_type == "adamw":
        return (
            torch.optim.AdamW(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            ),
            opt_type,
        )
    raise ValueError(f"Unsupported optimizer type: {opt_type}")


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: dict,
    epochs: int,
) -> tuple[torch.optim.lr_scheduler._LRScheduler, str]:
    sched_cfg = dict(cfg.get("scheduler", {}))
    sched_type = str(sched_cfg.get("type", "cosine")).lower()
    if sched_type == "cosine":
        eta_min = float(sched_cfg.get("eta_min", 1e-6))
        return (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(int(epochs), 1),
                eta_min=eta_min,
            ),
            sched_type,
        )
    if sched_type == "plateau":
        factor = float(sched_cfg.get("factor", 0.5))
        patience = int(sched_cfg.get("patience", 3))
        threshold = float(sched_cfg.get("threshold", 1e-4))
        cooldown = int(sched_cfg.get("cooldown", 0))
        min_lr = float(sched_cfg.get("min_lr", 1e-6))
        return (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=factor,
                patience=patience,
                threshold=threshold,
                cooldown=cooldown,
                min_lr=min_lr,
            ),
            sched_type,
        )
    raise ValueError(f"Unsupported scheduler type: {sched_type}")


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
    sep_loss_cfg: dict | None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    seen = 0

    meters = {
        "L_sep": AverageMeter(),
        "L_sep_jam": AverageMeter(),
        "L_sep_jam_unweighted": AverageMeter(),
        "L_sep_bg": AverageMeter(),
        "L_sep_sil": AverageMeter(),
        "L_sep_orth": AverageMeter(),
        "L_sep_bgtrue": AverageMeter(),
        "L_sep_bgenv": AverageMeter(),
        "L_sep_div": AverageMeter(),
        "L_sep_div_active": AverageMeter(),
        "L_sep_occ": AverageMeter(),
        "L_sep_edge": AverageMeter(),
        "hard_ratio": AverageMeter(),
        "case_sisdri_db": AverageMeter(),
        "SI_SDR_jam": AverageMeter(),
        "SI_SDR_bg": AverageMeter(),
    }
    skipped = 0

    for batch in loader:
        batch = _to_device(batch, device)
        bsz = batch["X"].shape[0]
        seen += int(bsz)

        with torch.set_grad_enabled(is_train):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
                enabled=amp_enabled,
            ):
                sep_out = model(batch["X"])
                cur_loss_cfg = dict(sep_loss_cfg or {})
                if not is_train:
                    cur_loss_cfg["hard_mining_enable"] = False
                losses = compute_sep_loss(batch=batch, sep_out=sep_out, loss_cfg=cur_loss_cfg)
                loss = losses["L_sep"]

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
            jam_sisdr, bg_sisdr = _compute_sep_quality(batch=batch, sep_out=sep_out, perm=losses["perm"])
            jam_sisdr = torch.nan_to_num(jam_sisdr, nan=0.0, posinf=0.0, neginf=0.0)
            bg_sisdr = torch.nan_to_num(bg_sisdr, nan=0.0, posinf=0.0, neginf=0.0)
            meters["SI_SDR_jam"].update(float(jam_sisdr.item()), n=bsz)
            meters["SI_SDR_bg"].update(float(bg_sisdr.item()), n=bsz)
            meters["L_sep"].update(float(torch.nan_to_num(losses["L_sep"], nan=0.0, posinf=0.0, neginf=0.0).item()), n=bsz)
            meters["L_sep_jam"].update(
                float(torch.nan_to_num(losses["L_sep_jam"], nan=0.0, posinf=0.0, neginf=0.0).item()),
                n=bsz,
            )
            if "L_sep_jam_unweighted" in losses:
                meters["L_sep_jam_unweighted"].update(
                    float(torch.nan_to_num(losses["L_sep_jam_unweighted"], nan=0.0, posinf=0.0, neginf=0.0).item()),
                    n=bsz,
                )
            meters["L_sep_bg"].update(
                float(torch.nan_to_num(losses["L_sep_bg"], nan=0.0, posinf=0.0, neginf=0.0).item()),
                n=bsz,
            )
            if "L_sep_sil" in losses:
                meters["L_sep_sil"].update(
                    float(torch.nan_to_num(losses["L_sep_sil"], nan=0.0, posinf=0.0, neginf=0.0).item()),
                    n=bsz,
                )
            if "L_sep_orth" in losses:
                meters["L_sep_orth"].update(
                    float(torch.nan_to_num(losses["L_sep_orth"], nan=0.0, posinf=0.0, neginf=0.0).item()),
                    n=bsz,
                )
            if "L_sep_bgtrue" in losses:
                meters["L_sep_bgtrue"].update(
                    float(torch.nan_to_num(losses["L_sep_bgtrue"], nan=0.0, posinf=0.0, neginf=0.0).item()),
                    n=bsz,
                )
            if "L_sep_bgenv" in losses:
                meters["L_sep_bgenv"].update(
                    float(torch.nan_to_num(losses["L_sep_bgenv"], nan=0.0, posinf=0.0, neginf=0.0).item()),
                    n=bsz,
                )
            if "L_sep_div" in losses:
                meters["L_sep_div"].update(
                    float(torch.nan_to_num(losses["L_sep_div"], nan=0.0, posinf=0.0, neginf=0.0).item()),
                    n=bsz,
                )
            if "L_sep_div_active" in losses:
                meters["L_sep_div_active"].update(
                    float(torch.nan_to_num(losses["L_sep_div_active"], nan=0.0, posinf=0.0, neginf=0.0).item()),
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
            if "hard_ratio" in losses:
                meters["hard_ratio"].update(
                    float(torch.nan_to_num(losses["hard_ratio"], nan=0.0, posinf=0.0, neginf=0.0).item()),
                    n=bsz,
                )
            if "sep_case_sisdri_db_mean" in losses:
                meters["case_sisdri_db"].update(
                    float(torch.nan_to_num(losses["sep_case_sisdri_db_mean"], nan=0.0, posinf=0.0, neginf=0.0).item()),
                    n=bsz,
                )

    out = {k: float(v.avg) for k, v in meters.items()}
    out["N_skip"] = float(skipped)
    out["AllSkipped"] = float(seen > 0 and skipped >= seen)
    return out


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

    optimizer, optimizer_type = _build_optimizer(model=model, cfg=cfg)
    scheduler, scheduler_type = _build_scheduler(optimizer=optimizer, cfg=cfg, epochs=epochs)
    amp_enabled = bool(cfg["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device=device.type, enabled=amp_enabled)
    grad_clip = float(cfg["grad_clip"])
    patience = int(cfg["early_stop_patience"])
    sep_loss_cfg = dict(cfg.get("loss_sep", {}))

    best_val = float("inf")
    best_epoch = 0
    stale = 0
    best_ckpt_path = ckpt_dir / "best.pt"
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
            sep_loss_cfg=sep_loss_cfg,
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
                sep_loss_cfg=sep_loss_cfg,
            )
        if scheduler_type == "plateau":
            scheduler.step(val_m["L_sep"])
        else:
            scheduler.step()

        rec = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], "train": train_m, "val": val_m}
        append_jsonl(metrics_path, rec)
        logger.info(
            "Epoch %03d | train L_sep=%.5f SIjam=%.3f skip=%d | val L_sep=%.5f SIjam=%.3f skip=%d",
            epoch,
            train_m["L_sep"],
            train_m["SI_SDR_jam"],
            int(train_m["N_skip"]),
            val_m["L_sep"],
            val_m["SI_SDR_jam"],
            int(val_m["N_skip"]),
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

        val_metric = float("inf") if bool(val_m.get("AllSkipped", 0.0)) else float(val_m["L_sep"])
        if val_metric < best_val:
            best_val = val_metric
            best_epoch = epoch
            stale = 0
            _save_ckpt(
                best_ckpt_path,
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

    worst_cases_to_export = int(cfg.get("worst_cases_to_export", 0))
    worst_cases_split = str(cfg.get("worst_cases_split", "val")).lower()
    worst_summary: dict[str, Any] | None = None
    if worst_cases_to_export > 0:
        if best_ckpt_path.exists():
            ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state"], strict=True)
        target_loader = val_loader if worst_cases_split == "val" else train_loader
        try:
            worst_summary = export_worst_cases(
                model=model,
                loader=target_loader,
                device=device,
                run_dir=run_dir,
                num_cases=worst_cases_to_export,
                split_name=worst_cases_split,
            )
            logger.info(
                "Exported worst cases: split=%s num=%d mean_case_sisdri=%.3f",
                worst_cases_split,
                int(worst_summary.get("num_exported", 0)),
                float(worst_summary.get("mean_case_sisdri_db", 0.0)),
            )
        except Exception as exc:  # pragma: no cover - best-effort diagnostics
            logger.exception("Worst-case export failed: %s", exc)

    logger.info("Stage1 done. best_epoch=%d best_val=%.6f", best_epoch, best_val)
    ret = {
        "best_epoch": best_epoch,
        "best_val": best_val,
        "elapsed_seconds": elapsed,
        "optimizer_type": optimizer_type,
        "scheduler_type": scheduler_type,
        "best_checkpoint": str(ckpt_dir / "best.pt"),
        "last_checkpoint": str(ckpt_dir / "last.pt"),
    }
    if worst_summary is not None:
        ret["worst_case_summary"] = worst_summary
    return ret
