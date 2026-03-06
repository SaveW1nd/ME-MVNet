"""Train ME-MVSepPE jointly (stage 2).

Usage:
    python scripts/13_train_seppe_joint.py --sep-ckpt runs/exp_sep_formal/checkpoints/best.pt --mode smoke
    python scripts/13_train_seppe_joint.py --sep-ckpt runs/exp_sep_formal/checkpoints/best.pt --mode formal
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset_npz_composite import CompositeISRJDataset
from src.models.builders import build_separator
from src.models.penet import MVSepPE, build_pe
from src.train.trainer_seppe import fit_seppe_joint
from src.utils.io import dump_yaml, ensure_dir, load_yaml, next_experiment_dir
from src.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--sep-ckpt", type=str, required=True)
    p.add_argument("--init-joint-ckpt", type=str, default=None)
    p.add_argument("--data-config", type=str, default="configs/data_composite.yaml")
    p.add_argument("--sep-config", type=str, default="configs/model_sep.yaml")
    p.add_argument("--pe-config", type=str, default="configs/model_pe.yaml")
    p.add_argument("--train-config", type=str, default="configs/train_joint.yaml")
    p.add_argument("--mode", type=str, choices=["smoke", "formal"], default="formal")
    p.add_argument("--exp-name", type=str, default=None)
    return p.parse_args()


def _select_device(train_cfg: dict) -> torch.device:
    dev = str(train_cfg["device"]).lower()
    if dev == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(dev)


def _build_run_dir(exp_name: str | None) -> Path:
    root = ensure_dir("runs")
    if exp_name:
        run_dir = root / exp_name
        if run_dir.exists():
            raise FileExistsError(f"Run dir already exists: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=False)
    else:
        run_dir = next_experiment_dir(root, prefix="exp_joint_")
    for sub in ["checkpoints", "logs", "figures", "tables", "predictions"]:
        ensure_dir(run_dir / sub)
    return run_dir


def _compute_overlap_ratio(g: np.ndarray) -> np.ndarray:
    """Per-sample overlap ratio from gate tensor (B,3,N)."""
    active_sum = g.sum(axis=(1, 2)).astype(np.float32)
    union = (g.sum(axis=1) > 0).sum(axis=1).astype(np.float32)
    return np.where(active_sum > 0, 1.0 - union / np.maximum(active_sum, 1.0), 0.0).astype(np.float32)


def _compute_min_active_jnr(jnr_db: np.ndarray, nf: np.ndarray) -> np.ndarray:
    active = nf > 0
    masked = np.where(active, jnr_db, np.full_like(jnr_db, 1e9))
    return masked.min(axis=1).astype(np.float32)


def _build_train_sampler(train_ds: CompositeISRJDataset, train_cfg: dict) -> WeightedRandomSampler | None:
    samp_cfg = dict(train_cfg.get("sampling", {}))
    if not bool(samp_cfg.get("enabled", False)):
        return None

    weights = np.ones(len(train_ds), dtype=np.float32)
    k_active = train_ds.k_active.astype(np.int64)
    overlap_ratio = _compute_overlap_ratio(train_ds.g)
    min_active_jnr = _compute_min_active_jnr(train_ds.jnr_db, train_ds.nf)

    if float(samp_cfg.get("multi_weight", 1.0)) != 1.0:
        weights *= np.where(k_active == 3, float(samp_cfg.get("multi_weight", 1.0)), 1.0).astype(np.float32)

    overlap_thr = float(samp_cfg.get("overlap_threshold", 0.5))
    overlap_weight = float(samp_cfg.get("overlap_weight", 1.0))
    if overlap_weight != 1.0:
        weights *= np.where(overlap_ratio >= overlap_thr, overlap_weight, 1.0).astype(np.float32)

    low_jnr_thr = float(samp_cfg.get("low_jnr_threshold_db", -5.0))
    low_jnr_weight = float(samp_cfg.get("low_jnr_weight", 1.0))
    if low_jnr_weight != 1.0:
        weights *= np.where(min_active_jnr < low_jnr_thr, low_jnr_weight, 1.0).astype(np.float32)

    weights = np.clip(weights, a_min=1e-6, a_max=None)
    return WeightedRandomSampler(
        weights=torch.from_numpy(weights).double(),
        num_samples=len(weights),
        replacement=True,
    )


def main() -> None:
    args = parse_args()
    data_cfg = load_yaml(args.data_config)
    sep_cfg = load_yaml(args.sep_config)
    pe_cfg = load_yaml(args.pe_config)
    train_cfg_all = load_yaml(args.train_config)
    train_cfg = train_cfg_all["train_joint"]
    loss_cfg = train_cfg_all["loss"]

    set_global_seed(int(train_cfg["seed"]))
    device = _select_device(train_cfg)
    run_dir = _build_run_dir(args.exp_name)

    data_dir = Path(data_cfg["dataset"]["output_dir"])
    train_npz = data_dir / "train.npz"
    val_npz = data_dir / "val.npz"
    if not train_npz.exists() or not val_npz.exists():
        raise FileNotFoundError(f"Missing composite dataset in {data_dir}. Run script 10 first.")

    train_ds = CompositeISRJDataset(train_npz, normalize_x=True)
    val_ds = CompositeISRJDataset(val_npz, normalize_x=True)
    train_sampler = _build_train_sampler(train_ds, train_cfg)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=int(train_cfg["num_workers"]),
        pin_memory=(device.type == "cuda"),
    )
    train_eval_num_samples = int(train_cfg.get("train_eval_num_samples", 0))
    if train_eval_num_samples > 0:
        g = torch.Generator().manual_seed(int(train_cfg["seed"]))
        perm = torch.randperm(len(train_ds), generator=g).tolist()
        keep = perm[: min(train_eval_num_samples, len(train_ds))]
        train_eval_ds = Subset(train_ds, keep)
        train_eval_loader = DataLoader(
            train_eval_ds,
            batch_size=int(train_cfg["batch_size"]),
            shuffle=False,
            num_workers=int(train_cfg["num_workers"]),
            pin_memory=(device.type == "cuda"),
        )
    else:
        train_eval_loader = None
    val_loader = DataLoader(
        val_ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(train_cfg["num_workers"]),
        pin_memory=(device.type == "cuda"),
    )

    sepnet = build_separator(sep_cfg)
    ckpt = torch.load(args.sep_ckpt, map_location="cpu", weights_only=False)
    sepnet.load_state_dict(ckpt["model_state"], strict=True)
    penet = build_pe(pe_cfg)
    model = MVSepPE(sepnet=sepnet, penet=penet).to(device)
    if args.init_joint_ckpt:
        joint_ckpt = torch.load(args.init_joint_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(joint_ckpt["model_state"], strict=True)

    epochs = int(train_cfg["epochs_smoke"] if args.mode == "smoke" else train_cfg["epochs_formal"])

    dump_yaml(
        {
            "mode": args.mode,
            "data": data_cfg,
            "model_sep": sep_cfg,
            "model_pe": pe_cfg,
            "train_joint": train_cfg_all,
            "init_sep_ckpt": args.sep_ckpt,
            "init_joint_ckpt": args.init_joint_ckpt,
        },
        run_dir / "config_dump.yaml",
    )

    summary = fit_seppe_joint(
        model=model,
        train_loader=train_loader,
        train_eval_loader=train_eval_loader,
        val_loader=val_loader,
        device=device,
        train_cfg=train_cfg,
        loss_cfg=loss_cfg,
        run_dir=run_dir,
        epochs=epochs,
    )

    print(f"Run dir: {run_dir}")
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
