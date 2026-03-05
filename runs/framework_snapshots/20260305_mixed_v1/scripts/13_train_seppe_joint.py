"""Train ME-MVSepPE jointly (stage 2).

Usage:
    python scripts/13_train_seppe_joint.py --sep-ckpt runs/exp_sep_formal/checkpoints/best.pt --mode smoke
    python scripts/13_train_seppe_joint.py --sep-ckpt runs/exp_sep_formal/checkpoints/best.pt --mode formal
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset_npz_composite import CompositeISRJDataset
from src.models.builders import build_separator
from src.models.penet import MVSepPE, PENet
from src.train.trainer_seppe import fit_seppe_joint
from src.utils.io import dump_yaml, ensure_dir, load_yaml, next_experiment_dir
from src.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--sep-ckpt", type=str, required=True)
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

    train_loader = DataLoader(
        train_ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(train_cfg["num_workers"]),
        pin_memory=(device.type == "cuda"),
    )
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
    penet = PENet(pe_cfg)
    model = MVSepPE(sepnet=sepnet, penet=penet).to(device)

    epochs = int(train_cfg["epochs_smoke"] if args.mode == "smoke" else train_cfg["epochs_formal"])

    dump_yaml(
        {
            "mode": args.mode,
            "data": data_cfg,
            "model_sep": sep_cfg,
            "model_pe": pe_cfg,
            "train_joint": train_cfg_all,
            "init_sep_ckpt": args.sep_ckpt,
        },
        run_dir / "config_dump.yaml",
    )

    summary = fit_seppe_joint(
        model=model,
        train_loader=train_loader,
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
