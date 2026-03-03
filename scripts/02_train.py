"""Train ME-MVNet.

Usage:
    python scripts/02_train.py --mode smoke
    python scripts/02_train.py --mode formal --exp-name exp_formal
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from src.data.dataset_npz import ISRJDataset, NF_VALUES
from src.models.memvnet import MEMVNet
from src.train.trainer import fit
from src.utils.io import dump_yaml, ensure_dir, load_yaml, next_experiment_dir
from src.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-config", type=str, default="configs/data.yaml")
    p.add_argument("--model-config", type=str, default="configs/model.yaml")
    p.add_argument("--train-config", type=str, default="configs/train.yaml")
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
        run_dir = next_experiment_dir(root)
    for sub in ["checkpoints", "logs", "predictions", "figures"]:
        ensure_dir(run_dir / sub)
    return run_dir


def main() -> None:
    args = parse_args()
    data_cfg = load_yaml(args.data_config)
    model_cfg = load_yaml(args.model_config)
    train_cfg_all = load_yaml(args.train_config)
    train_cfg = train_cfg_all["train"]
    loss_cfg = train_cfg_all["loss"]

    set_global_seed(int(train_cfg["seed"]))
    device = _select_device(train_cfg)
    run_dir = _build_run_dir(args.exp_name)

    data_dir = Path(data_cfg["dataset"]["output_dir"])
    train_path = data_dir / "train.npz"
    val_path = data_dir / "val.npz"
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(
            f"Missing dataset split files in {data_dir}. Run scripts/01_generate_dataset.py first."
        )

    train_ds = ISRJDataset(train_path, normalize_iq=True)
    val_ds = ISRJDataset(val_path, normalize_iq=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(train_cfg["num_workers"]),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(train_cfg["num_workers"]),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model = MEMVNet(model_cfg).to(device)
    epochs = int(train_cfg["epochs_smoke"] if args.mode == "smoke" else train_cfg["epochs_formal"])

    cfg_dump = {
        "mode": args.mode,
        "data": data_cfg,
        "model": model_cfg,
        "train": train_cfg_all,
    }
    dump_yaml(cfg_dump, run_dir / "config_dump.yaml")

    summary = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        train_cfg=train_cfg,
        loss_cfg=loss_cfg,
        nf_values=NF_VALUES,
        run_dir=run_dir,
        epochs=epochs,
    )

    print(f"Run dir: {run_dir}")
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
