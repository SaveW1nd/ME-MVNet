"""Evaluate ME-MVNet checkpoint on a split.

Usage:
    python scripts/03_eval.py --ckpt runs/exp_001/checkpoints/best.pt --split test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.dataset_npz import ISRJDataset, NF_VALUES
from src.eval.metrics import (
    compute_jnr_bucket_metrics,
    compute_nf_confusion,
    compute_overall_metrics,
)
from src.models.memvnet import MEMVNet
from src.utils.io import ensure_dir, load_yaml, save_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--split", type=str, choices=["train", "val", "test"], default="test")
    p.add_argument("--data-config", type=str, default="configs/data.yaml")
    p.add_argument("--model-config", type=str, default="configs/model.yaml")
    p.add_argument("--eval-config", type=str, default="configs/eval.yaml")
    p.add_argument("--run-dir", type=str, default=None)
    return p.parse_args()


def _select_device(eval_cfg: dict) -> torch.device:
    dev = str(eval_cfg["device"]).lower()
    if dev == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(dev)


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
    return out


def _run_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, np.ndarray]:
    model.eval()
    out: dict[str, list[np.ndarray]] = {
        "Tl_true": [],
        "Tl_pred": [],
        "Tf_true": [],
        "Tf_pred": [],
        "NF_true": [],
        "NF_pred": [],
        "NF_probs": [],
        "JNR_dB": [],
        "mask_gt": [],
        "mask_hat": [],
    }

    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            pred = model(batch["X"])

            nf_prob = torch.softmax(pred["NF_logits"], dim=1)
            nf_idx = torch.argmax(nf_prob, dim=1)
            nf_pred_val = torch.tensor(NF_VALUES, device=device)[nf_idx]

            out["Tl_true"].append(batch["Tl_s"].cpu().numpy())
            out["Tl_pred"].append(pred["Tl_hat"].cpu().numpy())
            out["Tf_true"].append(batch["Tf_s"].cpu().numpy())
            out["Tf_pred"].append(pred["Tf_hat"].cpu().numpy())
            out["NF_true"].append(batch["NF_value"].cpu().numpy())
            out["NF_pred"].append(nf_pred_val.cpu().numpy())
            out["NF_probs"].append(nf_prob.cpu().numpy())
            out["JNR_dB"].append(batch["JNR_dB"].cpu().numpy())
            out["mask_gt"].append(batch["mask"].cpu().numpy())
            out["mask_hat"].append(pred["mask_hat"].cpu().numpy())

    return {k: np.concatenate(v, axis=0) for k, v in out.items()}


def main() -> None:
    args = parse_args()
    data_cfg = load_yaml(args.data_config)
    model_cfg = load_yaml(args.model_config)
    eval_cfg = load_yaml(args.eval_config)["eval"]
    device = _select_device(eval_cfg)

    ckpt_path = Path(args.ckpt)
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        run_dir = ckpt_path.parent.parent
    pred_dir = ensure_dir(run_dir / "predictions")
    table_dir = ensure_dir(run_dir / "tables")

    split_path = Path(data_cfg["dataset"]["output_dir"]) / f"{args.split}.npz"
    if not split_path.exists():
        raise FileNotFoundError(f"Missing split file: {split_path}")

    ds = ISRJDataset(split_path, normalize_iq=True)
    loader = DataLoader(
        ds,
        batch_size=int(eval_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(eval_cfg["num_workers"]),
        pin_memory=(device.type == "cuda"),
    )

    model = MEMVNet(model_cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    pred = _run_inference(model=model, loader=loader, device=device)

    overall = compute_overall_metrics(
        tl_true=pred["Tl_true"],
        tl_pred=pred["Tl_pred"],
        tf_true=pred["Tf_true"],
        tf_pred=pred["Tf_pred"],
        nf_true=pred["NF_true"],
        nf_pred=pred["NF_pred"],
    )
    per_jnr = compute_jnr_bucket_metrics(
        jnr_db=pred["JNR_dB"],
        tl_true=pred["Tl_true"],
        tl_pred=pred["Tl_pred"],
        tf_true=pred["Tf_true"],
        tf_pred=pred["Tf_pred"],
        nf_true=pred["NF_true"],
        nf_pred=pred["NF_pred"],
    )
    cm = compute_nf_confusion(pred["NF_true"], pred["NF_pred"])

    np.savez_compressed(pred_dir / f"{args.split}_pred.npz", **pred)
    save_json(
        {
            "split": args.split,
            "overall": overall,
            "confusion_matrix_labels": [1, 2, 4],
            "confusion_matrix": cm.tolist(),
        },
        table_dir / f"{args.split}_metrics.json",
    )
    pd.DataFrame(per_jnr).to_csv(table_dir / f"{args.split}_metrics_by_jnr.csv", index=False)
    pd.DataFrame([overall]).to_csv(table_dir / f"{args.split}_metrics_overall.csv", index=False)

    print("Evaluation done.")
    print(json.dumps(overall, indent=2))
    print(f"Predictions: {pred_dir / f'{args.split}_pred.npz'}")
    print(f"Tables: {table_dir}")


if __name__ == "__main__":
    main()
