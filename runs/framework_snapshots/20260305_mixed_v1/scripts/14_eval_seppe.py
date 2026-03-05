"""Evaluate ME-MVSepPE model on composite dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset_npz_composite import CompositeISRJDataset
from src.eval.metrics_seppe import (
    compute_metrics_by_nf,
    compute_metrics_cond_nf_correct,
    compute_metrics_by_jnr,
    compute_metrics_by_kactive,
    compute_metrics_overall,
    compute_nf_confusion_4,
)
from src.models.builders import build_separator
from src.models.penet import MVSepPE, PENet
from src.models.pit_perm import align_true_by_perm, best_perm_from_pairwise, pairwise_sep_cost
from src.utils.io import ensure_dir, load_yaml, save_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--split", type=str, choices=["train", "val", "test"], default="test")
    p.add_argument("--data-config", type=str, default="configs/data_composite.yaml")
    p.add_argument("--sep-config", type=str, default="configs/model_sep.yaml")
    p.add_argument("--pe-config", type=str, default="configs/model_pe.yaml")
    p.add_argument("--eval-config", type=str, default="configs/eval_composite.yaml")
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


def _infer(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, np.ndarray]:
    model.eval()
    out: dict[str, list[np.ndarray]] = {
        "X": [],
        "j_hat": [],
        "j_true": [],
        "g_hat": [],
        "g_true": [],
        "Tl_pred_us": [],
        "Tl_true_us": [],
        "Ts_pred_us": [],
        "Ts_true_us": [],
        "NF_pred": [],
        "NF_true": [],
        "JNR_dB": [],
        "K_active": [],
    }

    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            pred = model(batch["X"])

            pair_cost = pairwise_sep_cost(pred["j_hat"], batch["J"], batch["NF"])
            perm, _ = best_perm_from_pairwise(pair_cost)

            j_true = align_true_by_perm(batch["J"], perm)
            g_true = align_true_by_perm(batch["G"], perm)
            tl_true = align_true_by_perm(batch["Tl_us"], perm)
            ts_true = align_true_by_perm(batch["Ts_us"], perm)
            nf_true = align_true_by_perm(batch["NF"], perm)
            jnr_true = align_true_by_perm(batch["JNR_dB"], perm)

            nf_pred = torch.argmax(pred["NF_logits"], dim=-1)

            out["X"].append(batch["X"].cpu().numpy())
            out["j_hat"].append(pred["j_hat"].cpu().numpy())
            out["j_true"].append(j_true.cpu().numpy())
            out["g_hat"].append(pred["g_hat"].cpu().numpy())
            out["g_true"].append(g_true.cpu().numpy())
            out["Tl_pred_us"].append(pred["Tl_hat_us"].cpu().numpy())
            out["Tl_true_us"].append(tl_true.cpu().numpy())
            out["Ts_pred_us"].append(pred["Ts_hat_us"].cpu().numpy())
            out["Ts_true_us"].append(ts_true.cpu().numpy())
            out["NF_pred"].append(nf_pred.cpu().numpy())
            out["NF_true"].append(nf_true.cpu().numpy())
            out["JNR_dB"].append(jnr_true.cpu().numpy())
            out["K_active"].append(batch["K_active"].cpu().numpy())

    return {k: np.concatenate(v, axis=0) for k, v in out.items()}


def main() -> None:
    args = parse_args()
    data_cfg = load_yaml(args.data_config)
    sep_cfg = load_yaml(args.sep_config)
    pe_cfg = load_yaml(args.pe_config)
    eval_cfg_all = load_yaml(args.eval_config)
    eval_cfg = eval_cfg_all["eval"]
    tol_cfg = eval_cfg_all["tolerance"]

    device = _select_device(eval_cfg)
    ckpt_path = Path(args.ckpt)
    run_dir = Path(args.run_dir) if args.run_dir else ckpt_path.parent.parent
    pred_dir = ensure_dir(run_dir / "predictions")
    table_dir = ensure_dir(run_dir / "tables")

    data_dir = Path(data_cfg["dataset"]["output_dir"])
    split_path = data_dir / f"{args.split}.npz"
    if not split_path.exists():
        raise FileNotFoundError(f"Missing split file: {split_path}")

    ds = CompositeISRJDataset(split_path, normalize_x=True)
    loader = DataLoader(
        ds,
        batch_size=int(eval_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(eval_cfg["num_workers"]),
        pin_memory=(device.type == "cuda"),
    )

    model = MVSepPE(build_separator(sep_cfg), PENet(pe_cfg)).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"], strict=True)

    pred = _infer(model=model, loader=loader, device=device)

    overall = compute_metrics_overall(
        tl_true_us=pred["Tl_true_us"],
        tl_pred_us=pred["Tl_pred_us"],
        ts_true_us=pred["Ts_true_us"],
        ts_pred_us=pred["Ts_pred_us"],
        nf_true=pred["NF_true"],
        nf_pred=pred["NF_pred"],
        tl_tol_us=float(tol_cfg["tl_us"]),
        ts_tol_us=float(tol_cfg["ts_us"]),
    )
    by_jnr = compute_metrics_by_jnr(
        tl_true_us=pred["Tl_true_us"],
        tl_pred_us=pred["Tl_pred_us"],
        ts_true_us=pred["Ts_true_us"],
        ts_pred_us=pred["Ts_pred_us"],
        nf_true=pred["NF_true"],
        nf_pred=pred["NF_pred"],
        jnr_db=pred["JNR_dB"],
        tl_tol_us=float(tol_cfg["tl_us"]),
        ts_tol_us=float(tol_cfg["ts_us"]),
    )
    by_kactive = compute_metrics_by_kactive(
        tl_true_us=pred["Tl_true_us"],
        tl_pred_us=pred["Tl_pred_us"],
        ts_true_us=pred["Ts_true_us"],
        ts_pred_us=pred["Ts_pred_us"],
        nf_true=pred["NF_true"],
        nf_pred=pred["NF_pred"],
        k_active=pred["K_active"],
        tl_tol_us=float(tol_cfg["tl_us"]),
        ts_tol_us=float(tol_cfg["ts_us"]),
    )
    by_nf = compute_metrics_by_nf(
        tl_true_us=pred["Tl_true_us"],
        tl_pred_us=pred["Tl_pred_us"],
        ts_true_us=pred["Ts_true_us"],
        ts_pred_us=pred["Ts_pred_us"],
        nf_true=pred["NF_true"],
        nf_pred=pred["NF_pred"],
        tl_tol_us=float(tol_cfg["tl_us"]),
        ts_tol_us=float(tol_cfg["ts_us"]),
    )
    cond_nf = compute_metrics_cond_nf_correct(
        tl_true_us=pred["Tl_true_us"],
        tl_pred_us=pred["Tl_pred_us"],
        ts_true_us=pred["Ts_true_us"],
        ts_pred_us=pred["Ts_pred_us"],
        nf_true=pred["NF_true"],
        nf_pred=pred["NF_pred"],
        tl_tol_us=float(tol_cfg["tl_us"]),
        ts_tol_us=float(tol_cfg["ts_us"]),
    )
    cm = compute_nf_confusion_4(pred["NF_true"], pred["NF_pred"])

    np.savez_compressed(pred_dir / f"{args.split}_pred.npz", **pred)
    save_json(
        {
            "split": args.split,
            "overall": overall,
            "nf_classes": [0, 1, 2, 3],
            "confusion_matrix": cm.tolist(),
        },
        table_dir / "test_metrics.json",
    )
    pd.DataFrame(by_jnr).to_csv(table_dir / "test_metrics_by_jnr.csv", index=False)
    pd.DataFrame(by_kactive).to_csv(table_dir / "test_metrics_by_kactive.csv", index=False)
    pd.DataFrame(by_nf).to_csv(table_dir / "metrics_by_nf.csv", index=False)
    pd.DataFrame(cond_nf).to_csv(table_dir / "metrics_cond_nf_correct.csv", index=False)

    print("Composite evaluation done.")
    print(json.dumps(overall, indent=2))
    print(f"Saved predictions: {pred_dir / f'{args.split}_pred.npz'}")
    print(f"Saved tables: {table_dir}")


if __name__ == "__main__":
    main()
