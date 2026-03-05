"""Evaluate CISRJ separator on composite dataset (separation metrics only).

Outputs:
    - tables/cisrj_eval_summary.json
    - tables/cisrj_eval_by_nf.csv
    - tables/cisrj_eval_by_jnr.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset_npz_composite import CompositeISRJDataset
from src.models.builders import build_separator
from src.models.pit_perm import align_true_by_perm, best_perm_from_pairwise, pairwise_sep_cost
from src.models.sisdr import si_sdr
from src.utils.io import ensure_dir, load_yaml, save_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--split", type=str, choices=["train", "val", "test"], default="test")
    p.add_argument("--scenario", type=str, choices=["dual", "multi", "all"], default="all")
    p.add_argument("--data-config", type=str, default="configs/data_composite.yaml")
    p.add_argument("--model-config", type=str, default="configs/model_sep_cisrj_repro.yaml")
    p.add_argument("--run-dir", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


def _select_device(device_arg: str) -> torch.device:
    dev = str(device_arg).lower()
    if dev == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(dev)


def _subset_by_scenario(ds: CompositeISRJDataset, scenario: str) -> Subset | CompositeISRJDataset:
    if scenario == "all":
        return ds
    target = 2 if scenario == "dual" else 3
    idx = np.where(ds.k_active == target)[0]
    if idx.size == 0:
        raise RuntimeError(f"No samples for scenario={scenario}")
    return Subset(ds, idx.tolist())


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
    return out


def _bin_by_jnr(jnr: np.ndarray, step: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    jmin = float(np.floor(np.min(jnr)))
    jmax = float(np.ceil(np.max(jnr)))
    edges = np.arange(jmin, jmax + step, step, dtype=np.float32)
    if edges.size < 2:
        edges = np.array([jmin, jmin + step], dtype=np.float32)
    mids = 0.5 * (edges[:-1] + edges[1:])
    return edges, mids


def main() -> None:
    args = parse_args()
    data_cfg = load_yaml(args.data_config)
    model_cfg = load_yaml(args.model_config)
    device = _select_device(args.device)

    ckpt_path = Path(args.ckpt)
    run_dir = Path(args.run_dir) if args.run_dir else ckpt_path.parent.parent
    table_dir = ensure_dir(run_dir / "tables")

    data_dir = Path(data_cfg["dataset"]["output_dir"])
    split_path = data_dir / f"{args.split}.npz"
    if not split_path.exists():
        raise FileNotFoundError(f"Missing split file: {split_path}")

    ds_full = CompositeISRJDataset(split_path, normalize_x=True, normalize_targets=True)
    ds = _subset_by_scenario(ds_full, args.scenario)
    loader = DataLoader(
        ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
    )

    model = build_separator(model_cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    sisdr_est_all: list[np.ndarray] = []
    sisdr_mix_all: list[np.ndarray] = []
    sisdri_all: list[np.ndarray] = []
    nf_all: list[np.ndarray] = []
    jnr_all: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            out = model(batch["X"])

            pair_cost = pairwise_sep_cost(out["j_hat"], batch["J"], batch["NF"])
            perm, _ = best_perm_from_pairwise(pair_cost)
            j_true = align_true_by_perm(batch["J"], perm)
            nf_true = align_true_by_perm(batch["NF"], perm)
            jnr_true = align_true_by_perm(batch["JNR_dB"], perm)

            for k in range(3):
                active = nf_true[:, k] > 0
                if not torch.any(active):
                    continue
                est = si_sdr(out["j_hat"][:, k], j_true[:, k])[active]
                mix = si_sdr(batch["X"], j_true[:, k])[active]
                imp = est - mix

                sisdr_est_all.append(est.detach().cpu().numpy())
                sisdr_mix_all.append(mix.detach().cpu().numpy())
                sisdri_all.append(imp.detach().cpu().numpy())
                nf_all.append(nf_true[:, k][active].detach().cpu().numpy())
                jnr_all.append(jnr_true[:, k][active].detach().cpu().numpy())

    if not sisdri_all:
        raise RuntimeError("No active jammer sources found for evaluation")

    sisdr_est = np.concatenate(sisdr_est_all, axis=0).astype(np.float64)
    sisdr_mix = np.concatenate(sisdr_mix_all, axis=0).astype(np.float64)
    sisdri = np.concatenate(sisdri_all, axis=0).astype(np.float64)
    nf = np.concatenate(nf_all, axis=0).astype(np.int64)
    jnr = np.concatenate(jnr_all, axis=0).astype(np.float64)

    summary = {
        "scenario": args.scenario,
        "split": args.split,
        "count_active_sources": int(sisdri.size),
        "SI_SDR_est_mean_dB": float(np.mean(sisdr_est)),
        "SI_SDR_mix_mean_dB": float(np.mean(sisdr_mix)),
        "SI_SDRi_mean_dB": float(np.mean(sisdri)),
        "SI_SDRi_std_dB": float(np.std(sisdri)),
    }

    rows_nf = []
    for nf_v in [1, 2, 3]:
        idx = np.where(nf == nf_v)[0]
        if idx.size == 0:
            rows_nf.append({"NF_true": nf_v, "Count": 0, "SI_SDR_mean_dB": 0.0, "SI_SDRi_mean_dB": 0.0})
            continue
        rows_nf.append(
            {
                "NF_true": nf_v,
                "Count": int(idx.size),
                "SI_SDR_mean_dB": float(np.mean(sisdr_est[idx])),
                "SI_SDRi_mean_dB": float(np.mean(sisdri[idx])),
            }
        )

    edges, _ = _bin_by_jnr(jnr, step=2.0)
    rows_jnr = []
    for i in range(edges.size - 1):
        lo = float(edges[i])
        hi = float(edges[i + 1])
        idx = np.where((jnr >= lo) & (jnr < hi))[0]
        if idx.size == 0:
            continue
        rows_jnr.append(
            {
                "JNR_lo": lo,
                "JNR_hi": hi,
                "Count": int(idx.size),
                "SI_SDR_mean_dB": float(np.mean(sisdr_est[idx])),
                "SI_SDRi_mean_dB": float(np.mean(sisdri[idx])),
            }
        )

    save_json(summary, table_dir / "cisrj_eval_summary.json")
    pd.DataFrame(rows_nf).to_csv(table_dir / "cisrj_eval_by_nf.csv", index=False)
    pd.DataFrame(rows_jnr).to_csv(table_dir / "cisrj_eval_by_jnr.csv", index=False)

    print("CISRJ separation evaluation done.")
    print(json.dumps(summary, indent=2))
    print(f"Saved tables: {table_dir}")


if __name__ == "__main__":
    main()
