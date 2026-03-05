"""Run scenario-split evaluation (dual/multi) for a joint SepPE checkpoint.

Usage:
    python scripts/23_eval_seppe_by_scenario.py \
      --ckpt runs/exp_joint_formal_need2_e2_v1/checkpoints/best.pt \
      --split test \
      --run-dir runs/exp_joint_formal_need2_e2_v1
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io import ensure_dir, save_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--split", type=str, choices=["train", "val", "test"], default="test")
    p.add_argument("--scenarios", type=str, default="dual,multi")
    p.add_argument("--data-config", type=str, default="configs/data_composite.yaml")
    p.add_argument("--sep-config", type=str, default="configs/model_sep.yaml")
    p.add_argument("--pe-config", type=str, default="configs/model_pe.yaml")
    p.add_argument("--eval-config", type=str, default="configs/eval_composite.yaml")
    p.add_argument("--run-dir", type=str, default=None)
    return p.parse_args()


def _split_scenarios(s: str) -> list[str]:
    valid = {"dual", "multi", "all"}
    out = []
    for x in [v.strip().lower() for v in s.split(",") if v.strip()]:
        if x not in valid:
            raise ValueError(f"Invalid scenario: {x}")
        out.append(x)
    if not out:
        raise ValueError("No scenarios specified")
    return out


def _run_eval_one(args: argparse.Namespace, scenario: str) -> None:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "14_eval_seppe.py"),
        "--ckpt",
        str(args.ckpt),
        "--split",
        str(args.split),
        "--scenario",
        scenario,
        "--data-config",
        str(args.data_config),
        "--sep-config",
        str(args.sep_config),
        "--pe-config",
        str(args.pe_config),
        "--eval-config",
        str(args.eval_config),
    ]
    if args.run_dir:
        cmd.extend(["--run-dir", str(args.run_dir)])
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    scenarios = _split_scenarios(args.scenarios)
    ckpt_path = Path(args.ckpt)
    run_dir = Path(args.run_dir) if args.run_dir else ckpt_path.parent.parent

    rows: list[dict] = []
    for scenario in scenarios:
        _run_eval_one(args=args, scenario=scenario)

        if scenario == "all":
            metric_path = run_dir / "tables" / "test_metrics.json"
        else:
            metric_path = run_dir / "tables" / f"scenario_{scenario}" / "test_metrics.json"
        if not metric_path.exists():
            raise FileNotFoundError(f"Missing metrics file: {metric_path}")
        rec = json.loads(metric_path.read_text(encoding="utf-8"))
        overall = rec["overall"]
        rows.append(
            {
                "Scenario": scenario,
                "Split": rec.get("split", args.split),
                "A_total": float(overall["A_total"]),
                "A_Tl": float(overall["A_Tl"]),
                "A_Ts": float(overall["A_Ts"]),
                "A_NF": float(overall["A_NF"]),
                "NF_Acc": float(overall["NF_Acc"]),
                "NF_macroF1": float(overall["NF_macroF1"]),
                "Tl_MAE_us": float(overall["Tl_MAE_us"]),
                "Ts_MAE_us": float(overall["Ts_MAE_us"]),
                "MetricsPath": str(metric_path).replace("\\", "/"),
            }
        )

    out_csv = run_dir / "tables" / "test_metrics_by_scenario.csv"
    ensure_dir(out_csv.parent)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    save_json({"run_dir": str(run_dir), "rows": rows}, run_dir / "tables" / "test_metrics_by_scenario.json")

    print("Scenario-split evaluation done.")
    print(f"Run dir: {run_dir}")
    print(f"Saved summary CSV: {out_csv}")
    print(f"Saved summary JSON: {run_dir / 'tables' / 'test_metrics_by_scenario.json'}")


if __name__ == "__main__":
    main()
