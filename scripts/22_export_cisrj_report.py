"""Export CISRJ reproduction summary table from dual/multi runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io import ensure_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dual-run", type=str, required=True)
    p.add_argument("--multi-run", type=str, required=True)
    p.add_argument("--out-csv", type=str, default="paper/tables/cisrj_repro_summary.csv")
    return p.parse_args()


def _load_summary(run_dir: Path) -> dict:
    p = run_dir / "tables" / "cisrj_eval_summary.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing summary file: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    dual = _load_summary(Path(args.dual_run))
    multi = _load_summary(Path(args.multi_run))

    # Reference values from paper Table 4.
    paper_ref = {
        "dual": 20.9,
        "multi": 17.8,
    }

    rows = []
    for s in [dual, multi]:
        scenario = str(s["scenario"]).lower()
        ref = paper_ref.get(scenario, None)
        val = float(s["SI_SDRi_mean_dB"])
        rows.append(
            {
                "Scenario": scenario,
                "RunDir": str(Path(args.dual_run if scenario == "dual" else args.multi_run)).replace("\\", "/"),
                "SI_SDRi_mean_dB": val,
                "SI_SDR_mean_dB": float(s["SI_SDR_est_mean_dB"]),
                "Paper_Table4_SI_SDRi_dB": ref if ref is not None else "",
                "Gap_to_Paper_dB": (val - ref) if ref is not None else "",
                "Count_active_sources": int(s["count_active_sources"]),
            }
        )

    out_csv = Path(args.out_csv)
    ensure_dir(out_csv.parent)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"CISRJ report exported: {out_csv}")


if __name__ == "__main__":
    main()
