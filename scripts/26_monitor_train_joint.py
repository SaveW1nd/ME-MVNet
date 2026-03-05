"""Monitored launcher for joint training (script 13).

Features:
- Launch stage-2 training attempts sequentially.
- Poll metrics.jsonl in real-time.
- Stop an attempt early if quality is clearly below threshold.
- Retry with another seed and keep all logs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class AttemptResult:
    attempt: int
    exp_name: str
    seed: int
    status: str
    reason: str
    best_epoch: int
    best_a_total: float
    last_epoch: int
    last_a_total: float
    launcher_log: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--sep-ckpt", type=str, required=True)
    p.add_argument("--data-config", type=str, default="configs/data_composite.yaml")
    p.add_argument("--sep-config", type=str, default="configs/model_sep.yaml")
    p.add_argument("--pe-config", type=str, default="configs/model_pe.yaml")
    p.add_argument("--train-config", type=str, default="configs/train_joint.yaml")
    p.add_argument("--mode", type=str, choices=["smoke", "formal"], default="formal")
    p.add_argument("--base-exp-name", type=str, default="exp_joint_monitor")
    p.add_argument("--attempts", type=int, default=2)
    p.add_argument("--seed-list", type=str, default="20260304,20260314,20260324")
    p.add_argument("--poll-seconds", type=float, default=5.0)
    p.add_argument("--bad-check-epoch", type=int, default=20)
    p.add_argument("--min-best-a-total", type=float, default=0.25)
    p.add_argument("--plateau-patience", type=int, default=10)
    p.add_argument("--accept-a-total", type=float, default=0.45)
    p.add_argument("--max-runtime-minutes", type=float, default=240.0)
    return p.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _dump_yaml(obj: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=False)


def _read_metrics(metrics_path: Path) -> tuple[int, float, int, float]:
    if not metrics_path.exists():
        return 0, -1.0, 0, -1.0
    best_epoch = 0
    best_a = -1.0
    last_epoch = 0
    last_a = -1.0
    with metrics_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ep = int(rec.get("epoch", 0))
            val = rec.get("val", {})
            a = float(val.get("A_total", -1.0))
            last_epoch = ep
            last_a = a
            if a > best_a:
                best_a = a
                best_epoch = ep
    return best_epoch, best_a, last_epoch, last_a


def _terminate_process(proc: subprocess.Popen[Any]) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=15)
        return
    except subprocess.TimeoutExpired:
        pass
    proc.kill()
    proc.wait(timeout=10)


def _monitor_attempt(
    *,
    args: argparse.Namespace,
    attempt: int,
    seed: int,
    summary_md: Path,
) -> AttemptResult:
    exp_name = f"{args.base_exp_name}_a{attempt:02d}"
    run_dir = ROOT / "runs" / exp_name
    logs_dir = ROOT / "runs" / "experiment_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    launcher_log = logs_dir / f"{exp_name}.log"

    train_cfg_path = ROOT / args.train_config
    train_cfg = _load_yaml(train_cfg_path)
    train_cfg["train_joint"]["seed"] = int(seed)
    temp_cfg = ROOT / "runs" / "tmp_monitor" / f"{exp_name}_train_joint.yaml"
    _dump_yaml(train_cfg, temp_cfg)

    cmd = [
        sys.executable,
        "scripts/13_train_seppe_joint.py",
        "--sep-ckpt",
        args.sep_ckpt,
        "--data-config",
        args.data_config,
        "--sep-config",
        args.sep_config,
        "--pe-config",
        args.pe_config,
        "--train-config",
        str(temp_cfg),
        "--mode",
        args.mode,
        "--exp-name",
        exp_name,
    ]

    start_ts = time.time()
    with launcher_log.open("w", encoding="utf-8") as lf:
        lf.write(f"[{datetime.now().isoformat(timespec='seconds')}] launch: {' '.join(cmd)}\n")
        lf.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            stdout=lf,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
        )

    metrics_path = run_dir / "logs" / "metrics.jsonl"
    last_reported_epoch = -1
    reason = "completed"
    status = "completed"

    while proc.poll() is None:
        best_epoch, best_a, last_epoch, last_a = _read_metrics(metrics_path)

        if last_epoch > last_reported_epoch and last_epoch > 0 and last_a >= 0.0:
            line = (
                f"[attempt {attempt:02d}] epoch={last_epoch} val_A_total={last_a:.4f} "
                f"best={best_a:.4f}@{best_epoch}"
            )
            print(line, flush=True)
            with summary_md.open("a", encoding="utf-8") as f:
                f.write(f"- {line}\n")
            last_reported_epoch = last_epoch

        if (
            last_epoch >= int(args.bad_check_epoch)
            and best_a >= 0.0
            and best_a < float(args.min_best_a_total)
        ):
            reason = (
                f"underperform: epoch>={args.bad_check_epoch} and best_A_total<{args.min_best_a_total}"
            )
            status = "stopped"
            _terminate_process(proc)
            break

        if (
            best_epoch > 0
            and (last_epoch - best_epoch) >= int(args.plateau_patience)
            and best_a < float(args.accept_a_total)
            and last_epoch >= int(args.bad_check_epoch)
        ):
            reason = (
                f"plateau: no improvement for {args.plateau_patience} epochs and "
                f"best_A_total<{args.accept_a_total}"
            )
            status = "stopped"
            _terminate_process(proc)
            break

        elapsed_min = (time.time() - start_ts) / 60.0
        if elapsed_min >= float(args.max_runtime_minutes):
            reason = f"timeout: exceeded {args.max_runtime_minutes} minutes"
            status = "stopped"
            _terminate_process(proc)
            break

        time.sleep(float(args.poll_seconds))

    if proc.poll() is None:
        _terminate_process(proc)

    # Final read
    best_epoch, best_a, last_epoch, last_a = _read_metrics(metrics_path)
    if status == "completed" and proc.returncode not in (0, None):
        status = "failed"
        reason = f"trainer_exit_code={proc.returncode}"

    return AttemptResult(
        attempt=attempt,
        exp_name=exp_name,
        seed=int(seed),
        status=status,
        reason=reason,
        best_epoch=int(best_epoch),
        best_a_total=float(best_a),
        last_epoch=int(last_epoch),
        last_a_total=float(last_a),
        launcher_log=str(launcher_log.relative_to(ROOT)),
    )


def _parse_seed_list(seed_list: str, attempts: int) -> list[int]:
    seeds = []
    for token in seed_list.split(","):
        token = token.strip()
        if not token:
            continue
        seeds.append(int(token))
    if not seeds:
        seeds = [20260304]
    while len(seeds) < attempts:
        seeds.append(seeds[-1] + 11)
    return seeds[:attempts]


def main() -> None:
    args = parse_args()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_md = ROOT / "runs" / "experiment_logs" / f"{args.base_exp_name}_monitor_{ts}.md"
    summary_md.parent.mkdir(parents=True, exist_ok=True)

    with summary_md.open("w", encoding="utf-8") as f:
        f.write(f"# Joint Monitor Run ({ts})\n\n")
        f.write(f"- mode: `{args.mode}`\n")
        f.write(f"- attempts: `{args.attempts}`\n")
        f.write(f"- sep_ckpt: `{args.sep_ckpt}`\n")
        f.write(f"- bad_check_epoch: `{args.bad_check_epoch}`\n")
        f.write(f"- min_best_a_total: `{args.min_best_a_total}`\n")
        f.write(f"- plateau_patience: `{args.plateau_patience}`\n")
        f.write(f"- accept_a_total: `{args.accept_a_total}`\n\n")
        f.write("## Live\n")

    seeds = _parse_seed_list(args.seed_list, args.attempts)
    results: list[AttemptResult] = []
    success = False

    for i in range(1, args.attempts + 1):
        result = _monitor_attempt(args=args, attempt=i, seed=seeds[i - 1], summary_md=summary_md)
        results.append(result)

        with summary_md.open("a", encoding="utf-8") as f:
            f.write(
                "\n"
                f"- attempt={result.attempt:02d} seed={result.seed} status={result.status} "
                f"best_A_total={result.best_a_total:.4f}@{result.best_epoch} "
                f"last_A_total={result.last_a_total:.4f}@{result.last_epoch} "
                f"reason={result.reason} log={result.launcher_log}\n"
            )

        if result.status == "completed" and result.best_a_total >= float(args.accept_a_total):
            success = True
            break

    with summary_md.open("a", encoding="utf-8") as f:
        f.write("\n## Final Summary\n")
        f.write("| attempt | seed | status | best_A_total | best_epoch | last_A_total | last_epoch | reason |\n")
        f.write("|---|---:|---|---:|---:|---:|---:|---|\n")
        for r in results:
            f.write(
                f"| {r.attempt:02d} | {r.seed} | {r.status} | {r.best_a_total:.4f} | "
                f"{r.best_epoch} | {r.last_a_total:.4f} | {r.last_epoch} | {r.reason} |\n"
            )
        f.write(f"\n- success: `{success}`\n")

    print(f"Monitor summary: {summary_md.relative_to(ROOT)}")
    for r in results:
        print(
            f"attempt={r.attempt:02d} seed={r.seed} status={r.status} "
            f"best_A_total={r.best_a_total:.4f}@{r.best_epoch} reason={r.reason}"
        )


if __name__ == "__main__":
    main()
