"""Composite ISRJ dataset generator for ME-MVSepPE.

Each split NPZ stores:
    X: float32 (num, 2, N)
    J: float32 (num, 3, 2, N)
    G: uint8   (num, 3, N)
    Tl_us: float32 (num, 3)
    Ts_us: float32 (num, 3)
    NF: int32 (num, 3), values in {0,1,2,3}
    JNR_dB: float32 (num, 3)
    K_active: int32 (num,), values in {2,3}
    meta_json: str
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.io import ensure_dir, save_json


def make_lfm_pri(fs: float, tp: float, pri: float, b: float) -> np.ndarray:
    """Build PRI-length baseband sequence with LFM in pulse window only."""
    n = int(round(fs * pri))
    npulse = int(round(fs * tp))
    kr = b / tp

    s_full = np.zeros((n,), dtype=np.complex64)
    t = (np.arange(npulse, dtype=np.float64) - npulse / 2.0) / fs
    s_full[:npulse] = np.exp(1j * np.pi * kr * t**2).astype(np.complex64)
    return s_full


def shift_right_zero_pad(x: np.ndarray, delay: int) -> np.ndarray:
    """Right shift with zero padding."""
    n = x.shape[0]
    if delay <= 0:
        return x.copy()
    if delay >= n:
        return np.zeros_like(x)
    out = np.zeros_like(x)
    out[delay:] = x[: n - delay]
    return out


def _sample_safe_right_delay(mask: np.ndarray, rng: np.random.Generator) -> int:
    """Sample right-shift delay while keeping at least one active sample in window.

    With zero-pad shifting, unconstrained delay can move the whole jammer outside
    the observation window, causing NF>0 labels with zero-energy jammer targets.
    """
    active_idx = np.flatnonzero(mask)
    if active_idx.size == 0:
        return 0
    n = int(mask.shape[0])
    max_delay = n - 1 - int(active_idx[-1])
    if max_delay <= 0:
        return 0
    return int(rng.integers(0, max_delay + 1))


def _sample_offset_with_cycle(
    rng: np.random.Generator,
    npulse: int,
    nl: int,
    nu: int,
) -> tuple[int, int]:
    # Keep the strict offset range [0, npulse-nl], but ensure at least one cycle.
    upper = npulse - nl
    if upper < 0:
        raise ValueError(f"Invalid nl={nl} for npulse={npulse}")

    for _ in range(64):
        offset = int(rng.integers(0, upper + 1))
        k_cycles = int((npulse - offset) // nu)
        if k_cycles > 0:
            return offset, k_cycles

    # Fallback (very unlikely): force offset to 0.
    offset = 0
    k_cycles = int((npulse - offset) // nu)
    if k_cycles <= 0:
        k_cycles = 1
    return offset, k_cycles


def generate_one_jammer(
    *,
    s_full: np.ndarray,
    fs: float,
    b: float,
    nf: int,
    tl_us: float,
    rng: np.random.Generator,
    enable_right_shift: bool = True,
    enable_freq_shift: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate one jammer component and forwarding gate mask.

    Returns:
        j: complex64 (N,)
        g: uint8 (N,)
    """
    n = s_full.shape[0]
    npulse = int(np.count_nonzero(np.abs(s_full) > 0))
    nl = int(round(tl_us * 1e-6 * fs))
    nl = max(1, nl)

    nu = (nf + 1) * nl
    offset, k_cycles = _sample_offset_with_cycle(rng, npulse=npulse, nl=nl, nu=nu)

    j = np.zeros((n,), dtype=np.complex64)
    g = np.zeros((n,), dtype=np.uint8)

    for c in range(k_cycles):
        base = offset + c * nu
        sl = s_full[base : base + nl]
        for m in range(1, nf + 1):
            st = base + m * nl
            ed = st + nl
            if st >= n:
                continue
            ed = min(ed, n)
            valid = ed - st
            if valid <= 0:
                continue
            j[st:ed] += sl[:valid]
            g[st:ed] = 1

    if enable_right_shift:
        delay = _sample_safe_right_delay(g, rng=rng)
        if delay > 0:
            j = shift_right_zero_pad(j, delay=delay)
            g = shift_right_zero_pad(g, delay=delay)

    if enable_freq_shift:
        fd = float(rng.uniform(-0.2 * b, 0.2 * b))
        idx = np.arange(n, dtype=np.float64)
        phase = np.exp(1j * 2.0 * np.pi * fd * idx / fs).astype(np.complex64)
        j = j * phase
    return j.astype(np.complex64), g.astype(np.uint8)


def _scale_to_power(x: np.ndarray, p_target: float, eps: float = 1e-8) -> np.ndarray:
    p0 = float(np.mean(np.abs(x) ** 2))
    if p0 <= eps:
        return np.zeros_like(x)
    scale = np.sqrt(p_target / (p0 + eps)).astype(np.float32)
    return (x * scale).astype(np.complex64)


def _sample_k_active_list(count: int, dual_ratio: float, rng: np.random.Generator) -> np.ndarray:
    n_dual = int(round(count * dual_ratio))
    n_dual = min(max(n_dual, 0), count)
    n_triple = count - n_dual
    arr = np.array([2] * n_dual + [3] * n_triple, dtype=np.int32)
    rng.shuffle(arr)
    return arr


def _build_k_active_list(
    *,
    count: int,
    grid: dict[str, Any],
    rng: np.random.Generator,
) -> np.ndarray:
    """Build K_active array.

    - default: mixed dual/multi by `k_active_dual_ratio`
    - fixed mode: all samples use `k_active_fixed` in {2, 3}
    """
    if "k_active_fixed" in grid:
        k_fixed = int(grid["k_active_fixed"])
        if k_fixed not in (2, 3):
            raise ValueError(f"grid.k_active_fixed must be 2 or 3, got {k_fixed}")
        return np.full((count,), k_fixed, dtype=np.int32)

    dual_ratio = float(grid["k_active_dual_ratio"])
    return _sample_k_active_list(count, dual_ratio=dual_ratio, rng=rng)


def _sample_overlap_ratio(gates: np.ndarray, k_active: int) -> float:
    active_gates = gates[:k_active].astype(bool)
    active_sum = float(active_gates.sum())
    if active_sum <= 0.0:
        return 0.0
    union = float(np.any(active_gates, axis=0).sum())
    return float(1.0 - union / max(active_sum, 1.0))


def _source_overlap_ratios(gates: np.ndarray, k_active: int) -> np.ndarray:
    active_gates = gates[:k_active].astype(bool)
    vals: list[float] = []
    for k in range(k_active):
        own = active_gates[k]
        own_len = int(own.sum())
        if own_len <= 0:
            vals.append(0.0)
            continue
        other = np.any(active_gates[np.arange(k_active) != k], axis=0)
        vals.append(float(np.logical_and(own, other).sum() / own_len))
    return np.asarray(vals, dtype=np.float32)


def _overlap_penalty(
    *,
    gates: np.ndarray,
    k_active: int,
    sample_overlap_max: float | None,
    source_overlap_max: float | None,
) -> float:
    penalty = 0.0
    if sample_overlap_max is not None:
        sample_ov = _sample_overlap_ratio(gates, k_active)
        penalty += max(0.0, sample_ov - sample_overlap_max)
    if source_overlap_max is not None:
        source_ov = _source_overlap_ratios(gates, k_active)
        if source_ov.size > 0:
            penalty += max(0.0, float(source_ov.max()) - source_overlap_max)
    return float(penalty)


def _overlap_control(cfg: dict[str, Any]) -> dict[str, Any]:
    raw = cfg.get("overlap_control", {})
    enabled = bool(raw.get("enabled", False))
    sample_overlap_max = raw.get("sample_overlap_max")
    source_overlap_max = raw.get("source_overlap_max")
    return {
        "enabled": enabled,
        "sample_overlap_max": None if sample_overlap_max is None else float(sample_overlap_max),
        "source_overlap_max": None if source_overlap_max is None else float(source_overlap_max),
        "max_retries": int(raw.get("max_retries", 64)),
    }


def _meta_dict(cfg: dict[str, Any]) -> dict[str, Any]:
    signal = cfg["signal"]
    grid = cfg["grid"]
    if "k_active_fixed" in grid:
        scenario_mode: str | int = int(grid["k_active_fixed"])
    else:
        scenario_mode = "mixed"
    return {
        "dataset_name": cfg["dataset"]["name"],
        "seed": int(cfg["dataset"]["seed"]),
        "signal": signal,
        "grid": grid,
        "augment": cfg.get("augment", {}),
        "scenario_mode": scenario_mode,
        "background": cfg["background"],
        "split": cfg["split"],
        "definition": {
            "K": 3,
            "input": "x=(B,2,N) IQ",
            "mixture": "x=sum_{k=1..K}(j_k)+b",
            "Ts": "Ts=(NF+1)*Tl",
            "NF_classes": [0, 1, 2, 3],
            "NF_0": "silence source for inactive third slot in dual samples",
        },
    }


def generate_composite_split(
    cfg: dict[str, Any],
    split_name: str,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Generate one split arrays."""
    num = int(cfg["split"][split_name])
    signal = cfg["signal"]
    grid = cfg["grid"]
    bg_cfg = cfg["background"]

    fs = float(signal["fs"])
    tp = float(signal["tp"])
    pri = float(signal["pri"])
    b = float(signal["b"])
    augment = cfg.get("augment", {})
    overlap_ctl = _overlap_control(cfg)
    enable_right_shift = bool(augment.get("enable_right_shift", True))
    enable_freq_shift = bool(augment.get("enable_freq_shift", True))
    n = int(round(fs * pri))
    if n != int(signal["n"]):
        raise ValueError(f"N mismatch: computed {n}, config {signal['n']}")
    npulse = int(round(fs * tp))

    s_full = make_lfm_pri(fs=fs, tp=tp, pri=pri, b=b)

    x_arr = np.zeros((num, 2, n), dtype=np.float32)
    j_arr = np.zeros((num, 3, 2, n), dtype=np.float32)
    g_arr = np.zeros((num, 3, n), dtype=np.uint8)
    tl_us_arr = np.zeros((num, 3), dtype=np.float32)
    ts_us_arr = np.zeros((num, 3), dtype=np.float32)
    nf_arr = np.zeros((num, 3), dtype=np.int32)
    jnr_arr = np.zeros((num, 3), dtype=np.float32)

    k_active_arr = _build_k_active_list(count=num, grid=grid, rng=rng)

    nf_values = np.array(grid["nf_values"], dtype=np.int32)
    tl_min_us = float(grid["tl_us_min"])
    tl_max_us = float(grid["tl_us_max"])
    jnr_min = float(grid["jnr_db_min"])
    jnr_max = float(grid["jnr_db_max"])

    pn = float(bg_cfg["noise_power"])
    snr_echo_db = float(bg_cfg["echo_snr_db"])
    pe_target = pn * (10.0 ** (snr_echo_db / 10.0))

    for i in range(num):
        k_active = int(k_active_arr[i])
        n_trials = overlap_ctl["max_retries"] if overlap_ctl["enabled"] else 1
        best_penalty = float("inf")
        best_state: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None

        for _ in range(n_trials):
            jams = np.zeros((3, n), dtype=np.complex64)
            gates = np.zeros((3, n), dtype=np.uint8)
            nf_row = np.zeros((3,), dtype=np.int32)
            tl_row = np.zeros((3,), dtype=np.float32)
            ts_row = np.zeros((3,), dtype=np.float32)
            jnr_row = np.zeros((3,), dtype=np.float32)

            for k in range(3):
                if k >= k_active:
                    continue

                nf = int(rng.choice(nf_values))
                tl_us = float(rng.uniform(tl_min_us, tl_max_us))
                jnr_db = float(rng.uniform(jnr_min, jnr_max))

                j_raw, g = generate_one_jammer(
                    s_full=s_full,
                    fs=fs,
                    b=b,
                    nf=nf,
                    tl_us=tl_us,
                    rng=rng,
                    enable_right_shift=enable_right_shift,
                    enable_freq_shift=enable_freq_shift,
                )
                p_target = pn * (10.0 ** (jnr_db / 10.0))
                j = _scale_to_power(j_raw, p_target=p_target)

                jams[k, :] = j
                gates[k, :] = g
                nf_row[k] = np.int32(nf)
                tl_row[k] = np.float32(tl_us)
                ts_row[k] = np.float32((nf + 1) * tl_us)
                jnr_row[k] = np.float32(jnr_db)

            penalty = _overlap_penalty(
                gates=gates,
                k_active=k_active,
                sample_overlap_max=overlap_ctl["sample_overlap_max"],
                source_overlap_max=overlap_ctl["source_overlap_max"],
            )
            if penalty < best_penalty:
                best_penalty = penalty
                best_state = (jams, gates, nf_row, tl_row, ts_row, jnr_row)
            if penalty <= 0.0:
                break

        if best_state is None:
            raise RuntimeError("Failed to sample composite jammer state.")

        jams, gates, nf_row, tl_row, ts_row, jnr_row = best_state
        nf_arr[i, :] = nf_row
        tl_us_arr[i, :] = tl_row
        ts_us_arr[i, :] = ts_row
        jnr_arr[i, :] = jnr_row

        # Background echo
        delay_echo = int(rng.integers(0, n))
        e0 = shift_right_zero_pad(s_full, delay=delay_echo)
        e = _scale_to_power(e0, p_target=pe_target)

        # AWGN
        noise = (
            rng.standard_normal(n, dtype=np.float32)
            + 1j * rng.standard_normal(n, dtype=np.float32)
        ) / np.sqrt(2.0)
        noise = noise.astype(np.complex64) * np.float32(np.sqrt(pn))

        x = np.sum(jams, axis=0) + e + noise

        x_arr[i, 0, :] = x.real.astype(np.float32)
        x_arr[i, 1, :] = x.imag.astype(np.float32)

        j_arr[i, :, 0, :] = jams.real.astype(np.float32)
        j_arr[i, :, 1, :] = jams.imag.astype(np.float32)
        g_arr[i, :, :] = gates

    # Strict dual constraints: third source silence.
    dual_idx = np.where(k_active_arr == 2)[0]
    if dual_idx.size > 0:
        j_arr[dual_idx, 2, :, :] = 0.0
        g_arr[dual_idx, 2, :] = 0
        nf_arr[dual_idx, 2] = 0
        tl_us_arr[dual_idx, 2] = 0.0
        ts_us_arr[dual_idx, 2] = 0.0
        jnr_arr[dual_idx, 2] = 0.0

    # Consistency for active sources.
    active = nf_arr > 0
    ts_ref = (nf_arr.astype(np.float32) + 1.0) * tl_us_arr
    ts_us_arr[active] = ts_ref[active]

    return {
        "X": x_arr,
        "J": j_arr,
        "G": g_arr,
        "Tl_us": tl_us_arr,
        "Ts_us": ts_us_arr,
        "NF": nf_arr,
        "JNR_dB": jnr_arr,
        "K_active": k_active_arr.astype(np.int32),
    }


def save_composite_split(
    out_dir: Path,
    split_name: str,
    arrays: dict[str, np.ndarray],
    meta_json: str,
) -> None:
    """Save one split to NPZ."""
    np.savez_compressed(
        out_dir / f"{split_name}.npz",
        X=arrays["X"],
        J=arrays["J"],
        G=arrays["G"],
        Tl_us=arrays["Tl_us"],
        Ts_us=arrays["Ts_us"],
        NF=arrays["NF"],
        JNR_dB=arrays["JNR_dB"],
        K_active=arrays["K_active"],
        meta_json=meta_json,
    )


def generate_and_save_composite_dataset(cfg: dict[str, Any]) -> dict[str, Any]:
    """Generate train/val/test composite dataset and save NPZ files."""
    out_dir = ensure_dir(cfg["dataset"]["output_dir"])
    seed = int(cfg["dataset"]["seed"])
    rng = np.random.default_rng(seed)

    meta = _meta_dict(cfg)
    meta_json = json.dumps(meta, ensure_ascii=False, indent=2)
    save_json(meta, out_dir / "meta.json")

    summary: dict[str, Any] = {"output_dir": str(out_dir), "seed": seed}
    for split_name in ["train", "val", "test"]:
        arrays = generate_composite_split(cfg, split_name=split_name, rng=rng)
        save_composite_split(out_dir=out_dir, split_name=split_name, arrays=arrays, meta_json=meta_json)
        summary[f"{split_name}_samples"] = int(arrays["X"].shape[0])

    return summary
