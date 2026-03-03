"""ISRJ dataset generation.

Output split files (`train.npz`, `val.npz`, `test.npz`) contain:
    - X: float32, shape (num, 2, 4000)
    - mask: uint8, shape (num, 4000)
    - Tl_s: float32, shape (num,)
    - Tf_s: float32, shape (num,)
    - NF: int32, shape (num,)
    - JNR_dB: int32, shape (num,)
    - meta_json: UTF-8 JSON string
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.io import ensure_dir, save_json


def make_baseband_lfm(fs: float, tp: float, kr: float) -> np.ndarray:
    """Baseband LFM chirp: exp(j*pi*kr*t^2), t in [-tp/2, tp/2)."""
    n = int(round(fs * tp))
    t = (np.arange(n, dtype=np.float64) - n / 2.0) / fs
    return np.exp(1j * np.pi * kr * t**2).astype(np.complex64)


def make_single_isrj(s: np.ndarray, nl: int, nf: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate one ISRJ-only waveform and forwarding mask.

    Args:
        s: complex64 array, shape (N,), transmitted LFM pulse.
        nl: slice width in samples.
        nf: forwarding times, value in {1,2,4}.

    Returns:
        j: complex64 ISRJ waveform, shape (N,)
        mask: uint8 forwarding mask, shape (N,)
    """
    n = int(s.shape[0])
    nu = (nf + 1) * nl
    k_cycles = n // nu

    j = np.zeros(n, dtype=np.complex64)
    mask = np.zeros(n, dtype=np.uint8)

    for k in range(k_cycles):
        base = k * nu
        sl = s[base : base + nl]
        for m in range(1, nf + 1):
            st = base + m * nl
            ed = st + nl
            j[st:ed] += sl
            mask[st:ed] = 1
    return j, mask


def add_awgn_by_jnr(j: np.ndarray, jnr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Add complex AWGN by JNR definition: JNR=10log10(PJ/PN)."""
    pj = float(np.mean(np.abs(j) ** 2))
    if pj <= 0:
        raise ValueError("PJ must be positive for valid JNR noise injection.")
    pn = pj / (10.0 ** (jnr_db / 10.0))
    noise = (
        rng.standard_normal(j.shape[0], dtype=np.float32)
        + 1j * rng.standard_normal(j.shape[0], dtype=np.float32)
    ) / np.sqrt(2.0)
    return j + noise.astype(np.complex64) * np.float32(np.sqrt(pn))


def _build_nl_list(nl_min: int, nl_max: int, points: int) -> np.ndarray:
    values = np.unique(np.round(np.linspace(nl_min, nl_max, points)).astype(np.int32))
    if len(values) != points:
        raise ValueError(
            f"Nl list collapsed after rounding: expected {points}, got {len(values)}"
        )
    return values


def _jnr_list(cfg: dict[str, Any]) -> np.ndarray:
    g = cfg["grid"]
    start = int(g["jnr_db_start"])
    stop = int(g["jnr_db_stop"])
    step = int(g["jnr_db_step"])
    return np.arange(start, stop + step, step, dtype=np.int32)


def _build_meta(cfg: dict[str, Any], nl_nf12: np.ndarray, nl_nf4: np.ndarray) -> dict[str, Any]:
    fs = float(cfg["signal"]["fs"])
    tp = float(cfg["signal"]["tp"])
    b = float(cfg["signal"]["b"])
    kr = b / tp
    meta = {
        "dataset_name": cfg["dataset"]["name"],
        "fs": fs,
        "tp": tp,
        "b": b,
        "kr": kr,
        "n": int(round(fs * tp)),
        "nf_list": [int(v) for v in cfg["grid"]["nf_list"]],
        "jnr_db_list": _jnr_list(cfg).tolist(),
        "mc_repeats": int(cfg["grid"]["mc_repeats"]),
        "nl_list_nf12": nl_nf12.tolist(),
        "nl_list_nf4": nl_nf4.tolist(),
        "seed": int(cfg["dataset"]["seed"]),
        "label_definition": {
            "Tl": "slice width (seconds), Tl = Nl/fs",
            "NF": "forwarding times (integer), NF = M",
            "Tf": "forwarding width per cycle (seconds), Tf = NF*Tl",
            "Tu": "interrupted-sampling interval, Tu = (NF+1)*Tl",
        },
    }
    return meta


def generate_dataset_arrays(cfg: dict[str, Any]) -> dict[str, Any]:
    """Generate full 6000-sample dataset arrays according to config."""
    fs = float(cfg["signal"]["fs"])
    tp = float(cfg["signal"]["tp"])
    b = float(cfg["signal"]["b"])
    kr = b / tp
    n = int(round(fs * tp))
    expected_n = int(cfg["signal"]["n"])
    if n != expected_n:
        raise ValueError(f"Signal N mismatch: computed {n}, config {expected_n}")

    nf_list = np.array(cfg["grid"]["nf_list"], dtype=np.int32)
    jnr_list = _jnr_list(cfg)
    points = int(cfg["grid"]["nl_points_per_nf"])
    nl_nf12 = _build_nl_list(
        int(cfg["grid"]["nl_min_nf12"]),
        int(cfg["grid"]["nl_max_nf12"]),
        points,
    )
    nl_nf4 = _build_nl_list(
        int(cfg["grid"]["nl_min_nf4"]),
        int(cfg["grid"]["nl_max_nf4"]),
        points,
    )
    mc_repeats = int(cfg["grid"]["mc_repeats"])

    rng = np.random.default_rng(int(cfg["dataset"]["seed"]))
    s = make_baseband_lfm(fs=fs, tp=tp, kr=kr)

    samples: list[tuple[int, int, int, int]] = []
    for nf in nf_list:
        nl_list = nl_nf4 if int(nf) == 4 else nl_nf12
        for nl in nl_list:
            for jnr_db in jnr_list:
                for mc in range(mc_repeats):
                    samples.append((int(nf), int(nl), int(jnr_db), int(mc)))

    expected = len(nf_list) * points * len(jnr_list) * mc_repeats
    if len(samples) != expected:
        raise ValueError(f"Sample count mismatch: {len(samples)} vs {expected}")

    x = np.zeros((len(samples), 2, n), dtype=np.float32)
    mask = np.zeros((len(samples), n), dtype=np.uint8)
    tl_s = np.zeros((len(samples),), dtype=np.float32)
    tf_s = np.zeros((len(samples),), dtype=np.float32)
    nf_y = np.zeros((len(samples),), dtype=np.int32)
    jnr_y = np.zeros((len(samples),), dtype=np.int32)

    for idx, (nf, nl, jnr_db, _) in enumerate(samples):
        j, mk = make_single_isrj(s=s, nl=nl, nf=nf)
        xn = add_awgn_by_jnr(j, jnr_db=jnr_db, rng=rng)

        x[idx, 0, :] = xn.real.astype(np.float32)
        x[idx, 1, :] = xn.imag.astype(np.float32)
        mask[idx, :] = mk
        tl_s[idx] = np.float32(nl / fs)
        tf_s[idx] = np.float32(nf * nl / fs)
        nf_y[idx] = np.int32(nf)
        jnr_y[idx] = np.int32(jnr_db)

    split_cfg = cfg["split"]
    train_ratio = float(split_cfg["train"])
    val_ratio = float(split_cfg["val"])
    test_ratio = float(split_cfg["test"])
    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-9:
        raise ValueError(f"Split ratio must sum to 1.0, got {ratio_sum}")

    train_idx = []
    val_idx = []
    test_idx = []
    for nf in nf_list:
        idx_nf = np.where(nf_y == nf)[0]
        rng.shuffle(idx_nf)

        n_total = idx_nf.shape[0]
        n_train = int(round(train_ratio * n_total))
        n_val = int(round(val_ratio * n_total))
        n_test = n_total - n_train - n_val
        if n_test < 0:
            raise ValueError("Negative n_test caused by split ratio rounding.")

        train_idx.append(idx_nf[:n_train])
        val_idx.append(idx_nf[n_train : n_train + n_val])
        test_idx.append(idx_nf[n_train + n_val :])

    split_indices = {
        "train": np.concatenate(train_idx),
        "val": np.concatenate(val_idx),
        "test": np.concatenate(test_idx),
    }
    rng.shuffle(split_indices["train"])
    rng.shuffle(split_indices["val"])
    rng.shuffle(split_indices["test"])

    meta = _build_meta(cfg, nl_nf12=nl_nf12, nl_nf4=nl_nf4)
    return {
        "X": x,
        "mask": mask,
        "Tl_s": tl_s,
        "Tf_s": tf_s,
        "NF": nf_y,
        "JNR_dB": jnr_y,
        "split_indices": split_indices,
        "meta": meta,
    }


def save_dataset(cfg: dict[str, Any], arrays: dict[str, Any]) -> dict[str, Any]:
    """Save generated arrays to split NPZ files and meta.json."""
    out_dir = ensure_dir(cfg["dataset"]["output_dir"])
    meta = arrays["meta"]
    meta_json = json.dumps(meta, ensure_ascii=False, indent=2)
    save_json(meta, out_dir / "meta.json")

    def save_split(name: str, idxs: np.ndarray) -> None:
        np.savez_compressed(
            out_dir / f"{name}.npz",
            X=arrays["X"][idxs],
            mask=arrays["mask"][idxs],
            Tl_s=arrays["Tl_s"][idxs],
            Tf_s=arrays["Tf_s"][idxs],
            NF=arrays["NF"][idxs],
            JNR_dB=arrays["JNR_dB"][idxs],
            meta_json=meta_json,
        )

    for split_name, split_idxs in arrays["split_indices"].items():
        save_split(split_name, split_idxs)

    summary = {
        "output_dir": str(out_dir),
        "train_samples": int(len(arrays["split_indices"]["train"])),
        "val_samples": int(len(arrays["split_indices"]["val"])),
        "test_samples": int(len(arrays["split_indices"]["test"])),
        "total_samples": int(arrays["X"].shape[0]),
    }
    return summary


def generate_and_save_dataset(cfg: dict[str, Any]) -> dict[str, Any]:
    """Run full generation and persist split files."""
    arrays = generate_dataset_arrays(cfg)
    return save_dataset(cfg, arrays)
