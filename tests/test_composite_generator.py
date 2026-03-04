from __future__ import annotations

import numpy as np

from src.data.isrj_composite_generator import generate_composite_split


def _tiny_cfg() -> dict:
    return {
        "dataset": {"name": "tiny", "output_dir": "unused", "seed": 123},
        "signal": {"fs": 100e6, "tp": 15e-6, "pri": 40e-6, "b": 10e6, "n": 4000},
        "grid": {
            "nf_values": [1, 2, 3],
            "tl_us_min": 1.0,
            "tl_us_max": 3.0,
            "jnr_db_min": -10.0,
            "jnr_db_max": 20.0,
            "k_active_dual_ratio": 0.5,
        },
        "background": {"echo_snr_db": 0.0, "noise_power": 1.0},
        "split": {"train": 20, "val": 8, "test": 8},
        "sanity": {"ts_tl_eps": 1e-6},
    }


def test_composite_generator_shapes_and_rules() -> None:
    cfg = _tiny_cfg()
    rng = np.random.default_rng(42)
    d = generate_composite_split(cfg, split_name="train", rng=rng)

    assert d["X"].shape == (20, 2, 4000)
    assert d["J"].shape == (20, 3, 2, 4000)
    assert d["G"].shape == (20, 3, 4000)
    assert d["Tl_us"].shape == (20, 3)
    assert d["Ts_us"].shape == (20, 3)
    assert d["NF"].shape == (20, 3)
    assert d["JNR_dB"].shape == (20, 3)
    assert d["K_active"].shape == (20,)

    assert d["X"].dtype == np.float32
    assert d["J"].dtype == np.float32
    assert d["G"].dtype == np.uint8
    assert d["NF"].dtype == np.int32

    idx_dual = np.where(d["K_active"] == 2)[0]
    if idx_dual.size > 0:
        assert np.allclose(d["J"][idx_dual, 2], 0.0)
        assert np.all(d["G"][idx_dual, 2] == 0)
        assert np.all(d["NF"][idx_dual, 2] == 0)
        assert np.allclose(d["Tl_us"][idx_dual, 2], 0.0)
        assert np.allclose(d["Ts_us"][idx_dual, 2], 0.0)

    active = d["NF"] > 0
    ts_ref = (d["NF"].astype(np.float32) + 1.0) * d["Tl_us"]
    max_err = float(np.max(np.abs(d["Ts_us"][active] - ts_ref[active]))) if np.any(active) else 0.0
    assert max_err <= 1e-6
