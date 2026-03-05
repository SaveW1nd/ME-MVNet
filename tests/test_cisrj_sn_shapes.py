from __future__ import annotations

import torch

from src.models.builders import build_separator
from src.utils.io import load_yaml


def test_cisrj_sn_shapes_and_mixture_consistency() -> None:
    cfg = load_yaml("configs/model_sep_cisrj_repro.yaml")
    # Reduce model size for unit test speed.
    cfg["model_sep"]["encoder_channels"] = 64
    cfg["model_sep"]["tcn_blocks"] = 2
    cfg["model_sep"]["max_length"] = 1024
    model = build_separator(cfg).eval()

    x = torch.randn(2, 2, 4000, dtype=torch.float32)
    out = model(x)

    assert out["j_hat"].shape == (2, 3, 2, 4000)
    assert out["b_hat"].shape == (2, 2, 4000)
    assert out["masks"].shape[0] == 2
    assert out["masks"].shape[1] == 4

    rec = torch.sum(out["j_hat"], dim=1) + out["b_hat"]
    err = torch.mean(torch.abs(rec - x)).item()
    assert err < 1e-5
