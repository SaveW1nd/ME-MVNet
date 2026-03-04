from __future__ import annotations

import torch

from src.models.sepnet import SepNet
from src.utils.io import load_yaml


def test_sepnet_shapes_and_mixture_consistency() -> None:
    cfg = load_yaml("configs/model_sep.yaml")
    model = SepNet(cfg).eval()
    x = torch.randn(3, 2, 4000, dtype=torch.float32)
    out = model(x)

    assert out["j_hat"].shape == (3, 3, 2, 4000)
    assert out["b_hat"].shape == (3, 2, 4000)
    assert out["masks"].shape[0] == 3
    assert out["masks"].shape[1] == 4

    rec = torch.sum(out["j_hat"], dim=1) + out["b_hat"]
    err = torch.mean(torch.abs(rec - x)).item()
    assert err < 1e-5
