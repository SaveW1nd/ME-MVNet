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


def test_sepnet_jam_only_shapes_and_mixture_consistency() -> None:
    cfg = load_yaml("configs/model_sep.yaml")
    cfg["model_sep"]["jam_only"] = True
    model = SepNet(cfg).eval()
    x = torch.randn(2, 2, 4000, dtype=torch.float32)
    out = model(x)

    assert out["j_hat"].shape == (2, 3, 2, 4000)
    assert out["b_hat"].shape == (2, 2, 4000)
    assert out["masks"].shape[1] == 3

    rec = torch.sum(out["j_hat"], dim=1) + out["b_hat"]
    err = torch.mean(torch.abs(rec - x)).item()
    assert err < 1e-5


def test_sepnet_grouped_decoder_shapes_and_mixture_consistency() -> None:
    cfg = load_yaml("configs/model_sep.yaml")
    cfg["model_sep"]["decoder_grouped"] = True
    cfg["model_sep"]["bg_residual_scale"] = 0.2
    model = SepNet(cfg).eval()
    x = torch.randn(2, 2, 4000, dtype=torch.float32)
    out = model(x)

    assert out["j_hat"].shape == (2, 3, 2, 4000)
    assert out["b_hat"].shape == (2, 2, 4000)
    rec = torch.sum(out["j_hat"], dim=1) + out["b_hat"]
    err = torch.mean(torch.abs(rec - x)).item()
    assert err < 1e-5


def test_sepnet_bg_dedicated_shapes_and_mixture_consistency() -> None:
    cfg = load_yaml("configs/model_sep.yaml")
    cfg["model_sep"]["decoder_grouped"] = True
    cfg["model_sep"]["bg_dedicated"] = True
    cfg["model_sep"]["bg_residual_ratio"] = 0.2
    model = SepNet(cfg).eval()
    x = torch.randn(2, 2, 4000, dtype=torch.float32)
    out = model(x)

    assert out["j_hat"].shape == (2, 3, 2, 4000)
    assert out["b_hat"].shape == (2, 2, 4000)
    assert out["masks"].shape[1] == 3
    rec = torch.sum(out["j_hat"], dim=1) + out["b_hat"]
    err = torch.mean(torch.abs(rec - x)).item()
    assert err < 1e-5
