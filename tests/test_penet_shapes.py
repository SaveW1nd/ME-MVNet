from __future__ import annotations

import torch

from src.models.penet import PENet
from src.utils.io import load_yaml


def test_penet_shapes() -> None:
    cfg = load_yaml("configs/model_pe.yaml")
    model = PENet(cfg).eval()
    x = torch.randn(4, 2, 4000, dtype=torch.float32)
    out = model(x)

    assert out["g_logit"].shape == (4, 4000)
    assert out["g_hat"].shape == (4, 4000)
    assert out["Tl_hat_us"].shape == (4,)
    assert out["NF_logits"].shape == (4, 4)
    assert out["Ts_hat_us"].shape == (4,)

    assert torch.all(out["g_hat"] >= 0)
    assert torch.all(out["g_hat"] <= 1)
    assert torch.all(out["Tl_hat_us"] > 0)
    assert torch.all(out["Ts_hat_us"] > 0)
