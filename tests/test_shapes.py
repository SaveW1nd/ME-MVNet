from __future__ import annotations

import torch

from src.models.memvnet import MEMVNet
from src.utils.io import load_yaml


def test_model_forward_shapes_and_dtypes() -> None:
    cfg = load_yaml("configs/model.yaml")
    model = MEMVNet(cfg).eval()
    x = torch.randn(4, 2, 4000, dtype=torch.float32)
    out = model(x)

    assert out["Tl_hat"].shape == (4,)
    assert out["Tf_hat"].shape == (4,)
    assert out["NF_logits"].shape == (4, 3)
    assert out["mask_hat"].shape == (4, 4000)

    assert out["Tl_hat"].dtype == torch.float32
    assert out["Tf_hat"].dtype == torch.float32
    assert out["NF_logits"].dtype == torch.float32
    assert out["mask_hat"].dtype == torch.float32

    assert torch.all(out["Tl_hat"] > 0)
    assert torch.all(out["Tf_hat"] > 0)
    assert torch.all(out["mask_hat"] >= 0)
    assert torch.all(out["mask_hat"] <= 1)
