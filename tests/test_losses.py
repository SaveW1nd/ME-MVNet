from __future__ import annotations

import torch

from src.models.losses import compute_total_loss


def test_loss_outputs_and_backward() -> None:
    b = 6
    n = 4000
    pred = {
        "Tl_hat": torch.rand(b, requires_grad=True) + 1e-3,
        "Tf_hat": torch.rand(b, requires_grad=True) + 1e-3,
        "NF_logits": torch.randn(b, 3, requires_grad=True),
        "mask_hat": torch.rand(b, n, requires_grad=True),
    }
    batch = {
        "Tl_s": torch.rand(b),
        "Tf_s": torch.rand(b),
        "NF_index": torch.randint(0, 3, size=(b,)),
        "mask": torch.randint(0, 2, size=(b, n)).float(),
    }
    loss_cfg = {"w_tl": 1.0, "w_tf": 1.0, "w_nf": 1.0, "w_mask": 0.5, "w_phy": 0.2}
    losses = compute_total_loss(pred, batch, loss_cfg, nf_values=[1, 2, 4])
    assert set(losses.keys()) == {"L_Tl", "L_Tf", "L_NF", "L_mask", "L_phy", "L_total"}
    losses["L_total"].backward()
