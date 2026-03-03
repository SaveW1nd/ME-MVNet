from __future__ import annotations

import numpy as np
import torch

from src.data.dataset_npz import ISRJDataset


def test_npz_loading_shapes(tmp_path) -> None:
    num = 8
    n = 4000
    path = tmp_path / "toy.npz"
    np.savez_compressed(
        path,
        X=np.random.randn(num, 2, n).astype(np.float32),
        mask=np.random.randint(0, 2, size=(num, n), dtype=np.uint8),
        Tl_s=np.random.rand(num).astype(np.float32),
        Tf_s=np.random.rand(num).astype(np.float32),
        NF=np.random.choice([1, 2, 4], size=(num,)).astype(np.int32),
        JNR_dB=np.random.choice(np.arange(-10, 30, 2), size=(num,)).astype(np.int32),
        meta_json="{}",
    )

    ds = ISRJDataset(path, normalize_iq=True)
    assert len(ds) == num
    sample = ds[0]
    assert sample["X"].shape == (2, n)
    assert sample["mask"].shape == (n,)
    assert sample["X"].dtype == torch.float32
