"""Composite NPZ dataset loader for ME-MVSepPE."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .transforms import standardize_iq


NF_CLASSES = [0, 1, 2, 3]


class CompositeISRJDataset(Dataset):
    """Load one composite split from NPZ.

    Fields in NPZ:
        X: float32 (B,2,N)
        J: float32 (B,3,2,N)
        G: uint8   (B,3,N)
        Tl_us: float32 (B,3)
        Ts_us: float32 (B,3)
        NF: int32 (B,3)
        JNR_dB: float32 (B,3)
        K_active: int32 (B,)
    """

    def __init__(self, npz_path: str | Path, normalize_x: bool = True) -> None:
        d = np.load(npz_path, allow_pickle=False)
        self.x = d["X"].astype(np.float32)
        self.j = d["J"].astype(np.float32)
        self.g = d["G"].astype(np.float32)
        self.tl_us = d["Tl_us"].astype(np.float32)
        self.ts_us = d["Ts_us"].astype(np.float32)
        self.nf = d["NF"].astype(np.int64)
        self.jnr_db = d["JNR_dB"].astype(np.float32)
        self.k_active = d["K_active"].astype(np.int64)
        self.normalize_x = normalize_x

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> dict[str, Any]:
        x = self.x[idx]
        if self.normalize_x:
            x = standardize_iq(x)

        return {
            "X": torch.from_numpy(x),
            "J": torch.from_numpy(self.j[idx]),
            "G": torch.from_numpy(self.g[idx]),
            "Tl_us": torch.from_numpy(self.tl_us[idx]),
            "Ts_us": torch.from_numpy(self.ts_us[idx]),
            "NF": torch.from_numpy(self.nf[idx]).long(),
            "JNR_dB": torch.from_numpy(self.jnr_db[idx]),
            "K_active": torch.tensor(int(self.k_active[idx]), dtype=torch.long),
        }
