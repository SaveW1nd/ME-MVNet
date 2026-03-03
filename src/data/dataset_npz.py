"""NPZ dataset loader.

Expected file fields:
    X: float32, (B, 2, 4000)
    mask: uint8, (B, 4000)
    Tl_s: float32, (B,)
    Tf_s: float32, (B,)
    NF: int32, (B,) with values in {1,2,4}
    JNR_dB: int32, (B,)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .transforms import standardize_iq


NF_VALUES = [1, 2, 4]
NF_TO_INDEX = {v: i for i, v in enumerate(NF_VALUES)}


class ISRJDataset(Dataset):
    """Load one split file into memory."""

    def __init__(
        self,
        npz_path: str | Path,
        normalize_iq: bool = True,
    ) -> None:
        data = np.load(npz_path, allow_pickle=False)
        self.x = data["X"].astype(np.float32)
        self.mask = data["mask"].astype(np.float32)
        self.tl_s = data["Tl_s"].astype(np.float32)
        self.tf_s = data["Tf_s"].astype(np.float32)
        self.nf = data["NF"].astype(np.int32)
        self.jnr_db = data["JNR_dB"].astype(np.int32)
        self.normalize_iq = normalize_iq

        self.nf_index = np.array([NF_TO_INDEX[int(v)] for v in self.nf], dtype=np.int64)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> dict[str, Any]:
        x = self.x[idx]
        if self.normalize_iq:
            x = standardize_iq(x)

        return {
            "X": torch.from_numpy(x),
            "mask": torch.from_numpy(self.mask[idx]),
            "Tl_s": torch.tensor(self.tl_s[idx], dtype=torch.float32),
            "Tf_s": torch.tensor(self.tf_s[idx], dtype=torch.float32),
            "NF_index": torch.tensor(self.nf_index[idx], dtype=torch.long),
            "NF_value": torch.tensor(self.nf[idx], dtype=torch.int64),
            "JNR_dB": torch.tensor(self.jnr_db[idx], dtype=torch.int64),
        }
