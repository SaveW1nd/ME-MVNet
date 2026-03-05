"""GateFormer: transformer-based parameter reader from gate periodicity."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GateFormer(nn.Module):
    """Read periodic gate structure and estimate Tl/NF for one separated source."""

    def __init__(
        self,
        seq_dim: int,
        tf_dim: int,
        mech_dim: int,
        periodic_dim: int,
        num_nf_classes: int,
        cfg: dict | None = None,
    ) -> None:
        super().__init__()
        cfg = cfg or {}

        self.seq_dim = int(seq_dim)
        self.periodic_dim = int(periodic_dim)
        self.d_model = int(cfg.get("d_model", self.seq_dim))
        self.num_layers = int(cfg.get("num_layers", 2))
        self.num_heads = int(cfg.get("num_heads", 4))
        self.dropout = float(cfg.get("dropout", 0.1))
        self.ff_mult = int(cfg.get("ff_mult", 2))
        self.max_seq_len = int(cfg.get("max_seq_len", 1024))
        head_hidden = int(cfg.get("head_hidden_dim", self.d_model))

        self.input_proj = nn.Linear(self.seq_dim + 1, self.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.max_seq_len, self.d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.num_heads,
            dim_feedforward=self.d_model * self.ff_mult,
            dropout=self.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=self.num_layers)
        self.enc_norm = nn.LayerNorm(self.d_model)

        self.cond_proj = nn.Sequential(
            nn.Linear(int(tf_dim) + int(mech_dim) + self.periodic_dim, self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )
        self.query_tokens = nn.Parameter(torch.randn(2, self.d_model) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=self.num_heads,
            dropout=self.dropout,
            batch_first=True,
        )
        self.query_norm1 = nn.LayerNorm(self.d_model)
        self.query_ffn = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * self.ff_mult),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model * self.ff_mult, self.d_model),
        )
        self.query_norm2 = nn.LayerNorm(self.d_model)

        self.tl_head = nn.Sequential(
            nn.Linear(self.d_model, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, 1),
        )
        self.nf_head = nn.Sequential(
            nn.Linear(self.d_model, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, int(num_nf_classes)),
        )
        self.softplus = nn.Softplus()

    def _position_tokens(self, length: int, dtype: torch.dtype) -> torch.Tensor:
        if length <= self.max_seq_len:
            pos = self.pos_emb[:, :length, :]
        else:
            pos = F.interpolate(
                self.pos_emb.transpose(1, 2),
                size=length,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        return pos.to(dtype=dtype)

    def forward(
        self,
        seq_feat: torch.Tensor,
        g_logit: torch.Tensor,
        z_tf: torch.Tensor,
        z_mech: torch.Tensor,
        z_periodic: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        # seq_feat: (B,L,D), g_logit: (B,N)
        if seq_feat.ndim != 3:
            raise ValueError(f"Expected seq_feat (B,L,D), got {tuple(seq_feat.shape)}")
        if g_logit.ndim != 2:
            raise ValueError(f"Expected g_logit (B,N), got {tuple(g_logit.shape)}")

        bsz, length, _ = seq_feat.shape
        g_ds = F.adaptive_avg_pool1d(g_logit.unsqueeze(1), output_size=length).squeeze(1)
        u = torch.cat([seq_feat, g_ds.unsqueeze(-1)], dim=-1)
        u = self.input_proj(u)
        u = u + self._position_tokens(length=length, dtype=u.dtype)
        u = self.encoder(u)
        u = self.enc_norm(u)

        if z_periodic is None:
            z_periodic = z_mech.new_zeros((z_mech.shape[0], self.periodic_dim))
        cond = self.cond_proj(torch.cat([z_tf, z_mech, z_periodic], dim=1))
        q = self.query_tokens.unsqueeze(0).expand(bsz, -1, -1) + cond.unsqueeze(1)
        q_attn, _ = self.cross_attn(query=q, key=u, value=u, need_weights=False)
        q = self.query_norm1(q + q_attn)
        q = self.query_norm2(q + self.query_ffn(q))

        h_tl = q[:, 0, :]
        h_nf = q[:, 1, :]
        tl_hat_us = self.softplus(self.tl_head(h_tl)).squeeze(1)
        nf_logits = self.nf_head(h_nf)
        return {"Tl_hat_us": tl_hat_us, "NF_logits": nf_logits}

