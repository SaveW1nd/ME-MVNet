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
        self.num_nf_classes = int(num_nf_classes)
        self.d_model = int(cfg.get("d_model", self.seq_dim))
        self.num_layers = int(cfg.get("num_layers", 2))
        self.num_heads = int(cfg.get("num_heads", 4))
        self.dropout = float(cfg.get("dropout", 0.1))
        self.ff_mult = int(cfg.get("ff_mult", 2))
        self.max_seq_len = int(cfg.get("max_seq_len", 1024))
        head_hidden = int(cfg.get("head_hidden_dim", self.d_model))
        self.min_tl_us = float(cfg.get("min_tl_us", 0.2))
        self.max_tl_us = float(cfg.get("max_tl_us", 3.5))
        self.min_ts_us = float(cfg.get("min_ts_us", 0.4))
        self.max_ts_us = float(cfg.get("max_ts_us", 8.0))
        self.ts_direct_enabled = bool(cfg.get("ts_direct_enabled", False))
        self.ts_residual_enabled = bool(cfg.get("ts_residual_enabled", False))
        self.ts_residual_max_abs_us = float(cfg.get("ts_residual_max_abs_us", 0.8))
        self.nf_head_mode = str(cfg.get("nf_head_mode", "flat")).lower()
        self.tl_use_periodic_fusion = bool(
            cfg.get("tl_use_periodic_fusion", self.periodic_dim > 0)
        )
        if self.ts_direct_enabled and self.ts_residual_enabled:
            raise ValueError("ts_direct_enabled and ts_residual_enabled cannot both be true")

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
        query_n = 3 if self.ts_direct_enabled else 2
        self.query_tokens = nn.Parameter(torch.randn(query_n, self.d_model) * 0.02)
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
        if self.ts_direct_enabled:
            self.ts_head = nn.Sequential(
                nn.Linear(self.d_model, head_hidden),
                nn.GELU(),
                nn.Linear(head_hidden, 1),
            )
        else:
            self.ts_head = None
        if self.ts_residual_enabled:
            self.ts_residual_head = nn.Sequential(
                nn.Linear(self.d_model * 2, head_hidden),
                nn.GELU(),
                nn.Linear(head_hidden, 1),
            )
        else:
            self.ts_residual_head = None
        if self.periodic_dim > 0 and self.tl_use_periodic_fusion:
            tl_aux_hidden = max(32, head_hidden // 2)
            self.tl_periodic_head = nn.Sequential(
                nn.Linear(self.periodic_dim, tl_aux_hidden),
                nn.GELU(),
                nn.Linear(tl_aux_hidden, 1),
            )
            self.tl_blend_head = nn.Sequential(
                nn.Linear(self.d_model + self.periodic_dim, tl_aux_hidden),
                nn.GELU(),
                nn.Linear(tl_aux_hidden, 1),
            )
        else:
            self.tl_periodic_head = None
            self.tl_blend_head = None

        if self.nf_head_mode == "hier":
            if self.num_nf_classes < 2:
                raise ValueError(f"num_nf_classes must be >=2 for hier mode, got {self.num_nf_classes}")
            self.nf_active_head = nn.Sequential(
                nn.Linear(self.d_model, head_hidden),
                nn.GELU(),
                nn.Linear(head_hidden, 1),
            )
            self.nf_cls_head = nn.Sequential(
                nn.Linear(self.d_model, head_hidden),
                nn.GELU(),
                nn.Linear(head_hidden, self.num_nf_classes - 1),
            )
            self.nf_head = None
        else:
            self.nf_head = nn.Sequential(
                nn.Linear(self.d_model, head_hidden),
                nn.GELU(),
                nn.Linear(head_hidden, self.num_nf_classes),
            )
            self.nf_active_head = None
            self.nf_cls_head = None
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
        tl_feat_us = self.softplus(self.tl_head(h_tl)).squeeze(1)
        if self.tl_periodic_head is not None and self.tl_blend_head is not None:
            tl_periodic_us = self.softplus(self.tl_periodic_head(z_periodic)).squeeze(1)
            tl_blend = torch.sigmoid(self.tl_blend_head(torch.cat([h_tl, z_periodic], dim=1))).squeeze(1)
            tl_hat_us = tl_blend * tl_feat_us + (1.0 - tl_blend) * tl_periodic_us
        else:
            tl_hat_us = tl_feat_us
            tl_periodic_us = tl_hat_us
            tl_blend = torch.ones_like(tl_hat_us)
        tl_hat_us = torch.clamp(tl_hat_us, min=self.min_tl_us, max=self.max_tl_us)

        if self.nf_head_mode == "hier":
            active_logit = self.nf_active_head(h_nf).squeeze(1)  # (B,)
            cls_logits = self.nf_cls_head(h_nf)  # (B,C-1)
            nf_logits = torch.cat(
                [(-active_logit).unsqueeze(1), active_logit.unsqueeze(1) + cls_logits],
                dim=1,
            )
        else:
            active_logit = None
            nf_logits = self.nf_head(h_nf)
        slot_feat = torch.cat([h_tl, h_nf], dim=1)
        out: dict[str, torch.Tensor] = {
            "Tl_hat_us": tl_hat_us,
            "NF_logits": nf_logits,
            "slot_feat": slot_feat,
        }
        if self.ts_head is not None:
            h_ts = q[:, 2, :]
            ts_direct = self.softplus(self.ts_head(h_ts)).squeeze(1)
            ts_direct = torch.clamp(ts_direct, min=self.min_ts_us, max=self.max_ts_us)
            ts_direct = torch.maximum(ts_direct, tl_hat_us + 0.05)
            out["Ts_direct_us"] = ts_direct
        if self.ts_residual_head is not None:
            ts_residual = self.ts_residual_max_abs_us * torch.tanh(self.ts_residual_head(slot_feat)).squeeze(1)
            out["Ts_residual_us"] = ts_residual
        if active_logit is not None:
            out["NF_active_logit"] = active_logit
        if self.tl_periodic_head is not None:
            out["Tl_periodic_us"] = tl_periodic_us
            out["Tl_blend"] = tl_blend
        return out

