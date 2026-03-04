"""ME-MVNet implementation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.stft import iq_to_logmag_stft
from .blocks_1d import ConvBNAct1d, TCNResidualBlock
from .blocks_2d import ConvBNAct2d, Residual2dBlock
from .fusion import FusionMLP


class RawIQBranch(nn.Module):
    """Raw IQ branch: 1D CNN/TCN + Transformer + mask head."""

    def __init__(self, cfg: dict, n_samples: int) -> None:
        super().__init__()
        c1, c2 = [int(v) for v in cfg["stem_channels"]]
        n_layers = int(cfg["transformer_layers"])
        n_heads = int(cfg["transformer_heads"])
        dropout = float(cfg["dropout"])

        self.stem = nn.Sequential(
            ConvBNAct1d(2, c1, k=7, s=2),
            ConvBNAct1d(c1, c2, k=5, s=2),
        )
        self.tcn = nn.Sequential(
            TCNResidualBlock(c2, dilation=1, dropout=dropout),
            TCNResidualBlock(c2, dilation=2, dropout=dropout),
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=c2,
            nhead=n_heads,
            dim_feedforward=c2 * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.post_norm = nn.LayerNorm(c2)

        self.mask_head = nn.Sequential(
            nn.Conv1d(c2, int(cfg.get("mask_hidden_dim", 64)), kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(int(cfg.get("mask_hidden_dim", 64)), 1, kernel_size=1),
        )
        self.n_samples = int(n_samples)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self.stem(x)  # (B,C,L)
        feat = self.tcn(feat)

        tokens = feat.transpose(1, 2)  # (B,L,C)
        tokens = self.transformer(tokens)
        tokens = self.post_norm(tokens)
        seq_feat = tokens.transpose(1, 2)  # (B,C,L)

        emb = seq_feat.mean(dim=-1)  # (B,C)

        mask_logits = self.mask_head(seq_feat)  # (B,1,L)
        mask_logits = F.interpolate(
            mask_logits, size=self.n_samples, mode="linear", align_corners=False
        ).squeeze(1)
        mask_hat = torch.sigmoid(mask_logits)
        return emb, mask_hat, mask_logits


class TFBranch(nn.Module):
    """STFT + 2D CNN encoder."""

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.n_fft = int(cfg["n_fft"])
        self.hop_length = int(cfg["hop_length"])
        self.win_length = int(cfg["win_length"])
        c1, c2, c3 = [int(v) for v in cfg["channels"]]

        self.encoder = nn.Sequential(
            ConvBNAct2d(1, c1, k=3, s=1),
            Residual2dBlock(c1, c2, stride=2),
            Residual2dBlock(c2, c3, stride=2),
            Residual2dBlock(c3, c3, stride=1),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tf = iq_to_logmag_stft(
            x=x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
        )
        feat = self.encoder(tf)
        return self.pool(feat).flatten(1)


class MechanismBranch(nn.Module):
    """Mechanism feature branch from |x| periodic/statistics features."""

    def __init__(self, feature_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
        )

    @staticmethod
    def _extract_features(x: torch.Tensor) -> torch.Tensor:
        """Extract fixed 32-dim feature vector from |x|."""
        x = x.float()
        amp = torch.sqrt(torch.clamp(x[:, 0] ** 2 + x[:, 1] ** 2, min=1e-8))  # (B,N)
        bsz = amp.shape[0]

        mean = amp.mean(dim=1)
        std = amp.std(dim=1) + 1e-6
        max_v = amp.max(dim=1).values
        min_v = amp.min(dim=1).values
        median = amp.median(dim=1).values
        q10 = torch.quantile(amp, 0.10, dim=1)
        q25 = torch.quantile(amp, 0.25, dim=1)
        q75 = torch.quantile(amp, 0.75, dim=1)
        q90 = torch.quantile(amp, 0.90, dim=1)
        rms = torch.sqrt((amp**2).mean(dim=1) + 1e-8)
        crest = max_v / (rms + 1e-6)
        centered = amp - mean.unsqueeze(1)
        skew = (centered**3).mean(dim=1) / (std**3 + 1e-6)
        kurt = (centered**4).mean(dim=1) / (std**4 + 1e-6)
        d1 = torch.diff(amp, dim=1)
        d1_mean = d1.mean(dim=1)
        d1_std = d1.std(dim=1)
        d1_abs_mean = d1.abs().mean(dim=1)

        stat_feats = torch.stack(
            [
                mean,
                std,
                max_v,
                min_v,
                median,
                q10,
                q25,
                q75,
                q90,
                rms,
                crest,
                skew,
                kurt,
                d1_mean,
                d1_std,
                d1_abs_mean,
            ],
            dim=1,
        )

        lags = [16, 24, 32, 48, 64, 96, 128, 160, 192, 224, 256, 320, 384, 448, 512, 640]
        ac_feats = []
        power = (amp**2).mean(dim=1) + 1e-8
        for lag in lags:
            lag = min(lag, amp.shape[1] - 1)
            ac = (amp[:, :-lag] * amp[:, lag:]).mean(dim=1) / power
            ac_feats.append(ac)
        ac_feats = torch.stack(ac_feats, dim=1)

        feats = torch.cat([stat_feats, ac_feats], dim=1)
        feats = torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        if feats.shape[1] != 32:
            raise RuntimeError(f"Expected 32 mechanism features, got {feats.shape[1]}")
        return feats.reshape(bsz, 32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self._extract_features(x)
        return self.mlp(feats)


class MEMVNet(nn.Module):
    """ME-MVNet with multi-view encoders and multi-head outputs."""

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        model_cfg = cfg["model"]
        self.n_samples = int(model_cfg["n_samples"])
        self.nf_values = [int(v) for v in model_cfg["nf_values"]]

        raw_cfg = dict(model_cfg["raw_branch"])
        raw_cfg["mask_hidden_dim"] = int(model_cfg["mask_head"]["hidden_dim"])
        self.raw_branch = RawIQBranch(raw_cfg, n_samples=self.n_samples)
        self.tf_branch = TFBranch(model_cfg["tf_branch"])
        self.mech_branch = MechanismBranch(
            feature_dim=int(model_cfg["mech_branch"]["feature_dim"]),
            hidden_dim=int(model_cfg["mech_branch"]["hidden_dim"]),
        )

        raw_dim = int(model_cfg["raw_branch"]["stem_channels"][1])
        tf_dim = int(model_cfg["tf_branch"]["channels"][-1])
        mech_dim = int(model_cfg["mech_branch"]["hidden_dim"])
        fused_in = raw_dim + tf_dim + mech_dim
        fused_hidden = int(model_cfg["fusion"]["hidden_dim"])

        self.fusion = FusionMLP(in_dim=fused_in, hidden_dim=fused_hidden)
        self.tl_head = nn.Linear(fused_hidden, 1)
        self.tf_head = nn.Linear(fused_hidden, 1)
        self.nf_head = nn.Linear(fused_hidden, len(self.nf_values))
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 3 or x.shape[1] != 2:
            raise ValueError(f"Expected input shape (B,2,N), got {tuple(x.shape)}")

        raw_emb, mask_hat, mask_logits = self.raw_branch(x)
        tf_emb = self.tf_branch(x)
        mech_emb = self.mech_branch(x)

        fused = self.fusion(torch.cat([raw_emb, tf_emb, mech_emb], dim=1))
        tl_hat = self.softplus(self.tl_head(fused)).squeeze(1)
        tf_hat = self.softplus(self.tf_head(fused)).squeeze(1)
        nf_logits = self.nf_head(fused)

        return {
            "Tl_hat": tl_hat,
            "Tf_hat": tf_hat,
            "NF_logits": nf_logits,
            "mask_hat": mask_hat,
            "mask_logits": mask_logits,
        }
