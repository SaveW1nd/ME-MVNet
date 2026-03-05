"""Parameter estimation network (PE-Net) for separated jammer sources."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.stft import iq_to_logmag_stft
from .blocks_1d import ConvBNAct1d, TCNResidualBlock
from .blocks_2d import ConvBNAct2d, Residual2dBlock
from .gateformer import GateFormer


class RawBranchPE(nn.Module):
    """Raw IQ branch with gate head."""

    def __init__(self, cfg: dict, n_samples: int) -> None:
        super().__init__()
        c1, c2 = [int(v) for v in cfg["stem_channels"]]
        n_layers = int(cfg["transformer_layers"])
        n_heads = int(cfg["transformer_heads"])
        dropout = float(cfg["dropout"])
        gate_hidden = int(cfg["gate_hidden_dim"])
        self.n_samples = int(n_samples)

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

        self.gate_head = nn.Sequential(
            nn.Conv1d(c2, gate_hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(gate_hidden, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.stem(x)
        feat = self.tcn(feat)

        tok = feat.transpose(1, 2)
        tok = self.transformer(tok)
        tok = self.post_norm(tok)
        seq_feat = tok.transpose(1, 2)  # (B,C,L)

        z_iq = seq_feat.mean(dim=-1)

        g_logit = self.gate_head(seq_feat)  # (B,1,L)
        g_logit = F.interpolate(g_logit, size=self.n_samples, mode="linear", align_corners=False).squeeze(1)
        g_hat = torch.sigmoid(g_logit)
        return {"z_iq": z_iq, "seq_feat": seq_feat, "g_logit": g_logit, "g_hat": g_hat}


class TFBranchPE(nn.Module):
    """TF branch from STFT log-mag."""

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


class MechanismBranchPE(nn.Module):
    """Mechanism branch: 32-dim stats/autocorr features."""

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
        # Force feature extraction in float32 for autocast stability.
        x = x.float()
        amp = torch.sqrt(torch.clamp(x[:, 0] ** 2 + x[:, 1] ** 2, min=1e-8))
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
        cen = amp - mean.unsqueeze(1)
        skew = (cen**3).mean(dim=1) / (std**3 + 1e-6)
        kurt = (cen**4).mean(dim=1) / (std**4 + 1e-6)
        d1 = torch.diff(amp, dim=1)
        d1_mean = d1.mean(dim=1)
        d1_std = d1.std(dim=1)
        d1_abs = d1.abs().mean(dim=1)

        stats = torch.stack(
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
                d1_abs,
            ],
            dim=1,
        )
        lags = [16, 24, 32, 48, 64, 96, 128, 160, 192, 224, 256, 320, 384, 448, 512, 640]
        ac = []
        power = (amp**2).mean(dim=1) + 1e-8
        for lag in lags:
            lag = min(lag, amp.shape[1] - 1)
            ac.append((amp[:, :-lag] * amp[:, lag:]).mean(dim=1) / power)
        ac = torch.stack(ac, dim=1)
        feats = torch.cat([stats, ac], dim=1)
        feats = torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        if feats.shape[1] != 32:
            raise RuntimeError(f"Expected 32 features, got {feats.shape[1]}")
        return feats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self._extract_features(x))


class PENet(nn.Module):
    """PE-Net shared across 3 separated jammer slots."""

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        m = cfg["model_pe"]
        self.n_samples = int(m["n_samples"])
        self.nf_values = [int(v) for v in m["nf_values"]]

        raw_cfg = dict(m["raw_branch"])
        raw_cfg["gate_hidden_dim"] = int(m["gate_head"]["hidden_dim"])
        self.raw = RawBranchPE(raw_cfg, n_samples=self.n_samples)
        self.tf = TFBranchPE(m["tf_branch"])
        self.mech = MechanismBranchPE(
            feature_dim=int(m["mech_branch"]["feature_dim"]),
            hidden_dim=int(m["mech_branch"]["hidden_dim"]),
        )

        raw_dim = int(m["raw_branch"]["stem_channels"][1])
        tf_dim = int(m["tf_branch"]["channels"][-1])
        mech_dim = int(m["mech_branch"]["hidden_dim"])
        self.gateformer = GateFormer(
            seq_dim=raw_dim,
            tf_dim=tf_dim,
            mech_dim=mech_dim,
            num_nf_classes=len(self.nf_values),
            cfg=dict(m.get("gateformer", {})),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 3 or x.shape[1] != 2:
            raise ValueError(f"Expected (B,2,N), got {tuple(x.shape)}")

        raw_out = self.raw(x)
        seq_feat = raw_out["seq_feat"]
        g_logit = raw_out["g_logit"]
        g_hat = raw_out["g_hat"]

        z_tf = self.tf(x)
        z_mech = self.mech(x)
        param_out = self.gateformer(
            seq_feat=seq_feat.transpose(1, 2),
            g_logit=g_logit,
            z_tf=z_tf,
            z_mech=z_mech,
        )
        tl_hat_us = param_out["Tl_hat_us"]
        nf_logits = param_out["NF_logits"]

        nf_vals = torch.tensor(self.nf_values, device=x.device, dtype=nf_logits.dtype)
        nf_prob = torch.softmax(nf_logits, dim=1)
        expected_nf = torch.sum(nf_prob * nf_vals.unsqueeze(0), dim=1)
        ts_hat_us = (expected_nf + 1.0) * tl_hat_us

        return {
            "g_logit": g_logit,
            "g_hat": g_hat,
            "Tl_hat_us": tl_hat_us,
            "NF_logits": nf_logits,
            "Ts_hat_us": ts_hat_us,
        }


class MVSepPE(nn.Module):
    """Combined model wrapper: SepNet + shared-weight PENet."""

    def __init__(self, sepnet: nn.Module, penet: PENet) -> None:
        super().__init__()
        self.sepnet = sepnet
        self.penet = penet

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        sep_out = self.sepnet(x)
        j_hat = sep_out["j_hat"]  # (B,3,2,N)

        g_logit_list = []
        g_hat_list = []
        tl_list = []
        ts_list = []
        nf_logit_list = []
        for k in range(3):
            pe = self.penet(j_hat[:, k, :, :])
            g_logit_list.append(pe["g_logit"])
            g_hat_list.append(pe["g_hat"])
            tl_list.append(pe["Tl_hat_us"])
            ts_list.append(pe["Ts_hat_us"])
            nf_logit_list.append(pe["NF_logits"])

        out = dict(sep_out)
        out["g_logit"] = torch.stack(g_logit_list, dim=1)  # (B,3,N)
        out["g_hat"] = torch.stack(g_hat_list, dim=1)  # (B,3,N)
        out["Tl_hat_us"] = torch.stack(tl_list, dim=1)  # (B,3)
        out["Ts_hat_us"] = torch.stack(ts_list, dim=1)  # (B,3)
        out["NF_logits"] = torch.stack(nf_logit_list, dim=1)  # (B,3,4)
        return out
