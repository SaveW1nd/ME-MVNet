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


class PeriodicBranchPE(nn.Module):
    """Periodic evidence branch used as one extra conditioning feature."""

    def __init__(self, cfg: dict, n_samples: int) -> None:
        super().__init__()
        self.n_samples = int(n_samples)
        self.max_lag = int(cfg.get("max_lag", 1400))
        self.max_lag = min(max(self.max_lag, 64), self.n_samples - 1)
        out_dim = int(cfg.get("out_dim", 64))
        c1 = int(cfg.get("conv_channels_1", 32))
        c2 = int(cfg.get("conv_channels_2", 64))

        self.conv = nn.Sequential(
            ConvBNAct1d(2, c1, k=7, s=2),
            ConvBNAct1d(c1, c2, k=5, s=2),
            nn.Conv1d(c2, out_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )

    @staticmethod
    def _next_pow2(v: int) -> int:
        out = 1
        while out < v:
            out <<= 1
        return out

    def _autocorr_norm(self, sig: torch.Tensor) -> torch.Tensor:
        """Normalized autocorrelation for (B,N) -> (B,max_lag+1)."""
        bsz, n = sig.shape
        x = sig.float()
        x = x - x.mean(dim=1, keepdim=True)
        fft_len = self._next_pow2(2 * n)
        spec = torch.fft.rfft(x, n=fft_len, dim=1)
        ac = torch.fft.irfft(spec * torch.conj(spec), n=fft_len, dim=1)[:, : self.max_lag + 1]
        denom = torch.clamp(ac[:, :1], min=1e-6)
        ac = ac / denom
        return torch.nan_to_num(ac, nan=0.0, posinf=0.0, neginf=0.0)

    def forward(self, x: torch.Tensor, g_logit: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != 2:
            raise ValueError(f"Expected x (B,2,N), got {tuple(x.shape)}")
        if g_logit.ndim != 2:
            raise ValueError(f"Expected g_logit (B,N), got {tuple(g_logit.shape)}")

        amp = torch.sqrt(torch.clamp(x[:, 0] * x[:, 0] + x[:, 1] * x[:, 1], min=1e-8))
        g = torch.sigmoid(g_logit)
        weighted = amp * g

        ac_gate = self._autocorr_norm(g)
        ac_weighted = self._autocorr_norm(weighted)
        ac_pair = torch.stack([ac_gate[:, 1:], ac_weighted[:, 1:]], dim=1)  # (B,2,L-1), drop lag=0
        return self.conv(ac_pair).squeeze(-1)


class SlotPeriodicBranchPE(nn.Module):
    """Encode separator-provided slot-periodic lag weights."""

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.input_dim = int(cfg.get("input_dim", 10))
        self.out_dim = int(cfg.get("out_dim", 32))
        self.hidden_dim = int(cfg.get("hidden_dim", max(32, self.out_dim * 2)))
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.out_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"Expected slot periodic weights (B,L), got {tuple(x.shape)}")
        if int(x.shape[1]) != self.input_dim:
            raise ValueError(f"Expected slot periodic input dim {self.input_dim}, got {int(x.shape[1])}")
        x = x.float()
        x = x / torch.clamp(x.sum(dim=1, keepdim=True), min=1e-6)
        return self.mlp(x)


class RawLiteBranchPE(nn.Module):
    """Lightweight raw branch: local conv + TCN, no transformer."""

    def __init__(self, cfg: dict, n_samples: int) -> None:
        super().__init__()
        c1, c2 = [int(v) for v in cfg["stem_channels"]]
        dropout = float(cfg.get("dropout", 0.1))
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
        self.proj = nn.Sequential(
            nn.Conv1d(c2, c2, kernel_size=1),
            nn.GELU(),
        )
        self.gate_head = nn.Sequential(
            nn.Conv1d(c2, gate_hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(gate_hidden, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.proj(self.tcn(self.stem(x)))
        z_raw = feat.mean(dim=-1)
        g_logit = self.gate_head(feat)
        g_logit = F.interpolate(g_logit, size=self.n_samples, mode="linear", align_corners=False).squeeze(1)
        g_hat = torch.sigmoid(g_logit)
        return {"z_raw": z_raw, "g_logit": g_logit, "g_hat": g_hat}


class GateStatsBranchPE(nn.Module):
    """Explicit gate/period statistics branch."""

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.lags = [int(v) for v in cfg.get("lags", [16, 24, 32, 48, 64, 96, 128, 160, 192, 224, 256, 320])]
        self.hidden_dim = int(cfg.get("hidden_dim", 96))
        in_dim = 6 + 2 * len(self.lags)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
        )

    @staticmethod
    def _safe_ac(sig: torch.Tensor, lag: int) -> torch.Tensor:
        lag = min(max(int(lag), 1), sig.shape[1] - 1)
        num = (sig[:, :-lag] * sig[:, lag:]).mean(dim=1)
        den = torch.sqrt(
            torch.clamp((sig[:, :-lag] * sig[:, :-lag]).mean(dim=1), min=1e-8)
            * torch.clamp((sig[:, lag:] * sig[:, lag:]).mean(dim=1), min=1e-8)
        )
        return num / torch.clamp(den, min=1e-6)

    def forward(self, x: torch.Tensor, g_logit: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != 2:
            raise ValueError(f"Expected x (B,2,N), got {tuple(x.shape)}")
        if g_logit.ndim != 2:
            raise ValueError(f"Expected g_logit (B,N), got {tuple(g_logit.shape)}")

        x = x.float()
        g = torch.sigmoid(g_logit.float())
        amp = torch.sqrt(torch.clamp(x[:, 0] * x[:, 0] + x[:, 1] * x[:, 1], min=1e-8))
        edge = F.pad(torch.abs(torch.diff(g, dim=1)), (1, 0))
        gated_amp = amp * g

        stats = [
            g.mean(dim=1),
            g.std(dim=1),
            edge.mean(dim=1),
            edge.std(dim=1),
            gated_amp.mean(dim=1),
            gated_amp.std(dim=1),
        ]
        g0 = g - g.mean(dim=1, keepdim=True)
        ga0 = gated_amp - gated_amp.mean(dim=1, keepdim=True)
        for lag in self.lags:
            stats.append(self._safe_ac(g0, lag))
            stats.append(self._safe_ac(ga0, lag))

        feat = torch.stack(stats, dim=1)
        feat = torch.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
        return self.mlp(feat)


class SepGateBranchPE(nn.Module):
    """Encode separator-provided gate occupancy and boundary evidence."""

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        c1 = int(cfg.get("conv_channels_1", 32))
        c2 = int(cfg.get("conv_channels_2", 64))
        out_dim = int(cfg.get("out_dim", 32))
        self.out_dim = out_dim
        self.encoder = nn.Sequential(
            ConvBNAct1d(2, c1, k=7, s=2),
            TCNResidualBlock(c1, dilation=1, dropout=0.1),
            ConvBNAct1d(c1, c2, k=5, s=2),
            TCNResidualBlock(c2, dilation=2, dropout=0.1),
            nn.Conv1d(c2, out_dim, kernel_size=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, g_hint: torch.Tensor) -> torch.Tensor:
        if g_hint.ndim != 2:
            raise ValueError(f"Expected g_hint (B,N), got {tuple(g_hint.shape)}")
        g = torch.clamp(g_hint.float(), 0.0, 1.0)
        edge = F.pad(torch.abs(torch.diff(g, dim=1)), (1, 0))
        feat = torch.stack([g, edge], dim=1)
        return self.encoder(feat).squeeze(-1)


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
        tf_cfg = dict(m.get("tf_branch", {}))
        self.use_tf = bool(tf_cfg.get("enabled", True))
        tf_hidden_dim = int(tf_cfg.get("channels", [0])[-1]) if self.use_tf else 0
        if self.use_tf and tf_hidden_dim > 0:
            self.tf = TFBranchPE(tf_cfg)
        else:
            self.tf = None
            tf_hidden_dim = 0
        mech_cfg = dict(m.get("mech_branch", {}))
        self.use_mech = bool(mech_cfg.get("enabled", True))
        mech_hidden_dim = int(mech_cfg.get("hidden_dim", 0)) if self.use_mech else 0
        if self.use_mech and mech_hidden_dim > 0:
            self.mech = MechanismBranchPE(
                feature_dim=int(mech_cfg["feature_dim"]),
                hidden_dim=mech_hidden_dim,
            )
        else:
            self.mech = None
            mech_hidden_dim = 0
        periodic_cfg = dict(m.get("periodic_branch", {}))
        self.use_periodic = bool(periodic_cfg.get("enabled", False))
        self.periodic_detach_gate_logit = bool(periodic_cfg.get("detach_gate_logit", True))
        periodic_dim = int(periodic_cfg.get("out_dim", 0)) if self.use_periodic else 0
        if self.use_periodic and periodic_dim > 0:
            self.periodic = PeriodicBranchPE(periodic_cfg, n_samples=self.n_samples)
        else:
            self.periodic = None
            periodic_dim = 0
        slot_periodic_cfg = dict(m.get("slot_periodic_branch", {}))
        self.use_slot_periodic = bool(slot_periodic_cfg.get("enabled", False))
        slot_periodic_dim = int(slot_periodic_cfg.get("out_dim", 0)) if self.use_slot_periodic else 0
        if self.use_slot_periodic and slot_periodic_dim > 0:
            self.slot_periodic = SlotPeriodicBranchPE(slot_periodic_cfg)
        else:
            self.slot_periodic = None
            slot_periodic_dim = 0
        sep_gate_branch_cfg = dict(m.get("sep_gate_branch", {}))
        self.use_sep_gate_branch = bool(sep_gate_branch_cfg.get("enabled", False))
        sep_gate_dim = int(sep_gate_branch_cfg.get("out_dim", 0)) if self.use_sep_gate_branch else 0
        if self.use_sep_gate_branch and sep_gate_dim > 0:
            self.sep_gate_branch = SepGateBranchPE(sep_gate_branch_cfg)
        else:
            self.sep_gate_branch = None
            sep_gate_dim = 0

        raw_dim = int(m["raw_branch"]["stem_channels"][1])
        tf_dim = tf_hidden_dim
        mech_dim = mech_hidden_dim
        self.gateformer = GateFormer(
            seq_dim=raw_dim,
            tf_dim=tf_dim,
            mech_dim=mech_dim,
            periodic_dim=periodic_dim + slot_periodic_dim + sep_gate_dim,
            num_nf_classes=len(self.nf_values),
            cfg=dict(m.get("gateformer", {})),
        )
        self.slot_feat_dim = int(self.gateformer.d_model) * 2
        slot_ref_cfg = dict(m.get("slot_refiner", {}))
        self.slot_refiner_enabled = bool(slot_ref_cfg.get("enabled", False))
        self.slot_refiner_d_model = int(slot_ref_cfg.get("d_model", 128))
        self.slot_refiner_num_heads = int(slot_ref_cfg.get("num_heads", 4))
        self.slot_refiner_num_layers = int(slot_ref_cfg.get("num_layers", 1))
        self.slot_refiner_dropout = float(slot_ref_cfg.get("dropout", 0.1))
        self.slot_refiner_nf_delta_scale = float(slot_ref_cfg.get("nf_delta_scale", 0.6))
        self.slot_refiner_tl_delta_scale = float(slot_ref_cfg.get("tl_delta_scale", 0.20))
        self.slot_refiner_min_tl_us = float(slot_ref_cfg.get("min_tl_us", 0.2))
        self.slot_refiner_max_tl_us = float(slot_ref_cfg.get("max_tl_us", 3.5))
        slot_set_cfg = dict(m.get("slot_set_decoder", {}))
        self.slot_set_decoder_enabled = bool(slot_set_cfg.get("enabled", False))
        self.slot_set_decoder_d_model = int(slot_set_cfg.get("d_model", 160))
        self.slot_set_decoder_num_heads = int(slot_set_cfg.get("num_heads", 4))
        self.slot_set_decoder_num_layers = int(slot_set_cfg.get("num_layers", 2))
        self.slot_set_decoder_dropout = float(slot_set_cfg.get("dropout", 0.1))
        self.slot_set_decoder_head_hidden_dim = int(slot_set_cfg.get("head_hidden_dim", 128))
        self.slot_set_decoder_min_tl_us = float(slot_set_cfg.get("min_tl_us", 0.2))
        self.slot_set_decoder_max_tl_us = float(slot_set_cfg.get("max_tl_us", 3.5))

        kactive_cfg = dict(m.get("kactive_head", {}))
        self.kactive_head_enabled = bool(kactive_cfg.get("enabled", False))
        self.kactive_hidden_dim = int(kactive_cfg.get("hidden_dim", 128))
        self.kactive_logit_scale = float(kactive_cfg.get("logit_scale", 1.0))
        ts_blend_cfg = dict(m.get("ts_blend", {}))
        self.ts_direct_enabled = bool(getattr(self.gateformer, "ts_direct_enabled", False))
        self.ts_direct_alpha = float(ts_blend_cfg.get("alpha", 0.5))
        self.ts_direct_alpha = min(max(self.ts_direct_alpha, 0.0), 1.0)
        self.ts_min_margin_us = float(ts_blend_cfg.get("min_margin_us", 0.05))
        self.ts_blend_learnable = bool(ts_blend_cfg.get("learnable", False))
        anchor_cfg = dict(m.get("periodic_anchor_refiner", {}))
        self.periodic_anchor_enabled = bool(anchor_cfg.get("enabled", False))
        self.periodic_anchor_lag_unit_us = float(anchor_cfg.get("lag_unit_us", 0.04))
        self.periodic_anchor_hidden_dim = int(anchor_cfg.get("hidden_dim", 128))
        self.periodic_anchor_nf_bias_scale = float(anchor_cfg.get("nf_bias_scale", 1.0))
        self.periodic_anchor_nf_bias_tau_us = float(anchor_cfg.get("nf_bias_tau_us", 0.35))
        self.periodic_anchor_min_tl_us = float(anchor_cfg.get("min_tl_us", 0.2))
        self.periodic_anchor_max_tl_us = float(anchor_cfg.get("max_tl_us", 3.5))
        gate_hint_cfg = dict(m.get("sep_gate_hint", {}))
        self.sep_gate_hint_enabled = bool(gate_hint_cfg.get("enabled", False))
        self.sep_gate_hint_alpha = float(gate_hint_cfg.get("alpha", 0.35))
        if self.ts_direct_enabled and self.ts_blend_learnable:
            hid = max(32, int(self.slot_feat_dim // 4))
            self.ts_blend_head = nn.Sequential(
                nn.Linear(int(self.slot_feat_dim), hid),
                nn.GELU(),
                nn.Linear(hid, 1),
            )
        else:
            self.ts_blend_head = None

    def forward(
        self,
        x: torch.Tensor,
        z_slot_periodic_input: torch.Tensor | None = None,
        g_hint: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if x.ndim != 3 or x.shape[1] != 2:
            raise ValueError(f"Expected (B,2,N), got {tuple(x.shape)}")

        raw_out = self.raw(x)
        seq_feat = raw_out["seq_feat"]
        g_logit = raw_out["g_logit"]
        if self.sep_gate_hint_enabled and g_hint is not None:
            g_hint = g_hint.to(device=g_logit.device, dtype=g_logit.dtype)
            g_hint = torch.clamp(g_hint, min=1e-4, max=1.0 - 1e-4)
            g_hint_logit = torch.log(g_hint) - torch.log1p(-g_hint)
            alpha = min(max(self.sep_gate_hint_alpha, 0.0), 1.0)
            g_logit = (1.0 - alpha) * g_logit + alpha * g_hint_logit
        g_hat = torch.sigmoid(g_logit)

        if self.tf is not None:
            z_tf = self.tf(x)
        else:
            z_tf = g_logit.new_zeros((g_logit.shape[0], 0))
        if self.mech is not None:
            z_mech = self.mech(x)
        else:
            z_mech = z_tf.new_zeros((z_tf.shape[0], 0))
        if self.periodic is not None:
            g_for_periodic = g_logit.detach() if self.periodic_detach_gate_logit else g_logit
            z_periodic = self.periodic(x=x, g_logit=g_for_periodic)
        else:
            z_periodic = None
        if self.slot_periodic is not None and z_slot_periodic_input is not None:
            z_slot_periodic = self.slot_periodic(z_slot_periodic_input)
        elif self.slot_periodic is not None:
            z_slot_periodic = z_mech.new_zeros((z_mech.shape[0], self.slot_periodic.out_dim))
        else:
            z_slot_periodic = None
        if self.sep_gate_branch is not None and g_hint is not None:
            z_sep_gate = self.sep_gate_branch(g_hint)
        elif self.sep_gate_branch is not None:
            z_sep_gate = z_mech.new_zeros((z_mech.shape[0], self.sep_gate_branch.out_dim))
        else:
            z_sep_gate = None

        z_periodic_parts = [z for z in [z_periodic, z_slot_periodic, z_sep_gate] if z is not None]
        z_periodic_all = torch.cat(z_periodic_parts, dim=1) if z_periodic_parts else None
        param_out = self.gateformer(
            seq_feat=seq_feat.transpose(1, 2),
            g_logit=g_logit,
            z_tf=z_tf,
            z_mech=z_mech,
            z_periodic=z_periodic_all,
        )
        tl_hat_us = param_out["Tl_hat_us"]
        nf_logits = param_out["NF_logits"]

        nf_vals = torch.tensor(self.nf_values, device=x.device, dtype=nf_logits.dtype)
        nf_prob = torch.softmax(nf_logits, dim=1)
        expected_nf = torch.sum(nf_prob * nf_vals.unsqueeze(0), dim=1)
        ts_struct_us = (expected_nf + 1.0) * tl_hat_us
        ts_hat_us = ts_struct_us
        ts_direct_us = None
        ts_residual_us = None
        ts_blend = None
        if "Ts_residual_us" in param_out:
            ts_residual_us = param_out["Ts_residual_us"]
            ts_hat_us = ts_struct_us + ts_residual_us
            ts_hat_us = torch.maximum(ts_hat_us, torch.full_like(ts_hat_us, self.gateformer.min_ts_us))
            ts_hat_us = torch.maximum(ts_hat_us, tl_hat_us + self.ts_min_margin_us)
            ts_hat_us = torch.clamp(ts_hat_us, max=self.gateformer.max_ts_us)
        if "Ts_direct_us" in param_out:
            ts_direct_us = param_out["Ts_direct_us"]
            if self.ts_blend_head is not None:
                ts_blend = torch.sigmoid(self.ts_blend_head(param_out["slot_feat"]).squeeze(1))
            else:
                ts_blend = torch.full_like(ts_struct_us, self.ts_direct_alpha)
            ts_hat_us = ts_blend * ts_direct_us + (1.0 - ts_blend) * ts_struct_us
            ts_hat_us = torch.maximum(ts_hat_us, tl_hat_us + self.ts_min_margin_us)

        out = {
            "g_logit": g_logit,
            "g_hat": g_hat,
            "Tl_hat_us": tl_hat_us,
            "NF_logits": nf_logits,
            "Ts_hat_us": ts_hat_us,
            "Ts_struct_us": ts_struct_us,
            "slot_feat": param_out["slot_feat"],
            "NF_active_logit": param_out["NF_active_logit"] if "NF_active_logit" in param_out else None,
        }
        if ts_direct_us is not None:
            out["Ts_hat_direct_us"] = ts_direct_us
        if ts_residual_us is not None:
            out["Ts_residual_us"] = ts_residual_us
        if ts_blend is not None:
            out["Ts_blend"] = ts_blend
        return out


class GateStatPE(nn.Module):
    """Simplified PE: gate + periodic statistics + lightweight MLP."""

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        m = cfg["model_pe"]
        self.n_samples = int(m["n_samples"])
        self.nf_values = [int(v) for v in m["nf_values"]]

        raw_cfg = dict(m["raw_branch"])
        raw_cfg["gate_hidden_dim"] = int(m["gate_head"]["hidden_dim"])
        self.raw = RawLiteBranchPE(raw_cfg, n_samples=self.n_samples)
        self.mech = MechanismBranchPE(
            feature_dim=int(m["mech_branch"]["feature_dim"]),
            hidden_dim=int(m["mech_branch"]["hidden_dim"]),
        )
        gate_stats_cfg = dict(m.get("gate_stats_branch", {}))
        self.gate_stats = GateStatsBranchPE(gate_stats_cfg)

        hidden_dim = int(m.get("simple_head", {}).get("hidden_dim", 160))
        raw_dim = int(m["raw_branch"]["stem_channels"][1])
        mech_dim = int(m["mech_branch"]["hidden_dim"])
        gate_dim = int(self.gate_stats.hidden_dim)
        self.fusion = nn.Sequential(
            nn.Linear(raw_dim + mech_dim + gate_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(m.get("simple_head", {}).get("dropout", 0.1))),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.slot_feat_dim = hidden_dim
        self.tl_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.nf_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, len(self.nf_values)),
        )
        self.softplus = nn.Softplus()

        gateformer_cfg = dict(m.get("gateformer", {}))
        self.min_tl_us = float(gateformer_cfg.get("min_tl_us", 0.2))
        self.max_tl_us = float(gateformer_cfg.get("max_tl_us", 3.5))
        self.ts_min_margin_us = float(m.get("ts_blend", {}).get("min_margin_us", 0.05))

        self.slot_refiner_enabled = False
        self.slot_set_decoder_enabled = False
        self.kactive_head_enabled = False
        self.periodic_anchor_enabled = False
        self.ts_direct_enabled = False

    def forward(
        self,
        x: torch.Tensor,
        z_slot_periodic_input: torch.Tensor | None = None,
        g_hint: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        del z_slot_periodic_input, g_hint
        raw_out = self.raw(x)
        z_raw = raw_out["z_raw"]
        g_logit = raw_out["g_logit"]
        g_hat = raw_out["g_hat"]
        z_mech = self.mech(x)
        z_gate = self.gate_stats(x=x, g_logit=g_logit)
        slot_feat = self.fusion(torch.cat([z_raw, z_mech, z_gate], dim=1))
        tl_hat_us = torch.clamp(self.softplus(self.tl_head(slot_feat)).squeeze(1), min=self.min_tl_us, max=self.max_tl_us)
        nf_logits = self.nf_head(slot_feat)
        return {
            "g_logit": g_logit,
            "g_hat": g_hat,
            "Tl_hat_us": tl_hat_us,
            "NF_logits": nf_logits,
            "slot_feat": slot_feat,
        }


class MVSepPE(nn.Module):
    """Combined model wrapper: SepNet + shared-weight PENet."""

    def __init__(self, sepnet: nn.Module, penet: nn.Module) -> None:
        super().__init__()
        self.sepnet = sepnet
        self.penet = penet
        self.register_buffer(
            "nf_values_tensor",
            torch.tensor([float(v) for v in self.penet.nf_values], dtype=torch.float32),
            persistent=False,
        )

        if bool(getattr(self.penet, "slot_refiner_enabled", False)):
            d_model = int(self.penet.slot_refiner_d_model)
            self.slot_refiner_in = nn.Linear(int(self.penet.slot_feat_dim), d_model)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=int(self.penet.slot_refiner_num_heads),
                dim_feedforward=d_model * 2,
                dropout=float(self.penet.slot_refiner_dropout),
                batch_first=True,
                norm_first=True,
                activation="gelu",
            )
            self.slot_refiner = nn.TransformerEncoder(
                enc_layer,
                num_layers=int(self.penet.slot_refiner_num_layers),
            )
            self.slot_refiner_norm = nn.LayerNorm(d_model)
            self.slot_nf_delta_head = nn.Linear(d_model, len(self.penet.nf_values))
            self.slot_tl_delta_head = nn.Linear(d_model, 1)
            self.slot_nf_delta_scale = float(self.penet.slot_refiner_nf_delta_scale)
            self.slot_tl_delta_scale = float(self.penet.slot_refiner_tl_delta_scale)
            self.slot_tl_min = float(self.penet.slot_refiner_min_tl_us)
            self.slot_tl_max = float(self.penet.slot_refiner_max_tl_us)
        else:
            self.slot_refiner_in = None
            self.slot_refiner = None
            self.slot_refiner_norm = None
            self.slot_nf_delta_head = None
            self.slot_tl_delta_head = None

        if bool(getattr(self.penet, "slot_set_decoder_enabled", False)):
            nf_dim = len(self.penet.nf_values)
            desc_dim = int(self.penet.slot_feat_dim) + nf_dim + 5
            d_model = int(self.penet.slot_set_decoder_d_model)
            head_hidden = int(self.penet.slot_set_decoder_head_hidden_dim)
            self.slot_set_in = nn.Linear(desc_dim, d_model)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=int(self.penet.slot_set_decoder_num_heads),
                dim_feedforward=d_model * 2,
                dropout=float(self.penet.slot_set_decoder_dropout),
                batch_first=True,
                norm_first=True,
                activation="gelu",
            )
            self.slot_set_encoder = nn.TransformerEncoder(
                enc_layer,
                num_layers=int(self.penet.slot_set_decoder_num_layers),
            )
            self.slot_set_norm = nn.LayerNorm(d_model)
            self.slot_set_nf_head = nn.Sequential(
                nn.Linear(d_model, head_hidden),
                nn.GELU(),
                nn.Linear(head_hidden, nf_dim),
            )
            self.slot_set_tl_head = nn.Sequential(
                nn.Linear(d_model, head_hidden),
                nn.GELU(),
                nn.Linear(head_hidden, 1),
            )
            self.slot_set_nf_blend_head = nn.Linear(d_model, 1)
            self.slot_set_tl_blend_head = nn.Linear(d_model, 1)
            self.slot_set_softplus = nn.Softplus()
            self.slot_set_tl_min = float(self.penet.slot_set_decoder_min_tl_us)
            self.slot_set_tl_max = float(self.penet.slot_set_decoder_max_tl_us)
        else:
            self.slot_set_in = None
            self.slot_set_encoder = None
            self.slot_set_norm = None
            self.slot_set_nf_head = None
            self.slot_set_tl_head = None
            self.slot_set_nf_blend_head = None
            self.slot_set_tl_blend_head = None
            self.slot_set_softplus = None

        if bool(getattr(self.penet, "kactive_head_enabled", False)):
            in_dim = int(self.penet.slot_feat_dim) * 3 + 3
            hid = int(self.penet.kactive_hidden_dim)
            self.kactive_head = nn.Sequential(
                nn.Linear(in_dim, hid),
                nn.GELU(),
                nn.Linear(hid, 1),
            )
            self.kactive_logit_scale = float(self.penet.kactive_logit_scale)
        else:
            self.kactive_head = None

        if bool(getattr(self.penet, "periodic_anchor_enabled", False)):
            lag_list = [float(v) for v in getattr(self.sepnet, "periodic_lags", [])]
            if not lag_list:
                raise ValueError("periodic_anchor_refiner requires separator.periodic_lags")
            lag_unit_us = float(getattr(self.penet, "periodic_anchor_lag_unit_us", 0.04))
            lag_us = [v * lag_unit_us for v in lag_list]
            self.register_buffer(
                "periodic_anchor_lag_us",
                torch.tensor(lag_us, dtype=torch.float32),
                persistent=False,
            )
            desc_dim = int(self.penet.slot_feat_dim) + len(lag_list) + len(self.penet.nf_values) + 2
            hid = int(self.penet.periodic_anchor_hidden_dim)
            self.anchor_tl_blend_head = nn.Sequential(
                nn.Linear(desc_dim, hid),
                nn.GELU(),
                nn.Linear(hid, 1),
            )
            self.anchor_ts_blend_head = nn.Sequential(
                nn.Linear(desc_dim, hid),
                nn.GELU(),
                nn.Linear(hid, 1),
            )
            self.anchor_nf_bias_scale = float(self.penet.periodic_anchor_nf_bias_scale)
            self.anchor_nf_bias_tau_us = float(self.penet.periodic_anchor_nf_bias_tau_us)
            self.anchor_tl_min = float(self.penet.periodic_anchor_min_tl_us)
            self.anchor_tl_max = float(self.penet.periodic_anchor_max_tl_us)
        else:
            self.anchor_tl_blend_head = None
            self.anchor_ts_blend_head = None

    def _recompute_ts(
        self,
        *,
        tl_hat_us: torch.Tensor,
        nf_logits: torch.Tensor,
        ts_direct_us: torch.Tensor | None = None,
        ts_blend: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        nf_vals = self.nf_values_tensor.to(device=nf_logits.device, dtype=nf_logits.dtype).view(1, 1, -1)
        nf_prob = torch.softmax(nf_logits, dim=-1)
        ts_struct_us = (torch.sum(nf_prob * nf_vals, dim=-1) + 1.0) * tl_hat_us
        if ts_direct_us is None:
            return ts_struct_us, ts_struct_us

        if ts_blend is None:
            alpha = float(getattr(self.penet, "ts_direct_alpha", 0.5))
            ts_blend = torch.full_like(ts_struct_us, alpha)
        ts_hat_us = ts_blend * ts_direct_us + (1.0 - ts_blend) * ts_struct_us
        margin = float(getattr(self.penet, "ts_min_margin_us", 0.05))
        ts_hat_us = torch.maximum(ts_hat_us, tl_hat_us + margin)
        return ts_hat_us, ts_struct_us

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        sep_out = self.sepnet(x)
        j_hat = sep_out["j_hat"]  # (B,3,2,N)

        g_logit_list = []
        g_hat_list = []
        tl_list = []
        nf_logit_list = []
        slot_feat_list = []
        nf_active_logit_list = []
        ts_direct_list = []
        ts_blend_list = []
        for k in range(3):
            if sep_out.get("slot_periodic_weights", None) is not None and sep_out["slot_periodic_weights"].shape[1] > k:
                z_slot_periodic = sep_out["slot_periodic_weights"][:, k, :]
            else:
                z_slot_periodic = None
            if sep_out.get("masks", None) is not None and sep_out["masks"].shape[1] > k:
                mask_k = sep_out["masks"][:, k, ...]
                if mask_k.ndim == 3:
                    g_hint = mask_k.mean(dim=1)
                else:
                    g_hint = mask_k
                g_hint = F.interpolate(g_hint.unsqueeze(1), size=x.shape[-1], mode="linear", align_corners=False).squeeze(1)
            else:
                g_hint = None
            pe = self.penet(j_hat[:, k, :, :], z_slot_periodic_input=z_slot_periodic, g_hint=g_hint)
            g_logit_list.append(pe["g_logit"])
            g_hat_list.append(pe["g_hat"])
            tl_list.append(pe["Tl_hat_us"])
            nf_logit_list.append(pe["NF_logits"])
            slot_feat_list.append(pe["slot_feat"])
            if pe.get("NF_active_logit", None) is not None:
                nf_active_logit_list.append(pe["NF_active_logit"])
            if pe.get("Ts_hat_direct_us", None) is not None:
                ts_direct_list.append(pe["Ts_hat_direct_us"])
            if pe.get("Ts_blend", None) is not None:
                ts_blend_list.append(pe["Ts_blend"])

        out = dict(sep_out)
        out["g_logit"] = torch.stack(g_logit_list, dim=1)  # (B,3,N)
        out["g_hat"] = torch.stack(g_hat_list, dim=1)  # (B,3,N)
        out["Tl_hat_us"] = torch.stack(tl_list, dim=1)  # (B,3)
        out["NF_logits"] = torch.stack(nf_logit_list, dim=1)  # (B,3,4)
        if nf_active_logit_list:
            out["NF_active_logit"] = torch.stack(nf_active_logit_list, dim=1)  # (B,3)
        if len(ts_direct_list) == 3:
            out["Ts_hat_direct_us"] = torch.stack(ts_direct_list, dim=1)  # (B,3)
        if len(ts_blend_list) == 3:
            out["Ts_blend"] = torch.stack(ts_blend_list, dim=1)  # (B,3)
        ts_hat_us, ts_struct_us = self._recompute_ts(
            tl_hat_us=out["Tl_hat_us"],
            nf_logits=out["NF_logits"],
            ts_direct_us=out.get("Ts_hat_direct_us"),
            ts_blend=out.get("Ts_blend"),
        )
        out["Ts_hat_us"] = ts_hat_us
        out["Ts_struct_us"] = ts_struct_us

        slot_feat = torch.stack(slot_feat_list, dim=1)  # (B,3,F)
        if self.kactive_head is not None:
            slot_energy = torch.mean(j_hat * j_hat, dim=(2, 3))  # (B,3)
            k_in = torch.cat([slot_feat.reshape(slot_feat.shape[0], -1), slot_energy], dim=1)
            out["K_active_logit"] = self.kactive_logit_scale * self.kactive_head(k_in).squeeze(1)

        if self.slot_refiner_in is not None and self.slot_refiner is not None:
            h = self.slot_refiner_in(slot_feat)
            h = self.slot_refiner(h)
            h = self.slot_refiner_norm(h)

            nf_delta = self.slot_nf_delta_scale * self.slot_nf_delta_head(h)  # (B,3,4)
            nf_logits = out["NF_logits"] + nf_delta

            tl_delta = self.slot_tl_delta_scale * torch.tanh(self.slot_tl_delta_head(h).squeeze(-1))
            tl_hat = torch.clamp(out["Tl_hat_us"] + tl_delta, min=self.slot_tl_min, max=self.slot_tl_max)

            out["NF_logits"] = nf_logits
            out["Tl_hat_us"] = tl_hat
            ts_hat, ts_struct = self._recompute_ts(
                tl_hat_us=tl_hat,
                nf_logits=nf_logits,
                ts_direct_us=out.get("Ts_hat_direct_us"),
                ts_blend=out.get("Ts_blend"),
            )
            out["Ts_hat_us"] = ts_hat
            out["Ts_struct_us"] = ts_struct
            out["NF_logits_pre_refine"] = torch.stack(nf_logit_list, dim=1)
            out["Tl_hat_us_pre_refine"] = torch.stack(tl_list, dim=1)

        if self.slot_set_in is not None and self.slot_set_encoder is not None:
            nf_logits = out["NF_logits"]
            tl_hat = out["Tl_hat_us"]
            ts_struct = out["Ts_struct_us"]
            slot_energy = torch.mean(j_hat * j_hat, dim=(2, 3))
            gate_density = out["g_hat"].mean(dim=-1)
            if "NF_active_logit" in out and out["NF_active_logit"] is not None:
                active_logit = out["NF_active_logit"]
            else:
                active_logit = torch.logsumexp(nf_logits[..., 1:], dim=-1) - nf_logits[..., 0]

            slot_desc = torch.cat(
                [
                    slot_feat,
                    nf_logits,
                    tl_hat.unsqueeze(-1),
                    ts_struct.unsqueeze(-1),
                    slot_energy.unsqueeze(-1),
                    gate_density.unsqueeze(-1),
                    active_logit.unsqueeze(-1),
                ],
                dim=-1,
            )
            h_set = self.slot_set_in(slot_desc)
            h_set = self.slot_set_encoder(h_set)
            h_set = self.slot_set_norm(h_set)

            nf_logits_set = self.slot_set_nf_head(h_set)
            tl_hat_set = self.slot_set_softplus(self.slot_set_tl_head(h_set).squeeze(-1))
            tl_hat_set = torch.clamp(tl_hat_set, min=self.slot_set_tl_min, max=self.slot_set_tl_max)
            nf_alpha = torch.sigmoid(self.slot_set_nf_blend_head(h_set))
            tl_alpha = torch.sigmoid(self.slot_set_tl_blend_head(h_set)).squeeze(-1)

            out["NF_logits_pre_set"] = nf_logits
            out["Tl_hat_us_pre_set"] = tl_hat
            out["NF_logits"] = (1.0 - nf_alpha) * nf_logits + nf_alpha * nf_logits_set
            out["Tl_hat_us"] = torch.clamp(
                (1.0 - tl_alpha) * tl_hat + tl_alpha * tl_hat_set,
                min=self.slot_set_tl_min,
                max=self.slot_set_tl_max,
            )
            out["slot_set_alpha_nf"] = nf_alpha.squeeze(-1)
            out["slot_set_alpha_tl"] = tl_alpha
            ts_hat, ts_struct = self._recompute_ts(
                tl_hat_us=out["Tl_hat_us"],
                nf_logits=out["NF_logits"],
                ts_direct_us=out.get("Ts_hat_direct_us"),
                ts_blend=out.get("Ts_blend"),
            )
            out["Ts_hat_us"] = ts_hat
            out["Ts_struct_us"] = ts_struct

        if (
            self.anchor_tl_blend_head is not None
            and sep_out.get("slot_periodic_weights", None) is not None
            and sep_out["slot_periodic_weights"].shape[1] >= 3
        ):
            slot_w = sep_out["slot_periodic_weights"][:, :3, :].to(dtype=out["Tl_hat_us"].dtype)
            lag_us = self.periodic_anchor_lag_us.to(device=slot_w.device, dtype=slot_w.dtype).view(1, 1, -1)
            ts_anchor_us = torch.sum(slot_w * lag_us, dim=-1)

            nf_logits_pre = out["NF_logits"]
            tl_pre = out["Tl_hat_us"]
            ts_pre = out["Ts_hat_us"]
            nf_vals = self.nf_values_tensor.to(device=nf_logits_pre.device, dtype=nf_logits_pre.dtype).view(1, 1, -1)
            nf_resid = torch.abs((nf_vals + 1.0) * tl_pre.unsqueeze(-1) - ts_anchor_us.unsqueeze(-1))
            nf_bias = -nf_resid / max(self.anchor_nf_bias_tau_us, 1e-6)
            nf_logits = nf_logits_pre + self.anchor_nf_bias_scale * nf_bias

            nf_prob = torch.softmax(nf_logits, dim=-1)
            expected_nf = torch.sum(nf_prob * nf_vals, dim=-1)
            tl_anchor_us = torch.clamp(
                ts_anchor_us / (expected_nf + 1.0),
                min=self.anchor_tl_min,
                max=self.anchor_tl_max,
            )

            desc = torch.cat(
                [
                    slot_feat,
                    slot_w,
                    torch.softmax(nf_logits_pre, dim=-1),
                    tl_pre.unsqueeze(-1),
                    ts_pre.unsqueeze(-1),
                ],
                dim=-1,
            )
            alpha_tl = torch.sigmoid(self.anchor_tl_blend_head(desc).squeeze(-1))
            alpha_ts = torch.sigmoid(self.anchor_ts_blend_head(desc).squeeze(-1))
            tl_hat = torch.clamp(
                (1.0 - alpha_tl) * tl_pre + alpha_tl * tl_anchor_us,
                min=self.anchor_tl_min,
                max=self.anchor_tl_max,
            )
            ts_hat, ts_struct = self._recompute_ts(
                tl_hat_us=tl_hat,
                nf_logits=nf_logits,
                ts_direct_us=out.get("Ts_hat_direct_us"),
                ts_blend=out.get("Ts_blend"),
            )
            ts_anchor_clip = torch.maximum(ts_anchor_us, tl_hat + float(getattr(self.penet, "ts_min_margin_us", 0.05)))
            ts_hat = (1.0 - alpha_ts) * ts_hat + alpha_ts * ts_anchor_clip

            out["NF_logits_pre_anchor"] = nf_logits_pre
            out["Tl_hat_us_pre_anchor"] = tl_pre
            out["Ts_hat_us_pre_anchor"] = ts_pre
            out["NF_logits"] = nf_logits
            out["Tl_hat_us"] = tl_hat
            out["Ts_hat_us"] = ts_hat
            out["Ts_struct_us"] = ts_struct
            out["Tl_anchor_us"] = tl_anchor_us
            out["Ts_anchor_us"] = ts_anchor_us
            out["anchor_alpha_tl"] = alpha_tl
            out["anchor_alpha_ts"] = alpha_ts
        return out


def build_pe(cfg: dict) -> nn.Module:
    reader_type = str(cfg.get("model_pe", {}).get("reader_type", "gateformer")).lower()
    if reader_type in {"gateformer", "default", "penet"}:
        return PENet(cfg)
    if reader_type in {"gatestat", "gate_stat", "simple_gate_stats", "simple"}:
        return GateStatPE(cfg)
    raise ValueError(f"Unsupported model_pe.reader_type: {reader_type}")
