"""Model builders for separator frontends."""

from __future__ import annotations

from .cisrj_sn import CISRJSN
from .itersepnet import IterSepNet
from .periodfirst_sepnet import PeriodFirstSepNet, SoftPeriodFirstSepNet
from .replayperiod_sepnet import ReplayPeriodSepNet
from .sepnet import SepNet


def build_separator(cfg: dict):
    """Build separator frontend from config.

    Supported model types in cfg["model_sep"]["model_type"]:
    - "sepnet" (default)
    - "itersepnet"
    - "cisrj_sn"
    - "periodfirstsepnet"
    """
    m = cfg.get("model_sep", {})
    model_type = str(m.get("model_type", "sepnet")).lower()
    if model_type == "sepnet":
        return SepNet(cfg)
    if model_type in {"itersepnet", "iter-sepnet", "iter_sepnet"}:
        return IterSepNet(cfg)
    if model_type in {"cisrj_sn", "cisrj-sn", "cisrj"}:
        return CISRJSN(cfg)
    if model_type in {"periodfirstsepnet", "period-first-sepnet", "period_first_sepnet", "periodfirst"}:
        return PeriodFirstSepNet(cfg)
    if model_type in {"softperiodsepnet", "soft-period-sepnet", "soft_period_sepnet", "softperiod"}:
        return SoftPeriodFirstSepNet(cfg)
    if model_type in {"replayperiodsepnet", "replay-period-sepnet", "replay_period_sepnet", "replayperiod"}:
        return ReplayPeriodSepNet(cfg)
    raise ValueError(f"Unsupported model_sep.model_type: {model_type}")
