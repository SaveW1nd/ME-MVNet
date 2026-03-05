"""Model builders for separator frontends."""

from __future__ import annotations

from .cisrj_sn import CISRJSN
from .sepnet import SepNet


def build_separator(cfg: dict):
    """Build separator frontend from config.

    Supported model types in cfg["model_sep"]["model_type"]:
    - "sepnet" (default)
    - "cisrj_sn"
    """
    m = cfg.get("model_sep", {})
    model_type = str(m.get("model_type", "sepnet")).lower()
    if model_type == "sepnet":
        return SepNet(cfg)
    if model_type in {"cisrj_sn", "cisrj-sn", "cisrj"}:
        return CISRJSN(cfg)
    raise ValueError(f"Unsupported model_sep.model_type: {model_type}")
