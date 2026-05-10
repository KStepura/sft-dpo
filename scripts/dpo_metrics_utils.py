#!/usr/bin/env python3
"""Extract final-step signals from TRL DPO trainer_state.json (loss, margins; KL if logged)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


def find_last_checkpoint(dpo_out: Path) -> Optional[Path]:
    cks = sorted(dpo_out.glob("checkpoint-*"))
    return cks[-1] if cks else None


def extract_dpo_metrics(dpo_out: Path) -> Dict[str, Any]:
    ckpt = find_last_checkpoint(dpo_out)
    out: Dict[str, Any] = {
        "kl_final": None,
        "loss_final": None,
        "reward_margin_final": None,
        "num_log_records": 0,
        "checkpoint": str(ckpt) if ckpt else "",
        "kl_candidates": {},
    }
    if not ckpt:
        return out
    state_path = ckpt / "trainer_state.json"
    if not state_path.exists():
        return out
    try:
        st = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return out
    hist = st.get("log_history") or []
    if not isinstance(hist, list) or not hist:
        return out
    last = hist[-1]
    out["num_log_records"] = len(hist)
    if not isinstance(last, dict):
        return out
    out["loss_final"] = last.get("loss")
    out["reward_margin_final"] = last.get("rewards/margins")
    # TRL versions differ: explicit kl vs implicit_kl vs *_kl keys
    if "kl" in last:
        out["kl_final"] = last.get("kl")
    for k, v in last.items():
        if not isinstance(v, (int, float)):
            continue
        lk = str(k).lower()
        if "kl" in lk:
            out["kl_candidates"][k] = v
            if out["kl_final"] is None:
                out["kl_final"] = v
    return out


def enrich_run_meta(d: Dict[str, Any], *, beta: float, max_steps: int, dpo_data_path: str) -> Dict[str, Any]:
    x = dict(d)
    x.update({"beta": beta, "max_steps": max_steps, "dpo_data_path": dpo_data_path})
    return x
