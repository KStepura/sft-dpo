#!/usr/bin/env python3
"""Shared checks before long experiment runs."""
from __future__ import annotations

import sys
from typing import Optional


def cuda_status() -> tuple[bool, str]:
    try:
        import torch
    except ImportError as e:
        return False, f"torch import failed: {e}"
    ok = bool(torch.cuda.is_available())
    if not ok:
        return False, "torch.cuda.is_available() is False"
    try:
        name = torch.cuda.get_device_name(0)
    except Exception as exc:
        return False, f"CUDA device error: {exc}"
    return True, name


def require_cuda(*, allow_cpu: bool, label: str = "experiment") -> None:
    ok, msg = cuda_status()
    if ok:
        print(f"[{label}] CUDA OK: {msg}", flush=True)
        return
    if allow_cpu:
        print(
            f"[{label}] WARNING: {msg}. Continuing on CPU (--allow-cpu). Training will be very slow.",
            file=sys.stderr,
            flush=True,
        )
        return
    print(
        f"[{label}] ERROR: GPU required but {msg}.\n"
        "Fix drivers/CUDA, run: source env.sh && python -c \"import torch; print(torch.cuda.is_available())\"\n"
        "Or pass --allow-cpu only for smoke/debug.",
        file=sys.stderr,
        flush=True,
    )
    raise SystemExit(2)
