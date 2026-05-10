#!/usr/bin/env python3
"""Recompute auto-metrics with the patched eval_metrics.py for all existing
benchmark compare files. Skips the LLM judge (no GPU/Ollama needed) and writes
each result into results/<exp_id>/metrics_beta_*.clean.json.

After it finishes, run build_raw_clean_table.py to print the side-by-side
RAW (existing metrics.json) vs CLEAN (newly computed) comparison.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = str(REPO_ROOT / ".venv" / "bin" / "python")
EVAL_METRICS = str(REPO_ROOT / "scripts" / "eval_metrics.py")
OUT_ROOT = REPO_ROOT / "results"

EXPERIMENTS = [
    ("benchmark_dialog_base_20260503_182808", "dialog"),
    ("benchmark_summarization_base_20260503_232644", "summarization"),
    ("benchmark_mbpp_code_base_20260504_042309", "code"),
]


def main() -> int:
    for exp_id, track in EXPERIMENTS:
        exp_dir = REPO_ROOT / "results" / "experiments" / exp_id
        refs = exp_dir / "refs.jsonl"
        if not refs.exists():
            print(f"[skip] {exp_id}: no refs.jsonl")
            continue
        compare_dir = exp_dir / "compare"
        out_dir = OUT_ROOT / exp_id
        out_dir.mkdir(parents=True, exist_ok=True)

        for compare_path in sorted(compare_dir.glob("compare_beta_*.jsonl")):
            beta_slug = compare_path.stem.replace("compare_beta_", "")
            out_path = out_dir / f"metrics_beta_{beta_slug}.clean.json"
            cmd = [
                PYTHON, EVAL_METRICS,
                "--track", track,
                "--compare_path", str(compare_path),
                "--refs_path", str(refs),
                "--out_path", str(out_path),
            ]
            print(f"[{exp_id} β={beta_slug}] {' '.join(cmd[:2])} ...")
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                print(f"  FAIL: {proc.stderr}")
                return 1
            if proc.stdout.strip():
                for line in proc.stdout.strip().splitlines():
                    print(f"  {line}")
    print("\nDone. CLEAN metrics in:", OUT_ROOT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
