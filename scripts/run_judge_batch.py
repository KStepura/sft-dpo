#!/usr/bin/env python3
"""
Batch-run pairwise LLM judge on all existing experiment compare files.

Scans results/experiments/ for experiments that have:
  - summary.json   (to read track)
  - refs.jsonl
  - compare/compare_beta_*.jsonl

For each match, calls eval_metrics.py with judge flags and writes:
  metrics/metrics_beta_*.json  (creates or overwrites, preserves track metrics)

Usage:
  python scripts/run_judge_batch.py \\
      --judge_model qwen2.5:7b \\
      --judge_base_url http://127.0.0.1:11434/v1 \\
      [--judge_api_key sk-...] \\
      [--judge_sleep_sec 0.15] \\
      [--judge_timeout_sec 120] \\
      [--judge_json_mode] \\
      [--exp_root results/experiments] \\
      [--limit N]   # judge only first N prompts per compare file (for testing) \\
      [--dry_run]
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple


def find_tasks(
    root: Path,
) -> List[Tuple[Path, str, Path, List[Path]]]:
    """Return list of (exp_dir, track, refs_path, compare_files)."""
    tasks = []
    for summary_path in sorted(root.glob("*/summary.json")):
        exp_dir = summary_path.parent
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            print(f"  [skip] {exp_dir.name}: cannot read summary.json")
            continue

        track = summary.get("track", "")
        if not track:
            print(f"  [skip] {exp_dir.name}: track not set in summary.json")
            continue

        refs = exp_dir / "refs.jsonl"
        if not refs.exists():
            print(f"  [skip] {exp_dir.name}: no refs.jsonl")
            continue

        compare_dir = exp_dir / "compare"
        compare_files = sorted(compare_dir.glob("compare_beta_*.jsonl")) if compare_dir.exists() else []
        if not compare_files:
            print(f"  [skip] {exp_dir.name}: no compare_beta_*.jsonl files")
            continue

        tasks.append((exp_dir, track, refs, compare_files))
    return tasks


def count_prompts(compare_path: Path) -> int:
    n = 0
    with compare_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch pairwise LLM judge on existing experiment results.")
    ap.add_argument("--exp_root", default="results/experiments", help="Root dir with per-experiment folders")
    ap.add_argument("--judge_model", required=True, help="Judge model name (OpenAI-compatible API)")
    ap.add_argument("--judge_base_url", default="https://api.openai.com/v1", help="OpenAI-compatible endpoint")
    ap.add_argument("--judge_api_key", default="", help="API key (or set JUDGE_API_KEY env var)")
    ap.add_argument("--judge_json_mode", action="store_true", help="Use response_format=json_object")
    ap.add_argument("--judge_sleep_sec", type=float, default=0.15, help="Sleep between judge API calls")
    ap.add_argument("--judge_timeout_sec", type=int, default=120, help="HTTP timeout per judge request")
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        metavar="N",
        help="Judge only the first N prompts per compare file (0 = all; useful for smoke-testing)",
    )
    ap.add_argument("--exp_filter", default="", metavar="SUBSTR",
                    help="Only process experiments whose directory name contains SUBSTR (e.g. 'v2_track')")
    ap.add_argument("--dry_run", action="store_true", help="Print commands without executing them")
    args = ap.parse_args()

    root = Path(args.exp_root)
    if not root.exists():
        raise SystemExit(f"exp_root not found: {root}")

    print(f"Scanning {root} ...\n")
    tasks = find_tasks(root)
    if args.exp_filter:
        before = len(tasks)
        tasks = [(ed, tr, rf, cf) for ed, tr, rf, cf in tasks if args.exp_filter in ed.name]
        print(f"Filter '{args.exp_filter}': {before} → {len(tasks)} experiments\n")

    if not tasks:
        print("No experiments found with track + refs.jsonl + compare files. Nothing to do.")
        return

    total_compare = sum(len(cf) for _, _, _, cf in tasks)
    total_prompts = sum(count_prompts(cf) for _, _, _, cfs in tasks for cf in cfs)
    effective_prompts = min(total_prompts, args.limit * total_compare) if args.limit else total_prompts
    total_api_calls = effective_prompts * 3 * 2  # 3 pairs × AB + BA

    print(f"Experiments : {len(tasks)}")
    print(f"Compare files: {total_compare}")
    print(f"Prompts      : {total_prompts}" + (f"  (capped to {args.limit}/file → {effective_prompts})" if args.limit else ""))
    print(f"API calls    : ~{total_api_calls}  (3 pairs × 2 directions × prompts)")
    print(f"Sleep/call   : {args.judge_sleep_sec}s  →  sleep budget ~{total_api_calls * args.judge_sleep_sec:.0f}s")
    if args.dry_run:
        print("\n[DRY RUN — no commands will be executed]\n")
    else:
        print()

    t_start = time.time()
    n_ok = 0
    n_err = 0

    for exp_dir, track, refs, compare_files in tasks:
        print(f"=== {exp_dir.name}  track={track}  ({len(compare_files)} beta(s)) ===")
        metrics_dir = exp_dir / "metrics"
        metrics_dir.mkdir(exist_ok=True)

        for compare_path in compare_files:
            beta_slug = compare_path.stem.replace("compare_", "")
            out_path = metrics_dir / f"metrics_{beta_slug}.json"
            n_prompts = count_prompts(compare_path)
            effective = min(n_prompts, args.limit) if args.limit else n_prompts
            calls = effective * 3 * 2

            print(f"  {compare_path.name}  ({n_prompts} prompts, ~{calls} calls)  → {out_path.name}")

            cmd = [
                sys.executable, "scripts/eval_metrics.py",
                "--track", track,
                "--compare_path", str(compare_path),
                "--refs_path", str(refs),
                "--out_path", str(out_path),
                "--judge_model", args.judge_model,
                "--judge_base_url", args.judge_base_url,
                "--judge_sleep_sec", str(args.judge_sleep_sec),
                "--judge_timeout_sec", str(args.judge_timeout_sec),
            ]
            if args.judge_api_key:
                cmd.extend(["--judge_api_key", args.judge_api_key])
            if args.judge_json_mode:
                cmd.append("--judge_json_mode")
            if args.limit:
                cmd.extend(["--judge_limit", str(args.limit)])

            if args.dry_run:
                print(f"    $ {' '.join(cmd)}")
                continue

            proc = subprocess.run(cmd, check=False)
            if proc.returncode == 0:
                n_ok += 1
            else:
                print(f"    [ERROR] exit={proc.returncode}")
                n_err += 1

    if not args.dry_run:
        elapsed = time.time() - t_start
        print(f"\nDone in {elapsed:.0f}s.  ok={n_ok}  errors={n_err}")


if __name__ == "__main__":
    main()
