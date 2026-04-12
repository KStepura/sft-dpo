#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


def iter_summaries(root: Path):
    for summary_path in sorted(root.glob("*/summary.json")):
        try:
            yield json.loads(summary_path.read_text(encoding="utf-8")), summary_path
        except Exception:
            continue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/experiments", help="Root dir with per-experiment folders")
    ap.add_argument("--out_csv", default="results/experiments/summary_table.csv")
    args = ap.parse_args()

    root = Path(args.root)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for summary, summary_path in iter_summaries(root):
        exp_id = summary.get("exp_id", summary_path.parent.name)
        created_at = summary.get("created_at", "")
        total_seconds = round(sum(step.get("seconds", 0) for step in summary.get("steps", [])), 2)
        for rec in summary.get("compare", []):
            stats = rec.get("stats", {})
            rows.append(
                {
                    "exp_id": exp_id,
                    "created_at": created_at,
                    "beta": rec.get("beta", ""),
                    "num_prompts": stats.get("num_prompts", 0),
                    "avg_len_base": stats.get("avg_len_base", ""),
                    "avg_len_sft": stats.get("avg_len_sft", ""),
                    "avg_len_dpo": stats.get("avg_len_dpo", ""),
                    "total_runtime_sec": total_seconds,
                    "summary_path": str(summary_path),
                    "compare_path": rec.get("path", ""),
                }
            )

    fieldnames = [
        "exp_id",
        "created_at",
        "beta",
        "num_prompts",
        "avg_len_base",
        "avg_len_sft",
        "avg_len_dpo",
        "total_runtime_sec",
        "summary_path",
        "compare_path",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows to {out_csv}")


if __name__ == "__main__":
    main()
