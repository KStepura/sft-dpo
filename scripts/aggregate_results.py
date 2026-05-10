#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def iter_summaries(root: Path):
    for summary_path in sorted(root.glob("*/summary.json")):
        try:
            yield json.loads(summary_path.read_text(encoding="utf-8")), summary_path
        except Exception:
            continue


def _safe_read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return None


def _clean_stats(stats_path: Optional[str]) -> Tuple[str, str, str]:
    if not stats_path:
        return "", "", ""
    obj = _safe_read_json(Path(stats_path))
    if not obj:
        return "", "", ""
    kept = obj.get("kept", "")
    dropped = obj.get("dropped", "")
    dr = obj.get("drop_rate")
    if dr == "" or dr is None:
        try:
            k = float(kept)
            d = float(dropped)
            dr = round(d / max(k + d, 1.0), 4) if (k or d) else ""
        except Exception:
            dr = ""
    return str(kept), str(dropped), "" if dr == "" else str(dr)


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
        track = summary.get("track", "")
        clean_kept, clean_dropped, clean_drop_rate = _clean_stats(summary.get("clean_stats_path"))

        for rec in summary.get("compare", []):
            stats = rec.get("stats", {})
            metrics_obj = rec.get("metrics") or {}
            pm = metrics_obj.get("primary_metric") or ""
            dpo_metrics = rec.get("dpo_metrics") or {}

            judge_obj = (metrics_obj.get("llm_judge") or {}) if metrics_obj else {}
            judge_win_rates = judge_obj.get("win_rates") or {}

            row = {
                "exp_id": exp_id,
                "created_at": created_at,
                "track": track,
                "beta": rec.get("beta", ""),
                "num_prompts": stats.get("num_prompts", 0),
                "avg_len_base": stats.get("avg_len_base", ""),
                "avg_len_sft": stats.get("avg_len_sft", ""),
                "avg_len_dpo": stats.get("avg_len_dpo", ""),
                "primary_metric": pm,
                "metric_base": "",
                "metric_sft": "",
                "metric_dpo": "",
                "judge_model": judge_obj.get("judge_model", ""),
                "judge_win_rate_base": "" if judge_win_rates.get("base") is None else str(judge_win_rates["base"]),
                "judge_win_rate_sft": "" if judge_win_rates.get("sft") is None else str(judge_win_rates["sft"]),
                "judge_win_rate_dpo": "" if judge_win_rates.get("dpo") is None else str(judge_win_rates["dpo"]),
                "judge_failures": "" if judge_obj.get("judge_failures") is None else str(judge_obj["judge_failures"]),
                "dpo_kl_final": "" if dpo_metrics.get("kl_final") is None else str(dpo_metrics.get("kl_final")),
                "dpo_loss_final": "" if dpo_metrics.get("loss_final") is None else str(dpo_metrics.get("loss_final")),
                "dpo_reward_margin_final": ""
                if dpo_metrics.get("reward_margin_final") is None
                else str(dpo_metrics.get("reward_margin_final")),
                "clean_kept": clean_kept,
                "clean_dropped": clean_dropped,
                "clean_drop_rate": clean_drop_rate,
                "total_runtime_sec": total_seconds,
                "summary_path": str(summary_path),
                "compare_path": rec.get("path", ""),
            }
            if pm and metrics_obj:
                for side in ("base", "sft", "dpo"):
                    side_obj = metrics_obj.get(side) or {}
                    v = side_obj.get(pm)
                    row[f"metric_{side}"] = "" if v is None else str(v)
            rows.append(row)

    fieldnames = [
        "exp_id",
        "created_at",
        "track",
        "beta",
        "num_prompts",
        "avg_len_base",
        "avg_len_sft",
        "avg_len_dpo",
        "primary_metric",
        "metric_base",
        "metric_sft",
        "metric_dpo",
        "judge_model",
        "judge_win_rate_base",
        "judge_win_rate_sft",
        "judge_win_rate_dpo",
        "judge_failures",
        "dpo_kl_final",
        "dpo_loss_final",
        "dpo_reward_margin_final",
        "clean_kept",
        "clean_dropped",
        "clean_drop_rate",
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
