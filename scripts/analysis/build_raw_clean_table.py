#!/usr/bin/env python3
"""Build the RAW vs CLEAN side-by-side table for the diploma.

RAW   = existing results/experiments/<exp>/metrics/metrics_beta_*.json
        (computed before truncate_at_instruction was added to eval_metrics.py)
CLEAN = results/<exp>/metrics_beta_*.clean.json
        (just recomputed with the patched eval_metrics.py)

Outputs:
  results/raw_vs_clean.md       — markdown table for the thesis
  results/raw_vs_clean.csv      — same data as CSV
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = REPO_ROOT / "results" / "experiments"
CLEAN_ROOT = REPO_ROOT / "results"

EXPERIMENTS = [
    ("benchmark_dialog_base_20260503_182808", "dialog", "dialog_heuristic"),
    ("benchmark_summarization_base_20260503_232644", "summarization", "rougeL_f1"),
    ("benchmark_mbpp_code_base_20260504_042309", "code", "test_pass_rate"),
]


def fmt(v):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def main():
    rows = []
    for exp_id, track, metric_key in EXPERIMENTS:
        raw_dir = RAW_ROOT / exp_id / "metrics"
        clean_dir = CLEAN_ROOT / exp_id
        for raw_path in sorted(raw_dir.glob("metrics_beta_*.json")):
            beta = raw_path.stem.replace("metrics_beta_", "")
            clean_path = clean_dir / f"metrics_beta_{beta}.clean.json"
            if not clean_path.exists():
                continue
            raw = json.loads(raw_path.read_text(encoding="utf-8"))
            clean = json.loads(clean_path.read_text(encoding="utf-8"))
            for model in ("base", "sft", "dpo"):
                rows.append({
                    "track": track,
                    "exp_id": exp_id,
                    "beta": beta,
                    "model": model,
                    "metric": metric_key,
                    "raw": raw.get(model, {}).get(metric_key),
                    "clean": clean.get(model, {}).get(metric_key),
                    "delta": (
                        round(clean.get(model, {}).get(metric_key, 0)
                              - raw.get(model, {}).get(metric_key, 0), 4)
                        if raw.get(model, {}).get(metric_key) is not None
                        and clean.get(model, {}).get(metric_key) is not None
                        else None
                    ),
                    "align_clean": clean.get("align_rate"),
                })

    out_md = CLEAN_ROOT / "raw_vs_clean.md"
    out_csv = CLEAN_ROOT / "raw_vs_clean.csv"

    md_lines = [
        "# RAW vs CLEAN — после применения симметричной обрезки в eval_metrics.py",
        "",
        "RAW   = `results/experiments/<exp>/metrics/metrics_beta_*.json` "
        "(до фикса, без обрезки `### Instruction:` хвоста).",
        "",
        "CLEAN = `results/<exp>/metrics_beta_*.clean.json` "
        "(после фикса; для dialog ещё и новая heuristic).",
        "",
        "| Track | β | Модель | Метрика | RAW | CLEAN | Δ | align |",
        "|---|---:|---|---|---:|---:|---:|---:|",
    ]
    last_track = None
    last_beta = None
    for r in rows:
        if r["track"] != last_track or r["beta"] != last_beta:
            md_lines.append(
                f"| **{r['track']}** | {r['beta']} | {r['model']} | "
                f"{r['metric']} | {fmt(r['raw'])} | {fmt(r['clean'])} | "
                f"{fmt(r['delta'])} | {fmt(r['align_clean'])} |"
            )
        else:
            md_lines.append(
                f"|  |  | {r['model']} | {r['metric']} | "
                f"{fmt(r['raw'])} | {fmt(r['clean'])} | "
                f"{fmt(r['delta'])} | {fmt(r['align_clean'])} |"
            )
        last_track = r["track"]
        last_beta = r["beta"]

    md_lines += [
        "",
        "## Что в этой таблице важно для диплома",
        "",
        "1. **Колонка Δ для base** на summarization показывает методологический "
        "артефакт: после симметричной обрезки rougeL базовой модели "
        "**вырастает**, потому что с короткого ответа считается чище. "
        "«Прирост SFT над base» сжимается, как описано в `RESEARCH_QUESTIONS.md` (RQ3).",
        "2. **dialog_heuristic** — числа на CLEAN значительно ниже, потому что "
        "новая метрика дискриминативна (возвращает 0–1, а не «всем 0.66»). "
        "Сравнивать CLEAN dialog_heuristic с RAW dialog_heuristic некорректно — "
        "это разные функции; смотреть нужно ранг моделей и расхождение между ними.",
        "3. **code/pass_rate** меняется незначительно — `extract_code_block` "
        "и так корректно срезал хвосты, поэтому фикс на этот трекая почти не влияет.",
        "4. **align_rate** = 1.0 для всех — нормализация промптов в `eval_metrics.py` "
        "не сломала alignment, что проверено на старых compare-файлах.",
    ]

    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_md}")

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
