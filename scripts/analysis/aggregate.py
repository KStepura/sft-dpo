#!/usr/bin/env python3
"""Aggregate every v2 run into one CSV + markdown, and plot DPO dynamics.

Sources:
- results/experiments/v2_*_<ts>/metrics/metrics_beta_*.json — auto-metrics
- results/experiments/v2_*_<ts>/summary.json — config snapshot
- outputs/v2_*_<ts>/dpo_beta_*/checkpoint-*/trainer_state.json — per-step signals

Outputs:
- results/aggregate.csv
- results/aggregate.md
- figures/dpo_dynamics_<run>.png  (loss, logps, margins, accuracies)
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO / "results" / "experiments"
OUT_DIR = REPO / "results"
FIG_DIR = REPO / "figures"
OUTPUTS_ROOT = REPO / "outputs"

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Treat any results/experiments dir starting with v2_ as a run we care about.
V2_RUN_PREFIXES = ("v2_track_", "v2_sum_", "v2_smoke_")


def load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def find_runs() -> List[Path]:
    return sorted(
        d for d in RESULTS_ROOT.iterdir()
        if d.is_dir() and any(d.name.startswith(p) for p in V2_RUN_PREFIXES)
    )


def collect_dpo_log_history(exp_id: str, beta_slug: str) -> List[Dict[str, Any]]:
    """Stitch trainer_state.json log_history across all saved checkpoints
    (TRL writes the cumulative history into each checkpoint's trainer_state)."""
    out_dir = OUTPUTS_ROOT / exp_id / f"dpo_beta_{beta_slug}"
    if not out_dir.exists():
        return []
    cks = sorted(out_dir.glob("checkpoint-*"),
                 key=lambda p: int(p.name.split("-")[-1]))
    if not cks:
        return []
    # last checkpoint has the longest log_history (cumulative).
    state = load_json(cks[-1] / "trainer_state.json")
    return state.get("log_history") or []


def main():
    runs = find_runs()
    rows = []
    for exp_dir in runs:
        exp_id = exp_dir.name
        summary = load_json(exp_dir / "summary.json")
        track = summary.get("track")
        cfg = load_json(exp_dir / "resolved_config.json")
        seed = cfg.get("seed")
        policy_source = cfg.get("data", {}).get("policy_source", "baseline")
        loss_type = cfg.get("dpo", {}).get("loss_type")
        max_steps_dpo = cfg.get("dpo", {}).get("max_steps")
        for entry in summary.get("compare", []):
            beta = entry.get("beta")
            beta_slug = str(beta).replace(".", "p")
            metrics = entry.get("metrics") or {}
            # find the primary metric value per model
            primary = metrics.get("primary_metric")
            row = {
                "exp_id": exp_id,
                "track": track,
                "seed": seed,
                "policy_source": policy_source,
                "loss_type": loss_type,
                "max_steps_dpo": max_steps_dpo,
                "beta": beta,
                "primary_metric": primary,
                "num_eval": metrics.get("num_eval_items"),
                "align_rate": metrics.get("align_rate"),
            }
            for m in ("base", "sft", "dpo"):
                row[f"{m}_metric"] = (metrics.get(m) or {}).get(primary) if primary else None
            # quick DPO trainer signals at end (prefer trainer_state directly,
            # fall back to summary's dpo_metrics block if the checkpoint dir
            # was reorganised by select-best).
            hist = collect_dpo_log_history(exp_id, beta_slug)
            if hist:
                last = hist[-1]
                row["dpo_loss_final"] = last.get("loss")
                row["dpo_reward_margin_final"] = last.get("rewards/margins")
                row["dpo_logps_rejected_final"] = last.get("logps/rejected")
                row["dpo_logps_chosen_final"] = last.get("logps/chosen")
                row["dpo_accuracy_final"] = last.get("rewards/accuracies")
            else:
                dpo_metrics = entry.get("dpo_metrics") or {}
                row["dpo_loss_final"] = dpo_metrics.get("loss_final")
                row["dpo_reward_margin_final"] = dpo_metrics.get("reward_margin_final")
                row["dpo_logps_rejected_final"] = None
                row["dpo_logps_chosen_final"] = None
                row["dpo_accuracy_final"] = None
            row["dpo_selected_ckpt"] = (entry.get("dpo_metrics") or {}).get("selected_checkpoint", "")
            rows.append(row)

            if hist:
                _plot_dynamics(exp_id, beta_slug, hist, track, policy_source, loss_type, beta)

    # Write CSV.
    if not rows:
        print("no runs found")
        return
    csv_path = OUT_DIR / "aggregate.csv"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {csv_path} ({len(rows)} rows)")

    # Write markdown.
    md = ["# Сводка v2 экспериментов",
          "",
          "Все строки — финальный compare на eval-наборе (после симметричной обрезки).",
          "Метрика — `rougeL_f1` для summarization, `dialog_heuristic` для dialog.",
          ""]
    md.append("| exp | track | seed | policy | loss | β | base | sft | dpo | DPO−SFT | loss_final | margin | logps/rej | acc |")
    md.append("|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        diff = (r["dpo_metric"] - r["sft_metric"]) if (r["dpo_metric"] is not None and r["sft_metric"] is not None) else None
        md.append(
            f"| {_short(r['exp_id'])} | {r['track']} | {r['seed']} | {r['policy_source']} | "
            f"{r['loss_type']} | {r['beta']} | {_fmt(r['base_metric'])} | "
            f"{_fmt(r['sft_metric'])} | **{_fmt(r['dpo_metric'])}** | {_fmt(diff)} | "
            f"{_fmt(r['dpo_loss_final'])} | {_fmt(r['dpo_reward_margin_final'])} | "
            f"{_fmt(r.get('dpo_logps_rejected_final'))} | {_fmt(r.get('dpo_accuracy_final'))} |"
        )

    md.extend([
        "",
        "## DPO trainer signals — динамика",
        "",
        "Каждый прогон → отдельный PNG в `figures/dpo_dynamics_*.png`. ",
        "На графиках по шагам: `loss`, `logps/chosen`, `logps/rejected`, `rewards/margins`, `rewards/accuracies`.",
        "",
        "**Главное наблюдение для диплома:** в on-policy IPO прогоне `logps/rejected` "
        "стабильно держится в районе −90…−100, тогда как в v1 baseline-pair sigmoid-DPO "
        "к 200-му шагу падал до −272 (см. AUDIT_FINDINGS.md). Графики позволяют визуально "
        "показать, что коллапс предотвращён.",
    ])

    md_path = OUT_DIR / "aggregate.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {md_path}")


def _short(exp_id: str) -> str:
    # drop trailing timestamp ("_20260507_043001") for readability
    parts = exp_id.split("_")
    if len(parts) >= 2 and parts[-1].isdigit() and parts[-2].isdigit():
        return "_".join(parts[:-2])
    return exp_id


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _plot_dynamics(exp_id, beta_slug, hist, track, policy_source, loss_type, beta):
    """One figure with 5 stacked subplots: loss, logps/c, logps/r, margin, acc."""
    steps = [r.get("step") for r in hist if "step" in r]
    if not steps:
        return
    keys = [
        ("loss",                 "loss"),
        ("logps/chosen",         "logps/chosen"),
        ("logps/rejected",       "logps/rejected"),
        ("rewards/margins",      "rewards/margins"),
        ("rewards/accuracies",   "rewards/accuracies"),
    ]
    fig, axes = plt.subplots(len(keys), 1, figsize=(7.0, 9.0), sharex=True)
    title = (f"DPO dynamics — {_short(exp_id)} (track={track}, "
             f"policy={policy_source}, loss={loss_type}, β={beta})")
    fig.suptitle(title, fontsize=10)
    for ax, (k, label) in zip(axes, keys):
        xs, ys = [], []
        for r in hist:
            if k in r and "step" in r:
                xs.append(r["step"])
                ys.append(r[k])
        if xs:
            ax.plot(xs, ys, marker="o", linewidth=1.2)
        ax.set_ylabel(label, fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("step")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = FIG_DIR / f"dpo_dynamics_{_short(exp_id)}_beta_{beta_slug}.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  fig: {out}")


if __name__ == "__main__":
    main()
