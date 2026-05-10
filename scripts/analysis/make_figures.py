#!/usr/bin/env python3
"""Собирает все графики для главы Results диплома.

Источники данных:
  - results/experiments/v2_*/metrics/metrics_beta_*.json — основные метрики
  - results/experiments/v2_*/summary.json — стат. длины + dpo_metrics
  - outputs/v2_*/dpo_beta_*/checkpoint-N/trainer_state.json — динамика DPO
  - results/cross_domain_d1_sum_to_mbpp/metrics/metrics.json — D1
  - draft/outputs_experiments/track_*/dpo_beta_*/dpo_metrics.json — v1 финальные точки

Сохраняет PDF + PNG в figures/.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

COLORS = {"base": "#7f7f7f", "sft": "#1f77b4", "dpo": "#d62728"}


def save(fig, name: str) -> None:
    pdf = FIG_DIR / f"{name}.pdf"
    png = FIG_DIR / f"{name}.png"
    fig.tight_layout()
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {pdf.relative_to(ROOT)} and {png.relative_to(ROOT)}")


def load_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def metrics_for(exp_dir: str, beta_slug: str) -> dict:
    p = ROOT / "results" / "experiments" / exp_dir / "metrics" / f"metrics_beta_{beta_slug}.json"
    return load_json(p)


def trainer_history(exp_dir: str, beta_slug: str) -> list[dict]:
    """Возвращает log_history последнего checkpoint (содержит все шаги)."""
    out_dir = ROOT / "outputs" / exp_dir / f"dpo_beta_{beta_slug}"
    ckpts = sorted(out_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    if not ckpts:
        return []
    state = load_json(ckpts[-1] / "trainer_state.json")
    return state.get("log_history", [])


def fig_beta_curve():
    """β-кривая для summarization on-policy (seed=42), 5 точек."""
    print("\n[fig_beta_curve]")
    candidates = [
        (0.05, "v2_sum_beta0p05_sweep", "0p05"),
        (0.10, "v2_sum_beta0p1_sweep", "0p1"),
        (0.30, "v2_track_summarization_nojudge", "0p3"),
        (0.50, "v2_sum_beta0p5_sweep", "0p5"),
        (0.70, "v2_sum_beta0p7_sweep", "0p7"),
    ]
    points = []
    for beta, prefix, slug in candidates:
        dirs = sorted((ROOT / "results/experiments").glob(f"{prefix}_*"))
        if not dirs:
            print(f"  skip β={beta} ({prefix} not found)")
            continue
        m_path = dirs[-1] / "metrics" / f"metrics_beta_{slug}.json"
        if not m_path.exists():
            print(f"  skip β={beta} (no metrics file)")
            continue
        points.append((beta, dirs[-1].name, slug))

    betas, base_vals, sft_vals, dpo_vals = [], [], [], []
    for beta, exp_dir, slug in points:
        m = metrics_for(exp_dir, slug)
        betas.append(beta)
        base_vals.append(m["base"]["rougeL_f1"])
        sft_vals.append(m["sft"]["rougeL_f1"])
        dpo_vals.append(m["dpo"]["rougeL_f1"])

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.axhline(y=np.mean(base_vals), color=COLORS["base"], linestyle=":", label=f"Base ({np.mean(base_vals):.3f})")
    ax.axhline(y=np.mean(sft_vals), color=COLORS["sft"], linestyle="--", label=f"SFT ({np.mean(sft_vals):.3f})")
    ax.plot(betas, dpo_vals, "o-", color=COLORS["dpo"], lw=2, markersize=8, label="DPO (on-policy IPO)")
    for b, v in zip(betas, dpo_vals):
        ax.annotate(f"{v:.3f}", (b, v), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=9)

    ax.set_xlabel(r"DPO regularization parameter $\beta$")
    ax.set_ylabel("ROUGE-L F1 (summarization)")
    ax.set_title(f"β-кривая on-policy DPO на summarization (seed=42, n={len(betas)} точек)")
    ax.set_xticks(betas)
    ax.set_xticklabels([f"{b:g}" for b in betas])
    ax.legend(loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.30))
    ax.grid(axis="y", alpha=0.3)
    save(fig, "fig_beta_curve_summarization")


def fig_dpo_dynamics():
    """Динамика DPO trainer signals: v2 onpolicy IPO (стабильна) vs v1 baseline+sigmoid (коллапс).

    v2 — берём log_history из последнего checkpoint.
    v1 — горизонтальные линии финальных значений из dpo_metrics.json.
    """
    print("\n[fig_dpo_dynamics]")
    runs = [
        ("v2_track_summarization_nojudge_20260506_210741", "0p3", "v2 onpolicy IPO (sum, β=0.3)", "#1f77b4"),
        ("v2_sum_baseline_pairs_nojudge_20260507_024739", "0p3", "v2 baseline-pairs IPO (sum, β=0.3)", "#ff7f0e"),
        ("v2_track_dialog_nojudge_20260507_000348", "0p3", "v2 onpolicy IPO (dialog, β=0.3)", "#2ca02c"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7), sharex=True)

    metrics_to_plot = [
        ("loss", "Loss"),
        ("logps/rejected", "logps/rejected"),
        ("rewards/margins", "rewards/margins"),
    ]

    for ax, (key, ylabel) in zip(axes, metrics_to_plot):
        for exp_dir, slug, label, color in runs:
            hist = trainer_history(exp_dir, slug)
            steps = [r["step"] for r in hist if key in r]
            vals = [r[key] for r in hist if key in r]
            if steps:
                ax.plot(steps, vals, "o-", label=label, color=color, lw=1.6, markersize=4)
        ax.set_xlabel("step")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(alpha=0.3)

    # для logps/rejected добавим v1 collapse-точку
    try:
        v1_dpo = load_json(
            ROOT / "draft/outputs_experiments/track_sum_v1_20260415_163422/dpo_beta_0p1/dpo_metrics.json"
        )
        if v1_dpo.get("loss_final") is not None:
            axes[0].axhline(
                y=v1_dpo["loss_final"], color="red", linestyle=":", lw=1.5,
                label=f"v1 sum-DPO loss_final={v1_dpo['loss_final']:.3f}",
            )
        if v1_dpo.get("reward_margin_final") is not None:
            axes[2].axhline(
                y=v1_dpo["reward_margin_final"], color="red", linestyle=":", lw=1.5,
                label=f"v1 sum-DPO margin_final={v1_dpo['reward_margin_final']:.1f}",
            )
    except Exception as e:
        print(f"  (skip v1 reference: {e})")

    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for hi, li in zip(h, l):
            if li not in labels:
                handles.append(hi)
                labels.append(li)
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.18))
    fig.suptitle("Динамика обучения DPO: on-policy IPO vs baseline-pairs (v2)", y=1.04)
    save(fig, "fig_dpo_dynamics")


def fig_multiseed():
    """Multi-seed устойчивость на ТРЁХ треках (summarization, dialog, code).

    Каждая панель: bars (base, SFT, DPO) для каждого seed. Подписи Δ(DPO−SFT).
    """
    print("\n[fig_multiseed]")

    sum_runs = [
        ("seed=42", "v2_track_summarization_nojudge_20260506_210741", "0p3"),
        ("seed=1337", "v2_track_summarization_seed1337_nojudge_20260507_043001", "0p3"),
    ]
    seed2024_dir = list((ROOT / "results/experiments").glob("v2_track_summarization_seed2024*"))
    if seed2024_dir:
        m_path = seed2024_dir[0] / "metrics" / "metrics_beta_0p3.json"
        if m_path.exists():
            sum_runs.append(("seed=2024", seed2024_dir[0].name, "0p3"))

    dial_runs = [
        ("seed=42", "v2_track_dialog_nojudge_20260507_000348", "0p3"),
    ]
    dial_1337 = sorted((ROOT / "results/experiments").glob("v2_track_dialog_nojudge_seed1337_reuse_*"))
    if dial_1337:
        dial_runs.append(("seed=1337", dial_1337[-1].name, "0p3"))

    code_runs = [
        ("seed=42", "v2_track_code_mbpp_20260507_164026", "0p3"),
    ]
    code_1337 = sorted((ROOT / "results/experiments").glob("v2_track_code_mbpp_seed1337_*"))
    if code_1337:
        code_runs.append(("seed=1337", code_1337[-1].name, "0p3"))

    panels = [
        ("Summarization (ROUGE-L F1)", "rougeL_f1", sum_runs, 0.27, 0.40),
        ("Dialog (heuristic)", "dialog_heuristic", dial_runs, 0.40, 0.62),
        ("Code MBPP (test_pass_rate)", "test_pass_rate", code_runs, 0.0, 0.025),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    w = 0.27
    for ax, (title, key, runs, ymin, ymax) in zip(axes, panels):
        labels, base_v, sft_v, dpo_v = [], [], [], []
        for lbl, exp_dir, slug in runs:
            m = metrics_for(exp_dir, slug)
            labels.append(lbl)
            base_v.append(m["base"][key])
            sft_v.append(m["sft"][key])
            dpo_v.append(m["dpo"][key])

        x = np.arange(len(labels))
        ax.bar(x - w, base_v, w, label="Base", color=COLORS["base"])
        ax.bar(x, sft_v, w, label="SFT", color=COLORS["sft"])
        ax.bar(x + w, dpo_v, w, label="DPO (β=0.3, on-policy IPO)", color=COLORS["dpo"])
        ymax_actual = max(max(base_v), max(sft_v), max(dpo_v))
        for xi, b, s, d in zip(x, base_v, sft_v, dpo_v):
            for off, val in [(-w, b), (0, s), (w, d)]:
                fmt = f"{val:.3f}" if ymax_actual > 0.05 else f"{val:.4f}"
                ax.text(xi + off, val + ymax_actual * 0.018, fmt, ha="center", fontsize=8)
            delta = d - s
            col = "darkgreen" if delta > 0 else ("darkred" if delta < 0 else "gray")
            ax.text(xi, max(s, d) + ymax_actual * 0.10, f"Δ={delta:+.3f}",
                    ha="center", fontsize=8, color=col, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(ymin, max(ymax, ymax_actual * 1.18))
        ax.grid(axis="y", alpha=0.3)
        ax.legend(loc="upper right", frameon=False, fontsize=8)

    fig.suptitle("Multi-seed устойчивость на трёх треках (on-policy IPO, β=0.3)", y=1.02)
    save(fig, "fig_multiseed_summarization")


def fig_cross_domain():
    """Полная 3x3 cross-domain матрица: 3 eval-домена × 3 train-источника + base.

    Артефакты:
       in-domain runs: v2_track_{summarization, dialog, code_mbpp}_*
       cross runs: results/cross_domain_{d1..d6}_*/metrics/metrics.json
    """
    print("\n[fig_cross_domain]")
    in_sum = metrics_for("v2_track_summarization_nojudge_20260506_210741", "0p3")
    in_dial = metrics_for("v2_track_dialog_nojudge_20260507_000348", "0p3")
    in_code = metrics_for("v2_track_code_mbpp_20260507_164026", "0p3")

    d1 = load_json(ROOT / "results/cross_domain_d1_sum_to_mbpp/metrics/metrics.json")
    d2 = load_json(ROOT / "results/cross_domain_d2_mbpp_to_sum/metrics/metrics.json")
    d3 = load_json(ROOT / "results/cross_domain_d3_dialog_to_mbpp/metrics/metrics.json")
    d4 = load_json(ROOT / "results/cross_domain_d4_dialog_to_sum/metrics/metrics.json")
    d5 = load_json(ROOT / "results/cross_domain_d5_code_to_dialog/metrics/metrics.json")
    d6 = load_json(ROOT / "results/cross_domain_d6_sum_to_dialog/metrics/metrics.json")

    panels = [
        ("Eval = summarization (ROUGE-L F1)", "rougeL_f1",
         [("In-domain (sum)", in_sum), ("D2: code→sum", d2), ("D4: dialog→sum", d4)]),
        ("Eval = dialog (heuristic)", "dialog_heuristic",
         [("In-domain (dialog)", in_dial), ("D5: code→dialog", d5), ("D6: sum→dialog", d6)]),
        ("Eval = code MBPP (test pass-rate)", "test_pass_rate",
         [("In-domain (code)", in_code), ("D1: sum→code", d1), ("D3: dialog→code", d3)]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.0))
    w = 0.27
    for ax, (title, key, rows) in zip(axes, panels):
        labels = [r[0] for r in rows]
        base_v = [r[1]["base"][key] for r in rows]
        sft_v = [r[1]["sft"][key] for r in rows]
        dpo_v = [r[1]["dpo"][key] for r in rows]
        x = np.arange(len(labels))
        ax.bar(x - w, base_v, w, label="Base", color=COLORS["base"])
        ax.bar(x, sft_v, w, label="SFT", color=COLORS["sft"])
        ax.bar(x + w, dpo_v, w, label="DPO (β=0.3)", color=COLORS["dpo"])
        ymax = max(max(base_v), max(sft_v), max(dpo_v))
        for xi, b, s, d in zip(x, base_v, sft_v, dpo_v):
            for off, val in [(-w, b), (0, s), (w, d)]:
                ax.text(xi + off, val + ymax * 0.018,
                        f"{val:.3f}" if ymax > 0.05 else f"{val:.4f}",
                        ha="center", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, ymax * 1.18)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(loc="upper right", frameon=False, fontsize=8)
    fig.suptitle("Полная 3×3 cross-domain матрица: in-domain (D-) vs cross-domain (D1–D6)", y=1.02)
    save(fig, "fig_cross_domain_mbpp")


def fig_sigmoid_vs_ipo():
    """Sigmoid-DPO vs IPO при той же on-policy схеме (β=0.3, seed=42)."""
    print("\n[fig_sigmoid_vs_ipo]")
    ipo = metrics_for("v2_track_summarization_nojudge_20260506_210741", "0p3")
    sig_dirs = sorted((ROOT / "results/experiments").glob("v2_sum_sigmoid_onpolicy_*"))
    if not sig_dirs:
        print("  v2_sum_sigmoid_onpolicy not found")
        return
    sigm = load_json(sig_dirs[-1] / "metrics" / "metrics_beta_0p3.json")

    labels = ["Base", "SFT", "DPO\n(IPO loss)", "DPO\n(sigmoid loss)"]
    vals = [ipo["base"]["rougeL_f1"], ipo["sft"]["rougeL_f1"], ipo["dpo"]["rougeL_f1"], sigm["dpo"]["rougeL_f1"]]
    colors = [COLORS["base"], COLORS["sft"], COLORS["dpo"], "#9467bd"]

    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    bars = ax.bar(labels, vals, color=colors)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.003, f"{val:.3f}", ha="center", fontsize=9)
    ax.set_ylabel("ROUGE-L F1 (summarization)")
    ax.set_title("Sigmoid vs IPO при on-policy preference-pairs (β=0.3, seed=42)")
    ax.set_ylim(0.27, max(vals) + 0.02)
    ax.grid(axis="y", alpha=0.3)
    save(fig, "fig_sigmoid_vs_ipo")


def fig_ablation_multiseed():
    """Multi-seed ablation: baseline-pairs vs on-policy на двух seed-ах."""
    print("\n[fig_ablation_multiseed]")
    onp_42 = metrics_for("v2_track_summarization_nojudge_20260506_210741", "0p3")
    onp_1337 = metrics_for("v2_track_summarization_seed1337_nojudge_20260507_043001", "0p3")
    bls_42 = metrics_for("v2_sum_baseline_pairs_nojudge_20260507_024739", "0p3")
    bls_1337_dirs = sorted((ROOT / "results/experiments").glob("v2_sum_baseline_pairs_seed1337_*"))
    if not bls_1337_dirs:
        print("  baseline-pairs seed=1337 not found")
        return
    bls_1337 = load_json(bls_1337_dirs[-1] / "metrics" / "metrics_beta_0p3.json")

    rows = [
        ("baseline-pairs\nseed=42", bls_42),
        ("baseline-pairs\nseed=1337", bls_1337),
        ("on-policy\nseed=42", onp_42),
        ("on-policy\nseed=1337", onp_1337),
    ]
    labels = [r[0] for r in rows]
    sft_v = [r[1]["sft"]["rougeL_f1"] for r in rows]
    dpo_v = [r[1]["dpo"]["rougeL_f1"] for r in rows]

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.bar(x - w / 2, sft_v, w, label="SFT", color=COLORS["sft"])
    ax.bar(x + w / 2, dpo_v, w, label="DPO (IPO, β=0.3)", color=COLORS["dpo"])
    for xi, s, d in zip(x, sft_v, dpo_v):
        ax.text(xi - w / 2, s + 0.003, f"{s:.3f}", ha="center", fontsize=8)
        ax.text(xi + w / 2, d + 0.003, f"{d:.3f}", ha="center", fontsize=8)
        delta = d - s
        col = "darkred" if delta < 0 else "darkgreen"
        ax.text(xi, max(s, d) + 0.018, f"Δ={delta:+.3f}", ha="center", fontsize=8, color=col, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("ROUGE-L F1 (summarization)")
    ax.set_title("Multi-seed apples-to-apples ablation: baseline-pairs vs on-policy\n(seed=42 и seed=1337)")
    ax.axvline(x=1.5, color="black", linestyle="--", alpha=0.4)
    ax.set_ylim(0.27, max(max(sft_v), max(dpo_v)) + 0.04)
    ax.legend(loc="lower right", frameon=False)
    ax.grid(axis="y", alpha=0.3)
    save(fig, "fig_ablation_multiseed")


def fig_judge_winrates():
    """LLM-judge winrates по трекам."""
    print("\n[fig_judge_winrates]")
    rows = [
        ("v2_track_summarization_nojudge_20260506_210741", "0p3", "summarization\n(seed=42)"),
        ("v2_track_summarization_seed1337_nojudge_20260507_043001", "0p3", "summarization\n(seed=1337)"),
        ("v2_sum_baseline_pairs_nojudge_20260507_024739", "0p3", "summarization\nbaseline-pairs"),
        ("v2_track_dialog_nojudge_20260507_000348", "0p3", "dialog\n(seed=42)"),
        ("v2_track_code_mbpp_20260507_164026", "0p3", "code (MBPP)\n(seed=42)"),
    ]
    labels, base_w, sft_w, dpo_w = [], [], [], []
    for exp_dir, slug, label in rows:
        m = metrics_for(exp_dir, slug)
        wr = m.get("llm_judge", {}).get("win_rates", {})
        if not wr:
            continue
        labels.append(label)
        base_w.append(wr.get("base", 0))
        sft_w.append(wr.get("sft", 0))
        dpo_w.append(wr.get("dpo", 0))

    x = np.arange(len(labels))
    w = 0.27
    fig, ax = plt.subplots(figsize=(8.0, 3.7))
    ax.bar(x - w, base_w, w, label="Base", color=COLORS["base"])
    ax.bar(x, sft_w, w, label="SFT", color=COLORS["sft"])
    ax.bar(x + w, dpo_w, w, label="DPO", color=COLORS["dpo"])

    for xi, b, s, d in zip(x, base_w, sft_w, dpo_w):
        for off, val in [(-w, b), (0, s), (w, d)]:
            if val > 0.005:
                ax.text(xi + off, val + 0.003, f"{val:.2f}", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("LLM-judge win-rate (vs ties)")
    ax.set_title("LLM-judge: глобальные win-rates по трекам (qwen2.5:7b, n=25)")
    ax.legend(loc="upper right", frameon=False)
    save(fig, "fig_judge_winrates")


def fig_raw_vs_clean():
    """RAW vs CLEAN base ROUGE-L: эффект симметричной обрезки."""
    print("\n[fig_raw_vs_clean]")
    raw_clean_md = ROOT / "results/raw_vs_clean.md"
    if not raw_clean_md.exists():
        print("  raw_vs_clean.md не найден — пропускаю")
        return

    points = [
        ("summarization\nrougeL_f1\nbase", 0.188, 0.294),
        ("dialog\ndialog_heuristic\nbase", 0.78, 0.575),
        ("code (MBPP)\ntest_pass_rate\nbase", 0.017, 0.017),
    ]

    labels = [p[0] for p in points]
    raw = [p[1] for p in points]
    clean = [p[2] for p in points]

    x = np.arange(len(labels))
    w = 0.4
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.bar(x - w / 2, raw, w, label="RAW (без обрезки)", color="#bcbd22")
    ax.bar(x + w / 2, clean, w, label="CLEAN (truncate_at_instruction)", color="#2ca02c")

    for xi, r, c in zip(x, raw, clean):
        ax.text(xi - w / 2, r + 0.01, f"{r:.3f}", ha="center", fontsize=8)
        ax.text(xi + w / 2, c + 0.01, f"{c:.3f}", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Метрика (base модель)")
    ax.set_title("Эффект симметричной постобработки: RAW vs CLEAN")
    ax.legend(loc="upper right", frameon=False)
    save(fig, "fig_raw_vs_clean")


def fig_track_summary():
    """Сводный bar по 3 трекам: base/SFT/DPO для главного v2 прогона."""
    print("\n[fig_track_summary]")
    rows = [
        ("Summarization\n(rougeL F1, seed=42)", "v2_track_summarization_nojudge_20260506_210741", "0p3", "rougeL_f1"),
        ("Dialog\n(heuristic, seed=42)", "v2_track_dialog_nojudge_20260507_000348", "0p3", "dialog_heuristic"),
        ("Code MBPP\n(pass-rate, seed=42)", "v2_track_code_mbpp_20260507_164026", "0p3", "test_pass_rate"),
    ]

    labels, base_v, sft_v, dpo_v = [], [], [], []
    for label, exp_dir, slug, key in rows:
        m = metrics_for(exp_dir, slug)
        labels.append(label)
        base_v.append(m["base"][key])
        sft_v.append(m["sft"][key])
        dpo_v.append(m["dpo"][key])

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, label, b, s, d in zip(axes, labels, base_v, sft_v, dpo_v):
        ax.bar(["Base", "SFT", "DPO"], [b, s, d], color=[COLORS["base"], COLORS["sft"], COLORS["dpo"]])
        for i, val in enumerate([b, s, d]):
            ax.text(i, val + max(b, s, d) * 0.02, f"{val:.3f}", ha="center", fontsize=9)
        ax.set_title(label)
        ax.set_ylim(0, max(b, s, d) * 1.18 + 1e-3)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Сводка v2 по трекам (on-policy IPO, β=0.3)", y=1.02)
    save(fig, "fig_track_summary")


def main():
    print(f"Saving figures to {FIG_DIR.relative_to(ROOT)}")
    fig_beta_curve()
    fig_dpo_dynamics()
    fig_multiseed()
    fig_cross_domain()
    fig_judge_winrates()
    fig_raw_vs_clean()
    fig_track_summary()
    fig_sigmoid_vs_ipo()
    fig_ablation_multiseed()
    print("\nDone.")


if __name__ == "__main__":
    main()
