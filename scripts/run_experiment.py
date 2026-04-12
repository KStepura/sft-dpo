#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def load_config(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except Exception as exc:
            raise SystemExit("YAML config requires PyYAML. Use JSON or install pyyaml.") from exc
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    raise SystemExit(f"Unsupported config format: {path}")


def slug_beta(beta: float) -> str:
    return str(beta).replace(".", "p")


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def run_step(step_name: str, cmd: list[str], log_path: Path):
    print(f"\n[{step_name}] {' '.join(cmd)}")
    t0 = time.time()
    with log_path.open("w", encoding="utf-8") as logf:
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, check=False)
    dt = time.time() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"Step '{step_name}' failed (exit={proc.returncode}). See {log_path}")
    return {"name": step_name, "seconds": round(dt, 2), "log_path": str(log_path)}


def quick_compare_stats(path: Path):
    n = 0
    lens = {"base": 0, "sft": 0, "dpo": 0}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            n += 1
            lens["base"] += len(rec.get("base", ""))
            lens["sft"] += len(rec.get("sft", ""))
            lens["dpo"] += len(rec.get("dpo", ""))
    if n == 0:
        return {"num_prompts": 0}
    return {
        "num_prompts": n,
        "avg_len_base": round(lens["base"] / n, 1),
        "avg_len_sft": round(lens["sft"] / n, 1),
        "avg_len_dpo": round(lens["dpo"] / n, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to JSON/YAML experiment config")
    ap.add_argument("--exp_id", default=None, help="Optional custom experiment id")
    args = ap.parse_args()

    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)

    exp_id = args.exp_id or f"{cfg.get('name', 'exp')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    exp_dir = Path("results") / "experiments" / exp_id
    logs_dir = exp_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    base_model_id = cfg["base_model_id"]
    dataset_id = cfg.get("dataset_id", "tatsu-lab/alpaca")
    seed = int(cfg.get("seed", 42))
    prompt_style = cfg.get("prompt_style", "chat")

    prompts = cfg["eval"]["prompts"]
    prompts_path = exp_dir / "prompts.txt"
    prompts_path.write_text("\n".join(prompts) + "\n", encoding="utf-8")

    data_raw = Path(cfg["paths"]["data_raw"])
    data_clean = Path(cfg["paths"]["data_clean"])
    outputs_root = Path(cfg["paths"]["outputs_root"]) / exp_id
    sft_out = outputs_root / "sft"
    compare_root = exp_dir / "compare"

    dpo_betas = [float(x) for x in cfg["dpo"]["betas"]]
    compare_summaries = []
    steps = []

    steps.append(
        run_step(
            "build_dpo_pairs",
            [
                sys.executable,
                "scripts/build_dpo_pairs.py",
                "--base_model_id",
                base_model_id,
                "--dataset_id",
                dataset_id,
                "--out_path",
                str(data_raw),
                "--num_examples",
                str(cfg["data"]["num_examples"]),
                "--seed",
                str(seed),
                "--max_new_tokens",
                str(cfg["data"]["max_new_tokens"]),
                "--temperature",
                str(cfg["data"]["temperature"]),
                "--top_p",
                str(cfg["data"]["top_p"]),
                "--prompt_style",
                str(prompt_style),
            ],
            logs_dir / "01_build_dpo_pairs.log",
        )
    )

    clean_args = [
        sys.executable,
        "scripts/clean_dpo_pairs_full.py",
        "--prompt_style",
        str(prompt_style),
        "--in_path",
        str(data_raw),
        "--out_path",
        str(data_clean),
        "--min_chars",
        str(cfg["data"].get("min_chars", 20)),
        "--stats_path",
        str(exp_dir / "clean_stats.json"),
    ]
    if cfg["data"].get("drop_if_truncated", True):
        clean_args.append("--drop_if_truncated")
    if cfg["data"].get("drop_if_equal", True):
        clean_args.append("--drop_if_equal")
    if cfg["data"].get("make_strict_prompt", True):
        clean_args.append("--make_strict_prompt")

    steps.append(run_step("clean_dpo_pairs", clean_args, logs_dir / "02_clean_dpo_pairs.log"))

    train_sft_cmd = [
        sys.executable,
        "scripts/train_sft.py",
        "--model_id",
        base_model_id,
        "--dataset_id",
        dataset_id,
        "--output_dir",
        str(sft_out),
        "--max_steps",
        str(cfg["sft"]["max_steps"]),
        "--lr",
        str(cfg["sft"]["lr"]),
        "--batch_size",
        str(cfg["sft"]["batch_size"]),
        "--grad_accum",
        str(cfg["sft"]["grad_accum"]),
        "--max_seq_len",
        str(cfg["sft"]["max_seq_len"]),
        "--seed",
        str(seed),
        "--prompt_style",
        str(prompt_style),
    ]
    mts = cfg["sft"].get("max_train_samples")
    if mts is not None and int(mts) > 0:
        train_sft_cmd.extend(["--max_train_samples", str(int(mts))])

    steps.append(run_step("train_sft", train_sft_cmd, logs_dir / "03_train_sft.log"))

    for beta in dpo_betas:
        beta_slug = slug_beta(beta)
        dpo_out = outputs_root / f"dpo_beta_{beta_slug}"
        compare_out = compare_root / f"compare_beta_{beta_slug}.jsonl"

        steps.append(
            run_step(
                f"train_dpo_beta_{beta_slug}",
                [
                    sys.executable,
                    "scripts/train_dpo.py",
                    "--base_model_id",
                    base_model_id,
                    "--sft_adapter_dir",
                    str(sft_out),
                    "--dpo_data_path",
                    str(data_clean),
                    "--output_dir",
                    str(dpo_out),
                    "--max_steps",
                    str(cfg["dpo"]["max_steps"]),
                    "--batch_size",
                    str(cfg["dpo"]["batch_size"]),
                    "--grad_accum",
                    str(cfg["dpo"]["grad_accum"]),
                    "--max_length",
                    str(cfg["dpo"]["max_length"]),
                    "--max_prompt_length",
                    str(cfg["dpo"]["max_prompt_length"]),
                    "--lr",
                    str(cfg["dpo"]["lr"]),
                    "--beta",
                    str(beta),
                    "--seed",
                    str(seed),
                ],
                logs_dir / f"04_train_dpo_beta_{beta_slug}.log",
            )
        )

        compare_cmd = [
            sys.executable,
            "scripts/compare_models.py",
            "--base_model_id",
            base_model_id,
            "--sft_adapter_dir",
            str(sft_out),
            "--dpo_adapter_dir",
            str(dpo_out),
            "--prompts_path",
            str(prompts_path),
            "--out_path",
            str(compare_out),
            "--max_new_tokens",
            str(cfg["eval"]["max_new_tokens"]),
            "--temperature",
            str(cfg["eval"]["temperature"]),
            "--top_p",
            str(cfg["eval"]["top_p"]),
            "--seed",
            str(seed),
        ]
        if cfg["eval"].get("do_sample", False):
            compare_cmd.append("--do_sample")

        steps.append(
            run_step(
                f"compare_beta_{beta_slug}",
                compare_cmd,
                logs_dir / f"05_compare_beta_{beta_slug}.log",
            )
        )
        compare_summaries.append({"beta": beta, "path": str(compare_out), "stats": quick_compare_stats(compare_out)})

    summary = {
        "exp_id": exp_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config_path": str(config_path),
        "seed": seed,
        "steps": steps,
        "compare": compare_summaries,
    }
    write_json(exp_dir / "summary.json", summary)
    write_json(exp_dir / "resolved_config.json", cfg)
    print(f"\nExperiment complete: {exp_dir}")


if __name__ == "__main__":
    main()
