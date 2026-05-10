#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from dpo_metrics_utils import enrich_run_meta, extract_dpo_metrics
from experiment_env import require_cuda


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
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    with log_path.open("w", encoding="utf-8") as logf:
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, check=False, env=env)
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
    ap.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow running without CUDA (debug only; extremely slow).",
    )
    args = ap.parse_args()

    require_cuda(allow_cpu=args.allow_cpu, label="run_experiment")

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

    eval_cfg = cfg["eval"]
    refs_src = eval_cfg.get("refs_path")
    if refs_src:
        refs_abs = Path(refs_src).resolve()
        if not refs_abs.exists():
            raise RuntimeError(f"refs_path not found: {refs_abs}")
        shutil.copy(refs_abs, exp_dir / "refs.jsonl")
        prompts_path = exp_dir / "refs.jsonl"
    else:
        prompts_path = exp_dir / "prompts.txt"
        prompts_path.write_text("\n".join(eval_cfg["prompts"]) + "\n", encoding="utf-8")

    data_raw = Path(cfg["paths"]["data_raw"])
    data_clean = Path(cfg["paths"]["data_clean"])
    outputs_root = Path(cfg["paths"]["outputs_root"]) / exp_id
    sft_out = outputs_root / "sft"
    # Allow reusing an existing SFT adapter (e.g. for β-sweep without retraining)
    if cfg["paths"].get("sft_adapter_dir"):
        sft_out = Path(cfg["paths"]["sft_adapter_dir"]).resolve()
    compare_root = exp_dir / "compare"

    skip_steps = set(cfg.get("skip_steps", []))

    dpo_betas = [float(x) for x in cfg["dpo"]["betas"]]
    judge_cfg = cfg.get("judge", {})
    compare_summaries = []
    steps = []

    data_cfg = cfg["data"]
    policy_source = data_cfg.get("policy_source", "baseline")

    def _skip(name: str):
        print(f"\n[{name}] SKIPPED (in skip_steps)")
        return {"name": name, "seconds": 0, "skipped": True, "log_path": ""}

    def build_step():
        if "build_dpo_pairs" in skip_steps:
            return _skip("build_dpo_pairs")
        cmd = [
            sys.executable,
            "scripts/build_dpo_pairs.py",
            "--base_model_id", base_model_id,
            "--dataset_id", dataset_id,
            "--out_path", str(data_raw),
            "--num_examples", str(data_cfg["num_examples"]),
            "--seed", str(seed),
            "--max_new_tokens", str(data_cfg["max_new_tokens"]),
            "--temperature", str(data_cfg["temperature"]),
            "--top_p", str(data_cfg["top_p"]),
            "--prompt_style", str(prompt_style),
            "--policy_source", policy_source,
        ]
        if policy_source != "baseline":
            cmd += [
                "--sft_adapter_dir", str(sft_out),
                "--chosen_temperature", str(data_cfg.get("chosen_temperature", 0.0)),
                "--rejected_temperature", str(data_cfg.get("rejected_temperature", 1.2)),
                "--rejected_top_p", str(data_cfg.get("rejected_top_p", 0.95)),
            ]
        return run_step("build_dpo_pairs", cmd, logs_dir / "01_build_dpo_pairs.log")

    def clean_step():
        if "clean_dpo_pairs" in skip_steps:
            return _skip("clean_dpo_pairs")
        cmd = [
            sys.executable,
            "scripts/clean_dpo_pairs_full.py",
            "--prompt_style", str(prompt_style),
            "--in_path", str(data_raw),
            "--out_path", str(data_clean),
            "--min_chars", str(data_cfg.get("min_chars", 20)),
            "--stats_path", str(exp_dir / "clean_stats.json"),
        ]
        if data_cfg.get("drop_if_truncated", True):
            cmd.append("--drop_if_truncated")
        if data_cfg.get("drop_if_equal", True):
            cmd.append("--drop_if_equal")
        if data_cfg.get("make_strict_prompt", True):
            cmd.append("--make_strict_prompt")
        return run_step("clean_dpo_pairs", cmd, logs_dir / "02_clean_dpo_pairs.log")

    def sft_step():
        if "train_sft" in skip_steps:
            return _skip("train_sft")
        cmd = [
            sys.executable,
            "scripts/train_sft.py",
            "--model_id", base_model_id,
            "--dataset_id", dataset_id,
            "--output_dir", str(sft_out),
            "--max_steps", str(cfg["sft"]["max_steps"]),
            "--lr", str(cfg["sft"]["lr"]),
            "--batch_size", str(cfg["sft"]["batch_size"]),
            "--grad_accum", str(cfg["sft"]["grad_accum"]),
            "--max_seq_len", str(cfg["sft"]["max_seq_len"]),
            "--seed", str(seed),
            "--prompt_style", str(prompt_style),
        ]
        mts = cfg["sft"].get("max_train_samples")
        if mts is not None and int(mts) > 0:
            cmd.extend(["--max_train_samples", str(int(mts))])
        # SFTTrainer's `packing=True` warns about cross-contamination between
        # samples when flash_attention_2 is unavailable. Default to no_packing
        # for clean gradients; configs can opt back in.
        if cfg["sft"].get("no_packing", True):
            cmd.append("--no_packing")
        return run_step("train_sft", cmd, logs_dir / "03_train_sft.log")

    if policy_source == "baseline":
        steps.append(build_step())
        steps.append(clean_step())
        steps.append(sft_step())
    else:
        # On-policy / mixed need the SFT adapter before pair generation.
        steps.append(sft_step())
        steps.append(build_step())
        steps.append(clean_step())

    dpo_cfg = cfg["dpo"]

    for beta in dpo_betas:
        beta_slug = slug_beta(beta)
        dpo_out = outputs_root / f"dpo_beta_{beta_slug}"
        compare_out = compare_root / f"compare_beta_{beta_slug}.jsonl"

        train_dpo_cmd = [
            sys.executable,
            "scripts/train_dpo.py",
            "--base_model_id", base_model_id,
            "--sft_adapter_dir", str(sft_out),
            "--dpo_data_path", str(data_clean),
            "--output_dir", str(dpo_out),
            "--max_steps", str(dpo_cfg["max_steps"]),
            "--batch_size", str(dpo_cfg["batch_size"]),
            "--grad_accum", str(dpo_cfg["grad_accum"]),
            "--max_length", str(dpo_cfg["max_length"]),
            "--max_prompt_length", str(dpo_cfg["max_prompt_length"]),
            "--lr", str(dpo_cfg["lr"]),
            "--beta", str(beta),
            "--seed", str(seed),
        ]
        loss_type = dpo_cfg.get("loss_type")
        if loss_type:
            train_dpo_cmd += ["--loss_type", str(loss_type)]
        save_steps = dpo_cfg.get("save_steps")
        if save_steps is not None:
            train_dpo_cmd += ["--save_steps", str(int(save_steps))]
        save_total_limit = dpo_cfg.get("save_total_limit")
        if save_total_limit is not None:
            train_dpo_cmd += ["--save_total_limit", str(int(save_total_limit))]

        steps.append(
            run_step(
                f"train_dpo_beta_{beta_slug}",
                train_dpo_cmd,
                logs_dir / f"04_train_dpo_beta_{beta_slug}.log",
            )
        )

        # Optional: pick the DPO checkpoint with the highest dev judge winrate
        # (or auto metric, when no judge is configured). DPO often peaks early
        # and then collapses into reward hacking — selecting on the final
        # weights penalises every method except the safest β.
        dpo_compare_target = dpo_out
        if dpo_cfg.get("select_checkpoint", False):
            select_cmd = [
                sys.executable,
                "scripts/analysis/select_best_dpo_checkpoint.py",
                "--base_model_id", base_model_id,
                "--sft_adapter_dir", str(sft_out),
                "--dpo_dir", str(dpo_out),
                "--refs_path", str(prompts_path),
                "--track", str(cfg.get("track", "")),
                "--prompt_style", str(prompt_style),
                "--max_new_tokens", str(eval_cfg["max_new_tokens"]),
                "--seed", str(seed),
                "--best_link", str(dpo_out.parent / f"dpo_beta_{beta_slug}_best"),
                "--report_path", str(dpo_out / "checkpoint_selection.json"),
            ]
            judge_model = judge_cfg.get("model", "")
            if judge_model:
                select_cmd += [
                    "--judge_model", judge_model,
                    "--judge_base_url", judge_cfg.get("base_url", "http://127.0.0.1:11434/v1"),
                    "--judge_sleep_sec", str(judge_cfg.get("sleep_sec", 0.15)),
                    "--judge_timeout_sec", str(judge_cfg.get("timeout_sec", 120)),
                ]
                if judge_cfg.get("api_key"):
                    select_cmd += ["--judge_api_key", judge_cfg["api_key"]]
                if judge_cfg.get("json_mode", False):
                    select_cmd.append("--judge_json_mode")
            steps.append(
                run_step(
                    f"select_best_dpo_beta_{beta_slug}",
                    select_cmd,
                    logs_dir / f"04b_select_best_dpo_beta_{beta_slug}.log",
                )
            )
            best_dir = dpo_out.parent / f"dpo_beta_{beta_slug}_best"
            if best_dir.exists():
                dpo_compare_target = best_dir

        dpo_metrics = enrich_run_meta(
            extract_dpo_metrics(dpo_compare_target),
            beta=beta,
            max_steps=int(dpo_cfg["max_steps"]),
            dpo_data_path=str(data_clean),
        )
        dpo_metrics["selected_checkpoint"] = str(dpo_compare_target)
        write_json(dpo_compare_target / "dpo_metrics.json", dpo_metrics)

        compare_cmd = [
            sys.executable,
            "scripts/compare_models.py",
            "--base_model_id",
            base_model_id,
            "--sft_adapter_dir",
            str(sft_out),
            "--dpo_adapter_dir",
            str(dpo_compare_target),
            "--prompts_path",
            str(prompts_path),
            "--out_path",
            str(compare_out),
            "--max_new_tokens",
            str(eval_cfg["max_new_tokens"]),
            "--temperature",
            str(eval_cfg["temperature"]),
            "--top_p",
            str(eval_cfg["top_p"]),
            "--prompt_style",
            str(prompt_style),
            "--seed",
            str(seed),
        ]
        if eval_cfg.get("do_sample", False):
            compare_cmd.append("--do_sample")
        # Symmetric truncation of hallucinated "### Instruction:" tails for ALL
        # three models. Without this the base column inflates length, which
        # tanks rougeL precision and makes any SFT delta look larger than it is.
        # Default ON so historical configs without the field are still corrected.
        if eval_cfg.get("postprocess_outputs", True):
            compare_cmd.append("--postprocess_outputs")

        steps.append(
            run_step(
                f"compare_beta_{beta_slug}",
                compare_cmd,
                logs_dir / f"05_compare_beta_{beta_slug}.log",
            )
        )
        entry = {
            "beta": beta,
            "path": str(compare_out),
            "stats": quick_compare_stats(compare_out),
            "dpo_train_dir": str(dpo_out),
            "dpo_compare_dir": str(dpo_compare_target),
            "dpo_metrics_path": str(dpo_compare_target / "dpo_metrics.json"),
            "dpo_metrics": dpo_metrics,
        }
        track = cfg.get("track")
        if track in ("dialog", "summarization", "code") and (exp_dir / "refs.jsonl").exists():
            metrics_path = exp_dir / "metrics" / f"metrics_beta_{beta_slug}.json"
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            eval_metrics_cmd = [
                sys.executable,
                "scripts/eval_metrics.py",
                "--track", track,
                "--compare_path", str(compare_out),
                "--refs_path", str(exp_dir / "refs.jsonl"),
                "--out_path", str(metrics_path),
            ]
            judge_model = judge_cfg.get("model", "")
            if judge_model:
                eval_metrics_cmd += [
                    "--judge_model", judge_model,
                    "--judge_base_url", judge_cfg.get("base_url", "http://127.0.0.1:11434/v1"),
                    "--judge_sleep_sec", str(judge_cfg.get("sleep_sec", 0.15)),
                    "--judge_timeout_sec", str(judge_cfg.get("timeout_sec", 120)),
                ]
                api_key = judge_cfg.get("api_key", "")
                if api_key:
                    eval_metrics_cmd += ["--judge_api_key", api_key]
                if judge_cfg.get("json_mode", False):
                    eval_metrics_cmd.append("--judge_json_mode")
            steps.append(
                run_step(
                    f"eval_metrics_beta_{beta_slug}",
                    eval_metrics_cmd,
                    logs_dir / f"06_eval_metrics_beta_{beta_slug}.log",
                )
            )
            try:
                entry["metrics_path"] = str(metrics_path)
                entry["metrics"] = json.loads(metrics_path.read_text(encoding="utf-8"))
            except Exception:
                entry["metrics_path"] = str(metrics_path)
                entry["metrics"] = None
        compare_summaries.append(entry)

    summary = {
        "exp_id": exp_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config_path": str(config_path),
        "seed": seed,
        "track": cfg.get("track", ""),
        "clean_stats_path": str(exp_dir / "clean_stats.json"),
        "steps": steps,
        "compare": compare_summaries,
    }
    write_json(exp_dir / "summary.json", summary)
    write_json(exp_dir / "resolved_config.json", cfg)
    print(f"\nExperiment complete: {exp_dir}")


if __name__ == "__main__":
    main()
