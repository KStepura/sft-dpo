#!/usr/bin/env python3
"""Pick the DPO checkpoint with the best dev-set score.

Why: DPO trained on a strong-vs-weak pair distribution often peaks early and
then collapses into reward hacking (`logps/rejected → −∞`). Training to the
final step penalises every β; instead, save every N steps and pick the
checkpoint that wins on a held-out dev split.

This script:
  1. Splits refs.jsonl into dev (last DEV_N rows) and the rest stays for eval.
  2. Loads base + tokenizer + frozen SFT adapter once.
  3. For each `checkpoint-*` in --dpo_dir, swaps the LoRA adapter, generates
     greedy completions on dev prompts, applies symmetric truncation and
     scores them with the track-specific auto-metric.
  4. Argmax over checkpoints → creates a symlink at --best_link.
  5. Writes a JSON report at --report_path.

LLM judge is intentionally NOT called per-checkpoint here: it is too slow
(O(N_ckpt × N_dev × 6 calls)). We rely on the auto-metric for selection and
let the final eval (post-selection) run the judge once on the chosen ckpt.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from prompt_utils import format_alpaca_prompt_prefix, format_chat_prompt_prefix  # noqa: E402

try:
    from peft import PeftModel
except ImportError as e:
    raise SystemExit("peft not installed") from e


DEV_N_DEFAULT = 8


def truncate_at_instruction(text: str) -> str:
    text = (text or "").strip()
    for cut_mark in ["\n\n### Instruction:", "\n### Instruction:"]:
        if cut_mark in text:
            text = text.split(cut_mark, 1)[0].strip()
    if text.count("### Response:") >= 2:
        text = text.split("### Response:")[-1].strip()
    return text


def extract_code_block(text: str) -> str:
    text = text or ""
    m = re.search(r"```(?:python)?\n(.*?)```", text, flags=re.S)
    if m:
        return m.group(1).strip()
    m2 = re.search(r"(?m)^(def |class |import |from )", text)
    if m2:
        text = text[m2.start():]
    return text.strip()


def run_python_tests(code: str, tests: List[str], timeout_sec: int = 8):
    if not tests:
        return 0, 0
    passed, total = 0, len(tests)
    for test in tests:
        script = code + "\n\n" + test + "\n"
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=True, encoding="utf-8") as tf:
            tf.write(script)
            tf.flush()
            proc = subprocess.run(
                [sys.executable, tf.name],
                capture_output=True, text=True,
                timeout=timeout_sec, check=False,
            )
        if proc.returncode == 0:
            passed += 1
    return passed, total


def make_prompt(prompt: str, tok, prompt_style: str) -> str:
    if prompt_style == "chat":
        return format_chat_prompt_prefix(tok, prompt, "")
    return format_alpaca_prompt_prefix(prompt, "")


def load_refs(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def score_dev(track: str, dev_rows, generations) -> float:
    if track == "summarization":
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        vals = []
        for ref, pred in zip(dev_rows, generations):
            s = scorer.score(ref["reference"], pred).get("rougeL")
            vals.append(s.fmeasure if s else 0.0)
        return sum(vals) / max(len(vals), 1)
    if track == "code":
        passed, total = 0, 0
        for ref, pred in zip(dev_rows, generations):
            code = extract_code_block(pred)
            p, t = run_python_tests(code, ref.get("tests") or [])
            passed += p
            total += t
        return (passed / total) if total else 0.0
    # dialog (or anything else): use a discriminative heuristic similar to
    # the patched eval_metrics.py — bounded length + Jaccard with reference.
    polite = ("please", "thank", "sorry", "appreciate", "kindly")
    scores = []
    for ref, pred in zip(dev_rows, generations):
        txt = pred.lower()
        ref_text = (ref.get("reference") or "").lower()
        s = 0.0
        n_words = len(txt.split())
        if 40 <= n_words <= 200:
            s += 0.3
        elif 20 <= n_words < 40 or 200 < n_words <= 350:
            s += 0.15
        if ref_text and txt:
            ref_t = set(ref_text.split())
            txt_t = set(txt.split())
            union = len(ref_t | txt_t)
            if union:
                s += min(0.4, len(ref_t & txt_t) / union * 2.0)
        if any(w in txt for w in polite):
            s += 0.15
        if "\n" in txt or "1." in txt:
            s += 0.15
        scores.append(min(s, 1.0))
    return sum(scores) / max(len(scores), 1)


@torch.inference_mode()
def generate(model, tok, prompt_text: str, max_new_tokens: int) -> str:
    inputs = tok(prompt_text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )
    decoded = tok.decode(out[0], skip_special_tokens=True)
    if decoded.startswith(prompt_text):
        decoded = decoded[len(prompt_text):]
    return truncate_at_instruction(decoded.strip())


def find_checkpoints(dpo_dir: Path) -> List[Path]:
    return sorted(
        (p for p in dpo_dir.glob("checkpoint-*") if p.is_dir()),
        key=lambda p: int(p.name.split("-")[-1]),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model_id", required=True)
    ap.add_argument("--sft_adapter_dir", required=True)
    ap.add_argument("--dpo_dir", required=True, help="Trainer output dir with checkpoint-* subdirs.")
    ap.add_argument("--refs_path", required=True, help="refs.jsonl with prompt/reference (and tests for code).")
    ap.add_argument("--track", required=True, choices=("dialog", "summarization", "code"))
    ap.add_argument("--prompt_style", default="alpaca", choices=("alpaca", "chat"))
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dev_n", type=int, default=DEV_N_DEFAULT,
                    help="How many last-rows from refs to use as dev for selection.")
    ap.add_argument("--best_link", required=True,
                    help="Path where a symlink to the chosen checkpoint will be created.")
    ap.add_argument("--report_path", required=True, help="Where to write the selection JSON report.")
    # judge args are accepted but ignored — we keep selection auto-metric-only
    # for speed (see module docstring).
    ap.add_argument("--judge_model", default="")
    ap.add_argument("--judge_base_url", default="")
    ap.add_argument("--judge_api_key", default="")
    ap.add_argument("--judge_sleep_sec", type=float, default=0.15)
    ap.add_argument("--judge_timeout_sec", type=int, default=120)
    ap.add_argument("--judge_json_mode", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    dpo_dir = Path(args.dpo_dir)
    refs_path = Path(args.refs_path)
    if not refs_path.exists():
        # When run_experiment didn't have refs.jsonl (e.g. plain prompts),
        # selection is meaningless — fall back to the trainer's last ckpt.
        print(f"[select] no refs at {refs_path}; falling back to final checkpoint.")
        cks = find_checkpoints(dpo_dir)
        target = cks[-1] if cks else dpo_dir
        _make_link(Path(args.best_link), target)
        Path(args.report_path).write_text(json.dumps({
            "track": args.track, "selected": str(target), "reason": "no refs.jsonl",
        }, indent=2), encoding="utf-8")
        return

    rows = load_refs(refs_path)
    if len(rows) < 4:
        print(f"[select] only {len(rows)} refs; selection unreliable; using final.")
        cks = find_checkpoints(dpo_dir)
        target = cks[-1] if cks else dpo_dir
        _make_link(Path(args.best_link), target)
        Path(args.report_path).write_text(json.dumps({
            "track": args.track, "selected": str(target), "reason": "too few refs",
        }, indent=2), encoding="utf-8")
        return

    dev_n = max(2, min(args.dev_n, len(rows) // 2))
    dev_rows = rows[-dev_n:]
    print(f"[select] dev_n={dev_n}, total refs={len(rows)}")

    checkpoints = find_checkpoints(dpo_dir)
    if not checkpoints:
        print(f"[select] no checkpoints under {dpo_dir}; using {dpo_dir} as-is.")
        _make_link(Path(args.best_link), dpo_dir)
        Path(args.report_path).write_text(json.dumps({
            "track": args.track, "selected": str(dpo_dir), "reason": "no subdir checkpoints",
        }, indent=2), encoding="utf-8")
        return

    has_cuda = torch.cuda.is_available()
    model_kwargs = {"device_map": "auto"} if has_cuda else {}
    if has_cuda:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    print(f"[select] loading base: {args.base_model_id}")
    tok = AutoTokenizer.from_pretrained(args.base_model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(args.base_model_id, **model_kwargs)
    base_model.eval()

    # Use the SFT adapter as the starting point (DPO checkpoints are LoRA
    # deltas on top of SFT). PeftModel lets us hot-swap adapters by name.
    pm = PeftModel.from_pretrained(base_model, args.sft_adapter_dir, adapter_name="sft")

    rows_by_ckpt: Dict[str, Dict[str, Any]] = {}
    for ckpt in checkpoints:
        adapter_name = f"dpo_{ckpt.name}"
        print(f"[select] loading adapter from {ckpt}")
        try:
            pm.load_adapter(str(ckpt), adapter_name=adapter_name)
        except Exception as e:
            print(f"  failed to load adapter: {e}")
            continue
        pm.set_adapter(adapter_name)
        pm.eval()

        gens: List[str] = []
        for row in dev_rows:
            prompt_text = make_prompt(row["prompt"], tok, args.prompt_style)
            try:
                gen = generate(pm, tok, prompt_text, args.max_new_tokens)
            except Exception as e:
                print(f"  generation failed: {e}")
                gen = ""
            gens.append(gen)
        score = score_dev(args.track, dev_rows, gens)
        rows_by_ckpt[ckpt.name] = {
            "path": str(ckpt),
            "score": round(float(score), 4),
            "n": len(dev_rows),
        }
        print(f"  {ckpt.name}: score={score:.4f}")

    if not rows_by_ckpt:
        target = checkpoints[-1]
        reason = "no checkpoint scored"
    else:
        best_name = max(rows_by_ckpt, key=lambda k: rows_by_ckpt[k]["score"])
        target = Path(rows_by_ckpt[best_name]["path"])
        reason = f"argmax dev-{args.track}"

    _make_link(Path(args.best_link), target)

    report = {
        "track": args.track,
        "dpo_dir": str(dpo_dir),
        "dev_n": dev_n,
        "checkpoints": rows_by_ckpt,
        "selected": str(target),
        "best_link": str(args.best_link),
        "reason": reason,
    }
    Path(args.report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[select] selected={target} reason={reason}")


def _make_link(link: Path, target: Path):
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.exists():
        try:
            if link.is_symlink() or link.is_file():
                link.unlink()
            else:
                # If a real directory exists at the link path, refuse to remove
                # it — let the user investigate. Plain file.unlink() would
                # raise IsADirectoryError, so handle explicitly.
                raise RuntimeError(
                    f"refusing to remove existing directory at {link}; "
                    f"please remove or rename it before re-running selection."
                )
        except FileNotFoundError:
            pass
    try:
        os.symlink(target.resolve(), link)
    except OSError:
        # Some filesystems disallow symlinks; fall back to writing a plain
        # text pointer the rest of the pipeline knows how to read.
        link.write_text(str(target.resolve()), encoding="utf-8")


if __name__ == "__main__":
    main()
