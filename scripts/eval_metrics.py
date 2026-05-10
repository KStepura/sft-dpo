#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib import request
from urllib.error import HTTPError, URLError

from rouge_score import rouge_scorer


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def truncate_at_instruction(text: str) -> str:
    # Base models without instruct-tuning sometimes hallucinate a follow-up
    # "### Instruction:" block after answering, inflating length and tanking
    # ROUGE precision. Mirrors postprocess_rejected in build_dpo_pairs.py.
    text = (text or "").strip()
    for cut_mark in ["\n\n### Instruction:", "\n### Instruction:"]:
        if cut_mark in text:
            text = text.split(cut_mark, 1)[0].strip()
    resp_mark = "### Response:"
    if text.count(resp_mark) >= 2:
        text = text.split(resp_mark)[-1].strip()
    return text


def extract_code_block(text: str) -> str:
    text = text or ""
    m = re.search(r"```(?:python)?\n(.*?)```", text, flags=re.S)
    if m:
        return m.group(1).strip()
    if "\nassistant\n" in text:
        text = text.split("\nassistant\n", 1)[1]
    m2 = re.search(r"(?m)^(def |class |import |from )", text)
    if m2:
        text = text[m2.start() :]
    return text.strip()


def run_python_tests(code: str, tests: List[str], timeout_sec: int = 8) -> Tuple[int, int]:
    if not tests:
        return 0, 0
    passed = 0
    total = len(tests)
    for test in tests:
        script = code + "\n\n" + test + "\n"
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=True, encoding="utf-8") as tf:
            tf.write(script)
            tf.flush()
            proc = subprocess.run(
                ["python", tf.name],
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                check=False,
            )
        if proc.returncode == 0:
            passed += 1
    return passed, total


def _chat_completion_openai_compat(
    model: str,
    messages: List[Dict[str, str]],
    api_key: str,
    base_url: str,
    timeout_sec: int = 45,
    json_mode: bool = False,
    max_retries: int = 6,
) -> str:
    body: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
    }
    if json_mode:
        # OpenAI-compatible JSON mode (not all local servers support this)
        body["response_format"] = {"type": "json_object"}
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        req = request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            method="POST",
            headers=headers,
        )
        try:
            with request.urlopen(req, timeout=timeout_sec) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            return payload["choices"][0]["message"]["content"]
        except HTTPError as e:
            last_err = e
            # Rate limit / transient server issues
            if e.code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                time.sleep(min(2**attempt, 30))
                continue
            raise
        except URLError as e:
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(min(2**attempt, 30))
                continue
            raise
    raise RuntimeError(f"chat completion failed after retries: {last_err}")


_RUBRICS: Dict[str, List[str]] = {
    "dialog": [
        "Helpfulness: does the answer fully address the user's request?",
        "Accuracy: is the content factually correct?",
        "Instruction-following: does it respect format, length, and tone constraints?",
        "Clarity: is it well-structured and easy to read?",
    ],
    "summarization": [
        "Faithfulness: does it accurately reflect the source without distortion?",
        "Coverage: are all key points from the original captured?",
        "Conciseness: is it free of unnecessary padding or repetition?",
        "Readability: is it fluent and clear?",
    ],
    "code": [
        "Correctness: would the code work correctly for the given task?",
        "Completeness: are edge cases and all requirements handled?",
        "Clarity: is the code readable and well-structured?",
        "Efficiency: is the approach reasonable and not wasteful?",
    ],
}

# Pairs evaluated; order determines which model is "A" vs "B" in the prompt.
_PAIRS = [("base", "sft"), ("base", "dpo"), ("sft", "dpo")]


def _pairwise_judge_messages(
    track: str,
    prompt: str,
    reference: str,
    answer_a: str,
    answer_b: str,
) -> List[Dict[str, str]]:
    criteria = "\n".join(f"  - {r}" for r in _RUBRICS.get(track, _RUBRICS["dialog"]))
    ref_block = (
        f"\nReference answer (for grounding only):\n{reference}\n"
        if (reference or "").strip()
        else ""
    )
    system = (
        "You are an expert evaluator comparing two AI-generated answers to the same prompt. "
        "First reason step by step, then output ONLY a JSON object — no markdown fences, no extra text — "
        'with keys "verdict" ("A", "B", or "tie") and "reason" (max 80 words).'
    )
    user = (
        f"Task type: {track}\n\n"
        f"Evaluation criteria:\n{criteria}\n\n"
        f"User prompt:\n{prompt}\n"
        f"{ref_block}\n"
        f"=== Answer A ===\n{answer_a}\n\n"
        f"=== Answer B ===\n{answer_b}\n\n"
        "Step 1 — Briefly analyse both answers against each criterion above.\n"
        "Step 2 — Output JSON:\n"
        '{"verdict": "A"|"B"|"tie", "reason": "<≤80 words>"}'
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _safe_verdict(raw: str) -> Tuple[Optional[str], str]:
    text = (raw or "").strip()
    if not text:
        return None, "empty"
    # Look for a flat JSON object containing "verdict" (no nested braces needed).
    m = re.search(r'\{[^{}]*"verdict"[^{}]*\}', text, re.IGNORECASE)
    if m:
        try:
            obj = json.loads(m.group(0))
            v = str(obj.get("verdict", "")).strip().upper()
            if v in ("A", "B", "TIE"):
                return ("tie" if v == "TIE" else v), str(obj.get("reason", ""))[:400]
        except Exception:
            pass
    # Plain-text fallback for models that ignore JSON instructions.
    tl = text.lower()
    for token, label in (
        ('"verdict":"a"', "A"), ('"verdict": "a"', "A"),
        ('"verdict":"b"', "B"), ('"verdict": "b"', "B"),
        ('"verdict":"tie"', "tie"), ('"verdict": "tie"', "tie"),
    ):
        if token in tl:
            return label, text[:400]
    return None, "parse_error"


def _flip(verdict: Optional[str]) -> Optional[str]:
    """Flip A↔B to normalise a BA-order verdict back to AB perspective."""
    if verdict == "A":
        return "B"
    if verdict == "B":
        return "A"
    return verdict  # "tie" or None unchanged


def compute_llm_judge(
    track: str,
    aligned: List[Tuple[Dict[str, Any], Dict[str, Any]]],
    judge_model: str,
    judge_api_key: str,
    judge_base_url: str,
    json_mode: bool,
    sleep_sec: float,
    timeout_sec: int,
) -> Dict[str, Any]:
    """Pairwise LLM judge.

    For every prompt, each pair of models (base/sft/dpo) is evaluated twice:
    once in AB order and once in BA order.  The BA verdict is flipped back to
    AB perspective so the two can be compared.  When both orderings agree the
    result is recorded as-is; when they disagree the item is counted as a tie
    (position-bias artefact).  This yields a position-bias-corrected win-rate
    for each model across all pairs it participates in.
    """
    pair_results: Dict[str, Any] = {}
    total_failures = 0

    for m_a, m_b in _PAIRS:
        pair_key = f"{m_a}_vs_{m_b}"
        items: List[Dict[str, Any]] = []
        wins: Dict[str, int] = {m_a: 0, m_b: 0, "tie": 0}
        n_consistent = 0

        for i, (ref, pred) in enumerate(aligned):
            prompt_text = ref.get("prompt", "")
            reference = ref.get("reference", "")
            ans_a = truncate_at_instruction(pred.get(m_a, ""))
            ans_b = truncate_at_instruction(pred.get(m_b, ""))

            # --- AB order ---
            raw_ab = _chat_completion_openai_compat(
                model=judge_model,
                messages=_pairwise_judge_messages(track, prompt_text, reference, ans_a, ans_b),
                api_key=judge_api_key,
                base_url=judge_base_url,
                timeout_sec=timeout_sec,
                json_mode=json_mode,
            )
            verdict_ab, reason_ab = _safe_verdict(raw_ab)
            if verdict_ab is None:
                total_failures += 1
            if sleep_sec > 0:
                time.sleep(sleep_sec)

            # --- BA order (position-bias check) ---
            raw_ba = _chat_completion_openai_compat(
                model=judge_model,
                messages=_pairwise_judge_messages(track, prompt_text, reference, ans_b, ans_a),
                api_key=judge_api_key,
                base_url=judge_base_url,
                timeout_sec=timeout_sec,
                json_mode=json_mode,
            )
            verdict_ba_raw, reason_ba = _safe_verdict(raw_ba)
            if verdict_ba_raw is None:
                total_failures += 1
            if sleep_sec > 0:
                time.sleep(sleep_sec)

            # Normalise BA verdict to AB perspective.
            verdict_ba_norm = _flip(verdict_ba_raw)

            # Consistent = both orderings agree (ties count as agreeing with each other).
            consistent = (
                verdict_ab is not None
                and verdict_ba_norm is not None
                and verdict_ab == verdict_ba_norm
            )
            if consistent:
                n_consistent += 1

            # Position-bias-corrected final verdict.
            if consistent:
                final = verdict_ab
            elif verdict_ab is not None or verdict_ba_norm is not None:
                final = "tie"
            else:
                final = None

            if final == "A":
                wins[m_a] += 1
            elif final == "B":
                wins[m_b] += 1
            elif final == "tie":
                wins["tie"] += 1

            items.append({
                "idx": i,
                "verdict_ab": verdict_ab,
                "verdict_ba_norm": verdict_ba_norm,
                "consistent": consistent,
                "final": final,
                "reason_ab": reason_ab,
                "reason_ba": reason_ba,
            })

        n = len(aligned)
        total_decided = wins[m_a] + wins[m_b] + wins["tie"]
        pair_results[pair_key] = {
            f"{m_a}_wins": wins[m_a],
            f"{m_b}_wins": wins[m_b],
            "ties": wins["tie"],
            f"{m_a}_win_rate": round(wins[m_a] / max(total_decided, 1), 4),
            f"{m_b}_win_rate": round(wins[m_b] / max(total_decided, 1), 4),
            "tie_rate": round(wins["tie"] / max(total_decided, 1), 4),
            "position_consistent_rate": round(n_consistent / max(n, 1), 4),
            "items": items,
        }

    # Aggregate win-rates per model across all pairs it participates in.
    model_wins: Dict[str, int] = {"base": 0, "sft": 0, "dpo": 0}
    model_total: Dict[str, int] = {"base": 0, "sft": 0, "dpo": 0}
    for m_a, m_b in _PAIRS:
        pk = f"{m_a}_vs_{m_b}"
        pr = pair_results[pk]
        pair_total = pr[f"{m_a}_wins"] + pr[f"{m_b}_wins"] + pr["ties"]
        for side in (m_a, m_b):
            model_wins[side] += pr[f"{side}_wins"]
            model_total[side] += pair_total

    win_rates = {
        m: round(model_wins[m] / max(model_total[m], 1), 4)
        for m in ("base", "sft", "dpo")
    }

    return {
        "judge_model": judge_model,
        "judge_base_url": judge_base_url,
        "judge_json_mode": json_mode,
        "num_judged_items": len(aligned),
        "judge_failures": total_failures,
        "win_rates": win_rates,
        "pairs": pair_results,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", required=True, choices=("dialog", "summarization", "code"))
    ap.add_argument("--compare_path", required=True)
    ap.add_argument("--refs_path", required=True)
    ap.add_argument("--out_path", required=True)
    ap.add_argument("--judge_model", default="", help="Optional LLM judge model (OpenAI-compatible API)")
    ap.add_argument("--judge_base_url", default="https://api.openai.com/v1")
    ap.add_argument("--judge_api_key", default="")
    ap.add_argument(
        "--judge_json_mode",
        action="store_true",
        help="Use response_format=json_object (OpenAI). Many local servers do not support this.",
    )
    ap.add_argument("--judge_sleep_sec", type=float, default=0.15, help="Sleep between judge calls to avoid 429")
    ap.add_argument("--judge_timeout_sec", type=int, default=120, help="HTTP timeout for judge requests")
    ap.add_argument("--judge_limit", type=int, default=0, metavar="N",
                    help="Judge only the first N aligned items (0 = all; for smoke-testing)")
    args = ap.parse_args()

    compare_rows = load_jsonl(Path(args.compare_path))
    ref_rows = load_jsonl(Path(args.refs_path))

    # Whitespace-tolerant alignment: compare_models.py and refs.jsonl can have
    # subtle whitespace differences (trailing newlines, doubled blank lines)
    # that silently turn `aligned` into the empty list and zero out every
    # metric. Normalise on both sides and report align_rate.
    def _norm_prompt(p: str) -> str:
        return " ".join((p or "").split())

    by_prompt = {_norm_prompt(r["prompt"]): r for r in compare_rows}
    aligned = [
        (r, by_prompt[_norm_prompt(r["prompt"])])
        for r in ref_rows
        if _norm_prompt(r["prompt"]) in by_prompt
    ]
    align_rate = round(len(aligned) / max(len(ref_rows), 1), 4)
    if len(aligned) < len(ref_rows):
        print(
            f"WARN: aligned {len(aligned)}/{len(ref_rows)} "
            f"(rate={align_rate}); check prompt normalisation between "
            f"refs and compare files."
        )

    metrics: Dict[str, Any] = {
        "track": args.track,
        "num_eval_items": len(aligned),
        "num_ref_items": len(ref_rows),
        "align_rate": align_rate,
        "base": {},
        "sft": {},
        "dpo": {},
    }

    if not aligned:
        out_path = Path(args.out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved empty metrics to {out_path}")
        return

    if args.track == "summarization":
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        for model_key in ("base", "sft", "dpo"):
            vals = []
            for ref, pred in aligned:
                pred_text = truncate_at_instruction(pred.get(model_key, ""))
                score = scorer.score(ref["reference"], pred_text).get("rougeL")
                vals.append(score.fmeasure if score else 0.0)
            metrics[model_key]["rougeL_f1"] = round(sum(vals) / max(len(vals), 1), 4)
        metrics["primary_metric"] = "rougeL_f1"

    elif args.track == "code":
        for model_key in ("base", "sft", "dpo"):
            passed = 0
            total = 0
            for ref, pred in aligned:
                pred_text = truncate_at_instruction(pred.get(model_key, ""))
                code = extract_code_block(pred_text)
                p, t = run_python_tests(code, ref.get("tests", []))
                passed += p
                total += t
            metrics[model_key]["test_pass_rate"] = round((passed / total) if total else 0.0, 4)
            metrics[model_key]["tests_total"] = total
        metrics["primary_metric"] = "test_pass_rate"

    else:
        # Old heuristic awarded +0.4 to anything ≥80 chars, which is essentially
        # everything — every model scored 0.66–0.69 and the metric was useless.
        # New version is a weak-but-discriminative sanity check; the primary
        # signal for the dialog track should still come from the LLM judge.
        polite_words = ("please", "thank", "sorry", "appreciate", "kindly")
        for model_key in ("base", "sft", "dpo"):
            scores = []
            for ref, pred in aligned:
                txt = truncate_at_instruction(pred.get(model_key, "")).lower()
                ref_text = (ref.get("reference") or "").lower()
                score = 0.0
                # 1) length in a reasonable band (penalises both too-short and
                # over-long base hallucinations).
                n_words = len(txt.split())
                if 40 <= n_words <= 200:
                    score += 0.3
                elif 20 <= n_words < 40 or 200 < n_words <= 350:
                    score += 0.15
                # 2) token Jaccard with reference — proxy for "addresses the
                # same content" without rewarding paraphrase-distance the way
                # rougeL would.
                if ref_text and txt:
                    ref_tokens = set(ref_text.split())
                    txt_tokens = set(txt.split())
                    union = len(ref_tokens | txt_tokens)
                    if union:
                        jacc = len(ref_tokens & txt_tokens) / union
                        score += min(0.4, jacc * 2.0)
                # 3) politeness cue.
                if any(w in txt for w in polite_words):
                    score += 0.15
                # 4) structural cue (newlines or list-style enumeration).
                if "\n" in txt or "1." in txt:
                    score += 0.15
                scores.append(min(score, 1.0))
            metrics[model_key]["dialog_heuristic"] = round(sum(scores) / max(len(scores), 1), 4)
        metrics["primary_metric"] = "dialog_heuristic"

    if args.judge_model:
        api_key = args.judge_api_key or os.getenv("JUDGE_API_KEY", "")
        # Local OpenAI-compatible servers often accept any bearer token, or no auth at all.
        # We still send a dummy value if needed.
        if not api_key and ("localhost" in args.judge_base_url or "127.0.0.1" in args.judge_base_url):
            api_key = "local"
        if not api_key:
            raise SystemExit("LLM judge requested but no API key provided (--judge_api_key or JUDGE_API_KEY).")
        judge_aligned = aligned[:args.judge_limit] if args.judge_limit else aligned
        judge = compute_llm_judge(
            track=args.track,
            aligned=judge_aligned,
            judge_model=args.judge_model,
            judge_api_key=api_key,
            judge_base_url=args.judge_base_url,
            json_mode=bool(args.judge_json_mode),
            sleep_sec=float(args.judge_sleep_sec),
            timeout_sec=int(args.judge_timeout_sec),
        )
        metrics["llm_judge"] = judge

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved metrics to {out_path}")


if __name__ == "__main__":
    main()
