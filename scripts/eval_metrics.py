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


def _judge_prompt(track: str, prompt: str, reference: str, answer: str) -> List[Dict[str, str]]:
    sys = (
        "You are a strict but fair evaluator. Return ONLY JSON with keys: "
        '"score" (integer 1-10) and "reason" (short string). No extra text.'
    )
    if track == "code":
        rubric = "Correctness first, then completeness and code quality."
    elif track == "summarization":
        rubric = "Faithfulness and coverage first, then clarity and concision."
    else:
        rubric = "Helpfulness, correctness, tone, and structure."
    user = (
        f"Task type: {track}\n"
        f"Rubric: {rubric}\n\n"
        f"Prompt:\n{prompt}\n\n"
        f"Reference:\n{reference}\n\n"
        f"Candidate answer:\n{answer}\n\n"
        'Return JSON exactly like: {"score": 7, "reason": "..."}.'
    )
    return [{"role": "system", "content": sys}, {"role": "user", "content": user}]


def _safe_score(raw: str) -> Tuple[Optional[int], str]:
    text = (raw or "").strip()
    if not text:
        return None, "empty"
    try:
        obj = json.loads(text)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            return None, "parse_error"
        try:
            obj = json.loads(m.group(0))
        except Exception:
            return None, "parse_error"
    try:
        score = int(obj.get("score"))
        if score < 1:
            score = 1
        if score > 10:
            score = 10
        reason = str(obj.get("reason", ""))[:500]
        return score, reason
    except Exception:
        return None, "parse_error"


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
    per_model_scores: Dict[str, List[int]] = {"base": [], "sft": [], "dpo": []}
    failures = 0
    for ref, pred in aligned:
        prompt = ref.get("prompt", "")
        reference = ref.get("reference", "")
        for model_key in ("base", "sft", "dpo"):
            answer = pred.get(model_key, "")
            raw = _chat_completion_openai_compat(
                model=judge_model,
                messages=_judge_prompt(track, prompt, reference, answer),
                api_key=judge_api_key,
                base_url=judge_base_url,
                timeout_sec=timeout_sec,
                json_mode=json_mode,
            )
            score, _reason = _safe_score(raw)
            if score is None:
                failures += 1
                continue
            per_model_scores[model_key].append(score)
            if sleep_sec > 0:
                time.sleep(sleep_sec)

    def avg(xs: List[int]) -> float:
        return round(sum(xs) / len(xs), 4) if xs else 0.0

    out: Dict[str, Any] = {
        "judge_model": judge_model,
        "judge_base_url": judge_base_url,
        "judge_json_mode": json_mode,
        "judge_sleep_sec": sleep_sec,
        "num_judged_items": len(aligned),
        "judge_failures": failures,
        "base": {"llm_judge_score": avg(per_model_scores["base"])},
        "sft": {"llm_judge_score": avg(per_model_scores["sft"])},
        "dpo": {"llm_judge_score": avg(per_model_scores["dpo"])},
    }
    return out


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
    args = ap.parse_args()

    compare_rows = load_jsonl(Path(args.compare_path))
    ref_rows = load_jsonl(Path(args.refs_path))
    by_prompt = {r["prompt"]: r for r in compare_rows}
    aligned = [(r, by_prompt[r["prompt"]]) for r in ref_rows if r["prompt"] in by_prompt]

    metrics: Dict[str, Any] = {
        "track": args.track,
        "num_eval_items": len(aligned),
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
                score = scorer.score(ref["reference"], pred.get(model_key, "")).get("rougeL")
                vals.append(score.fmeasure if score else 0.0)
            metrics[model_key]["rougeL_f1"] = round(sum(vals) / max(len(vals), 1), 4)
        metrics["primary_metric"] = "rougeL_f1"

    elif args.track == "code":
        for model_key in ("base", "sft", "dpo"):
            passed = 0
            total = 0
            for ref, pred in aligned:
                code = extract_code_block(pred.get(model_key, ""))
                p, t = run_python_tests(code, ref.get("tests", []))
                passed += p
                total += t
            metrics[model_key]["test_pass_rate"] = round((passed / total) if total else 0.0, 4)
            metrics[model_key]["tests_total"] = total
        metrics["primary_metric"] = "test_pass_rate"

    else:
        polite_words = ("please", "thank", "sorry", "appreciate", "kindly")
        for model_key in ("base", "sft", "dpo"):
            scores = []
            for _ref, pred in aligned:
                txt = (pred.get(model_key, "") or "").lower()
                score = 0.0
                if len(txt) >= 80:
                    score += 0.4
                if any(w in txt for w in polite_words):
                    score += 0.3
                if "\n" in txt or "1." in txt or "-" in txt:
                    score += 0.3
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
        judge = compute_llm_judge(
            track=args.track,
            aligned=aligned,
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
