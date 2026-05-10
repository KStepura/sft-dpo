#!/usr/bin/env python3
"""Prepare MBPP eval refs.jsonl (prompt + reference code + tests) and prompts list."""
import argparse
import ast
import json
from pathlib import Path

from datasets import load_dataset


def parse_tests(raw) -> list:
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            return ast.literal_eval(raw)
        except Exception:
            return []
    return []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_id", default="google-research-datasets/mbpp")
    ap.add_argument("--split", default="test")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_examples", type=int, default=40)
    ap.add_argument("--out_refs", type=str, required=True)
    ap.add_argument("--out_prompts_txt", type=str, required=True)
    args = ap.parse_args()

    ds = load_dataset(args.dataset_id, split=args.split).shuffle(seed=args.seed)
    n = min(args.num_examples, len(ds))
    ds = ds.select(range(n))

    refs = []
    prompt_lines = []
    for row in ds:
        prompt_text = (row.get("text") or "").strip()
        code = (row.get("code") or "").strip()
        tests = parse_tests(row.get("test_list"))
        refs.append({"prompt": prompt_text, "reference": code, "tests": tests})
        prompt_lines.append(prompt_text.replace("\n", " ").strip())

    out_refs = Path(args.out_refs)
    out_refs.parent.mkdir(parents=True, exist_ok=True)
    out_refs.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in refs) + "\n", encoding="utf-8")

    out_pt = Path(args.out_prompts_txt)
    out_pt.parent.mkdir(parents=True, exist_ok=True)
    out_pt.write_text("\n".join(prompt_lines) + "\n", encoding="utf-8")

    print(f"Wrote {n} eval items to {out_refs}")


if __name__ == "__main__":
    main()
