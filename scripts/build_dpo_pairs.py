import os
import json
import argparse

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from prompt_utils import format_alpaca_prompt_prefix, format_chat_prompt_prefix


def make_prompt(ex, tok, prompt_style: str) -> str:
    instr = ex.get("instruction") or ""
    inp = ex.get("input") or ""
    if prompt_style == "chat":
        return format_chat_prompt_prefix(tok, instr, inp)
    return format_alpaca_prompt_prefix(instr, inp)


def postprocess_rejected(text: str) -> str:
    text = (text or "").strip()
    for cut_mark in ["\n\n### Instruction:", "\n### Instruction:"]:
        if cut_mark in text:
            text = text.split(cut_mark, 1)[0].strip()
    resp_mark = "### Response:"
    if text.count(resp_mark) >= 2:
        text = text.split(resp_mark)[-1].strip()
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model_id", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--dataset_id", type=str, default="tatsu-lab/alpaca")
    ap.add_argument("--out_path", type=str, default="data/dpo_pairs.jsonl")
    ap.add_argument("--num_examples", type=int, default=800)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument(
        "--prompt_style",
        type=str,
        choices=("chat", "alpaca"),
        default="chat",
        help="Must match SFT / cleaning (chat for *-Instruct models).",
    )

    args = ap.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    print(f"Loading dataset: {args.dataset_id}")
    ds = load_dataset(args.dataset_id, split="train").shuffle(seed=args.seed)
    ds = ds.select(range(min(args.num_examples, len(ds))))
    print(f"Using {len(ds)} examples for DPO pairs")

    tok = AutoTokenizer.from_pretrained(args.base_model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    has_cuda = torch.cuda.is_available()
    quantization_config = None
    model_kwargs = {"device_map": "auto"} if has_cuda else {}
    if has_cuda:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = quantization_config

    model = AutoModelForCausalLM.from_pretrained(args.base_model_id, **model_kwargs)
    model.eval()

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)

    written = 0
    with open(args.out_path, "w", encoding="utf-8") as f:
        for i, ex in enumerate(ds):
            prompt = make_prompt(ex, tok, args.prompt_style)
            chosen = (ex.get("output") or "").strip()

            inputs = tok(prompt, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    pad_token_id=tok.pad_token_id,
                    eos_token_id=tok.eos_token_id,
                )

            decoded = tok.decode(out[0], skip_special_tokens=True)
            rejected = decoded[len(prompt):] if decoded.startswith(prompt) else decoded
            rejected = postprocess_rejected(rejected)

            if len(rejected) < 20:
                continue

            rec = {"prompt": prompt, "chosen": chosen, "rejected": rejected}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1

            if written % 50 == 0:
                print(f"Written {written} pairs")

    print(f"Done. Saved {written} pairs to {args.out_path}")


if __name__ == "__main__":
    main()
