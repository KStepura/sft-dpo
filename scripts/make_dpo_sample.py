import os, json, argparse
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def make_prompt(ex):
    instr = (ex.get("instruction") or "").strip()
    inp = (ex.get("input") or "").strip()
    if inp:
        return f"### Instruction:\n{instr}\n\n### Input:\n{inp}\n\n### Response:\n"
    return f"### Instruction:\n{instr}\n\n### Response:\n"

def postprocess(text: str) -> str:
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
    ap.add_argument("--model_id", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--dataset_id", default="tatsu-lab/alpaca")
    ap.add_argument("--out_path", default="data/dpo_pairs_sample.jsonl")
    ap.add_argument("--num_examples", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    args = ap.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    print("Loading dataset...")
    ds = load_dataset(args.dataset_id, split="train").shuffle(seed=args.seed)
    ds = ds.select(range(args.num_examples))

    print("Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print("Loading model (4-bit)...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        device_map="auto",
        quantization_config=bnb,
    )
    model.eval()

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)

    with open(args.out_path, "w", encoding="utf-8") as f:
        for i, ex in enumerate(ds):
            prompt = make_prompt(ex)
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
            rejected = postprocess(rejected)

            rec = {"prompt": prompt, "chosen": chosen, "rejected": rejected}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            print("\n" + "="*80)
            print(f"PAIR {i+1}")
            print("- PROMPT:\n", prompt)
            print("- CHOSEN:\n", chosen[:400], ("..." if len(chosen) > 400 else ""))
            print("- REJECTED:\n", rejected[:400], ("..." if len(rejected) > 400 else ""))

            if "\n### Instruction:" in rejected or "### Instruction:" in rejected:
                print("!! WARNING: rejected contains a new Instruction marker")

    print(f"\nSaved sample pairs to: {args.out_path}")

if __name__ == "__main__":
    main()
