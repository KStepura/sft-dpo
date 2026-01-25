import os, json, argparse, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def load_base(model_id: str, dtype, device_map):
    return AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device_map,
    )

def load_with_lora(base_model, lora_path: str, device_map):
    return PeftModel.from_pretrained(base_model, lora_path, device_map=device_map)

@torch.no_grad()
def generate(model, tok, prompt: str, max_new_tokens: int, temperature: float, top_p: float):
    inputs = tok(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True if temperature > 0 else False,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )
    return tok.decode(out[0], skip_special_tokens=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model_id", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--sft_lora", default="outputs/qwen2.5-3b-sft-lora")
    ap.add_argument("--dpo_lora", default="outputs/qwen2.5-3b-sft-dpo-lora")
    ap.add_argument("--out_path", default="results/compare_outputs.jsonl")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    args = ap.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)

    # Небольшой фикс: у Qwen часто pad_token не задан
    print("Loading tokenizer:", args.base_model_id)
    tok = AutoTokenizer.from_pretrained(args.base_model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    device_map = "auto"  # на GPU
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print("Loading BASE model...")
    base = load_base(args.base_model_id, dtype=dtype, device_map=device_map)
    base.eval()

    print("Loading SFT LoRA...")
    sft = load_with_lora(base, args.sft_lora, device_map=device_map)
    sft.eval()

    print("Loading DPO LoRA...")
    # Для DPO лучше загрузить отдельную базу (чтобы адаптеры не конфликтовали)
    base2 = load_base(args.base_model_id, dtype=dtype, device_map=device_map)
    base2.eval()
    dpo = load_with_lora(base2, args.dpo_lora, device_map=device_map)
    dpo.eval()

    prompts = [
        "Explain RLHF in simple terms.",
        "Write a polite email asking for an extension on a deadline.",
        "Give 5 bullet points on why unit tests matter.",
        "Translate to English: 'Мне нужно подготовить отчет по экспериментам.'",
        "Solve: If a train travels 120 km in 1.5 hours, what is its average speed?",
        "Summarize in 3 sentences: The internet changed how people communicate and learn.",
        "Write a short Python function to compute factorial.",
        "What are two pros and two cons of remote work?",
        "Explain what overfitting is and how to reduce it.",
        "Draft a short study plan for learning transformers in 2 weeks."
    ]
    prompts = prompts[:min(args.n, len(prompts))]

    t0 = time.time()
    with open(args.out_path, "w", encoding="utf-8") as f:
        for i, p in enumerate(prompts, 1):
            print(f"\n=== Prompt {i}/{len(prompts)} ===\n{p}\n")

            out_base = generate(base, tok, p, args.max_new_tokens, args.temperature, args.top_p)
            out_sft  = generate(sft,  tok, p, args.max_new_tokens, args.temperature, args.top_p)
            out_dpo  = generate(dpo,  tok, p, args.max_new_tokens, args.temperature, args.top_p)

            rec = {
                "prompt": p,
                "base": out_base,
                "sft": out_sft,
                "dpo": out_dpo,
                "gen": {"max_new_tokens": args.max_new_tokens, "temperature": args.temperature, "top_p": args.top_p},
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    dt = time.time() - t0
    print(f"\nSaved: {args.out_path}")
    print(f"Done in {dt:.1f}s")
    print("\nTip: open file with: head -n 1 results/compare_outputs.jsonl")

if __name__ == "__main__":
    main()
