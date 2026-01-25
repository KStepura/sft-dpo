import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--lora_path", default="outputs/qwen2.5-3b-sft-lora")
    ap.add_argument("--prompt", default="Explain what reinforcement learning from human feedback is.")
    ap.add_argument("--max_new_tokens", type=int, default=150)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--do_sample", action="store_true")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    device_map = "auto" if args.device == "auto" else args.device

    print("Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print("Loading base model...")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map=device_map,
        torch_dtype=torch.float16 if torch.cuda.is_available() and device_map != "cpu" else torch.float32,
    )

    print("Loading LoRA adapters...")
    model = PeftModel.from_pretrained(base, args.lora_path)
    model.eval()

    inputs = tok(args.prompt, return_tensors="pt")
    if device_map == "cpu":
        pass
    else:
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    print("Generating...")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature if args.do_sample else None,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )

    print("\n=== OUTPUT ===")
    print(tok.decode(out[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()