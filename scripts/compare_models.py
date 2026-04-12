#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_models.py

Сравнивает ответы трёх вариантов:
  1) base (Qwen/Qwen2.5-3B-Instruct)
  2) sft  (base + LoRA SFT)
  3) dpo  (base + LoRA DPO)

Сохраняет JSONL с полями: prompt, base, sft, dpo, gen

Пример:
python scripts/compare_models.py \
  --base_model_id Qwen/Qwen2.5-3B-Instruct \
  --sft_adapter_dir outputs/qwen2.5-3b-sft-lora \
  --dpo_adapter_dir outputs/qwen2.5-3b-sft-dpo-lora-strict2_400 \
  --prompts_path data/compare_prompts.txt \
  --out_path results/compare_outputs.jsonl \
  --max_new_tokens 180 --temperature 0.7 --top_p 0.9
"""

import os
import json
import argparse
from typing import List, Dict, Any, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed

try:
    from peft import PeftModel
except ImportError as e:
    raise SystemExit("peft не установлен. Установи: pip install peft") from e


DEFAULT_PROMPTS = [
    "Explain RLHF in simple terms.",
    "Draft a short study plan for learning transformers in 2 weeks.",
    "Summarize the concept of DPO (Direct Preference Optimization) in 5 bullet points.",
    "Rewrite this sentence to sound more formal: 'I wanna fix this asap.'",
    "Give 3 safety considerations when deploying an LLM in a customer support chatbot.",
]


def read_prompts(path: Optional[str]) -> List[str]:
    if not path:
        return DEFAULT_PROMPTS[:]
    prompts: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                continue
            prompts.append(s)
    return prompts or DEFAULT_PROMPTS[:]


def maybe_apply_chat_template(tok: AutoTokenizer, user_text: str) -> str:
    """
    Qwen-Instruct обычно лучше работает через chat template.
    Если шаблон есть — используем.
    """
    if hasattr(tok, "apply_chat_template") and tok.chat_template:
        msgs = [{"role": "user", "content": user_text}]
        # add_generation_prompt добавляет маркер начала ответа ассистента
        return tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
    # fallback: просто как есть
    return user_text


@torch.inference_mode()
def generate_one(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    prompt_text: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
) -> str:
    inputs = tok(prompt_text, return_tensors="pt")
    # device_map="auto" ->:
    if hasattr(model, "device"):
        device = model.device
        inputs = {k: v.to(device) for k, v in inputs.items()}
    else:
        inputs = {k: v.to("cuda" if torch.cuda.is_available() else "cpu") for k, v in inputs.items()}

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )

    decoded = tok.decode(out[0], skip_special_tokens=True)

    if decoded.startswith(prompt_text):
        return decoded[len(prompt_text):].strip()
    return decoded.strip()


def load_base_model(base_model_id: str) -> AutoModelForCausalLM:
    has_cuda = torch.cuda.is_available()
    model_kwargs = {"device_map": "auto"} if has_cuda else {}
    if has_cuda:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = bnb_cfg
        model_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        model_kwargs["torch_dtype"] = torch.float32
    model = AutoModelForCausalLM.from_pretrained(base_model_id, **model_kwargs)
    model.eval()
    return model


def load_tokenizer(base_model_id: str) -> AutoTokenizer:
    tok = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def attach_lora(base_model: AutoModelForCausalLM, adapter_dir: str) -> AutoModelForCausalLM:
    # PeftModel оборачивает base_model
    m = PeftModel.from_pretrained(base_model, adapter_dir)
    m.eval()
    return m


def ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model_id", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--sft_adapter_dir", type=str, required=True)
    ap.add_argument("--dpo_adapter_dir", type=str, required=True)

    ap.add_argument("--prompts_path", type=str, default=None, help="txt файл: 1 prompt = 1 строка")
    ap.add_argument("--out_path", type=str, default="results/compare_outputs.jsonl")

    ap.add_argument("--max_new_tokens", type=int, default=180)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--do_sample", action="store_true", help="если не задано — будет greedy")
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()
    set_seed(args.seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    prompts = read_prompts(args.prompts_path)
    ensure_dir(args.out_path)

    print(f"Loading tokenizer: {args.base_model_id}")
    tok = load_tokenizer(args.base_model_id)

    print(f"Loading base model: {args.base_model_id}")
    base_model = load_base_model(args.base_model_id)

    print(f"Loading SFT adapter: {args.sft_adapter_dir}")
    sft_model = attach_lora(base_model, args.sft_adapter_dir)

    # Чтобы DPO не “наследовал” SFT-адаптер внутри того же объекта, грузим отдельную копию base.
    # Это чуть дороже по памяти, но сравнение будет корректным.
    print(f"Reloading base model for DPO: {args.base_model_id}")
    base_for_dpo = load_base_model(args.base_model_id)

    print(f"Loading DPO adapter: {args.dpo_adapter_dir}")
    dpo_model = attach_lora(base_for_dpo, args.dpo_adapter_dir)

    gen_cfg: Dict[str, Any] = dict(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.do_sample,
        seed=args.seed,
    )

    print(f"Running {len(prompts)} prompts. Saving to: {args.out_path}")
    with open(args.out_path, "w", encoding="utf-8") as f:
        for i, user_prompt in enumerate(prompts, 1):
            prompt_text = maybe_apply_chat_template(tok, user_prompt)

            base_out = generate_one(
                base_model, tok, prompt_text,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=args.do_sample,
            )
            sft_out = generate_one(
                sft_model, tok, prompt_text,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=args.do_sample,
            )
            dpo_out = generate_one(
                dpo_model, tok, prompt_text,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=args.do_sample,
            )

            rec = {
                "prompt": user_prompt,
                "base": base_out,
                "sft": sft_out,
                "dpo": dpo_out,
                "gen": gen_cfg,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if i % 5 == 0 or i == len(prompts):
                print(f"  done {i}/{len(prompts)}")

    print("Done.")


if __name__ == "__main__":
    main()