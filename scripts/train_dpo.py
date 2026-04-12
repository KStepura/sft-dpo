import os
import argparse
import random

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed
from peft import PeftModel
from trl import DPOTrainer, DPOConfig

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model_id", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--sft_adapter_dir", type=str, default="outputs/qwen2.5-3b-sft-lora")
    ap.add_argument("--dpo_data_path", type=str, default="data/dpo_pairs.jsonl")
    ap.add_argument("--output_dir", type=str, default="outputs/qwen2.5-3b-sft-dpo-lora")
    ap.add_argument("--max_steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--max_prompt_length", type=int, default=512)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    has_cuda = torch.cuda.is_available()
    use_bf16 = has_cuda and torch.cuda.is_bf16_supported()
    use_fp16 = has_cuda and not use_bf16

    print("Loading DPO pairs:", args.dpo_data_path)
    ds = load_dataset("json", data_files=args.dpo_data_path, split="train")
    # Требуемые поля: prompt, chosen, rejected
    for col in ["prompt", "chosen", "rejected"]:
        if col not in ds.column_names:
            raise ValueError(f"Missing column '{col}' in dataset. Columns: {ds.column_names}")

    tok = AutoTokenizer.from_pretrained(args.base_model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    quantization_config = None
    model_kwargs = {"device_map": "auto"} if has_cuda else {}
    if has_cuda:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = quantization_config

    print("Loading base model (4-bit):", args.base_model_id)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_id,
        **model_kwargs,
    )

    print("Loading SFT adapter (trainable):", args.sft_adapter_dir)
    model = PeftModel.from_pretrained(base_model, args.sft_adapter_dir, is_trainable=True)

    print("Building reference model (frozen, same as SFT startpoint)")
    ref_base = AutoModelForCausalLM.from_pretrained(
        args.base_model_id,
        **model_kwargs,
    )
    ref_model = PeftModel.from_pretrained(ref_base, args.sft_adapter_dir, is_trainable=False)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    dpo_cfg = DPOConfig(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        bf16=use_bf16,
        fp16=use_fp16,
        report_to="none",
        optim="paged_adamw_8bit" if has_cuda else "adamw_torch",
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        beta=args.beta,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_cfg,
        train_dataset=ds,
        processing_class=tok,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)
    print("Done. Saved to:", args.output_dir)

if __name__ == "__main__":
    main()
