import os
import argparse
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
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
    args = ap.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    print("Loading DPO pairs:", args.dpo_data_path)
    ds = load_dataset("json", data_files=args.dpo_data_path, split="train")
    # Требуемые поля: prompt, chosen, rejected
    for col in ["prompt", "chosen", "rejected"]:
        if col not in ds.column_names:
            raise ValueError(f"Missing column '{col}' in dataset. Columns: {ds.column_names}")

    tok = AutoTokenizer.from_pretrained(args.base_model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print("Loading base model (4-bit):", args.base_model_id)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_id,
        device_map="auto",
        quantization_config=bnb_cfg,
    )

    print("Loading SFT adapter (trainable):", args.sft_adapter_dir)
    model = PeftModel.from_pretrained(base_model, args.sft_adapter_dir, is_trainable=True)

    print("Building reference model (frozen, same as SFT startpoint)")
    ref_base = AutoModelForCausalLM.from_pretrained(
        args.base_model_id,
        device_map="auto",
        quantization_config=bnb_cfg,
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
        bf16=True,
        fp16=False,
        report_to="none",
        optim="paged_adamw_8bit",
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
