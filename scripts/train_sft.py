import os
import argparse
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig

def formatting_fun(ex):
    instr = (ex.get("instruction") or "").strip()
    inp = (ex.get("input") or "").strip()
    out = (ex.get("output") or "").strip()
    if inp:
        return f"### Instruction:\n{instr}\n\n### Input:\n{inp}\n\n### Response:\n{out}"
    return f"### Instruction:\n{instr}\n\n### Response:\n{out}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--dataset_id", type=str, default="tatsu-lab/alpaca")
    ap.add_argument("--output_dir", type=str, default="outputs/qwen2.5-3b-sft-lora")
    ap.add_argument("--max_steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--max_seq_len", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    print(f"Loading dataset: {args.dataset_id}")
    ds = load_dataset(args.dataset_id)
    train = ds["train"].shuffle(seed=args.seed)

    print(f"Loading tokenizer/model: {args.model_id}")
    tok = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        device_map="auto",
        quantization_config=bnb_cfg,
    )

    peft_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    sft_cfg = SFTConfig(
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
        packing=True,
        report_to="none",
        optim="paged_adamw_8bit",
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",

        # ВАЖНО: длина задается здесь
        max_length=args.max_seq_len,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=train,
        peft_config=peft_cfg,
        processing_class=tok,
        formatting_func=formatting_fun,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)
    print("Done. Saved to:", args.output_dir)

if __name__ == "__main__":
    main()
import os
import argparse
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig

def formatting_fun(ex):
    instr = (ex.get("instruction") or "").strip()
    inp = (ex.get("input") or "").strip()
    out = (ex.get("output") or "").strip()
    if inp:
        return f"### Instruction:\n{instr}\n\n### Input:\n{inp}\n\n### Response:\n{out}"
    return f"### Instruction:\n{instr}\n\n### Response:\n{out}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--dataset_id", type=str, default="tatsu-lab/alpaca")
    ap.add_argument("--output_dir", type=str, default="outputs/qwen2.5-3b-sft-lora")
    ap.add_argument("--max_steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--max_seq_len", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    print(f"Loading dataset: {args.dataset_id}")
    ds = load_dataset(args.dataset_id)
    train = ds["train"].shuffle(seed=args.seed)

    print(f"Loading tokenizer/model: {args.model_id}")
    tok = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        device_map="auto",
        quantization_config=bnb_cfg,
    )

    peft_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    sft_cfg = SFTConfig(
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
        packing=True,
        report_to="none",
        optim="paged_adamw_8bit",
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",

        # ВАЖНО: длина задается здесь
        max_length=args.max_seq_len,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=train,
        peft_config=peft_cfg,
        processing_class=tok,
        formatting_func=formatting_fun,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)
    print("Done. Saved to:", args.output_dir)

if __name__ == "__main__":
    main()
