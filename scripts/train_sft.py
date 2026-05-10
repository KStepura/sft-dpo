import argparse
import os

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from prompt_utils import format_alpaca_supervised, format_chat_supervised


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
    ap.add_argument(
        "--prompt_style",
        type=str,
        choices=("chat", "alpaca"),
        default="chat",
        help="chat: tokenizer chat template (recommended for *-Instruct). alpaca: legacy ### blocks.",
    )
    ap.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="If set, use only this many shuffled training rows (saves RAM on CPU smoke).",
    )
    ap.add_argument("--dataset_split", type=str, default="train")
    ap.add_argument("--dataset_format", choices=("alpaca", "mbpp"), default="alpaca")
    ap.add_argument(
        "--no_packing",
        action="store_true",
        help="Disable example packing in SFTTrainer. Avoids cross-contamination "
             "between samples when flash_attention_2 is not available. "
             "Recommended for cleaner gradients at the cost of throughput.",
    )
    args = ap.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    has_cuda = torch.cuda.is_available()
    use_bf16 = has_cuda and torch.cuda.is_bf16_supported()
    use_fp16 = has_cuda and not use_bf16

    print(f"Loading dataset: {args.dataset_id} ({args.dataset_format}, split={args.dataset_split})")
    ds = load_dataset(args.dataset_id)
    train = ds[args.dataset_split].shuffle(seed=args.seed)
    if args.max_train_samples is not None and args.max_train_samples > 0:
        n = min(args.max_train_samples, len(train))
        train = train.select(range(n))
        print(f"Using subset: {n} train rows (max_train_samples={args.max_train_samples})")

    print(f"Loading tokenizer/model: {args.model_id}")
    tok = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if args.dataset_format == "mbpp":

        def formatting_fun(ex):
            instr = (ex.get("text") or "").strip()
            out = (ex.get("code") or "").strip()
            if args.prompt_style == "chat":
                return format_chat_supervised(tok, instr, "", out)
            return format_alpaca_supervised(instr, "", out)

    elif args.prompt_style == "chat":

        def formatting_fun(ex):
            return format_chat_supervised(
                tok,
                ex.get("instruction") or "",
                ex.get("input") or "",
                ex.get("output") or "",
            )

    else:

        def formatting_fun(ex):
            return format_alpaca_supervised(
                ex.get("instruction") or "",
                ex.get("input") or "",
                ex.get("output") or "",
            )

    quantization_config = None
    model_kwargs = {"device_map": "auto"} if has_cuda else {"torch_dtype": torch.float32}
    if has_cuda:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = quantization_config

    model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)
    if not has_cuda:
        model.gradient_checkpointing_enable()

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
        bf16=use_bf16,
        fp16=use_fp16,
        packing=(has_cuda and not args.no_packing),
        report_to="none",
        optim="paged_adamw_8bit" if has_cuda else "adamw_torch",
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
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
