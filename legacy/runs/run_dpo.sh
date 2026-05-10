#!/usr/bin/env bash
set -e
source /workspace/thesis-llm-alignment/env.sh

python scripts/train_dpo.py \
  --base_model_id Qwen/Qwen2.5-3B-Instruct \
  --sft_adapter_dir outputs/qwen2.5-3b-sft-lora \
  --dpo_data_path data/dpo_pairs.jsonl \
  --output_dir outputs/qwen2.5-3b-sft-dpo-lora \
  --max_steps 200 \
  --batch_size 1 \
  --grad_accum 16 \
  --max_length 1024 \
  --max_prompt_length 512 \
  --lr 1e-5 \
  --beta 0.1 \
  --seed 42
