#!/usr/bin/env bash
set -e
source /workspace/thesis-llm-alignment/env.sh

python scripts/train_sft.py \
  --model_id Qwen/Qwen2.5-3B-Instruct \
  --dataset_id tatsu-lab/alpaca \
  --output_dir outputs/qwen2.5-3b-sft-lora \
  --max_steps 200 \
  --batch_size 1 \
  --grad_accum 16 \
  --max_seq_len 1024 \
  --seed 42 \
  --prompt_style chat
