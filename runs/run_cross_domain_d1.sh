#!/usr/bin/env bash
# D1 cross-domain transfer test:
#   Берём base/SFT/DPO-best, обученные на summarization (alpaca, on-policy IPO, β=0.3, seed=42),
#   и прогоняем их на 40 MBPP eval-промптах (тот же набор, что в v2_track_code_mbpp).
#
# Цель — буквально проверить кросс-доменный перенос: помогает ли DPO/SFT,
# обученный на одном домене (суммаризация), на другом домене (генерация кода).
#
# Артефакты:
#   results/cross_domain_d1_sum_to_mbpp/compare/compare.jsonl
#   results/cross_domain_d1_sum_to_mbpp/metrics/metrics.json
#
# Время: ~25-30 мин на RTX 4090.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" && cd "$REPO"
source env.sh

SUM_FLAGSHIP=outputs/v2_track_summarization_nojudge_20260506_210741
SFT_DIR=$SUM_FLAGSHIP/sft
DPO_DIR=$SUM_FLAGSHIP/dpo_beta_0p3_best

CODE_EXP=results/experiments/v2_track_code_mbpp_20260507_164026
PROMPTS=$CODE_EXP/refs.jsonl
REFS=$CODE_EXP/refs.jsonl

OUT_DIR=results/cross_domain_d1_sum_to_mbpp
mkdir -p "$OUT_DIR/compare" "$OUT_DIR/metrics"

echo "[D1] SFT_DIR=$SFT_DIR"
echo "[D1] DPO_DIR=$DPO_DIR"
echo "[D1] PROMPTS=$PROMPTS"
echo "[D1] OUT_DIR=$OUT_DIR"

.venv/bin/python scripts/compare_models.py \
  --base_model_id Qwen/Qwen2.5-3B \
  --sft_adapter_dir "$SFT_DIR" \
  --dpo_adapter_dir "$DPO_DIR" \
  --prompts_path "$PROMPTS" \
  --out_path "$OUT_DIR/compare/compare.jsonl" \
  --max_new_tokens 220 \
  --temperature 0.0 \
  --top_p 1.0 \
  --prompt_style alpaca \
  --postprocess_outputs

.venv/bin/python scripts/eval_metrics.py \
  --track code \
  --compare_path "$OUT_DIR/compare/compare.jsonl" \
  --refs_path "$REFS" \
  --out_path "$OUT_DIR/metrics/metrics.json"

echo "[D1] DONE. Metrics:"
cat "$OUT_DIR/metrics/metrics.json"
