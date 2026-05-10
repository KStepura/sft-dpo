#!/usr/bin/env bash
# D2 cross-domain transfer test (обратное направление к D1):
#   Берём base/SFT/DPO-best, обученные на code/MBPP (on-policy IPO, β=0.3, seed=42),
#   и прогоняем их на 25 summarization eval-промптах (тот же набор, что в флагмане v2_track_summarization_nojudge).
#
# Цель — проверить, помогает ли SFT/DPO, обученный на узком code-домене,
# на широком summarization-домене (обратное направление к D1: sum→MBPP).
#
# Артефакты:
#   results/cross_domain_d2_mbpp_to_sum/compare/compare.jsonl
#   results/cross_domain_d2_mbpp_to_sum/metrics/metrics.json
#
# Время: ~10-15 мин на RTX 4090 (25 промптов, max_new_tokens=200).

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" && cd "$REPO"
source env.sh

CODE_FLAGSHIP=outputs/v2_track_code_mbpp_20260507_164026
SFT_DIR=$CODE_FLAGSHIP/sft
DPO_DIR=$CODE_FLAGSHIP/dpo_beta_0p3_best

SUM_EXP=results/experiments/v2_track_summarization_nojudge_20260506_210741
PROMPTS=$SUM_EXP/refs.jsonl
REFS=$SUM_EXP/refs.jsonl

OUT_DIR=results/cross_domain_d2_mbpp_to_sum
mkdir -p "$OUT_DIR/compare" "$OUT_DIR/metrics"

echo "[D2] SFT_DIR=$SFT_DIR"
echo "[D2] DPO_DIR=$DPO_DIR"
echo "[D2] PROMPTS=$PROMPTS"
echo "[D2] OUT_DIR=$OUT_DIR"

.venv/bin/python scripts/compare_models.py \
  --base_model_id Qwen/Qwen2.5-3B \
  --sft_adapter_dir "$SFT_DIR" \
  --dpo_adapter_dir "$DPO_DIR" \
  --prompts_path "$PROMPTS" \
  --out_path "$OUT_DIR/compare/compare.jsonl" \
  --max_new_tokens 200 \
  --temperature 0.0 \
  --top_p 1.0 \
  --prompt_style alpaca \
  --postprocess_outputs

.venv/bin/python scripts/eval_metrics.py \
  --track summarization \
  --compare_path "$OUT_DIR/compare/compare.jsonl" \
  --refs_path "$REFS" \
  --out_path "$OUT_DIR/metrics/metrics.json"

echo "[D2] DONE. Metrics:"
cat "$OUT_DIR/metrics/metrics.json"
