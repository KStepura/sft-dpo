#!/usr/bin/env bash
# Достраивает cross-domain матрицу до 3x3 (in-domain + 6 cross-domain cells).
# В работе уже сделаны: D1 (sum->MBPP) и D2 (code->sum).
# Здесь добавляются:
#   D3: dialog -> MBPP        (eval = code track)
#   D4: dialog -> summarization
#   D5: code   -> dialog
#   D6: sum    -> dialog
#
# Время: ~10-15 мин на каждое cell на RTX 4090. Итого ~50-60 мин.

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" && cd "$REPO"
source env.sh

DIALOG=outputs/v2_track_dialog_nojudge_20260507_000348
CODE=outputs/v2_track_code_mbpp_20260507_164026
SUM=outputs/v2_track_summarization_nojudge_20260506_210741

REFS_CODE=results/experiments/v2_track_code_mbpp_20260507_164026/refs.jsonl
REFS_SUM=results/experiments/v2_track_summarization_nojudge_20260506_210741/refs.jsonl
REFS_DIALOG=results/experiments/v2_track_dialog_nojudge_20260507_000348/refs.jsonl

run_cell () {
  local NAME=$1
  local SFT=$2
  local DPO=$3
  local PROMPTS=$4
  local TRACK=$5
  local MAXTOK=$6
  local OUT=results/cross_domain_${NAME}
  mkdir -p "$OUT/compare" "$OUT/metrics"
  echo
  echo "============================================================"
  echo "[$NAME] SFT=$SFT  DPO=$DPO  TRACK=$TRACK"
  echo "============================================================"
  .venv/bin/python scripts/compare_models.py \
    --base_model_id Qwen/Qwen2.5-3B \
    --sft_adapter_dir "$SFT" \
    --dpo_adapter_dir "$DPO" \
    --prompts_path "$PROMPTS" \
    --out_path "$OUT/compare/compare.jsonl" \
    --max_new_tokens "$MAXTOK" \
    --temperature 0.0 \
    --top_p 1.0 \
    --prompt_style alpaca \
    --postprocess_outputs

  .venv/bin/python scripts/eval_metrics.py \
    --track "$TRACK" \
    --compare_path "$OUT/compare/compare.jsonl" \
    --refs_path "$PROMPTS" \
    --out_path "$OUT/metrics/metrics.json"

  echo "[$NAME] METRICS:"
  cat "$OUT/metrics/metrics.json"
}

run_cell d3_dialog_to_mbpp \
  "$DIALOG/sft" "$DIALOG/dpo_beta_0p3_best" "$REFS_CODE" code 220

run_cell d4_dialog_to_sum \
  "$DIALOG/sft" "$DIALOG/dpo_beta_0p3_best" "$REFS_SUM" summarization 200

run_cell d5_code_to_dialog \
  "$CODE/sft" "$CODE/dpo_beta_0p3_best" "$REFS_DIALOG" dialog 220

run_cell d6_sum_to_dialog \
  "$SUM/sft" "$SUM/dpo_beta_0p3_best" "$REFS_DIALOG" dialog 220

echo
echo "[matrix] DONE all 4 new cells."
