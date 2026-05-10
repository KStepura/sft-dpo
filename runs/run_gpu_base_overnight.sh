#!/usr/bin/env bash
# Полный ночной прогон только для Qwen2.5-3B (base, Alpaca): смок → Alpaca-пилот → MBPP → dialog → summarization → CSV.
# Запуск: bash runs/run_gpu_base_overnight.sh
# Пропустить смок: RUN_SMOKE=0 bash runs/run_gpu_base_overnight.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck source=/dev/null
source "$ROOT/env.sh"

export PYTHONUNBUFFERED=1

RUN_SMOKE=${RUN_SMOKE:-1}
RUN_PILOT=${RUN_PILOT:-1}
RUN_MBPP=${RUN_MBPP:-1}
RUN_DIALOG=${RUN_DIALOG:-1}
RUN_SUMMARY=${RUN_SUMMARY:-1}
RUN_AGGREGATE=${RUN_AGGREGATE:-1}

echo "== Preflight"
bash "$ROOT/runs/preflight_gpu.sh"

python - <<'PY'
import sys
import torch
if not torch.cuda.is_available():
    print("FATAL: CUDA required. Install GPU PyTorch (bash runs/install_torch_cuda.sh).", file=sys.stderr)
    sys.exit(1)
print("CUDA:", torch.cuda.get_device_name(0))
PY

python scripts/validate_configs.py

LOGROOT="$ROOT/results/experiments/_batch_logs"
mkdir -p "$LOGROOT"
TS="$(date +%Y%m%d_%H%M%S)"

run_named () {
  local name="$1"
  shift
  echo ""
  echo "========================================"
  echo "BATCH: $name"
  echo "========================================"
  "$@" 2>&1 | tee "$LOGROOT/${TS}_${name}.log"
}

if [ "$RUN_SMOKE" = 1 ]; then
  run_named smoke_quick_base python scripts/run_experiment.py --config configs/smoke_quick_base.json
fi

if [ "$RUN_PILOT" = 1 ]; then
  run_named pilot_alpaca_sft_dpo_base python scripts/run_experiment.py --config configs/pilot_alpaca_sft_dpo_base.json
fi

if [ "$RUN_MBPP" = 1 ]; then
  run_named benchmark_mbpp_code_base python scripts/run_track_code_experiment.py --config configs/benchmark_mbpp_code_base.json
fi

if [ "$RUN_DIALOG" = 1 ]; then
  run_named benchmark_dialog_base python scripts/run_experiment.py --config configs/benchmark_dialog_base.json
fi

if [ "$RUN_SUMMARY" = 1 ]; then
  run_named benchmark_summarization_base python scripts/run_experiment.py --config configs/benchmark_summarization_base.json
fi

if [ "$RUN_AGGREGATE" = 1 ]; then
  run_named aggregate_results python scripts/aggregate_results.py
fi

echo ""
echo "Done. Logs: $LOGROOT/${TS}_*.log"
