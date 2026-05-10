#!/usr/bin/env bash
# Retry двух упавших шагов из overnight chain v3:
#   - dialog seed=1337 (reuse уже обученного SFT адаптера, ~80 мин)
#   - code (MBPP) seed=1337 (полный пайплайн с нуля, ~3.5 часа)
#
# Итого: ~4.5-5 часов.

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" && cd "$REPO"
source env.sh

LOG_DIR=logs
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)

run_step () {
  local NAME=$1
  shift
  local LOG="$LOG_DIR/retry_${NAME}_${TS}.log"
  echo
  echo "================================================================"
  echo "[$(date -u '+%H:%M:%S')] STEP: $NAME"
  echo "  log: $LOG"
  echo "================================================================"
  ( "$@" ) > "$LOG" 2>&1 && echo "[$(date -u '+%H:%M:%S')] $NAME: OK" || echo "[$(date -u '+%H:%M:%S')] $NAME: FAILED"
  sleep 10
}

run_step "01_dialog_seed1337_reuse" .venv/bin/python scripts/run_experiment.py \
  --config configs/dialog_seed1337.json

run_step "02_code_seed1337" .venv/bin/python scripts/run_track_code_experiment.py \
  --config configs/code_seed1337.json

echo
echo "[$(date -u '+%H:%M:%S')] RETRY DONE."
