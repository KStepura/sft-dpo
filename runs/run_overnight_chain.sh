#!/usr/bin/env bash
# Overnight chain v3 — финальная серия экспериментов для диплома.
#
# Очерёдность (приблизительное время на RTX 4090):
#   1. Cross-domain matrix D3–D6                          ~50 мин
#   2. β=0.05 sweep (reuse SFT+pairs от seed=42)          ~30 мин
#   3. β=0.7  sweep (reuse SFT+pairs от seed=42)          ~30 мин
#   4. Sigmoid-DPO ablation (on-policy, β=0.3, seed=42)   ~30 мин
#   5. baseline-pairs seed=1337                           ~3 часа
#   6. dialog seed=1337                                   ~3 часа
#   7. code (MBPP) seed=1337                              ~3.5 часа
#
# Итого: ~11.5 часов. Помещается в 12+ часов GPU-окно.
#
# Каждый шаг логируется отдельно в logs/overnight_v3_<step>_<ts>.log.
# Между шагами короткая пауза 10 сек для очистки GPU-памяти.
#
# При падении одного шага скрипт переходит к следующему (|| true) — лучше получить
# 6 из 7 результатов, чем потерять все из-за одной ошибки.

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" && cd "$REPO"
source env.sh

LOG_DIR=logs
mkdir -p "$LOG_DIR"
TS_BASE=$(date +%Y%m%d_%H%M%S)

run_step () {
  local NAME=$1
  shift
  local LOG="$LOG_DIR/overnight_v3_${NAME}_${TS_BASE}.log"
  echo
  echo "================================================================"
  echo "[$(date -u '+%H:%M:%S')] STEP: $NAME"
  echo "  log: $LOG"
  echo "  cmd: $*"
  echo "================================================================"
  ( "$@" ) > "$LOG" 2>&1 && echo "[$(date -u '+%H:%M:%S')] $NAME: OK" || echo "[$(date -u '+%H:%M:%S')] $NAME: FAILED (continue)"
  sleep 10
}

# 1. Cross-domain matrix (D3-D6)
run_step "01_cross_domain_matrix" bash runs/run_cross_domain_matrix.sh

# 2. β=0.05
run_step "02_beta0p05" .venv/bin/python scripts/run_experiment.py \
  --config configs/sum_beta_0p05.json

# 3. β=0.7
run_step "03_beta0p7" .venv/bin/python scripts/run_experiment.py \
  --config configs/sum_beta_0p7.json

# 4. Sigmoid-DPO ablation
run_step "04_sigmoid_onpolicy" .venv/bin/python scripts/run_experiment.py \
  --config configs/sum_sigmoid.json

# 5. baseline-pairs seed=1337
run_step "05_baseline_pairs_seed1337" .venv/bin/python scripts/run_experiment.py \
  --config configs/sum_baseline_pairs_seed1337.json

# 6. dialog seed=1337
run_step "06_dialog_seed1337" .venv/bin/python scripts/run_experiment.py \
  --config legacy/configs/dialog_seed1337_failed.json

# 7. code (MBPP) seed=1337
run_step "07_code_seed1337" .venv/bin/python scripts/run_track_code_experiment.py \
  --config configs/code_seed1337.json

echo
echo "[$(date -u '+%H:%M:%S')] OVERNIGHT CHAIN V3 DONE."
ls -lat logs/overnight_v3_*_${TS_BASE}.log
