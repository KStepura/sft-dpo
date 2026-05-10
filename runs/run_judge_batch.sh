#!/usr/bin/env bash
# Запуск pairwise LLM-judge на всех готовых результатах экспериментов.
# Требует: Ollama запущена локально с нужной моделью.
#
# Использование:
#   bash runs/run_judge_batch.sh
#
# Переменные окружения (опционально):
#   JUDGE_MODEL   — имя модели в Ollama (по умолчанию: qwen2.5:7b)
#   JUDGE_URL     — base_url Ollama (по умолчанию: http://127.0.0.1:11434/v1)
#   JUDGE_LIMIT   — прогнать только первые N промптов на файл (0 = все; для теста: 2)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck source=/dev/null
source "$ROOT/env.sh"

export PYTHONUNBUFFERED=1

JUDGE_MODEL="${JUDGE_MODEL:-qwen2.5:7b}"
JUDGE_URL="${JUDGE_URL:-http://127.0.0.1:11434/v1}"
JUDGE_LIMIT="${JUDGE_LIMIT:-0}"

echo "== LLM Judge batch =="
echo "   model : $JUDGE_MODEL"
echo "   url   : $JUDGE_URL"
echo "   limit : ${JUDGE_LIMIT:-all}"
echo ""

# Проверяем что Ollama отвечает
if ! curl -sf "$JUDGE_URL/../tags" > /dev/null 2>&1; then
  echo "FATAL: Ollama не отвечает на $JUDGE_URL"
  echo "       Запусти: ollama serve"
  exit 1
fi

# Проверяем что нужная модель загружена
if ! curl -sf "$JUDGE_URL/../tags" | python3 -c "
import json, sys
models = [m['name'] for m in json.load(sys.stdin).get('models', [])]
model = '$JUDGE_MODEL'
# Ollama может хранить модель без тега :latest
ok = any(m == model or m.split(':')[0] == model.split(':')[0] for m in models)
if not ok:
    print(f'FATAL: модель {model} не найдена в Ollama. Доступны: {models}', file=sys.stderr)
    sys.exit(1)
print(f'Модель OK: {model}')
"; then
  echo "Запусти: ollama pull $JUDGE_MODEL"
  exit 1
fi

LOGROOT="$ROOT/results/experiments/_batch_logs"
mkdir -p "$LOGROOT"
TS="$(date +%Y%m%d_%H%M%S)"
LOGFILE="$LOGROOT/${TS}_judge_batch.log"

EXTRA_ARGS=""
if [ "$JUDGE_LIMIT" != "0" ]; then
  EXTRA_ARGS="--limit $JUDGE_LIMIT"
fi

echo "Лог: $LOGFILE"
echo ""

python scripts/run_judge_batch.py \
  --judge_model "$JUDGE_MODEL" \
  --judge_base_url "$JUDGE_URL" \
  --exp_root results/experiments \
  $EXTRA_ARGS \
  2>&1 | tee "$LOGFILE"

echo ""
echo "Done. Результаты в results/experiments/*/metrics/metrics_beta_*.json"
