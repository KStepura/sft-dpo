#!/usr/bin/env bash
# Быстрый GPU-smoke: configs/smoke_gpu.json
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck source=/dev/null
source "$ROOT/env.sh"
exec python scripts/run_experiment.py --config configs/smoke_gpu.json "$@"
