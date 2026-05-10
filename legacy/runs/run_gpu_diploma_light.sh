#!/usr/bin/env bash
# Укороченная сетка: configs/diploma_grid_light.json
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck source=/dev/null
source "$ROOT/env.sh"
exec python scripts/run_experiment.py --config configs/diploma_grid_light.json "$@"
