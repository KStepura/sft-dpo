#!/usr/bin/env bash
# Проверки перед длительным GPU-прогоном. Запуск из любого каталога: bash runs/preflight_gpu.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ -f "$ROOT/env.sh" ]; then
  # shellcheck source=/dev/null
  source "$ROOT/env.sh"
fi

echo "== Repo root: $ROOT"
echo "== Python: $(command -v python)"
python --version

echo
echo "== Torch / CUDA"
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device 0:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
PY

echo
echo "== nvidia-smi (if available)"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || nvidia-smi
else
  echo "nvidia-smi not in PATH (OK for CPU-only images)"
fi

echo
echo "== Disk free (repo and HF cache)"
df -h "$ROOT" "${HF_HOME:-$HOME/.cache/huggingface}" 2>/dev/null || df -h "$ROOT"

echo
echo "Preflight done."
