#!/usr/bin/env bash
# Переустановка PyTorch с поддержкой CUDA (после CPU-пода или если pip поставил CPU-wheel).
# Использование: source env.sh && bash runs/install_torch_cuda.sh [cu124|cu121|cu118|...]
# Список индексов: https://pytorch.org/get-started/locally/
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck source=/dev/null
if [ -f "$ROOT/env.sh" ]; then
  source "$ROOT/env.sh"
fi

CUDA_TAG="${1:-cu124}"
INDEX_URL="https://download.pytorch.org/whl/${CUDA_TAG}"
echo "== Installing torch torchvision torchaudio from ${INDEX_URL}"
pip install --upgrade torch torchvision torchaudio --index-url "${INDEX_URL}"

python - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device 0:", torch.cuda.get_device_name(0))
    print("torch.version.cuda:", torch.version.cuda)
else:
    print("If you expected CUDA: check driver, CUDA tag (cu124 vs cu121), and nvidia-smi.")
PY
