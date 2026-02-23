#!/usr/bin/env bash
set -euo pipefail

WORKDIR="/workspace/thesis-llm-alignment"
VENV_PATH="$WORKDIR/.venv"
PYTHON_SYS_BIN="$(command -v python3)"

cd "$WORKDIR"

mkdir -p /workspace/tmp
mkdir -p /workspace/.cache/huggingface/hub
mkdir -p /workspace/.cache/huggingface/datasets

if [ ! -d "$VENV_PATH" ]; then
  "$PYTHON_SYS_BIN" -m venv "$VENV_PATH"
fi

# Repair broken venv python symlink if needed.
if [ ! -x "$VENV_PATH/bin/python" ]; then
  echo "Detected broken venv interpreter, recreating .venv..."
  rm -rf "$VENV_PATH"
  "$PYTHON_SYS_BIN" -m venv "$VENV_PATH"
fi

source "$VENV_PATH/bin/activate"
"$VENV_PATH/bin/python" -m pip install --upgrade pip wheel setuptools
"$VENV_PATH/bin/python" -m pip install -r requirements.txt
"$VENV_PATH/bin/python" -m pip install ipykernel

"$VENV_PATH/bin/python" -m ipykernel install --user \
  --name thesis-llm-alignment-venv \
  --display-name "Python (thesis-venv)"

"$VENV_PATH/bin/python" - <<'PY'
import importlib
pkgs = ["torch", "transformers", "datasets", "trl", "peft", "accelerate"]
missing = []
for p in pkgs:
    try:
        importlib.import_module(p)
    except Exception:
        missing.append(p)
if missing:
    raise SystemExit(f"Missing imports: {missing}")
print("Environment looks good.")
PY

echo
echo "Setup complete."
echo "Use in future sessions:"
echo "  source /workspace/thesis-llm-alignment/env.sh"
echo
echo "Notebook kernel registered as:"
echo "  Python (thesis-venv)"
