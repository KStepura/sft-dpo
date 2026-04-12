#!/usr/bin/env bash
set -euo pipefail

WORKDIR="/workspace/thesis-llm-alignment"
VENV_PATH="$WORKDIR/.venv"

# Prefer newer Python for modern ML stack compatibility.
if command -v python3.11 >/dev/null 2>&1; then
  PYTHON_SYS_BIN="$(command -v python3.11)"
elif command -v python3.10 >/dev/null 2>&1; then
  PYTHON_SYS_BIN="$(command -v python3.10)"
else
  PYTHON_SYS_BIN="$(command -v python3)"
fi

cd "$WORKDIR"
echo "Using system Python: $PYTHON_SYS_BIN ($("$PYTHON_SYS_BIN" --version 2>/dev/null || echo unknown))"

mkdir -p /workspace/tmp
mkdir -p /workspace/.cache/huggingface/hub
mkdir -p /workspace/.cache/huggingface/datasets

create_venv() {
  # Prefer stdlib venv.
  if "$PYTHON_SYS_BIN" -m venv "$VENV_PATH"; then
    return 0
  fi

  echo "python -m venv failed. Trying virtualenv fallback..."
  if "$PYTHON_SYS_BIN" -m pip --version >/dev/null 2>&1; then
    "$PYTHON_SYS_BIN" -m pip install --upgrade pip >/dev/null
    "$PYTHON_SYS_BIN" -m pip install virtualenv >/dev/null
    "$PYTHON_SYS_BIN" -m virtualenv "$VENV_PATH"
    return 0
  fi

  echo "ERROR: could not create virtual environment." >&2
  echo "Install system venv package, then rerun setup:" >&2
  echo "  apt-get update && apt-get install -y python3-venv" >&2
  return 1
}

is_venv_ready() {
  [ -x "$VENV_PATH/bin/python" ] && [ -f "$VENV_PATH/bin/activate" ]
}

if [ ! -d "$VENV_PATH" ]; then
  create_venv
fi

# Recover from partially created/broken .venv directories.
if ! is_venv_ready; then
  echo "Detected incomplete .venv, recreating..."
  rm -rf "$VENV_PATH"
  create_venv
fi

# Repair broken venv python symlink if needed.
if [ ! -x "$VENV_PATH/bin/python" ]; then
  echo "Detected broken venv interpreter, recreating .venv..."
  rm -rf "$VENV_PATH"
  create_venv
fi

source "$VENV_PATH/bin/activate"

# Some images create venv without pip. Bootstrap pip if missing.
if ! "$VENV_PATH/bin/python" -m pip --version >/dev/null 2>&1; then
  echo "pip is missing in .venv, bootstrapping with ensurepip..."
  "$VENV_PATH/bin/python" -m ensurepip --upgrade || true
fi

# If pip is still missing, fully recreate venv once.
if ! "$VENV_PATH/bin/python" -m pip --version >/dev/null 2>&1; then
  echo "pip bootstrap failed, recreating .venv..."
  rm -rf "$VENV_PATH"
  create_venv
  if ! is_venv_ready; then
    echo "ERROR: .venv exists but is incomplete after recreate." >&2
    exit 1
  fi
  source "$VENV_PATH/bin/activate"
fi

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
