#!/usr/bin/env bash
set +e

# Always work inside persistent storage
export WORKDIR=/workspace/thesis-llm-alignment

# Put temp files on /workspace (not on the 5GB overlay /)
export TMPDIR=/workspace/tmp
export TEMP=/workspace/tmp
export TMP=/workspace/tmp
mkdir -p /workspace/tmp

# Hugging Face caches on /workspace
export HF_HOME=/workspace/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/workspace/.cache/huggingface/hub
export HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets

# Avoid tokenizer thread spam
export TOKENIZERS_PARALLELISM=false

# Activate venv if exists
if [ -d "$WORKDIR/.venv" ]; then
  source "$WORKDIR/.venv/bin/activate"
fi

cd "$WORKDIR"
