# Reusable Environment Guide

This project is already configured to keep temp/cache data in `/workspace` via `env.sh`.

## One-time setup

Run once on a new machine/session:

```bash
bash runs/setup_env.sh
```

What it does:
- creates `.venv` if missing
- recreates `.venv` automatically if interpreter is broken
- installs `requirements.txt`
- installs `ipykernel` and registers notebook kernel `Python (thesis-venv)`
- validates core imports
- keeps Hugging Face cache in `/workspace/.cache/huggingface`

## Reuse in future sessions

Before running scripts/notebooks:

```bash
source /workspace/thesis-llm-alignment/env.sh
```

This will:
- export `WORKDIR`, `TMPDIR`, HF cache variables
- activate `.venv` automatically (if exists)
- `cd` into the project root

In notebooks, choose kernel:
- `Python (thesis-venv)`

## Suggested workflow

1. Quick validation run: open `smoke_pipeline_run.ipynb`
2. Full run: open `full_pipeline_run.ipynb`

## Optional reproducibility tip

After a successful setup, you can snapshot exact versions:

```bash
source /workspace/thesis-llm-alignment/env.sh
pip freeze > requirements.lock.txt
```

Then restore exactly with:

```bash
source /workspace/thesis-llm-alignment/env.sh
pip install -r requirements.lock.txt
```
