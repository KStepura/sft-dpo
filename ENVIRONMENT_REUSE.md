# Reusable Environment Guide

This project is already configured to keep temp/cache data in `/workspace` via `env.sh`.

**GPU:** после установки окружения см. пошаговый чеклист в [GPU_RUN.md](GPU_RUN.md) (проверка CUDA, порядок прогонов, агрегация результатов).

## One-time setup

Run once on a new machine/session:

```bash
bash runs/setup_env.sh
```

What it does:
- creates `.venv` if missing
- recreates `.venv` automatically if interpreter is broken
- prefers `python3.11`/`python3.10` over old `python3` when available
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

## CPU pod notes (different images)

Some CPU images do not include `python3-venv` / `ensurepip`, so `.venv` creation fails.
Also, some images default to Python 3.8, which may be too old for pinned ML package versions.

If `bash runs/setup_env.sh` fails with messages like `No module named ensurepip`:

```bash
apt-get update && apt-get install -y python3-venv
rm -rf /workspace/thesis-llm-alignment/.venv
bash /workspace/thesis-llm-alignment/runs/setup_env.sh
source /workspace/thesis-llm-alignment/env.sh
```

Quick validation:

```bash
python -m pip --version
python -c "import sys; print(sys.executable)"
```

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
