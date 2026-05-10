# Запуск на GPU

## 1. Подготовка окружения (один раз на машине)

```bash
bash runs/setup_env.sh
source env.sh
```

Проверка CUDA в PyTorch:

```bash
.venv/bin/python -c "import torch; print('cuda_available=', torch.cuda.is_available(), 'cuda_version=', torch.version.cuda)"
```

Если `cuda_available=False`, установить GPU-сборку PyTorch:

```bash
source env.sh
bash runs/install_torch_cuda.sh        # по умолчанию cu124
# или
bash runs/install_torch_cuda.sh cu121
```

## 2. Проверка перед запуском

```bash
source env.sh
bash runs/preflight_gpu.sh
```

Скрипт проверяет:

- `torch.cuda.is_available() == True`
- корректный вывод `nvidia-smi`
- наличие свободного места на диске

Также можно отдельно проверить JSON-конфиги без запуска:

```bash
.venv/bin/python scripts/validate_configs.py
```

## 3. Запуск экспериментов

### Один эксперимент

```bash
.venv/bin/python scripts/run_experiment.py --config configs/<name>.json
```

Доступные конфиги — см. [`PROJECT_MAP.md`](../PROJECT_MAP.md).

### Прогон базовых бенчмарков

```bash
bash runs/run_gpu_base_overnight.sh
```

Использует `configs/benchmark_dialog_base.json`, `benchmark_summarization_base.json`, `benchmark_mbpp_code_base.json`. Этапы можно отключать переменными окружения (`RUN_SMOKE=0`, `RUN_MBPP=0`, …) — см. сам скрипт.

### Полная ночная цепочка (cross-domain + β-sweep + multi-seed)

```bash
bash runs/run_overnight_chain.sh
```

### Cross-domain transfer

```bash
bash runs/run_cross_domain_d1.sh           # sum → MBPP
bash runs/run_cross_domain_d2.sh           # code → sum
bash runs/run_cross_domain_matrix.sh       # D3–D6
```

Длительным прогонам нужна GPU. По умолчанию `run_experiment.py` и `run_track_code_experiment.py` падают без CUDA. Для отладки есть флаг `--allow-cpu` (очень медленно).

## 4. Где смотреть результаты

После каждого запуска создаётся каталог:

```
results/experiments/<exp_id>/
├── summary.json
├── refs.jsonl
├── compare/compare_beta_*.jsonl
├── metrics/metrics_beta_*.json
└── logs/*.log
```

Адаптеры (LoRA-веса, `outputs/<exp_id>/sft`, `dpo_beta_*/checkpoint-*`) исключены из git как тяжёлые.

## 5. Сводная таблица

```bash
.venv/bin/python scripts/analysis/aggregate.py
# → results/aggregate.{csv,md}
```

## 6. Перегенерация фигур

```bash
.venv/bin/python scripts/analysis/make_figures.py
# → figures/*.{pdf,png}
```

## 7. Оценка метрик с LLM-judge

Пример (для уже готового compare-файла):

```bash
.venv/bin/python scripts/eval_metrics.py \
  --track summarization \
  --refs_path results/experiments/<exp_id>/refs.jsonl \
  --compare_path results/experiments/<exp_id>/compare/compare_beta_0p3.jsonl \
  --out_path results/experiments/<exp_id>/metrics/metrics_beta_0p3.json \
  --judge_model qwen2.5:7b \
  --judge_base_url http://127.0.0.1:11434/v1
```

Пакетный прогон judge на нескольких compare-файлах:

```bash
bash runs/run_judge_batch.sh
```

## 8. Частые ошибки

- `cuda_available=False` — установлен CPU-only PyTorch. Запустить `runs/install_torch_cuda.sh`.
- `CUDA out of memory` — уменьшить `max_seq_len`, `max_length`, `max_prompt_length`, `batch_size` в конфиге.
- Ошибки скачивания моделей — повторить запуск после восстановления сети.
- Падение шага пайплайна — открыть `results/experiments/<exp_id>/logs/<step>.log`.
