# GPU Run

## 1) Подготовка окружения (один раз на машине)

```bash
cd /workspace/thesis-llm-alignment
bash runs/setup_env.sh
source env.sh
```

Проверка CUDA в PyTorch:

```bash
python -c "import torch; print('cuda_available=', torch.cuda.is_available(), 'cuda_version=', torch.version.cuda)"
```

Если `cuda_available=False`, установить GPU-сборку PyTorch:

```bash
source env.sh
bash runs/install_torch_cuda.sh        # по умолчанию cu124
# или
bash runs/install_torch_cuda.sh cu121
```

## 2) Проверка перед запуском экспериментов

```bash
cd /workspace/thesis-llm-alignment
source env.sh
bash runs/preflight_gpu.sh
```

Проверить:
- `torch.cuda.is_available() == True`
- корректный вывод `nvidia-smi`
- достаточно свободного места на диске

## 3) Порядок прогонов

Рекомендуемый порядок:

```bash
source env.sh
bash runs/run_gpu_smoke.sh
bash runs/run_gpu_diploma_light.sh
bash runs/run_gpu_diploma.sh
```

Эквивалент вручную:

```bash
source env.sh
python scripts/run_experiment.py --config configs/smoke_gpu.json
python scripts/run_experiment.py --config configs/diploma_grid_light.json
python scripts/run_experiment.py --config configs/diploma_grid_v1.json
```

Запуск с пользовательским ID:

```bash
python scripts/run_experiment.py \
  --config configs/diploma_grid_v1.json \
  --exp_id my_run_2026_04_23
```

## 4) Где смотреть результаты

После каждого запуска создается каталог:

`results/experiments/<exp_id>/`

Основные файлы:
- `resolved_config.json`
- `summary.json`
- `logs/*.log`
- `compare/compare_beta_*.jsonl`

Чекпоинты/адаптеры:
- `outputs/experiments/<exp_id>/sft`
- `outputs/experiments/<exp_id>/dpo_beta_*`

## 5) Сводная таблица по всем экспериментам

```bash
source env.sh
python scripts/aggregate_results.py
```

Результат:
- `results/experiments/summary_table.csv`

## 6) Оценка метрик (включая LLM judge)

Пример для summarization:

```bash
source env.sh
python scripts/eval_metrics.py \
  --track summarization \
  --refs_path results/experiments/track_sum_v1_20260415_163422/refs.jsonl \
  --compare_path results/experiments/track_sum_v1_20260415_163422/compare/compare_beta_0p1.jsonl \
  --out_path results/metrics_with_judge_sum_0p1.json \
  --judge_model qwen2.5:7b \
  --judge_base_url http://127.0.0.1:11434/v1
```

## 7) Частые ошибки

- `cuda_available=False`: установлен CPU-only torch.
- `CUDA out of memory`: уменьшить `max_seq_len`, `max_length`, `max_prompt_length`, `batch_size`.
- ошибки скачивания моделей: повторить запуск после восстановления сети.
- падение шага пайплайна: открыть `results/experiments/<exp_id>/logs/<step>.log`.
