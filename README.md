# thesis-llm-alignment

Репозиторий для практического сравнения `base` vs `SFT` vs `DPO` (LoRA/TRL) на одной базовой модели:
- сбор preference-пар,
- обучение SFT и DPO (несколько `beta`),
- генерация сравнений,
- автоматическая оценка метрик,
- опциональная оценка `LLM-as-a-judge` через OpenAI-compatible API (включая локальный сервер).

Подробный запуск на GPU: `GPU_RUN.md`.

## Быстрый старт

```bash
bash runs/setup_env.sh
source env.sh
python scripts/run_experiment.py --config configs/smoke_cpu.json
```

`smoke_cpu` рассчитан на запуск без GPU (малая модель + ограниченные сэмплы).

## Что делает пайплайн

`scripts/run_experiment.py` запускает шаги по конфигу:
1. `build_dpo_pairs.py`
2. `clean_dpo_pairs_full.py`
3. `train_sft.py`
4. `train_dpo.py` для каждого `beta`
5. `compare_models.py` для каждого `beta`

Артефакты эксперимента:
- `results/experiments/<exp_id>/summary.json`
- `results/experiments/<exp_id>/resolved_config.json`
- `results/experiments/<exp_id>/compare/compare_beta_*.jsonl`
- `results/experiments/<exp_id>/logs/*.log`
- `outputs/experiments/<exp_id>/sft` и `outputs/experiments/<exp_id>/dpo_beta_*`

## Конфиги и типовые запуски

- CPU smoke: `configs/smoke_cpu.json`
- GPU smoke: `configs/smoke_gpu.json` или `bash runs/run_gpu_smoke.sh`
- Сетка диплома: `configs/diploma_grid_v1.json` или `bash runs/run_gpu_diploma.sh`
- Облегченная сетка: `configs/diploma_grid_light.json` или `bash runs/run_gpu_diploma_light.sh`

Пример:

```bash
python scripts/run_experiment.py --config configs/diploma_grid_v1.json
```

## Оценка метрик

`scripts/eval_metrics.py` принимает:
- `--track`: `dialog` | `summarization` | `code`
- `--refs_path`: эталонные данные (`prompt`, `reference`, для `code` также `tests`)
- `--compare_path`: JSONL с ответами `base/sft/dpo`
- `--out_path`: куда сохранить метрики

### Метрики по трекам

- `summarization`: `rougeL_f1`
- `dialog`: `dialog_heuristic` (простой эвристический скор)
- `code`: `test_pass_rate` (запуск тестов из `refs`)

### LLM judge (опционально)

Добавьте флаги:
- `--judge_model`
- `--judge_base_url` (OpenAI-compatible endpoint, например `http://127.0.0.1:11434/v1`)
- `--judge_api_key` (для localhost можно не задавать: подставляется `local`)

Дополнительно:
- `--judge_json_mode` (включать для OpenAI; для локальных серверов обычно не нужно)
- `--judge_sleep_sec` (по умолчанию 0.15)
- `--judge_timeout_sec` (по умолчанию 120)

Пример (локальный Ollama/vLLM gateway):

```bash
python scripts/eval_metrics.py \
  --track summarization \
  --refs_path results/experiments/track_sum_v1_20260415_163422/refs.jsonl \
  --compare_path results/experiments/track_sum_v1_20260415_163422/compare/compare_beta_0p1.jsonl \
  --out_path results/metrics_with_judge_sum_0p1.json \
  --judge_model qwen2.5:7b \
  --judge_base_url http://127.0.0.1:11434/v1
```

## Сводная таблица по экспериментам

```bash
python scripts/aggregate_results.py
```

Выход: `results/experiments/summary_table.csv`.

## Важные замечания

- Для instruct-моделей используйте `prompt_style: "chat"` в конфиге.
- Для старых alpaca-style шаблонов при ручной обработке можно использовать `prompt_style: "alpaca"`.
- Для воспроизводимой оценки в compare обычно держат `eval.do_sample: false`.

## Структура проекта

- `scripts/` — основной код пайплайна и метрик
- `configs/` — конфиги экспериментов
- `runs/` — shell-обертки для запусков
- `data/` — локальные наборы и промежуточные данные
- `outputs/` — адаптеры и модели после тренировки
- `results/` — сравнения, метрики, сводки

Окружение и переиспользование кэша: `ENVIRONMENT_REUSE.md`.
