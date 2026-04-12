# thesis-llm-alignment

Сравнение **SFT** и **DPO** (LoRA, TRL) на одной базовой модели: сбор preference-пар из датасета, обучение, сравнение генераций по конфигу.

**Запуск на GPU (пошагово, что сделать до прогона и после):** [GPU_RUN.md](GPU_RUN.md).

## Быстрый старт

```bash
bash runs/setup_env.sh
source env.sh
python scripts/run_experiment.py --config configs/smoke_cpu.json
```

`smoke_cpu` намеренно использует **Qwen2.5-0.5B-Instruct** и подвыборку Alpaca (`max_train_samples` в JSON), чтобы прогон помещался в RAM без GPU; на GPU для реальных экспериментов остаются `smoke_gpu` / `diploma_*` с **3B**.

Сводка по всем прогонам: `python scripts/aggregate_results.py` → `results/experiments/summary_table.csv`.

Окружение и кэш HF: [ENVIRONMENT_REUSE.md](ENVIRONMENT_REUSE.md).

### Конфиг-раннер и артефакты

- CPU smoke: `python scripts/run_experiment.py --config configs/smoke_cpu.json`
- GPU smoke: `bash runs/run_gpu_smoke.sh` или `python scripts/run_experiment.py --config configs/smoke_gpu.json`
- Сетка по β (полная): `bash runs/run_gpu_diploma.sh` или `configs/diploma_grid_v1.json`
- Сетка по β (облегчённая): `bash runs/run_gpu_diploma_light.sh` или `configs/diploma_grid_light.json`

После прогона: `results/experiments/<exp_id>/` — `resolved_config.json`, `summary.json`, `logs/`, `compare/compare_beta_*.jsonl`; адаптеры в `outputs/experiments/<exp_id>/sft` и `dpo_beta_*`.

Для сравнения нескольких экспериментов в одной таблице: `aggregate_results.py` (см. `--help`). Детерминированная оценка: в JSON держите `"do_sample": false` в секции `eval`.

## Важно: формат промптов

Для моделей вида `*-Instruct` в конфиге задавайте `"prompt_style": "chat"` (значение по умолчанию, если ключ опущен в `run_experiment.py`). Тогда SFT, генерация rejected для DPO и `compare_models.py` согласованы через **chat template** токенизатора.

Старые JSONL в формате Alpaca (`### Instruction` / `### Response`) при ручной очистке указывайте: `--prompt_style alpaca`.

## Соответствие задачам диплома (что уже есть / что добавить)

| Заявлено в работе                         | Сейчас в репозитории |
|------------------------------------------|----------------------|
| Пайплайны SFT и DPO                      | `train_sft.py`, `train_dpo.py`, `run_experiment.py` |
| Сетка по β (DPO)                         | `configs/diploma_grid_v1.json`, цикл в `run_experiment.py` |
| Оценка «до/после»                        | `compare_models.py`, длины в `summary.json` / CSV агрегатора |
| Диалоги / суммаризация / код как бенчмарки | Пока **один** instruction-датасет (Alpaca) и **ручные** промпты в конфиге; отдельных датасетов и метрик по доменам нет |
| Уровни сигнала вознаграждения (токен/ход/задача) | **Не реализовано** (классический DPO по парам; нет reward-модели и уровней) |
| KL / регуляризация в отчёте              | β задаётся; явный лог KL из тренера в агрегатор **не** тянется — стоит добавить логирование и колонки в CSV |
| Чек-лист выбора метода                   | Пока нет отдельного документа — выводы из экспериментов нужно оформить в тексте диплома |

## Структура

- `scripts/` — обучение, данные, сравнение, агрегация
- `configs/` — JSON для `run_experiment.py`
- `runs/` — вспомогательные shell-скрипты
- `data/`, `outputs/`, `results/`, `logs/` — артефакты (частично в `.gitignore`)
