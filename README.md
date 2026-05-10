# Пайплайн сравнения SFT и DPO на одной базовой модели. Кросс-доменная оценка на трёх треках.

| | |
|---|---|
| Модель | `Qwen/Qwen2.5-3B` (без instruction-tuning) |
| Адаптация | LoRA r=16, α=32 + 4-bit NF4 (QLoRA) |
| DPO loss | IPO (главный) или sigmoid |
| Треки | dialog (Alpaca), summarization (Alpaca-style), code (MBPP) |

## Быстрый старт

```bash
bash runs/setup_env.sh
source env.sh

# smoke (~10 мин на GPU)
.venv/bin/python scripts/run_experiment.py --config configs/smoke_quick.json

# главный прогон summarization (~3 ч на RTX 4090)
.venv/bin/python scripts/run_experiment.py --config configs/summarization.json
```

GPU и окружение — в [`docs/GPU.md`](docs/GPU.md) и [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md).

## Что делает один прогон

`scripts/run_experiment.py` запускает по JSON-конфигу:

1. подготовка eval-набора (промпты + референсы);
2. SFT (LoRA, 4-bit NF4, `packing=False`);
3. построение preference-пар (`baseline-pairs` или `on-policy`);
4. очистка пар;
5. DPO (IPO/sigmoid, несколько β); все чекпойнты сохраняются;
6. пост-hoc выбор лучшего DPO-чекпойнта на dev-сплите;
7. сравнительная генерация `base/SFT/DPO` с симметричной обрезкой хвостов;
8. метрики (ROUGE-L, `dialog_heuristic`, `test_pass_rate`) и опциональный LLM-judge.

Артефакты прогона:

```
results/experiments/<exp_id>/
├── summary.json
├── refs.jsonl
├── compare/compare_beta_<β>.jsonl
└── metrics/metrics_beta_<β>.json
outputs/<exp_id>/sft/                              ← LoRA-адаптеры (gitignored, ~1 GB на эксперимент)
outputs/<exp_id>/dpo_beta_<β>/checkpoint-*/
```

## Структура

```
.
├── README.md, PROJECT_MAP.md
├── env.sh, requirements.txt
├── full_pipeline_run.ipynb           ← end-to-end демо
│
├── scripts/                          ← основной пайплайн
│   ├── run_experiment.py             ← оркестратор
│   ├── train_sft.py, train_dpo.py
│   ├── build_dpo_pairs.py, clean_dpo_pairs_full.py
│   ├── compare_models.py, eval_metrics.py
│   ├── run_judge_batch.py            ← LLM-as-a-judge через Ollama
│   └── analysis/                     ← постанализ: фигуры, агрегатор, RAW/CLEAN
│
├── configs/                          ← все актуальные JSON-конфиги
├── runs/                             ← shell-обёртки (cross-domain, overnight, judge)
├── data/                             ← eval refs (DPO pairs gitignored)
├── patches/                          ← git-diff фиксов scripts/
│
├── results/                          ← финальные результаты
│   ├── aggregate.{csv,md}            ← сводка по всем прогонам
│   ├── raw_vs_clean.{csv,md}         ← эффект симметричной обрезки
│   ├── cross_domain_d{1..6}_*/       ← cross-domain transfer
│   ├── benchmark_*_base_*/           ← clean-recomputed base-метрики
│   └── experiments/<exp_id>/         ← per-experiment compare.jsonl и metrics
├── figures/                          ← 9 PDF + PNG (главные графики)
│
├── docs/                             ← документация
│   ├── AUDIT_FINDINGS.md             ← методологические ошибки ранней серии
│   ├── FIXES.md                      ← конкретные правки кода
│   ├── RESEARCH_QUESTIONS.md         ← RQ1–RQ4 и связь с экспериментами
│   ├── EXPERIMENT_DESIGN.md          ← дизайн, гиперпараметры, бюджет
│   └── GPU.md, ENVIRONMENT.md        ← запуск и окружение
│
└── legacy/                           ← старые конфиги и результаты, не вошедшие в финал
    ├── runs/                         ← старые shell-обёртки
    ├── configs/                      ← конфиги, не дошедшие до публикации
    └── results/                      ← старые pilot/smoke метрики
```

LoRA-адаптеры (`outputs/`, ~27 GB) и raw-логи (`logs/`) исключены из git как регенерируемые. См. [`.gitignore`](.gitignore).

## Главные результаты

| | |
|---|---|
| Главный эффект | Summarization, on-policy IPO, β=0.3, 3 seed: DPO стабильно > SFT, Δ = +0.012 ± 0.005 ROUGE-L F1 |
| Apples-to-apples ablation | baseline-pairs vs on-policy на 2 seed: знак Δ переворачивается с −0.008 ± 0.001 на +0.015 ± 0.001 |
| Sigmoid vs IPO (на on-policy) | sigmoid не коллапсирует, +0.004; IPO даёт +0.014 |
| β-сканирование (5 точек) | колоколообразная кривая, max при β=0.3 |
| Multi-seed dialog (n=2) | DPO > SFT по эвристике, Δ = +0.019 ± 0.005 |
| Multi-seed code (n=2) | in-domain alignment tax: SFT/DPO < base устойчиво на двух seed |
| Cross-domain 3×3 (D1–D6) | cross-domain SFT на code превосходит in-domain; dialog↔sum симметричен; code разрушает остальные домены |
| RAW vs CLEAN | симметричная обрезка хвостов «`### Instruction:`» поднимает ROUGE-L base с 0.188 до 0.294 |

Сводная таблица — [`results/aggregate.csv`](results/aggregate.csv) (а также `aggregate.md` после `scripts/analysis/aggregate.py`; `.md` в git не коммитится). Разбор фиксов v1 → v2 — [`docs/FIXES.md`](docs/FIXES.md) и [`docs/AUDIT_FINDINGS.md`](docs/AUDIT_FINDINGS.md).

## Воспроизведение

| Что нужно | Команда |
|---|---|
| Один эксперимент | `.venv/bin/python scripts/run_experiment.py --config configs/<name>.json` |
| Multi-seed на summarization | `configs/summarization.json`, `..._seed1337.json`, `..._seed2024.json` |
| β-сканирование | `configs/sum_beta_0p{05,1,5,7}.json` (β=0.3 покрыт главным конфигом) |
| Apples-to-apples | `configs/sum_baseline_pairs{,_seed1337}.json` vs `configs/summarization{,_seed1337}.json` |
| IPO vs sigmoid | `configs/sum_sigmoid.json` vs `configs/summarization.json` |
| Cross-domain D1–D6 | `bash runs/run_cross_domain_d1.sh && bash runs/run_cross_domain_d2.sh && bash runs/run_cross_domain_matrix.sh` |
| Полная ночная цепочка | `bash runs/run_overnight_chain.sh` |
| Перегенерация фигур | `.venv/bin/python scripts/analysis/make_figures.py` |
| Перегенерация сводки | `.venv/bin/python scripts/analysis/aggregate.py` |

## Цитирование

```
Степура Е.В. Адаптация языковой модели: сравнение Supervised Fine-Tuning и
Direct Preference Optimization на кросс-доменных бенчмарках. Бакалаврская
ВКР, НИУ ВШЭ, ФКН, 2026.
```
