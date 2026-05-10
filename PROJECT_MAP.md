# Карта репозитория

Подробная навигация по проекту. Главный `README.md` — короче; здесь — полный список файлов и где что лежит. У каждой папки есть свой `README.md` с кратким оглавлением.

Базовая модель: `Qwen/Qwen2.5-3B` (без instruction-tuning). Адаптация — LoRA r=16, α=32 + 4-bit NF4. Треки: dialog (Alpaca), summarization (Alpaca-style), code (MBPP). Финальная серия экспериментов завершена 9 мая 2026.

---

## Структура верхнего уровня

```
.
├── README.md, PROJECT_MAP.md
├── env.sh, requirements.txt, full_pipeline_run.ipynb
│
├── configs/        ← JSON-конфиги экспериментов
├── data/           ← refs.*.jsonl (eval) + dpo_pairs.*.jsonl (gitignored)
├── scripts/        ← код пайплайна
│   └── analysis/   ← постанализ: фигуры, агрегатор, RAW/CLEAN
├── runs/           ← shell-обёртки (cross-domain, overnight, judge, setup)
├── patches/        ← git-diff фиксов
│
├── results/        ← per-experiment метрики, cross-domain D1–D6, сводные таблицы
├── outputs/        ← LoRA-адаптеры, ~27 GB (gitignored)
├── logs/           ← raw stdout/stderr (gitignored)
├── figures/        ← 9 PDF и PNG для диплома + дополнительная динамика
│
├── docs/           ← документация: фиксы, RQ, дизайн, GPU
├── overleaf text/  ← LaTeX-исходник диплома
│
└── legacy/         ← старые конфиги, скрипты и результаты ранней серии
```

Папки `draft/`, `final/`, `text_diploma/`, `text_project_proposal/` остались от ранних итераций и в актуальной работе не используются.

---

## Конфиги в `configs/`

JSON-конфиги без префиксов; короткие имена. exp_id в `results/experiments/` и `outputs/` сохранили префикс `v2_*`, чтобы не ломать ссылки в логах и `summary.json`.

**Бенчмарки base-модели (clean-recomputed):** `benchmark_dialog_base.json`, `benchmark_summarization_base.json`, `benchmark_mbpp_code_base.json`.

**Smoke (быстрая проверка):** `smoke_quick.json`.

**Главные прогоны по трекам (seed=42):**

```
configs/summarization.json   → results/experiments/v2_track_summarization_nojudge_20260506_210741
configs/dialog.json          → results/experiments/v2_track_dialog_nojudge_20260507_000348
configs/code.json            → results/experiments/v2_track_code_mbpp_20260507_164026
```

**Multi-seed:**

```
configs/summarization_seed1337.json  → v2_track_summarization_seed1337_nojudge_20260507_043001
configs/summarization_seed2024.json  → v2_track_summarization_seed2024_nojudge_20260508_174815
configs/dialog_seed1337.json         → v2_track_dialog_nojudge_seed1337_reuse_20260509_022124
configs/code_seed1337.json           → v2_track_code_mbpp_seed1337_20260509_044029
```

**β-сканирование (summarization, on-policy):**

```
configs/sum_beta_0p05.json   β=0.05   v2_sum_beta0p05_sweep_20260508_232137
configs/sum_beta_0p1.json    β=0.10   v2_sum_beta0p1_sweep_20260507_162719
configs/summarization.json   β=0.30   (главный прогон)
configs/sum_beta_0p5.json    β=0.50   v2_sum_beta0p5_sweep_20260508_170941
configs/sum_beta_0p7.json    β=0.70   v2_sum_beta0p7_sweep_20260508_234024
```

**Apples-to-apples ablation (источник preference-пар):**

```
configs/sum_baseline_pairs.json          baseline-pairs   seed=42     v2_sum_baseline_pairs_nojudge_20260507_024739
configs/sum_baseline_pairs_seed1337.json baseline-pairs   seed=1337   v2_sum_baseline_pairs_seed1337_20260509_001802
configs/summarization.json               on-policy        seed=42     (главный прогон)
configs/summarization_seed1337.json      on-policy        seed=1337
```

**Sigmoid vs IPO ablation:**

```
configs/sum_sigmoid.json     sigmoid (on-policy)   v2_sum_sigmoid_onpolicy_20260508_235858
configs/summarization.json   IPO (on-policy)       (главный прогон)
```

**Cross-domain transfer (D1–D6).** JSON-конфигов нет, запускаются shell-скриптами; результаты в `results/cross_domain_d{N}_*/`.

```
D1   sum → MBPP code           runs/run_cross_domain_d1.sh
D2   code → summarization      runs/run_cross_domain_d2.sh
D3   dialog → MBPP code        runs/run_cross_domain_matrix.sh
D4   dialog → summarization    runs/run_cross_domain_matrix.sh
D5   code → dialog             runs/run_cross_domain_matrix.sh
D6   summarization → dialog    runs/run_cross_domain_matrix.sh
```

---

## Скрипты в `scripts/`

Основной пайплайн (запускается через `run_experiment.py`):

- `run_experiment.py` — оркестратор: build_pairs → clean → SFT → DPO (несколько β) → select_best → compare → eval_metrics
- `run_track_code_experiment.py` — то же, но с MBPP-специфичным препроцессингом
- `train_sft.py` — LoRA-SFT (по умолчанию `--no_packing`)
- `train_dpo.py` — DPO с поддержкой `--loss_type {sigmoid,ipo}`
- `build_dpo_pairs.py` — генерация preference-пар (`--policy_source {baseline,onpolicy,mixed}`)
- `clean_dpo_pairs_full.py` — фильтрация
- `compare_models.py` — инференс base/SFT/DPO с симметричной обрезкой (`--postprocess_outputs`)
- `eval_metrics.py` — ROUGE-L (sum), `dialog_heuristic` (dialog), test_pass_rate (code)
- `prepare_mbpp_eval.py` — eval-сплит MBPP test
- `run_judge_batch.py` — LLM-as-a-judge через Ollama
- `validate_configs.py` — линтер JSON-конфигов

Постанализ в `scripts/analysis/`:

- `make_figures.py` — все 9 фигур в `figures/`
- `aggregate.py` — сводная таблица: `results/aggregate.csv` + при необходимости `aggregate.md` (`.md` в `.gitignore`)
- `select_best_dpo_checkpoint.py` — пост-hoc выбор лучшего DPO-чекпойнта (вызывается из `run_experiment.py`)
- `recompute_metrics_clean.py` + `build_raw_clean_table.py` — RAW vs CLEAN: `results/raw_vs_clean.csv` + при необходимости `raw_vs_clean.md` (`.md` в `.gitignore`)

---

## Shell-обёртки в `runs/`

```
setup_env.sh                   создание .venv + установка пакетов
install_torch_cuda.sh          установка GPU-сборки PyTorch
preflight_gpu.sh               проверка CUDA, диска, конфигов

run_cross_domain_d1.sh         D1: sum → MBPP
run_cross_domain_d2.sh         D2: code → sum
run_cross_domain_matrix.sh     D3–D6
run_overnight_chain.sh         полная ночная цепочка
run_retry_dialog_code.sh       recovery после HF-DNS-сбоя 9 мая
run_judge_batch.sh             запуск judge на готовых compare.jsonl
run_gpu_base_overnight.sh      три бенчмарка base-модели
```

---

## Документация в `docs/`

- `AUDIT_FINDINGS.md` — методологические ошибки ранней серии; используется в `discussion.tex` диплома
- `FIXES.md` — конкретные правки кода для каждой ошибки
- `RESEARCH_QUESTIONS.md` — RQ1–RQ4 и связь с экспериментами
- `EXPERIMENT_DESIGN.md` — гиперпараметры, бюджет, дизайн матрицы
- `GPU.md` — запуск на GPU
- `ENVIRONMENT.md` — установка окружения, кэш HuggingFace

В `legacy/docs/` могут лежать `PLAN.md`, `REPORTING.md`, `EXPERIMENTS_OUTLINE.md` — служебные черновики; они в `.gitignore`, актуальное описание — в `docs/`.

---

## Результаты в `results/`

Сводки:

- `aggregate.csv` — все метрики; `aggregate.md` генерируется скриптом и не коммитится
- `raw_vs_clean.csv` — эффект обрезки; `raw_vs_clean.md` генерируется скриптом и не коммитится

Структура одного эксперимента:

```
results/experiments/v2_<name>_<timestamp>/
├── summary.json                 пути всех шагов и финальный статус
├── refs.jsonl                   eval-промпты с эталонами
├── clean_stats.json             статистика очистки DPO-пар
├── compare/compare_beta_<β>.jsonl
└── metrics/metrics_beta_<β>.json    base / SFT / DPO по основной метрике
```

Cross-domain (`results/cross_domain_d*/`):

```
cross_domain_d<N>_<src>_to_<dst>/
├── compare/compare.jsonl
└── metrics/metrics.json
```

LoRA-адаптеры (~27 GB на серию) лежат отдельно в `outputs/v2_<name>_<timestamp>/`:

```
sft/                                LoRA-адаптер после SFT
dpo_beta_<β>/checkpoint-{10..40}/   все DPO checkpoints (для select_best)
dpo_beta_<β>/trainer_state.json     log_history (loss, logps, margins)
dpo_beta_<β>/checkpoint_selection.json
dpo_beta_<β>_best/                  симлинк на лучший checkpoint
```

---

## Фигуры в `figures/`

Источник — `scripts/analysis/make_figures.py`. В `overleaf text/figures/` лежат копии тех же PDF.

```
fig_track_summary.pdf                bar-chart base/SFT/DPO по 3 трекам
fig_multiseed_summarization.pdf      multi-seed на 3 трека (sum n=3, dialog n=2, code n=2)
fig_beta_curve_summarization.pdf     β-кривая 5 точек, max при β=0.3
fig_ablation_multiseed.pdf           baseline-pairs vs on-policy на 2 seed
fig_sigmoid_vs_ipo.pdf               sigmoid vs IPO при on-policy
fig_cross_domain_mbpp.pdf            полная 3×3 cross-domain матрица (3 панели)
fig_dpo_dynamics.pdf                 loss / logps / margins: v2 стабилен, v1 коллапс
fig_judge_winrates.pdf               LLM-judge win-rates по трекам
fig_raw_vs_clean.pdf                 эффект симметричной обрезки
```

Перегенерация:

```bash
.venv/bin/python scripts/analysis/make_figures.py
cp figures/*.pdf "overleaf text/figures/"
```

Дополнительно в `figures/` лежит 15 файлов `dpo_dynamics_v2_*.png` — per-experiment динамика DPO trainer signals (входные данные для `fig_dpo_dynamics.pdf`).

---

## Текст диплома в `overleaf text/`

```
main.tex                       корневой документ
bibliography.bib
sections/
├── abstract.tex
├── introduction.tex
├── relevance.tex
├── related_work.tex            SFT, RLHF, DPO, IPO, KTO, ORPO, SimPO
├── methodology.tex             модель, LoRA, DPO/IPO формулы, on-policy схема
├── experimental_setup.tex
├── results.tex                 главные таблицы и фигуры
├── discussion.tex              интерпретация, ограничения
└── conclusion.tex              ответы на RQ, перспективы
figures/                        копии 9 PDF из figures/
```

---

## Cheatsheet

```bash
source env.sh                                                # активация .venv

.venv/bin/python scripts/run_experiment.py --config configs/<name>.json
.venv/bin/python scripts/analysis/aggregate.py
.venv/bin/python scripts/analysis/make_figures.py && cp figures/*.pdf "overleaf text/figures/"

bash runs/run_cross_domain_d1.sh
bash runs/run_cross_domain_d2.sh
bash runs/run_cross_domain_matrix.sh
```

Прочитать конкретный результат:

```bash
cat results/experiments/v2_track_summarization_nojudge_20260506_210741/metrics/metrics_beta_0p3.json
```

---

## Откуда берутся числа в дипломе

```
sum, DPO=0.309, SFT=0.295 (seed=42)        results/experiments/v2_track_summarization_nojudge_20260506_210741/metrics/metrics_beta_0p3.json
sum, DPO=0.318 (seed=1337)                 results/experiments/v2_track_summarization_seed1337_nojudge_20260507_043001/metrics/metrics_beta_0p3.json
sum, DPO=0.309 (seed=2024)                 results/experiments/v2_track_summarization_seed2024_nojudge_20260508_174815/metrics/metrics_beta_0p3.json

dialog, DPO=0.559 (seed=42)                results/experiments/v2_track_dialog_nojudge_20260507_000348/metrics/metrics_beta_0p3.json
dialog, DPO=0.552 (seed=1337)              results/experiments/v2_track_dialog_nojudge_seed1337_reuse_20260509_022124/metrics/metrics_beta_0p3.json

code, SFT=DPO=0.0083 (seed=42)             results/experiments/v2_track_code_mbpp_20260507_164026/metrics/metrics_beta_0p3.json
code, SFT=DPO=0.000 (seed=1337)            results/experiments/v2_track_code_mbpp_seed1337_20260509_044029/metrics/metrics_beta_0p3.json

baseline-pairs Δ=−0.009 (seed=42)          results/experiments/v2_sum_baseline_pairs_nojudge_20260507_024739/metrics/metrics_beta_0p3.json
baseline-pairs Δ=−0.007 (seed=1337)        results/experiments/v2_sum_baseline_pairs_seed1337_20260509_001802/metrics/metrics_beta_0p3.json
sigmoid (on-policy), DPO=0.299             results/experiments/v2_sum_sigmoid_onpolicy_20260508_235858/metrics/metrics_beta_0p3.json

β=0.05, DPO=0.299                          results/experiments/v2_sum_beta0p05_sweep_20260508_232137/metrics/metrics_beta_0p05.json
β=0.10, DPO=0.300                          results/experiments/v2_sum_beta0p1_sweep_20260507_162719/metrics/metrics_beta_0p1.json
β=0.50, DPO=0.299                          results/experiments/v2_sum_beta0p5_sweep_20260508_170941/metrics/metrics_beta_0p5.json
β=0.70, DPO=0.304                          results/experiments/v2_sum_beta0p7_sweep_20260508_234024/metrics/metrics_beta_0p7.json

D1 (sum→MBPP)                              results/cross_domain_d1_sum_to_mbpp/metrics/metrics.json
D2 (code→sum)                              results/cross_domain_d2_mbpp_to_sum/metrics/metrics.json
D3 (dialog→MBPP)                           results/cross_domain_d3_dialog_to_mbpp/metrics/metrics.json
D4 (dialog→sum)                            results/cross_domain_d4_dialog_to_sum/metrics/metrics.json
D5 (code→dialog)                           results/cross_domain_d5_code_to_dialog/metrics/metrics.json
D6 (sum→dialog)                            results/cross_domain_d6_sum_to_dialog/metrics/metrics.json

RAW vs CLEAN                               results/raw_vs_clean.csv
```

---

## Соответствие старых имён новым

После реорганизации 9 мая 2026 папка `experiment_v2/` расформирована. Если в логах или старых заметках встречается путь под `experiment_v2/...`, ниже его новое расположение.

```
experiment_v2/configs/v2_track_summarization_nojudge.json    →  configs/summarization.json
experiment_v2/configs/v2_track_dialog_nojudge.json           →  configs/dialog.json
experiment_v2/configs/v2_track_code_mbpp.json                →  configs/code.json
experiment_v2/configs/v2_sum_beta0p1_sweep.json              →  configs/sum_beta_0p1.json
experiment_v2/configs/v2_sum_baseline_pairs_nojudge.json     →  configs/sum_baseline_pairs.json
experiment_v2/configs/v2_sum_sigmoid_onpolicy.json           →  configs/sum_sigmoid.json
experiment_v2/configs/v2_smoke_quick.json                    →  configs/smoke_quick.json

experiment_v2/scripts/make_figures.py                        →  scripts/analysis/make_figures.py
experiment_v2/scripts/aggregate_v2.py                        →  scripts/analysis/aggregate.py
experiment_v2/scripts/select_best_dpo_checkpoint.py          →  scripts/analysis/select_best_dpo_checkpoint.py
experiment_v2/scripts/recompute_metrics_clean.py             →  scripts/analysis/recompute_metrics_clean.py
experiment_v2/scripts/build_raw_clean_table.py               →  scripts/analysis/build_raw_clean_table.py

experiment_v2/scripts/run_cross_domain_d1.sh                 →  runs/run_cross_domain_d1.sh
experiment_v2/scripts/run_cross_domain_d2.sh                 →  runs/run_cross_domain_d2.sh
experiment_v2/scripts/run_cross_domain_matrix.sh             →  runs/run_cross_domain_matrix.sh
experiment_v2/scripts/run_overnight_chain_v3.sh              →  runs/run_overnight_chain.sh
experiment_v2/scripts/run_retry_dialog_code.sh               →  runs/run_retry_dialog_code.sh

experiment_v2/results/cross_domain_d1_*                      →  results/cross_domain_d1_*
experiment_v2/results/aggregate_v2.{csv,md}                  →  results/aggregate.csv (+ локально aggregate.md)
experiment_v2/results/raw_vs_clean.{csv,md}                  →  results/raw_vs_clean.csv (+ локально raw_vs_clean.md)
experiment_v2/figures/*                                      →  figures/*
experiment_v2/outputs/v2_*                                   →  outputs/v2_*
experiment_v2/data/dpo_pairs.v2_*.jsonl                      →  data/dpo_pairs.v2_*.jsonl
experiment_v2/patches/*.diff                                 →  patches/*.diff

experiment_v2/{README, FIXES, RESEARCH_QUESTIONS, EXPERIMENT_DESIGN}.md   →  docs/
experiment_v2/{PLAN, REPORTING, EXPERIMENTS_OUTLINE}.md                   →  legacy/docs/

ENVIRONMENT_REUSE.md (root)   →   docs/ENVIRONMENT.md
GPU_RUN.md (root)             →   docs/GPU.md
```

Префикс `v2_` сохранён в exp_id и адаптерах. Это нужно чтобы ссылки внутри `summary.json`, `metrics.json`, `trainer_state.json` и логов остались рабочими.
