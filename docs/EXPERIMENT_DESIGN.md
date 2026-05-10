# Дизайн эксперимента

Серия отвечает на RQ1–RQ4 (см. [`RESEARCH_QUESTIONS.md`](RESEARCH_QUESTIONS.md)). Хочется минимально возможные артефакты и контролируемая стохастика.

---

## 1. Базовая установка

| Параметр | Значение |
|---|---|
| Модель | `Qwen/Qwen2.5-3B` (без instruct) |
| Квантизация | 4-bit nf4, double-quant, bf16 compute |
| LoRA | r=16, α=32, dropout=0.05, target=`q,k,v,o,gate,up,down`-proj |
| prompt_style | `alpaca` (у базы нет chat-template) |
| seed'ы | 42, 1337, 2024 (3 повтора) |
| GPU | один (как в текущих прогонах) |

Базовую модель не меняем — это сохраняет совместимость с уже сделанным аудитом.

---

## 2. Треки и данные

| Track | Train для SFT | DPO-пары | Eval refs |
|---|---|---|---|
| dialog | tatsu-lab/alpaca (full split) | 600 пар on-policy + 200 контр-парных | `data/refs.benchmark_dialog_base.jsonl` |
| summarization | tatsu-lab/alpaca (full split) | 600 пар on-policy + 200 контр-парных | `data/refs.benchmark_summarization_base.jsonl` |
| code | google-research-datasets/mbpp `train` (text→code) | 400 пар (см. ниже) | mbpp `test`, 40 задач |

Eval-refs не меняем — это сохраняет сопоставимость со старыми числами.

### 2.1. Как строятся DPO-пары

Три варианта пар, каждый — отдельная конфигурация.

**P-baseline (контроль).**

- `chosen` — alpaca/MBPP gold output;
- `rejected` — генерация **base**, T=0.7, top_p=0.9.

Именно эти пары приводят к коллапсу в ранней серии. Оставляем для воспроизведения проблемы.

**P-onpolicy (главный).**

- сначала обучаем SFT (200 шагов);
- `chosen` — генерация **SFT**, T=0.0 (greedy);
- `rejected` — генерация **SFT**, T=1.2, top_p=0.95;
- если `chosen ≈ rejected` (token-level Levenshtein < 10%), пара выкидывается.

**P-mixed (если есть бюджет).**

- `chosen` — SFT-greedy;
- `rejected` — base-output;
- `chosen` ближе к политике, `rejected` всё ещё «дальний», но `chosen`-distribution и ref-distribution совпадают (ref = SFT).

Реализовано через `--policy_source` и `--sft_adapter_dir` в `scripts/build_dpo_pairs.py` (см. [`FIXES.md`](FIXES.md) §4).

### 2.2. Фильтрация пар

В `scripts/clean_dpo_pairs_full.py`:

- удалить пары, где `len(rejected) < 0.3 * len(chosen)` или `len(rejected) > 5 * len(chosen)`;
- удалить пары, где token-overlap Jaccard < 0.05 или > 0.95;
- сохранять `clean_stats.json` (уже есть).

Так снижаем долю «лёгких» пар, на которых DPO моментально достигает 100% accuracy.

---

## 3. Конфигурации обучения

### 3.1. SFT (одинаковый для всех треков, поправка только на бюджет)

| Параметр | Значение | Изменение |
|---|---|---|
| `max_steps` | 200 (1000 для code) | без изменений |
| `lr` | 2e-4 | без изменений |
| `batch_size` × `grad_accum` | 1 × 16 | без изменений |
| `max_seq_len` | 1024 | без изменений |
| `packing` | **False** | изменено (или включить `flash_attn_2`) |
| `attn_implementation` | `flash_attention_2`, если установлен | новое |

### 3.2. DPO (главный эксперимент)

| Параметр | Значение | Зачем |
|---|---|---|
| `max_steps` | **40** | коллапс наступает на 20-м шаге; 40 даёт запас |
| `save_steps` | **10** | для пост-hoc выбора чекпойнта |
| `save_total_limit` | **6** | сохранить 10/20/30/40 + текущий + последний |
| `lr` | **2e-6** | в 5× ниже текущего 1e-5; против резкого ухода от ref |
| `beta` | `[0.1, 0.3]` | 0.3 сильно держит у ref; 0.1 — для сравнения с прошлыми прогонами |
| `loss_type` | `["sigmoid", "ipo"]` | IPO как контроль на устойчивость к лёгким парам |
| `batch_size` × `grad_accum` | 1 × 16 | без изменений |
| `max_length` | 1024 | без изменений |
| `max_prompt_length` | 512 | без изменений |

### 3.3. Eval / compare

| Параметр | Значение | Зачем |
|---|---|---|
| `do_sample` | False (greedy) | детерминированный compare |
| `max_new_tokens` | 200 (sum), 220 (dialog), 220 (code) | как сейчас |
| `--postprocess_outputs` | **True** | симметричная обрезка хвоста всем моделям |
| LLM judge | `qwen2.5:7b` через Ollama | как сейчас |
| Judge mode | pairwise, AB+BA | как сейчас |

---

## 4. Матрица прогонов

```
tracks = [dialog, summarization, code]
methods = [base, SFT, DPO_sigmoid_baseline, DPO_sigmoid_onpolicy, DPO_ipo_onpolicy]
betas = [0.1, 0.3]   # только для DPO
seeds = [42, 1337, 2024]
```

Полная матрица — 3 × (1 + 1 + 2×2 + 2×2) = 3 × 10 = 30 ячеек, каждая ×3 seed = 90 train-прогонов. Это слишком много на один GPU.

### 4.1. Бюджетный план (минимум)

Сокращаем до 30 ячеек × 1 seed = 30 train-прогонов:

- multi-seed только на главной ячейке: для каждого трека выбираем лучший DPO-вариант после первой волны (3 seeds), остальные ячейки — 1 seed;
- `β=0.1` опускаем, оставляем только `β=0.3` — экономит половину прогонов:

```
methods × β = base + SFT + DPO_sigmoid_baseline_β0.3
            + DPO_sigmoid_onpolicy_β0.3 + DPO_ipo_onpolicy_β0.3
            = 5
```

3 трека × 5 методов × 1 seed = 15 train-прогонов (плюс +6 для multi-seed на главной ячейке). Один SFT-чекпойнт переиспользуется для всех DPO-вариантов одного трека.

### 4.2. Бюджет времени (грубая оценка)

`results/experiments/benchmark_summarization_base_20260503_232644/summary.json`:

- `build_dpo_pairs` — ~70 мин (раз на трек),
- `train_sft` — ~46 мин,
- `train_dpo` (200 шагов) — ~36 мин (при 40 шагах ~7 мин),
- `compare` — ~9 мин,
- `eval_metrics` + judge — 2–5 мин.

Один трек с 5 методами:

```
build (70) + sft (46) + 4 × (dpo 7 + compare 9 + eval 4) = 70 + 46 + 80 ≈ 196 мин ≈ 3.3 ч
```

3 трека ≈ 10 часов. Multi-seed для главной ячейки — ещё ~3–4 часа. Итого ~14 часов на одной GPU. Помещается в одну ночь.

---

## 5. Критерии «эксперимент состоялся»

Перед сводкой результатов проверить:

1. На smoke-прогоне (`smoke_quick.json`) `aligned-rate` в `metrics.json` = 1.0.
2. После полного прогона `dpo_metrics.json` для каждого DPO-варианта содержит ненулевые `loss_final`, `reward_margin_final`. Если хоть один None — баг в `dpo_metrics_utils.py` (P2).
3. Position-bias rate (`position_consistent_rate`) у judge ≥ 0.6 на summarization и dialog. Меньше — judge нестабилен; нужно `temperature=0` и/или другая модель.
4. RAW vs CLEAN таблица посчитана для summarization. Обе строки в итоговой сводке.

---

## 6. Какие графики строятся

1. Training curves DPO. Для каждого трека по шагам — `loss`, `rewards/margins`, `logps/rejected`. Стек по 5 методам. Видно коллапс P-baseline и устойчивость P-onpolicy/IPO.
2. Per-track winrate matrix. 3×3 (base/sft/dpo) heatmap winrate по judge.
3. RAW vs CLEAN bar chart (summarization). Сравнение `rougeL` по 3 моделям.
4. Method × seed scatter. По главному треку — winrate vs DPO-вариант, ±std.
5. Judge agreement scatter. `rougeL` vs judge winrate, точка на каждый промпт.

Сохранять в `figures/*.pdf` (LaTeX) + `*.png` (презентация).

---

## 7. Чекпойнт-selection

DPO сохраняет чекпойнты каждые 10 шагов. После этого нужен ещё один шаг:

```
for ckpt in checkpoint-{10,20,30,40}:
    compare_models.py(adapter_dir=ckpt, prompts=dev_prompts) -> compare_dev.jsonl
    eval_metrics.py(--track ..., --judge_model ...) -> metrics_dev.json
choose ckpt with max judge_winrate(dpo) on dev.
copy chosen ckpt -> outputs/experiments/.../dpo_beta_X_best/
```

`dev_prompts` — отдельные 25 промптов из `refs.jsonl` (последние 25 после shuffle). `eval_prompts` — первые 25.

Это отдельный скрипт `scripts/analysis/select_best_dpo_checkpoint.py`.

Без него мы оцениваем DPO после коллапса, что и видно в [`AUDIT_FINDINGS.md`](AUDIT_FINDINGS.md).

---

## 8. Что попадает в `configs/`

- `smoke_quick.json` — sanity, 1 seed, 25 промптов, 50 SFT, 20 DPO.
- `dialog.json`, `summarization.json`, `code.json` — главные прогоны по трекам.

Конфиги расширяют схему `run_experiment.py` тремя новыми полями:

```jsonc
{
  "sft": { "no_packing": true },
  "dpo": {
    "loss_type": "ipo",            // новое
    "save_steps": 10,              // новое
    "save_total_limit": 6,         // новое
    "policy_source": "onpolicy"    // новое: "baseline" | "onpolicy" | "mixed"
  }
}
```

`run_experiment.py` нужно дописать чтобы пробрасывать эти поля (см. [`FIXES.md`](FIXES.md) §6).
