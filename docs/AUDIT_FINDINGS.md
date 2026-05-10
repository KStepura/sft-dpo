# Аудит ранней серии экспериментов

Дата: 2026-05-05.
Контекст: SFT + DPO пайплайн на `Qwen2.5-3B` (base, alpaca prompt style), ранняя версия.

Здесь описаны методологические ошибки ранней серии и фиксы, реализованные в финальной серии. Конкретные правки по строкам — в [`FIXES.md`](FIXES.md). Старые артефакты лежат в [`../legacy/`](../legacy/).

---

## Численные результаты ранней серии (после симметричной обрезки)

| Track | Метрика | base | sft | dpo (best β) | LLM judge wins (best β) |
|---|---|---:|---:|---:|---|
| Summarization | rougeL_f1 | **0.294** | 0.310 | 0.255 (β=0.5) | base 0.02–0.16, **sft 0.18–0.32**, dpo 0.04–0.14 |
| Code (MBPP) | test_pass_rate | 0.017 | **0.067** | 0.025 | **base 0.30–0.38**, sft 0.21–0.25, dpo 0.06–0.13 |
| Dialog | heuristic | 0.688 | 0.684 | 0.66 | все ≈ 0.14–0.24, ~70% ничьих |

RAW vs CLEAN на summarization:

- RAW (без обрезки): base = 0.188, sft = 0.310, dpo = 0.220.
- CLEAN (с обрезкой хвоста base): base = **0.294**, sft = 0.310, dpo = 0.220.

Кажущийся прирост SFT над base сжимается с +65% до +5% — большая часть «эффекта SFT» оказалась артефактом препроцессинга.

---

## Критические баги

### 1. DPO коллапсирует к loss = 0 за 20 шагов

Файл: [`scripts/train_dpo.py`](../scripts/train_dpo.py)

```
Шаг 10  (epoch 0.23):  loss = 0.297,  rewards/accuracies = 0.762
Шаг 20  (epoch 0.46):  loss = 0.0003, rewards/accuracies = 1.0    ← коллапс
Шаги 30–200:            loss = 0.0,    logps/rejected = -204→-272  ← reward hacking
```

Финальное `logps/rejected ≈ −272` означает вероятность ≈ e⁻²⁷² ≈ 0. Модель «забывает» отклонённые ответы и теряет полезное поведение.

Причина. `chosen` (alpaca ground truth) и `rejected` (генерация base-модели) слишком разные. Классификатор моментально достигает 100% accuracy. Дальше идёт переобучение и уход от reference.

В финальной серии. Перешли на on-policy preference-pairs: `chosen` и `rejected` оба от SFT-модели, разные температуры. Sigmoid-loss заменили на IPO. См. [`FIXES.md`](FIXES.md) §4.

---

### 2. SFT обучается с `packing=True` без flash_attention

Файл: [`scripts/train_sft.py`](../scripts/train_sft.py)

В логе TRL предупреждение:

> *"You are using packing... may lead to **cross-contamination between samples**"*

Внимание одного сэмпла течёт в соседний по упакованной последовательности. SFT учится на зашумлённых градиентах.

В финальной серии. По умолчанию `packing=False` (см. [`FIXES.md`](FIXES.md) §3). Альтернатива — установить `flash-attn` и `attn_implementation="flash_attention_2"`.

---

### 3. eval_metrics.py не обрезает хвосты base-модели

Файл: [`scripts/eval_metrics.py`](../scripts/eval_metrics.py)

Base-модель (без instruct-тюнинга) после ответа продолжает галлюцинировать. Выдумывает новые `### Instruction: ...` и отвечает на них:

```
PROMPT: Summarize in two sentences: Arctic sea ice...
BASE (1116 chars):
  "Arctic sea ice extent has been decreasing... fisheries.

  ### Instruction:
  Write a 1000-word essay on the importance of education...

  ### Response:
  Educa..."   ← модель сама задала новую задачу и отвечает
```

В [`scripts/build_dpo_pairs.py`](../scripts/build_dpo_pairs.py) есть `postprocess_rejected`. Он режет на `\n### Instruction:` для DPO-трейна. Но к compare-файлам при evaluation этот же постпроцессинг не применяется. Из-за длины ROUGE precision base-модели катастрофически низкий.

«Улучшение SFT над base» (0.19 → 0.31) на summarization во многом искусственное. SFT не «научилась лучше суммаризовать» — она «научилась останавливаться».

В финальной серии. Симметричная `truncate_at_instruction` для всех трёх моделей (base/SFT/DPO) и в `eval_metrics.py`, и в `compare_models.py`. См. [`FIXES.md`](FIXES.md) §1.

---

## Умеренные баги

### 4. `dpo_metrics = null` в старых прогонах

Файлы: [`scripts/dpo_metrics_utils.py`](../scripts/dpo_metrics_utils.py), [`scripts/run_experiment.py`](../scripts/run_experiment.py)

`run_experiment.py` ищет `outputs/experiments/.../checkpoint-*`. В старой серии файлы лежали в другой директории (legacy/draft). `trainer_state.json` не находится — все поля `None`.

Касается только старых прогонов. Новые конфиги воспроизводят корректно.

---

### 5. Устаревшие judge-результаты

Файлы вида `legacy/results/judge_v1/metrics_with_judge_*.json` получены из ранних track-экспериментов (апрель). Показывают `base == sft` по всем метрикам. Канонические результаты — в [`results/experiments/`](../results/) для каждого финального прогона.

---

### 6. dialog_heuristic малоинформативен

Файл: [`scripts/eval_metrics.py`](../scripts/eval_metrics.py)

Старая версия. Проверка `len(txt) >= 80` всегда True. Получаем +0.4 ко всем сэмплам. Эвристика не различает модели — все ≈ 0.66–0.69.

В финальной серии. Новая четырёхкомпонентная метрика: длина в диапазоне 40–200 слов + Jaccard token-overlap с reference + polite-маркеры + структура. См. [`FIXES.md`](FIXES.md) §2.

---

## Качественный анализ ответов в ранней серии

### Summarization

- base — суше, ближе к исходному тексту.
- sft — естественнее, но добавляет лишних деталей.
- dpo — галлюцинирует факты (выдумал «commitment to provide more resources, parental involvement», которых не было в исходнике).

Лучший: base или sft. DPO опасен — выдумывает.

### Dialog

- base — длиннее, формальнее.
- sft — короче, сфокусирован.
- dpo — самый «человечный» тон, естественные извинения.

Лучший: DPO по тону. Различия маленькие.

### Code

- base — корректный, читаемый код, иногда с примером вызова.
- sft — странный стиль (`max_Number(Digit)`, заглавные имена), баги.
- dpo — код сломан (`s = x*x + y*y + y*y` — повтор `y*y` вместо `z*z`; `" ".join` вместо `""`).

Лучший: base. SFT и DPO выучили мусорный стиль из Alpaca-пар.

---

## Численные сводки ранней серии

### Summarization rougeL_f1 (RAW vs CLEAN)

```
beta=0.05  RAW:   base=0.188 sft=0.310 dpo=0.220
beta=0.05  CLEAN: base=0.294 sft=0.310 dpo=0.220
beta=0.1   RAW:   base=0.188 sft=0.310 dpo=0.223
beta=0.1   CLEAN: base=0.294 sft=0.310 dpo=0.223
beta=0.2   RAW:   base=0.188 sft=0.310 dpo=0.227
beta=0.2   CLEAN: base=0.294 sft=0.310 dpo=0.227
beta=0.5   RAW:   base=0.188 sft=0.310 dpo=0.255
beta=0.5   CLEAN: base=0.294 sft=0.310 dpo=0.255
```

### Code test_pass_rate (RAW == CLEAN, extract_code_block уже корректен)

```
beta=0.1: base=0.017 sft=0.067 dpo=0.025
beta=0.2: base=0.017 sft=0.067 dpo=0.025
```

### Dialog heuristic (RAW vs CLEAN)

```
beta=0.05  RAW:   base=0.78  sft=0.684 dpo=0.592
beta=0.05  CLEAN: base=0.688 sft=0.684 dpo=0.592
beta=0.1   RAW:   base=0.78  sft=0.684 dpo=0.616
beta=0.1   CLEAN: base=0.688 sft=0.684 dpo=0.616
beta=0.2   RAW:   base=0.78  sft=0.684 dpo=0.664
beta=0.2   CLEAN: base=0.688 sft=0.684 dpo=0.664
beta=0.5   RAW:   base=0.78  sft=0.684 dpo=0.664
beta=0.5   CLEAN: base=0.688 sft=0.684 dpo=0.664
```

### Средние длины ответов после обрезки (chars)

```
dialog/20260503_182808         base=451  sft=262  dpo=270
mbpp_code/20260504_042309      base=151  sft=170  dpo=151
summarization/20260503_232644  base=205  sft=189  dpo=284
```

---

## Что показала ранняя серия

1. DPO на маленькой модели (3B) с короткими прогонами и Alpaca-парами склонен к коллапсу. `loss → 0` за 20 шагов, `logps/rejected → −272`. Модель не учится предпочтениям, а делает reward hacking, уходя от reference. Это известная проблема DPO, эмпирически воспроизведённая на ограниченном бюджете.
2. SFT улучшает качество не всегда. На code SFT увеличивает `pass_rate` в 4 раза, но качество читаемого кода падает (выученный мусорный стиль из Alpaca). На summarization кажущееся улучшение — артефакт метрики (base просто не умеет останавливаться). На dialog различий нет.
3. Метрика чувствительна к препроцессингу. Один и тот же `rougeL_f1` показывает «base = 0.19» или «base = 0.29» в зависимости от того, режется ли хвост. Без честной обрезки выводы об эффекте SFT неверны.
4. LLM-judge не всегда согласен с автоматическими метриками. На code тесты говорят «SFT лучше», judge — «base лучше». Judge оценивает читаемость, тесты — корректность. Метрики дополняют друг друга, а не заменяют.

Всё это учтено в финальной серии. Конкретные изменения кода — в [`FIXES.md`](FIXES.md). Финальные результаты — в [`../results/aggregate.csv`](../results/aggregate.csv) (или `aggregate.md` после `scripts/analysis/aggregate.py`; `.md` не в git).
