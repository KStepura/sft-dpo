# Конкретные фиксы (по строкам)

Каждый фикс — отдельный пункт. После применения проверяем smoke-прогоном `configs/smoke_quick.json`.

Все ссылки идут на текущий `main`. Готовые `.diff` лежат в [`../patches/`](../patches/).

---

## §1. Симметричная обрезка хвоста в `compare_models.py` (P0)

[`scripts/eval_metrics.py`](../scripts/eval_metrics.py) уже вызывает `truncate_at_instruction(...)` (строки 402, 413, 427). Но [`scripts/compare_models.py`](../scripts/compare_models.py):263 делает это только если передан `--postprocess_outputs`, а в [`scripts/run_experiment.py`](../scripts/run_experiment.py):252–275 этот флаг не пробрасывается.

Итог: compare-файлы хранят сырые ответы с галлюцинированными хвостами. Обрезает их потом только eval, а другие потребители (judge, аналитика, отчёты) видят сырые ответы.

Фикс:

1. В [`scripts/run_experiment.py:276`](../scripts/run_experiment.py#L276) добавить безусловно:
   ```python
   compare_cmd.append("--postprocess_outputs")
   ```
   Или пробрасывать через `cfg["eval"]["postprocess_outputs"]`, по умолчанию `True`.
2. Проверить, что `compare_models.py:264-266` обрезает все три модели одинаково — да, обрезает.
3. Альтернатива: убрать обрезку из `eval_metrics.py` и оставить только в `compare_models.py`. Тогда метрики и judge видят одно и то же. Логически правильнее, но требует осторожной миграции.

Готовый патч: [`patches/run_experiment.diff`](../patches/run_experiment.diff).

---

## §2. Заменить или убрать `dialog_heuristic` (P0)

[`scripts/eval_metrics.py:413-437`](../scripts/eval_metrics.py#L413). Условие `len(txt) >= 80` даёт +0.4 буквально всем — у current-base средняя длина 451 символ. Метрика не различает модели.

Минимальный фикс — сделать heuristic осмысленной:

```python
elif args.track == "dialog":
    polite_words = ("please", "thank", "sorry", "appreciate", "kindly")
    for model_key in ("base", "sft", "dpo"):
        scores = []
        for ref, pred in aligned:
            txt = truncate_at_instruction(pred.get(model_key, "")).lower()
            ref_text = (ref.get("reference") or "").lower()
            score = 0.0
            n = len(txt.split())
            if 40 <= n <= 200:
                score += 0.3
            elif 20 <= n < 40 or 200 < n <= 350:
                score += 0.15
            ref_tokens = set(ref_text.split())
            txt_tokens = set(txt.split())
            if ref_tokens and txt_tokens:
                jacc = len(ref_tokens & txt_tokens) / len(ref_tokens | txt_tokens)
                score += min(0.4, jacc * 2.0)
            if any(w in txt for w in polite_words):
                score += 0.15
            if "\n" in txt or "1." in txt:
                score += 0.15
            scores.append(min(score, 1.0))
        metrics[model_key]["dialog_heuristic"] = round(sum(scores) / max(len(scores), 1), 4)
    metrics["primary_metric"] = "dialog_heuristic"
```

Правильный фикс — убрать heuristic вовсе. В `RESEARCH_QUESTIONS.md` явно сказать: главная метрика для dialog — judge winrate. Heuristic — sanity-check baseline.

Рекомендуем второй вариант. Одна основная метрика лучше нескольких посредственных.

---

## §3. SFT cross-contamination (P1)

[`scripts/train_sft.py:129`](../scripts/train_sft.py#L129). При `packing=True` без flash_attention сэмплы упаковываются в одну последовательность, и attention одного течёт в другой.

Вариант A (предпочтительный) — `attn_implementation`:

```python
# scripts/train_sft.py:105
model = AutoModelForCausalLM.from_pretrained(
    args.model_id,
    attn_implementation="flash_attention_2" if has_cuda else "eager",
    **model_kwargs,
)
```

Требует установленного `flash-attn`. В `runs/preflight_gpu.sh`:

```bash
python -c "import flash_attn; print(flash_attn.__version__)" || pip install flash-attn --no-build-isolation
```

Вариант B (резервный) — передавать `--no_packing` всегда, когда flash-attn недоступен. Аргумент уже есть в `train_sft.py:39`. В `run_experiment.py:174-197` добавить:

```python
if cfg["sft"].get("no_packing", False):
    train_sft_cmd.append("--no_packing")
```

Вариант B рекомендуем по умолчанию — не зависит от установленных C++-расширений.

---

## §4. On-policy DPO в `build_dpo_pairs.py` (P1)

`chosen` берётся из gold-датасета, `rejected` — из base. Это самая дальняя из возможных пар. DPO быстро становится бинарным классификатором «alpaca vs base», а не «лучше vs хуже».

Расширяем `build_dpo_pairs.py`:

```python
ap.add_argument(
    "--policy_source",
    choices=("baseline", "onpolicy", "mixed"),
    default="baseline",
    help="baseline: chosen=gold, rejected=base. "
         "onpolicy: chosen=SFT-greedy, rejected=SFT-T1.2. "
         "mixed: chosen=SFT-greedy, rejected=base.",
)
ap.add_argument(
    "--sft_adapter_dir",
    type=str,
    default=None,
    help="LoRA dir for SFT model, used when policy_source != baseline.",
)
ap.add_argument("--chosen_temperature", type=float, default=0.0)
ap.add_argument("--rejected_temperature", type=float, default=1.2)
ap.add_argument("--rejected_top_p", type=float, default=0.95)
```

В `main()` вместо одной модели — две генерации:

```python
if args.policy_source == "baseline":
    # как сейчас: chosen=gold, rejected=base.generate(T=0.7)
else:
    sft_model = PeftModel.from_pretrained(base_model, args.sft_adapter_dir)
    sft_model.eval()
    if args.policy_source == "onpolicy":
        chosen = sft_model.generate(prompt, T=args.chosen_temperature, top_p=1.0, max_new_tokens=128)
        rejected = sft_model.generate(prompt, T=args.rejected_temperature, top_p=args.rejected_top_p, max_new_tokens=128)
    else:  # mixed
        chosen = sft_model.generate(prompt, T=0.0, top_p=1.0, max_new_tokens=128)
        rejected = base_model.generate(prompt, T=0.7, top_p=0.9, max_new_tokens=128)
    chosen = postprocess_rejected(decode(chosen)[len(prompt):])
    rejected = postprocess_rejected(decode(rejected)[len(prompt):])
```

В `run_experiment.py` запускать `build_dpo_pairs` после `train_sft` для on-policy/mixed случаев. Старый порядок: build → clean → SFT → DPO. Нужный: SFT → build → clean → DPO. Это структурное изменение (см. §6).

---

## §5. Чекпойнт-selection (P1)

Сейчас `train_dpo.py` сохраняет финальный + 1–2 чекпойнта (`save_total_limit=2`), а оцениваем мы только последний — после коллапса.

1. В конфиге передавать `dpo.save_steps=10`, `dpo.save_total_limit=6`. В `train_dpo.py` пробросить (аргументы уже есть на строках 41–55).
2. В `run_experiment.py:204-242` после `train_dpo_beta_*` добавить шаг `select_best_dpo_checkpoint`:

```python
# scripts/analysis/select_best_dpo_checkpoint.py
# для каждого ckpt в dpo_out/checkpoint-*/
#   compare на dev-промптах (последние 25)
#   eval_metrics с judge
#   собрать judge_winrate[ckpt] для модели "dpo"
# выбрать argmax, сделать symlink dpo_beta_X_best -> dpo_beta_X/checkpoint-Y
```

Финальный compare указывает на `..._best`, а не на корневой `dpo_beta_X`.

3. Кладём логи `selection_log.json` рядом с `dpo_metrics.json`.

---

## §6. Передача новых полей через `run_experiment.py`

В `scripts/run_experiment.py` добавить:

- В блоке SFT: `if cfg["sft"].get("no_packing", False): train_sft_cmd.append("--no_packing")`.
- В блоке DPO: `loss_type`, `save_steps`, `save_total_limit` через `--loss_type`, `--save_steps`, `--save_total_limit`. Аргументы в `train_dpo.py` уже есть.
- В блоке `build_dpo_pairs` (нужно перенести после SFT для onpolicy): `--policy_source`, `--sft_adapter_dir`, `--chosen_temperature`, `--rejected_temperature`.
- Шаг 4.5 — `select_best_dpo_checkpoint`.
- В `eval_cfg` уважать `eval.postprocess_outputs` (по умолчанию `True`).

Порядок шагов после фикса:

```
1. (если policy != baseline) prepare_sft_input  ← новый dummy шаг
2. train_sft
3. build_dpo_pairs(--policy_source ..., --sft_adapter_dir sft_out)
4. clean_dpo_pairs
5. for beta in betas:
     train_dpo(--save_steps 10 --loss_type ...)
     select_best_dpo_checkpoint(--dev_prompts ...)
     compare_models(--postprocess_outputs --dpo_adapter_dir dpo_best)
     eval_metrics(--judge_model qwen2.5:7b)
```

---

## §7. Корректный alignment refs ↔ compare (P2)

[`scripts/eval_metrics.py:379`](../scripts/eval_metrics.py#L379) — точное совпадение по строке `prompt`. Если в `refs.jsonl` промпт хранится без переноса, а `compare_models.py` нормализует его (например, `\n\n` → `\n`), `aligned` будет короче.

Фикс:

```python
def _norm(p: str) -> str:
    return " ".join((p or "").split())

by_prompt = {_norm(r["prompt"]): r for r in compare_rows}
aligned = [(r, by_prompt[_norm(r["prompt"])])
           for r in ref_rows if _norm(r["prompt"]) in by_prompt]

if len(aligned) < len(ref_rows):
    print(f"WARN: aligned {len(aligned)}/{len(ref_rows)} — check prompt normalisation")
```

Добавить в `metrics.json` поле `align_rate = aligned/ref_rows`.

---

## §8. Что не трогаем

- `extract_code_block` — корректен.
- LLM-judge pairwise (AB+BA) — корректен.
- `aggregate_results.py` — почти не зависит от фиксов; обновим только колонки в финале.
- `prompt_utils.py` — без изменений.
