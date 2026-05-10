# Исследовательские вопросы

Серия экспериментов отвечает на четыре связанных вопроса. К каждому привязана конкретная метрика и конкретная часть эксперимента.

---

## RQ1. Помогает ли SFT (LoRA, 200 шагов, Alpaca/MBPP) на маленькой не-инструкт модели?

Сравниваем `Qwen2.5-3B (base)` с её SFT-версией. Отдельно для каждого трека.

Метрики:

- summarization — `rougeL_f1` после симметричной обрезки + judge winrate.
- code — `test_pass_rate`.
- dialog — judge winrate (heuristic выбрасываем либо оставляем как sanity-check).

Гипотеза. SFT с alpaca-данными улучшает summarization и dialog: формат «текст → текст» близок к alpaca. На code улучшения по читаемости не ждём — alpaca-output для кода это короткие текстовые объяснения, а не сам код. На MBPP-данных SFT даст рост `test_pass_rate`.

Если SFT на summarization после симметричной обрезки не отличается от base, ответ на RQ1 для этого трека частично отрицательный — это валидный наблюдательный результат.

---

## RQ2. При каких условиях DPO даёт прирост над SFT?

Сравниваем SFT-baseline с разными вариантами DPO: sigmoid и IPO, β ∈ {0.1, 0.3}, baseline-pairs и on-policy.

Метрики:

- judge winrate относительно SFT.
- DPO-trainer signals: `rewards/margins`, `logps/rejected`, `kl`. Они показывают, не «улетела» ли модель.

Гипотеза:

1. DPO с `chosen=alpaca-gold`, `rejected=base-output` коллапсирует за десятки шагов. Даже при β=0.3. Пары слишком далёкие.
2. IPO loss смягчает коллапс. Теоретически он менее склонен к overfitting на лёгких парах.
3. On-policy DPO — `chosen` и `rejected` оба от SFT-модели, разные температуры — наиболее устойчив. Только он даёт стат. значимый плюс над SFT по judge winrate.

Эти три подгипотезы — основной методологический фокус серии.

Независимо от знака эффекта фиксируем:

- шаг, на котором `loss` обнуляется;
- шаг, на котором `logps/rejected` падает ниже −X (X — из smoke-прогона);
- расхождение AB↔BA в judge (position-bias rate).

---

## RQ3. Насколько результаты чувствительны к препроцессингу метрик?

Берём один и тот же compare-файл и считаем метрики двумя способами:

- RAW — ответы как есть из `compare_models.py`.
- CLEAN — после `truncate_at_instruction(...)` симметрично для base/sft/dpo.

Метрика — `rougeL_f1` (summarization), длина ответа в токенах, judge winrate.

Гипотеза. На summarization разница RAW↔CLEAN для base превышает разницу SFT↔base по `rougeL_f1`. Значит, выводы про «эффект SFT» меняют знак или сильно меняют величину в зависимости от препроцессинга.

Цифры из `AUDIT_FINDINGS.md` это уже подтверждают: base RAW=0.188, base CLEAN=0.294. Нужно повторить на финальном прогоне и оформить таблицей.

Это самостоятельный методологический результат: без обрезки хвостов ROUGE на не-инструкт base даёт систематически заниженную оценку.

---

## RQ4. Согласуется ли LLM-judge с автоматическими метриками?

Считаем corr(`rougeL_f1`, judge winrate) и corr(`pass_rate`, judge winrate). Per-prompt и per-model.

Гипотеза:

- На summarization judge и `rougeL` коррелируют слабо или умеренно. Judge ценит fluency, rouge — n-gram overlap.
- На code judge не согласуется с `pass_rate`. Уже видно, что judge оценивает читаемость, а тесты — корректность. Это конкретный эмпирический результат: «judge ≠ корректность».

---

## Связь RQ → артефакты

| RQ | Где смотреть |
|---|---|
| RQ1 | `results/aggregate.{csv,md}`, главные конфиги `configs/{summarization,dialog,code}.json` |
| RQ2 | Apples-to-apples ablation `configs/sum_baseline_pairs*.json` vs on-policy главный прогон; β-sweep `configs/sum_beta_*.json`; sigmoid vs IPO `configs/sum_sigmoid.json`; графики `figures/fig_dpo_dynamics.pdf`, `fig_sigmoid_vs_ipo.pdf`, `fig_beta_curve_summarization.pdf` |
| RQ3 | `results/raw_vs_clean.{csv,md}`, `figures/fig_raw_vs_clean.pdf` |
| RQ4 | Поле `llm_judge` в `metrics_beta_*.json`, `figures/fig_judge_winrates.pdf` |
