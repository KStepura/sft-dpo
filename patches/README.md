# patches/

Готовые `git diff`-ы для шести фиксов из [`../docs/FIXES.md`](../docs/FIXES.md). Они уже применены к актуальному коду в `scripts/`. Лежат отдельно как документация изменений и для приложения «правки в кодовой базе» в дипломе.

- `run_experiment.diff` — пробрасывание новых полей конфига; шаг `select_best_dpo_checkpoint`; `build_dpo_pairs` теперь идёт после SFT для on-policy.
- `build_dpo_pairs.diff` — флаг `--policy_source {baseline, onpolicy, mixed}`; генерация chosen/rejected от SFT-policy; симметричная обрезка.
- `train_dpo.diff` — `loss_type`, `save_steps`, `save_total_limit` через CLI.
- `eval_metrics.diff` — alignment по нормализованному prompt; обновлённая `dialog_heuristic`.
- `run_track_code_experiment.diff` — те же фиксы для MBPP-пайплайна.

Применить отдельный фикс к чистому коду:

```bash
git apply patches/<name>.diff
```
