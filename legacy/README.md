# legacy/

Старые конфиги, скрипты и результаты ранней серии. Хранятся отдельно, чтобы не мешать актуальной работе, но не удалены — нужны для сравнения «было / стало» и для рисунка `figures/fig_dpo_dynamics.pdf`, где ранний коллапс DPO показан красной линией.

- [`runs/`](runs/) — старые shell-обёртки: `run_dpo.sh`, `run_sft.sh`, `run_gpu_diploma*.sh`, `run_gpu_smoke.sh`, `run_overnight_chain_v1.sh`. Заменены файлами в [`runs/`](../runs/) корня.
- [`configs/`](configs/) — конфиги, не дошедшие до публикации: версии с judge на этапе обучения, CPU-smoke, pilot, упавший `dialog_seed1337_failed.json` (HF-DNS-сбой 9 мая).
- [`docs/`](docs/) — старые черновики (`PLAN.md`, `REPORTING.md`, `EXPERIMENTS_OUTLINE.md` и т.п.); `*.md` здесь в `.gitignore`, актуально — [`docs/`](../docs/) в корне.
- [`results/`](results/) — старые pilot/smoke метрики, judge-результаты на отдельных compare-файлах, лог `three_tracks_master.*`.

Финальные числа и фигуры — в [`results/`](../results/) и [`figures/`](../figures/) корня. Обзор методологических ошибок ранней серии — в [`docs/AUDIT_FINDINGS.md`](../docs/AUDIT_FINDINGS.md) и [`docs/FIXES.md`](../docs/FIXES.md).
