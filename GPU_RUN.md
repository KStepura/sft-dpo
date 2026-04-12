# Запуск экспериментов на GPU

Краткая инструкция: что сделать **один раз** на машине с GPU, затем **в каком порядке** гонять конфиги и где смотреть результаты.

## CPU RunPod сейчас, GPU потом — так нормально

**Да, это разумный и частый сценарий:** на CPU-поде дописываешь код, гоняешь `smoke_cpu` / ноутбук, коммитишь в git; на GPU-поде — только проверка CUDA и длинные прогоны (`run_gpu_smoke.sh` → `run_gpu_diploma*.sh`).

**Проще всего по переносу кода и данных:**

| Способ | Смысл |
|--------|--------|
| **Один persistent volume** | Тот же диск подключаешь к новому GPU-поду: репозиторий и кэш HF уже на месте. Останется на GPU-поде один раз проверить **torch с CUDA** (см. §0 ниже): на CPU-поде часто стоит CPU-сборка torch — переустанови wheel с CUDA или пересоздай `.venv` только на GPU. |
| **Только git** | Код в GitHub/GitLab: на GPU-поде `git clone` / `git pull`, `bash runs/setup_env.sh` — без переноса volume между типами подов. Кэш моделей скачается заново (или укажи тот же network volume только под `HF_HOME`). |

Выключить CPU-под и открыть **тот же volume** на GPU-поде — нормальная схема RunPod, если volume изначально network/persistent. Проще альтернативы по сути нет: либо общий volume, либо git + при необходимости общий кэш HF на volume.

## 0. Один раз: окружение

Из корня репозитория (у вас путь может отличаться от `/workspace/thesis-llm-alignment` — тогда поправьте `env.sh` или экспортируйте переменные кэша сами, см. [ENVIRONMENT_REUSE.md](ENVIRONMENT_REUSE.md)):

```bash
cd /path/to/thesis-llm-alignment
bash runs/setup_env.sh
```

Проверка, что в venv есть **PyTorch с CUDA** (иначе GPU не задействуется):

```bash
source env.sh
python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.version.cuda)"
```

Если `cuda: False`, переустановите torch под вашу версию CUDA (пример для CUDA 12.4):

```bash
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Подставьте нужный индекс с [pytorch.org](https://pytorch.org/get-started/locally/).

Опционально для приватных/гейтед моделей:

```bash
export HF_TOKEN=...   # huggingface-cli login — альтернатива
```

Модель `Qwen/Qwen2.5-3B-Instruct` обычно доступна без токена.

---

## 1. Перед каждым сеансом

```bash
cd /path/to/thesis-llm-alignment
source env.sh
bash runs/preflight_gpu.sh
```

Скрипт проверит `torch.cuda`, выведет `nvidia-smi` и свободное место на диске (кэш и веса занимают **много** гигабайт; ориентир **≥ 30–50 ГБ** свободного места под полный прогон сетки).

---

## 2. Рекомендуемый порядок прогонов

| Шаг | Команда | Назначение |
|-----|---------|------------|
| A (обязательно первый на GPU) | `bash runs/run_gpu_smoke.sh` | Быстрая проверка всего пайплайна (~минуты + загрузка весов). |
| B (опционально, быстрее полной сетки) | `bash runs/run_gpu_diploma_light.sh` | Укороченная сетка β и меньше примеров — для отладки отчёта. |
| C (основной для диплома) | `bash runs/run_gpu_diploma.sh` | Полный конфиг `diploma_grid_v1.json` (дольше по времени). |

Либо вручную:

```bash
source env.sh
python scripts/run_experiment.py --config configs/smoke_gpu.json
python scripts/run_experiment.py --config configs/diploma_grid_light.json
python scripts/run_experiment.py --config configs/diploma_grid_v1.json
```

Свой идентификатор эксперимента (имя папки в `results/experiments/`):

```bash
python scripts/run_experiment.py --config configs/diploma_grid_v1.json --exp_id my_run_2026_04_11
```

---

## 3. Что появится после прогона

Для каждого запуска создаётся каталог:

`results/experiments/<exp_id>/`

- `resolved_config.json` — фактически использованный конфиг  
- `summary.json` — шаги, время, пути к compare, краткая статистика длин ответов  
- `logs/*.log` — полный stdout/stderr по шагам  
- `compare/compare_beta_<slug>.jsonl` — ответы base / SFT / DPO по промптам из конфига  

Адаптеры:

`outputs/experiments/<exp_id>/sft`  
`outputs/experiments/<exp_id>/dpo_beta_<slug>/`

Сырые и очищенные пары лежат в `data/` по путям из секции `paths` в JSON (например `data/dpo_pairs.grid_v1.raw.jsonl`).

---

## 4. Сводная таблица по всем прогонам

```bash
source env.sh
python scripts/aggregate_results.py
```

Файл: `results/experiments/summary_table.csv` (можно открыть в Excel / LibreOffice / pandas).

---

## 5. Оценка времени и VRAM (очень грубо)

Зависит от GPU, диска и сети. Ориентир для **Qwen2.5-3B-Instruct** в 4-bit как в скриптах:

- **smoke_gpu**: порядка десятков минут после кэширования модели.  
- **diploma_grid_light**: заметно короче полной сетки.  
- **diploma_grid_v1**: **часы** (SFT + 4× DPO + сравнения + генерация 800 пар в начале); удобно запускать в `tmux`/`screen` и писать лог в файл:

```bash
source env.sh
mkdir -p logs
python scripts/run_experiment.py --config configs/diploma_grid_v1.json 2>&1 | tee logs/diploma_grid_v1_$(date +%Y%m%d_%H%M%S).log
```

VRAM: 4-bit + LoRA обычно укладывается в **одну** потребительскую карту 16–24 ГБ; при OOM уменьшите в JSON `max_seq_len` / `max_length` / `max_prompt_length` или `num_examples`.

---

## 6. Ноутбуки

Тот же пайплайн по шагам: `smoke_pipeline_run.ipynb` (короткий), `full_pipeline_run.ipynb` (полный). Ядро: **Python (thesis-venv)** после `setup_env.sh`.

---

## 7. Если что-то упало

1. Откройте соответствующий `results/experiments/<exp_id>/logs/*.log`.  
2. Частые причины: нет места на диске, нет CUDA в torch, обрыв сети при скачивании HF — повторите шаг после `huggingface-cli download` или с уже заполненным кэшем.  
3. Пары DPO: если после `clean` осталось мало строк — уменьшите `min_chars`, ослабьте `drop_if_truncated` в конфиге или увеличьте `num_examples` в `build_dpo_pairs`.

После правок конфига удобно копировать файл, например `configs/my_grid.json`, и вызывать `run_experiment.py --config configs/my_grid.json`.
