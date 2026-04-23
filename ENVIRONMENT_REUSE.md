# Reuse Environment

## 1) Первичная установка

```bash
cd /workspace/thesis-llm-alignment
bash runs/setup_env.sh
```

Что создается:
- `.venv`
- зависимости из `requirements.txt`
- Jupyter kernel `Python (thesis-venv)`

## 2) Подключение окружения в новой сессии

```bash
source /workspace/thesis-llm-alignment/env.sh
```

После выполнения:
- активируется `.venv`
- выставляются переменные путей и кэша
- рабочая директория переключается в корень проекта

## 3) Быстрая проверка окружения

```bash
source /workspace/thesis-llm-alignment/env.sh
python -m pip --version
python -c "import sys; print(sys.executable)"
python -c "import torch; print(torch.__version__)"
```

## 4) Проверка ядра Jupyter

В ноутбуках выбрать kernel:
- `Python (thesis-venv)`

## 5) Если не создается venv (ensurepip / python3-venv)

```bash
apt-get update
apt-get install -y python3-venv
rm -rf /workspace/thesis-llm-alignment/.venv
bash /workspace/thesis-llm-alignment/runs/setup_env.sh
source /workspace/thesis-llm-alignment/env.sh
```

## 6) Фиксация версий пакетов (опционально)

Сохранить:

```bash
source /workspace/thesis-llm-alignment/env.sh
pip freeze > requirements.lock.txt
```

Восстановить:

```bash
source /workspace/thesis-llm-alignment/env.sh
pip install -r requirements.lock.txt
```

## 7) Дальше

Для GPU-запусков использовать инструкцию:
- `GPU_RUN.md`
