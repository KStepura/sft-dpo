# Окружение и кэш

## 1. Первичная установка

```bash
bash runs/setup_env.sh
```

Скрипт создаёт `.venv`, ставит зависимости из `requirements.txt` и регистрирует Jupyter kernel `Python (thesis-venv)`. Корень проекта определяется через `BASH_SOURCE`, поэтому скрипт работает из любой директории.

## 2. Подключение в новой сессии

```bash
source env.sh
```

`env.sh` активирует `.venv`, выставляет переменные путей и кэша HuggingFace, и переключает текущую директорию в корень проекта.

## 3. Быстрая проверка окружения

```bash
source env.sh
.venv/bin/python -m pip --version
.venv/bin/python -c "import sys; print(sys.executable)"
.venv/bin/python -c "import torch; print(torch.__version__)"
```

## 4. Проверка ядра Jupyter

В ноутбуках выбрать kernel `Python (thesis-venv)`.

## 5. Если venv не создаётся (ensurepip / python3-venv)

```bash
sudo apt-get update
sudo apt-get install -y python3-venv
rm -rf .venv
bash runs/setup_env.sh
source env.sh
```

## 6. Фиксация версий пакетов (опционально)

Сохранить:

```bash
source env.sh
pip freeze > requirements.lock.txt
```

Восстановить:

```bash
source env.sh
pip install -r requirements.lock.txt
```

## 7. Дальше

GPU-запуски и состав конфигов — [`GPU.md`](GPU.md).
