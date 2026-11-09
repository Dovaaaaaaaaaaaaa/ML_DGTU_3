"""Конфигурация проекта: seed, пути, номер варианта, доли разбиения, вывод.

Структура выводов повторяет лабораторные №1 и №2: каждый пункт методички
печатается одной строкой
    [Задание 1.3] Матрица взаимных корреляций ... OK (6x6, max|r| = 0.997)
"""

import os
import random
import sys
from datetime import datetime

import numpy as np

# Выставляется ДО первого импорта matplotlib: иначе на машине с GUI каждый
# plt.show() открывает отдельное окно и скрипт «зависает» до ручного закрытия.
os.environ.setdefault("MPLBACKEND", "Agg")

# sklearn/numpy/torch по умолчанию распараллеливают вычисления на все ядра, что
# на ноутбуке даёт скачки загрузки. setdefault — значения можно переопределить
# вручную перед запуском.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

SEED = 42

# Номер варианта. Методичка: «Датасет включает в себя набор из 5 признаков
# (по вариантам) и столбец значений целевой переменной y». Вариант меняется
# флагом --variant, значение здесь — по умолчанию (вариант 7, как в лабах 1-2).
VARIANT = 7

DATA_DIR = "data"
OUTPUT_DIR = "output"

DATASET_PATH = os.path.join(DATA_DIR, "Lab3_Dataset.xlsx")

# Разбиение временного ряда. Методичка требует train >= 50 %; выбрано 70/15/15:
# train достаточно большой для обучения нейросетей, val — для подбора
# гиперпараметров и ранней остановки, test — для итогового сравнения.
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15

# Горизонт прогноза для нейросетей (п. 4): 1 шаг и 7 шагов вперёд.
HORIZONS = (1, 7)

os.makedirs(OUTPUT_DIR, exist_ok=True)


def set_seeds(seed: int = SEED) -> None:
    """Фиксирует seed для random, numpy и torch (если он установлен)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def timestamped_filename(base: str, ext: str) -> str:
    """output/{base}_{ГГГГ-ММ-ДД_ЧЧ-ММ-СС}.{ext} — каждый прогон создаёт новый
    файл и не перетирает результаты прошлых запусков."""
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return os.path.join(OUTPUT_DIR, f"{base}_{ts}.{ext}")


# ----------------------------------------------------------------------
# Примитивы вывода
# ----------------------------------------------------------------------
def banner(text: str) -> None:
    print("\n" + "=" * 78 + f"\n{text}\n" + "=" * 78, flush=True)


def run_step(number: str, title: str) -> None:
    """Печатает начало пункта. number — строка вида '1.1' или '4.2'."""
    print(f"[Задание {number:>3}] {title} ... ", end="", flush=True)


def step_done(info: str = "") -> None:
    print("OK" + (f" ({info})" if info else ""), flush=True)


def step_skipped(reason: str = "") -> None:
    print("ПРОПУЩЕНО" + (f" ({reason})" if reason else ""), flush=True)


def sub_step(title: str) -> None:
    """Подзаголовок внутри пункта (например, перебор гиперпараметров)."""
    print(f"      -- {title}", flush=True)


def force_utf8_stdio() -> None:
    """Переводит stdout/stderr в UTF-8.

    В консоли Windows по умолчанию cp1251, и печать символов Δ, ≈, ✓ роняет
    скрипт UnicodeEncodeError. Вызывается в самом начале main().
    """
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
