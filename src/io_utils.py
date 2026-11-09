"""Чтение/запись таблиц, сохранение фигур и текстовых отчётов.

Датасет лабораторной №3 устроен так же, как в лабораторных №1-2:
ДВУХУРОВНЕВЫЙ заголовок Excel. В первой строке — объединённые ячейки с
названием варианта («Вариант 7»), во второй — имена признаков (a, b, c, d, e, y).
Поэтому `header=[0, 1]`, а `pandas.read_excel(path)` без указания header читает
только первую строку и получает столбцы «Unnamed: 1» … «Unnamed: 143».
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import OUTPUT_DIR, timestamped_filename


# ----------------------------------------------------------------------
# Чтение датасета
# ----------------------------------------------------------------------
def read_workbook(path: str) -> pd.DataFrame:
    """Читает лист вариантов с двухуровневым заголовком и «расплющивает» его.

    Возвращает DataFrame с однозначными именами столбцов вида
    «Вариант 7 / a», «Вариант 7 / y».
    """
    p = Path(path)
    if p.suffix.lower() != ".xlsx":
        raise ValueError(f"Ожидается .xlsx, получено: {p.suffix}")

    raw = pd.read_excel(p, header=[0, 1])
    if not isinstance(raw.columns, pd.MultiIndex):
        top = pd.Series([str(c) for c in raw.columns])
        bottom = pd.Series([""] * len(raw.columns))
    else:
        top = pd.Series([str(a) for a, _ in raw.columns])
        bottom = pd.Series([str(b) for _, b in raw.columns])

    # Объединённые ячейки Excel при чтении дают NaN в первой строке заголовка:
    # они относятся к тому же варианту, что и соседний заполненный столбец.
    top = top.ffill()
    raw.columns = [f"{a} / {b}".strip() for a, b in zip(top, bottom)]
    return raw


def extract_variant(df: pd.DataFrame, variant: int) -> pd.DataFrame:
    """Оставляет только столбцы одного варианта, имена становятся a, b, c, d, e, y."""
    prefix = f"Вариант {variant}"
    cols = [c for c in df.columns if c.startswith(prefix + " /") or c == prefix]
    if not cols:
        available = sorted({c.split(" /")[0] for c in df.columns if " /" in c})
        raise KeyError(f"Вариант {variant} не найден. Доступны: {available}")

    out = df[cols].copy()
    out.columns = [c.split(" / ", 1)[-1].strip() or prefix for c in cols]
    out.columns = [f"y" if c in ("y", prefix) else c for c in out.columns]
    return out


def read_variant(path: str, variant: int) -> pd.DataFrame:
    """Полный путь: файл -> нужный вариант -> плоские имена столбцов."""
    return extract_variant(read_workbook(path), variant)


def available_variants(path: str) -> list[int]:
    """Возвращает номера вариантов, найденных в файле."""
    df = read_workbook(path)
    found = set()
    for col in df.columns:
        if " /" in col:
            head = col.split(" /", 1)[0].strip()
            if head.startswith("Вариант "):
                try:
                    found.add(int(head.split()[-1]))
                except ValueError:
                    pass
    return sorted(found)


# ----------------------------------------------------------------------
# Сохранение результатов
# ----------------------------------------------------------------------
def save_figure(fig, base_name: str, close: bool = True) -> str:
    """Сохраняет matplotlib-фигуру в output/ и возвращает путь к файлу."""
    path = timestamped_filename(base_name, "png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if close:
        import matplotlib.pyplot as plt
        plt.close(fig)
    return path


def write_table(df: pd.DataFrame, base_name: str, index: bool = False) -> dict:
    """Сохраняет DataFrame в csv / xlsx / txt с timestamp в имени файла."""
    paths = {}
    paths["csv"] = timestamped_filename(base_name, "csv")
    df.to_csv(paths["csv"], index=index, encoding="utf-8-sig")
    paths["xlsx"] = timestamped_filename(base_name, "xlsx")
    with pd.ExcelWriter(paths["xlsx"], engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="results", index=index)
    paths["txt"] = timestamped_filename(base_name, "txt")
    with open(paths["txt"], "w", encoding="utf-8") as fh:
        fh.write(df.to_string(index=index))
    return paths


def write_excel_sheets(base_name: str, sheets: dict[str, pd.DataFrame]) -> str:
    """Сохраняет несколько DataFrame в один xlsx (каждый на своём листе)."""
    path = timestamped_filename(base_name, "xlsx")
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            safe = str(name)[:31]
            frame.to_excel(writer, sheet_name=safe, index=False)
    return path


def write_fixed_table(df: pd.DataFrame, filename: str,
                      sheet_name: str = "results") -> str:
    """Сохраняет таблицу под ФИКСИРОВАННЫМ именем в output/ (без timestamp).

    Нужно для файлов, которые методичка называет явно:
    results_Comparison.xlsx.
    """
    path = os.path.join(OUTPUT_DIR, filename)
    if filename.lower().endswith(".xlsx"):
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    elif filename.lower().endswith(".csv"):
        df.to_csv(path, index=False, encoding="utf-8-sig")
    else:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(df.to_string(index=False))
    return path


def save_text(text: str, filename: str) -> str:
    """Сохраняет текст под фиксированным именем в output/."""
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path


def output_files() -> tuple[int, int, int]:
    """Возвращает (число png, число xlsx, число прочих файлов) в output/."""
    if not os.path.isdir(OUTPUT_DIR):
        return 0, 0, 0
    files = os.listdir(OUTPUT_DIR)
    n_png = len([f for f in files if f.endswith(".png")])
    n_xlsx = len([f for f in files if f.endswith(".xlsx")])
    n_other = len(files) - n_png - n_xlsx
    return n_png, n_xlsx, n_other


def correlation_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Матрица корреляций Пирсона (короткая обёртка для единообразия)."""
    return frame.corr()


def corr_safe(a: np.ndarray, b: np.ndarray) -> float:
    """Коэффициент корреляции Пирсона с защитой от нулевой дисперсии."""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])
