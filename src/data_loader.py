"""Загрузка датасета, разбиение временного ряда и подготовка окон прогноза.

Методичка (введение): «для временного ряда с фиксированным шагом будущее
значение может быть получено путём обучения модели со сдвигом на 1 шаг, так
чтобы входному тензору Xt соответствовал выходной тензор Yt+1». Поэтому во
ВСЕХ разделах (1-3 и 4) используется один и тот же принцип:

    X[t] = (a, b, c, d, e) в момент t
    Y[t] = y в момент t + 1                    (одношаговый прогноз)
    Y[t] = (y[t+1], ..., y[t+7])               (семишаговый прогноз)

Разбиение train/val/test выполняется ПО ВРЕМЕНИ и ДО построения окон: иначе
целевые значения валидации могли бы попасть в обучающие окна (утечка).
Масштабирование настраивается ТОЛЬКО на train и затем применяется к val/test.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             r2_score)
from sklearn.preprocessing import StandardScaler

from src.config import TRAIN_FRAC, VAL_FRAC
from src.io_utils import corr_safe, read_variant


@dataclass
class Dataset:
    """Числовой временной ряд одного варианта."""

    name: str
    X: np.ndarray                      # (n, p) признаки
    y: np.ndarray                      # (n,) целевая переменная
    feature_names: list
    target_name: str = "y"
    report: dict = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        return self.X.shape[0]

    @property
    def n_features(self) -> int:
        return self.X.shape[1]


def load_dataset(path: str, variant: int, name: str = "Regression",
                 feature_names: list | None = None) -> Dataset:
    """Читает вариант из xlsx, чистит и возвращает Dataset.

    Порядок строк сохраняется: это временной ряд, перемешивать нельзя.
    Строки с пропуском в любом признаке или в y удаляются целиком.
    """
    raw = read_variant(path, variant)
    report = {
        "файл": path,
        "вариант": variant,
        "исходная форма": raw.shape,
        "пропуски": raw.isna().sum().to_dict(),
    }

    df = raw.copy()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    report["нечисловых ячеек"] = int(
        df.isna().sum().sum() - raw.isna().sum().sum())

    if "y" not in df.columns:
        raise KeyError(f"В варианте {variant} нет столбца y: {list(df.columns)}")

    feature_names = feature_names or [c for c in df.columns if c != "y"]
    missing = [c for c in feature_names + ["y"] if c not in df.columns]
    if missing:
        raise KeyError(f"Нет столбцов: {missing}")

    before = len(df)
    df = df.dropna(subset=feature_names + ["y"]).reset_index(drop=True)
    report["строк до/после очистки"] = (before, len(df))
    report["удалено строк"] = before - len(df)
    if df.empty:
        raise ValueError(f"После очистки не осталось ни одной строки "
                         f"(вариант {variant})")

    X = df[feature_names].to_numpy(dtype=np.float64)
    y = df["y"].to_numpy(dtype=np.float64)
    report["признаки"] = feature_names
    report["диапазон y"] = (float(y.min()), float(y.max()))
    report["пропусков осталось"] = int(np.isnan(X).sum() + np.isnan(y).sum())

    return Dataset(name=name, X=X, y=y,
                   feature_names=list(feature_names),
                   target_name="y", report=report)


def print_report(ds: Dataset) -> None:
    """Печатает отчёт о загрузке датасета."""
    r = ds.report
    print(f"      файл            : {r['файл']} (вариант {r['вариант']})")
    print(f"      в файле         : {r['исходная форма'][0]} строк x "
          f"{r['исходная форма'][1]} столбцов (все варианты сразу)")
    print(f"      строк отброшено : {r['удалено строк']} "
          f"({r['строк до/после очистки'][0]} -> "
          f"{r['строк до/после очистки'][1]}), осталось {ds.n_samples}")
    print(f"      признаки        : {', '.join(ds.feature_names)}")
    print(f"      цель y          : диапазон "
          f"[{r['диапазон y'][0]:.3f}; {r['диапазон y'][1]:.3f}]")
    print("      порядок строк   : сохранён (временной ряд, без перемешивания)")


# ----------------------------------------------------------------------
# Разбиение и окна прогноза
# ----------------------------------------------------------------------
def split_indices(n: int, train_frac: float = TRAIN_FRAC,
                  val_frac: float = VAL_FRAC) -> dict[str, slice]:
    """Хронологическое разбиение индексов 0..n-1 на train/val/test.

    Возвращает словарь срезов. Гарантирует непустые части даже на коротком
    ряду (для датасета из 1000 строк доли 70/15/15 дают 700/150/150).
    """
    n_train = max(1, int(round(n * train_frac)))
    n_val = max(1, int(round(n * val_frac)))
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = max(1, n - n_train - 1)
    return {
        "train": slice(0, n_train),
        "val": slice(n_train, n_train + n_val),
        "test": slice(n_train + n_val, n),
    }


def make_horizon_windows(X: np.ndarray, y: np.ndarray,
                         horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """Строит пары (X_t, Y_{t+1..t+horizon}) внутри одного сегмента.

    Возвращает Xw формы (m, p) и Yw формы (m, horizon). Для horizon=1 —
    обычный сдвиг на один шаг; для horizon=7 — матрица из семи будущих
    значений y. Форма Y всегда двумерная, чтобы код моделей был единым.
    """
    horizon = int(horizon)
    if horizon < 1:
        raise ValueError("horizon должен быть >= 1")
    m = len(y) - horizon
    if m <= 0:
        raise ValueError(f"Ряд слишком короткий для горизонта {horizon}")
    Xw = X[:m]
    # vstack (а не column_stack): каждый элемент — строка длины horizon,
    # итоговая форма (m, horizon). Для horizon=1 получается (m, 1).
    Yw = np.vstack([y[k + 1:k + 1 + horizon] for k in range(m)])
    return Xw, Yw


def prepare_dataset(ds: Dataset, horizon: int = 1,
                    train_frac: float = TRAIN_FRAC,
                    val_frac: float = VAL_FRAC) -> dict:
    """Готовит train/val/test с окнами прогноза и масштабированием.

    Возвращает словарь:
        segments[name]['X_raw'], ['Y_raw']  — окна до масштабирования;
        segments[name]['X'],     ['Y']      — после StandardScaler;
        scaler_X, scaler_Y                   — обучены ТОЛЬКО на train;
        indices                              — срезы исходного ряда.
    """
    idx = split_indices(len(ds.y), train_frac, val_frac)
    segments: dict[str, dict] = {}
    for name, sl in idx.items():
        Xw, Yw = make_horizon_windows(ds.X[sl], ds.y[sl], horizon)
        segments[name] = {"X_raw": Xw, "Y_raw": Yw}

    scaler_X = StandardScaler().fit(segments["train"]["X_raw"])
    scaler_Y = StandardScaler().fit(segments["train"]["Y_raw"])
    for name in segments:
        segments[name]["X"] = scaler_X.transform(segments[name]["X_raw"])
        segments[name]["Y"] = scaler_Y.transform(segments[name]["Y_raw"])

    return {
        "horizon": int(horizon),
        "indices": idx,
        "segments": segments,
        "scaler_X": scaler_X,
        "scaler_Y": scaler_Y,
        "feature_names": list(ds.feature_names),
        "target_name": ds.target_name,
    }


def inverse_y(prep: dict, values: np.ndarray) -> np.ndarray:
    """Возвращает прогноз из масштабированных единиц в исходные единицы y."""
    return prep["scaler_Y"].inverse_transform(np.asarray(values, dtype=float))


def segment_sizes(prep: dict) -> dict[str, int]:
    """Число окон в каждом сегменте."""
    return {name: len(seg["Y_raw"]) for name, seg in prep["segments"].items()}


def print_preparation(prep: dict) -> None:
    """Печатает параметры разбиения и масштабирования."""
    sizes = segment_sizes(prep)
    print(f"      горизонт прогноза : {prep['horizon']} шаг(ов)")
    print(f"      окна train/val/test: {sizes['train']} / {sizes['val']} / "
          f"{sizes['test']}")
    print(f"      train доля        : "
          f"{sizes['train'] / sum(sizes.values()) * 100:.1f} %")
    print(f"      масштабирование  : StandardScaler, обучен только на train")
    print(f"      средние X (train): "
          f"{np.round(prep['scaler_X'].mean_, 3).tolist()}")
    print(f"      СКО X (train)    : "
          f"{np.round(prep['scaler_X'].scale_, 3).tolist()}")


# ----------------------------------------------------------------------
# Метрики
# ----------------------------------------------------------------------
def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """MAE, MSE, RMSE, R2 и коэффициент корреляции для любых форм массивов."""
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()
    mse = float(mean_squared_error(yt, yp))
    return {
        "n": int(yt.size),
        "MAE": float(mean_absolute_error(yt, yp)),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(yt, yp)),
        "corr": corr_safe(yt, yp),
    }


def metrics_by_horizon(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """Метрики по каждому шагу горизонта для матриц (n, horizon)."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if yt.ndim == 1:
        yt = yt.reshape(-1, 1)
        yp = yp.reshape(-1, 1)
    rows = []
    for h in range(yt.shape[1]):
        m = regression_metrics(yt[:, h], yp[:, h])
        m["горизонт"] = h + 1
        rows.append(m)
    return pd.DataFrame(rows)[["горизонт", "n", "MAE", "MSE", "RMSE", "R2",
                               "corr"]]


def mean_horizon_metric(table: pd.DataFrame) -> dict:
    """Средние MAE/MSE по всем шагам горизонта (для итоговой таблицы)."""
    return {
        "MAE": float(table["MAE"].mean()),
        "MSE": float(table["MSE"].mean()),
        "RMSE": float(np.sqrt(table["MSE"].mean())),
        "R2": float(table["R2"].mean()),
        "corr": float(table["corr"].mean()),
    }


def print_metrics(prefix: str, m: dict) -> None:
    """Печатает одну строку метрик."""
    print(f"      {prefix:<18} MAE = {m['MAE']:.4f}   MSE = {m['MSE']:.4f}   "
          f"RMSE = {m['RMSE']:.4f}   R2 = {m['R2']:.4f}   "
          f"corr = {m['corr']:.4f}")
