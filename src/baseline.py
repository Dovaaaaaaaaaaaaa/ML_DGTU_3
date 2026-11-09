"""Пункт 1 методички: предварительная обработка и регрессия базовыми методами.

    1.1  все признаки и целевая переменная на едином графике
    1.2  разбиение train/val/test и масштабирование (параметры только с train)
    1.3  матрица взаимных корреляций признаков и цели
    1.4  парная линейная регрессия по самому коррелированному признаку
    1.5  множественная линейная регрессия по всем признакам
    1.6  нелинейная регрессия через scipy.optimize.curve_fit() по всем признакам

Все модели обучаются на горизонте 1 шаг: X[t] -> y[t+1] (см. data_loader).
"""

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression

from src.config import run_step, step_done, sub_step
from src.data_loader import (inverse_y, print_metrics, regression_metrics,
                             segment_sizes)
from src.io_utils import corr_safe, save_figure

REG_FEATURE_COLOR = "#1565C0"
REG_TARGET_COLOR = "#C62828"
REG_FIT_COLOR = "#2E7D32"
REG_PRED_COLOR = "#EF6C00"


# ----------------------------------------------------------------------
# Вспомогательные функции
# ----------------------------------------------------------------------
def _as_1d(a: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=float).ravel()


def _eval_predictions(prep: dict, predict) -> tuple[dict, dict]:
    """Считает прогнозы и метрики на train/val/test.

    predict(segment_name) -> массив прогноза в исходных единицах y.
    """
    preds, metrics = {}, {}
    for name, seg in prep["segments"].items():
        yp = _as_1d(predict(name))
        preds[name] = yp
        metrics[name] = regression_metrics(seg["Y_raw"], yp)
    return preds, metrics


def _print_model_metrics(title: str, metrics: dict) -> None:
    print(f"      {title}:")
    for name in ("train", "val", "test"):
        print_metrics(name, metrics[name])


def _fit_simple_linear(prep: dict, feature_index: int):
    """Парная линейная регрессия по одному (масштабированному) признаку."""
    tr = prep["segments"]["train"]
    model = LinearRegression().fit(tr["X"][:, [feature_index]],
                                   tr["Y_raw"].ravel())
    return model


def _predict_simple(prep: dict, model: LinearRegression, feature_index: int) -> dict:
    return {name: model.predict(seg["X"][:, [feature_index]]).ravel()
            for name, seg in prep["segments"].items()}


def _fit_multiple(prep: dict) -> LinearRegression:
    tr = prep["segments"]["train"]
    return LinearRegression().fit(tr["X"], tr["Y_raw"].ravel())


def _predict_multiple(prep: dict, model: LinearRegression) -> dict:
    return {name: model.predict(seg["X"]).ravel()
            for name, seg in prep["segments"].items()}


# ----------------------------------------------------------------------
# Функции для curve_fit (п. 1.6)
# ----------------------------------------------------------------------
def f_quadratic(X, *p):
    """Полный квадратичный полином по всем пяти признакам.

    Содержит intercept, линейные члены, квадраты и все парные произведения
    (21 параметр). Выбран как основной кандидат: по графику связь y с
    признаками почти линейна, но с заметной кривизной.
    """
    a, b, c, d, e = X
    return (p[0]
            + p[1] * a + p[2] * b + p[3] * c + p[4] * d + p[5] * e
            + p[6] * a * a + p[7] * b * b + p[8] * c * c
            + p[9] * d * d + p[10] * e * e
            + p[11] * a * b + p[12] * a * c + p[13] * a * d
            + p[14] * a * e + p[15] * b * c + p[16] * b * d
            + p[17] * b * e + p[18] * c * d + p[19] * c * e
            + p[20] * d * e)


def f_exp_dominant(X, c0, c1, c2, c3, c4, c5, amp, rate):
    """Линейная часть + экспонента по самому сильному признаку a.

    Форма y = c0 + Σ ci*xi + amp*exp(rate*a) нелинейна по rate, поэтому
    параметры нельзя получить обычным МНК — используется curve_fit().
    """
    a, b, c, d, e = X
    expo = np.exp(np.clip(rate * a, -30.0, 30.0))
    return c0 + c1 * a + c2 * b + c3 * c + c4 * d + c5 * e + amp * expo


def _quadratic_design(X: np.ndarray) -> np.ndarray:
    """Матрица регрессоров полного квадратичного полинома (для p0)."""
    a, b, c, d, e = X.T
    return np.column_stack([
        np.ones(len(a)), a, b, c, d, e,
        a * a, b * b, c * c, d * d, e * e,
        a * b, a * c, a * d, a * e,
        b * c, b * d, b * e,
        c * d, c * e, d * e,
    ])


def _fit_nonlinear_candidates(prep: dict) -> tuple[dict, pd.DataFrame]:
    """Подбирает вид нелинейной функции по ошибке на валидации.

    Обе функции обучаются на train через curve_fit(), сравниваются по MSE на
    val в исходных единицах y. Возвращает лучший результат и таблицу сравнения.
    """
    tr = prep["segments"]["train"]
    val = prep["segments"]["val"]
    Xtr = tr["X"]
    ytr = tr["Y"].ravel()          # масштабированная цель
    ytr_raw = tr["Y_raw"].ravel()

    # --- стартовые приближения ---
    lin_design = np.column_stack([np.ones(len(Xtr)), Xtr])
    p0_lin = np.linalg.lstsq(lin_design, ytr, rcond=None)[0]
    p0_quad = np.linalg.lstsq(_quadratic_design(Xtr), ytr, rcond=None)[0]
    p0_exp = np.concatenate([p0_lin, [0.05, -0.5]])

    candidates = [
        ("квадратичный полином (21 параметр)", f_quadratic, p0_quad, None),
        ("линейная часть + экспонента (8 параметров)", f_exp_dominant, p0_exp,
         ([-np.inf] * 6 + [-10.0, -10.0], [np.inf] * 6 + [10.0, 10.0])),
    ]

    rows, fitted = [], {}
    for title, func, p0, bounds in candidates:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if bounds is None:
                    popt, _ = curve_fit(func, Xtr.T, ytr, p0=p0,
                                        maxfev=50000)
                else:
                    popt, _ = curve_fit(func, Xtr.T, ytr, p0=p0,
                                        bounds=bounds, maxfev=50000)
            train_pred = inverse_y(prep, func(Xtr.T, *popt)
                                   .reshape(-1, 1)).ravel()
            val_pred = inverse_y(prep, func(val["X"].T, *popt)
                                  .reshape(-1, 1)).ravel()
            m_tr = regression_metrics(ytr_raw, train_pred)
            m_val = regression_metrics(val["Y_raw"], val_pred)
            rows.append({"функция": title, "MSE train": m_tr["MSE"],
                         "MSE val": m_val["MSE"], "MAE val": m_val["MAE"],
                         "corr val": m_val["corr"], "параметров": len(popt)})
            fitted[title] = {"func": func, "popt": popt, "MSE_val": m_val["MSE"]}
        except Exception as exc:                        # noqa: BLE001
            rows.append({"функция": title, "MSE train": np.nan,
                         "MSE val": np.nan, "MAE val": np.nan,
                         "corr val": np.nan, "параметров": len(p0)})
            print(f"      {title}: не удалось обучить ({exc})")

    table = pd.DataFrame(rows).sort_values("MSE val", na_position="last")
    best_title = table.iloc[0]["функция"]
    if best_title not in fitted:
        raise RuntimeError("Ни одна нелинейная функция не обучилась через "
                           "curve_fit")
    return fitted[best_title] | {"title": best_title}, table


# ----------------------------------------------------------------------
# Пункт 1 целиком
# ----------------------------------------------------------------------
def run_baseline(ds, prep: dict, variant: int) -> dict:
    """Выполняет пункт 1 методички. Возвращает словарь с моделями и прогнозами."""
    results: dict = {"models": {}, "variant": variant,
                     "y_test": _as_1d(prep["segments"]["test"]["Y_raw"]),
                     "test_start": prep["indices"]["test"].start}

    # ================= 1.1 =================
    run_step("1.1", "Все признаки и целевая переменная на едином графике")
    frame = pd.DataFrame(ds.X, columns=ds.feature_names)
    frame["y"] = ds.y
    z = (frame - frame.mean()) / frame.std(ddof=0)

    fig, ax = plt.subplots(figsize=(14, 6))
    for col in ds.feature_names:
        ax.plot(z[col].to_numpy(), linewidth=1.0, alpha=0.85, label=col)
    ax.plot(z["y"].to_numpy(), color="black", linewidth=1.8, label="y")
    ax.set_xlabel("Номер наблюдения (время)")
    ax.set_ylabel("Значение (z-score)")
    ax.set_title(f"Все признаки и целевая переменная y (вариант {variant}); "
                 f"ряды нормированы для сопоставимости масштабов")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(ncol=6, fontsize=9, loc="upper center")
    fig.tight_layout()
    path_11 = save_figure(fig, "task1_1_all_series")

    corr_raw = frame.corr(numeric_only=True)
    results["series_corr"] = corr_raw
    print(f"      строк: {len(frame)}, признаки: {', '.join(ds.feature_names)}")
    print(f"      разброс y: {ds.y.min():.3f} .. {ds.y.max():.3f}; "
          f"разные признаки имеют разные масштабы, поэтому на графике "
          f"использованы z-оценки")
    step_done(os.path.basename(path_11))

    # ================= 1.2 =================
    run_step("1.2", "Разбиение train/val/test и масштабирование")
    sizes = segment_sizes(prep)
    print()
    print(f"      train / val / test = {sizes['train']} / {sizes['val']} / "
          f"{sizes['test']} окон (горизонт {prep['horizon']})")
    print(f"      пропорция          = 70 / 15 / 15 %, train >= 50 %")
    print(f"      scaler_X обучен на train: средние "
          f"{np.round(prep['scaler_X'].mean_, 3).tolist()}, СКО "
          f"{np.round(prep['scaler_X'].scale_, 3).tolist()}")
    print(f"      scaler_Y обучен на train: среднее "
          f"{_as_1d(prep['scaler_Y'].mean_)[0]:.3f}, СКО "
          f"{_as_1d(prep['scaler_Y'].scale_)[0]:.3f}")
    print("      параметры масштабирования train применяются к val и test "
          "(без переобучения)")

    idx = prep["indices"]
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.plot(np.arange(len(ds.y)), ds.y, color="black", linewidth=1.2)
    spans = (("train", "#2E7D32"), ("val", "#F9A825"), ("test", "#C62828"))
    for name, color in spans:
        sl = idx[name]
        ax.axvspan(sl.start, sl.stop - 1, color=color, alpha=0.10,
                   label=f"{name} ({sl.stop - sl.start} строк)")
    ax.set_xlabel("Номер наблюдения")
    ax.set_ylabel("y")
    ax.set_title(f"Хронологическое разбиение временного ряда (вариант {variant})")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(ncol=3)
    fig.tight_layout()
    path_12 = save_figure(fig, "task1_2_split")
    results["split_figure"] = path_12
    step_done(os.path.basename(path_12))

    # ================= 1.3 =================
    run_step("1.3", "Матрица взаимных корреляций признаков и цели")
    # Цель со сдвигом на 1 шаг — именно она прогнозируется моделями.
    X_shift, y_shift = ds.X[:-1], ds.y[1:]
    corr_frame = pd.DataFrame(X_shift, columns=ds.feature_names)
    corr_frame["y(t+1)"] = y_shift
    corr = corr_frame.corr()

    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)))
    ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.index)
    for i in range(len(corr)):
        for j in range(len(corr)):
            value = corr.iloc[i, j]
            ax.text(j, i, f"{value:.3f}", ha="center", va="center",
                    fontsize=8,
                    color="white" if abs(value) > 0.55 else "black")
    ax.set_title(f"Корреляции Пирсона, цель y(t+1) (вариант {variant})")
    fig.colorbar(im, ax=ax, shrink=0.8, label="r")
    fig.tight_layout()
    path_13 = save_figure(fig, "task1_3_correlation")

    target_corr = corr["y(t+1)"].drop("y(t+1)").abs().sort_values(ascending=False)
    best_feature = str(target_corr.index[0])
    best_index = ds.feature_names.index(best_feature)
    best_r = float(corr.loc[best_feature, "y(t+1)"])
    results["corr"] = corr
    results["best_feature"] = best_feature
    results["best_feature_index"] = best_index
    print()
    print(corr.round(3).to_string().replace("\n", "\n      "))
    print(f"      максимальная |r| с y(t+1): {abs(best_r):.4f} "
          f"у признака {best_feature} (r = {best_r:.4f})")
    step_done(f"{corr.shape[0]}x{corr.shape[1]}, выбран признак {best_feature}")

    # ================= 1.4 =================
    run_step("1.4", f"Парная линейная регрессия по признаку {best_feature}")
    simple = _fit_simple_linear(prep, best_index)
    preds = _predict_simple(prep, simple, best_index)
    metrics = {name: regression_metrics(prep["segments"][name]["Y_raw"], p)
               for name, p in preds.items()}
    _print_model_metrics("парная линейная", metrics)
    coef = float(simple.coef_.ravel()[0])
    intercept = float(simple.intercept_)
    print(f"      модель (масштабированные единицы): "
          f"y = {intercept:.4f} + {coef:.4f} * {best_feature}")
    results["models"]["Парная линейная регрессия"] = {
        "predictions": preds, "metrics": metrics, "family": "статистика",
        "params": f"{best_feature}: {intercept:.4f} + {coef:.4f}*x",
        "scaled": True,
    }

    fig, ax = plt.subplots(figsize=(8, 5.5))
    x_test = prep["segments"]["test"]["X_raw"][:, best_index]
    y_test = _as_1d(prep["segments"]["test"]["Y_raw"])
    order = np.argsort(x_test)
    ax.scatter(x_test, y_test, s=14, alpha=0.5, color=REG_FEATURE_COLOR,
               label="исходные данные (test)")
    ax.plot(x_test[order], preds["test"][order], color=REG_FIT_COLOR,
            linewidth=2.0, label="парная линейная регрессия")
    ax.set_xlabel(best_feature)
    ax.set_ylabel("y")
    ax.set_title(f"Парная линейная регрессия y ~ {best_feature} (test)")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    path_14 = save_figure(fig, "task1_4_simple_linear")
    results["models"]["Парная линейная регрессия"]["figure"] = path_14
    step_done(f"MSE test = {metrics['test']['MSE']:.4f}, "
              f"corr = {metrics['test']['corr']:.4f}")

    # ================= 1.5 =================
    run_step("1.5", "Множественная линейная регрессия по всем признакам")
    multiple = _fit_multiple(prep)
    preds = _predict_multiple(prep, multiple)
    metrics = {name: regression_metrics(prep["segments"][name]["Y_raw"], p)
               for name, p in preds.items()}
    _print_model_metrics("множественная линейная", metrics)
    coef_table = pd.DataFrame({
        "признак": ["intercept"] + ds.feature_names,
        "коэффициент": [float(multiple.intercept_)]
                       + [float(v) for v in multiple.coef_.ravel()],
    })
    print(coef_table.to_string(index=False).replace("\n", "\n      "))
    results["models"]["Множественная линейная регрессия"] = {
        "predictions": preds, "metrics": metrics, "family": "статистика",
        "params": "; ".join(f"{r['признак']}={r['коэффициент']:.4f}"
                            for _, r in coef_table.iterrows()),
        "scaled": True,
        "coefficients": coef_table,
    }

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    y_test = _as_1d(prep["segments"]["test"]["Y_raw"])
    y_pred = preds["test"]
    t = np.arange(len(y_test))
    axes[0].plot(t, y_test, color="black", linewidth=1.5, label="реальные y")
    axes[0].plot(t, y_pred, color=REG_PRED_COLOR, linewidth=1.3,
                 label="прогноз")
    axes[0].set_ylabel("y")
    axes[0].set_title("Множественная линейная регрессия: test")
    axes[0].grid(True, linestyle=":", alpha=0.5)
    axes[0].legend()
    axes[1].plot(t, y_test - y_pred, color=REG_FEATURE_COLOR, linewidth=1.0)
    axes[1].axhline(0, color="gray", linestyle="--", linewidth=0.8)
    axes[1].set_xlabel("Номер наблюдения в test")
    axes[1].set_ylabel("ошибка")
    axes[1].grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    path_15 = save_figure(fig, "task1_5_multiple_linear")
    results["models"]["Множественная линейная регрессия"]["figure"] = path_15
    step_done(f"MSE test = {metrics['test']['MSE']:.4f}, "
              f"corr = {metrics['test']['corr']:.4f}")

    # ================= 1.6 =================
    run_step("1.6", "Нелинейная регрессия curve_fit() по всем признакам")
    best_fit, candidate_table = _fit_nonlinear_candidates(prep)
    func, popt = best_fit["func"], best_fit["popt"]
    preds = {
        name: inverse_y(prep, func(seg["X"].T, *popt).reshape(-1, 1)).ravel()
        for name, seg in prep["segments"].items()
    }
    metrics = {name: regression_metrics(prep["segments"][name]["Y_raw"], p)
               for name, p in preds.items()}
    print()
    print(candidate_table.round(4).to_string(index=False)
          .replace("\n", "\n      "))
    print(f"      выбрана функция: {best_fit['title']}")
    _print_model_metrics("нелинейная", metrics)
    results["models"]["Нелинейная регрессия (curve_fit)"] = {
        "predictions": preds, "metrics": metrics, "family": "статистика",
        "params": f"{best_fit['title']}, {len(popt)} параметров",
        "scaled": True,
        "candidates": candidate_table,
        "popt": popt,
    }

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    y_test = _as_1d(prep["segments"]["test"]["Y_raw"])
    y_pred = preds["test"]
    t = np.arange(len(y_test))
    axes[0].plot(t, y_test, color="black", linewidth=1.5, label="реальные y")
    axes[0].plot(t, y_pred, color=REG_FIT_COLOR, linewidth=1.3,
                 label=f"нелинейная аппроксимация ({best_fit['title']})")
    axes[0].set_ylabel("y")
    axes[0].set_title("Нелинейная регрессия curve_fit(): test")
    axes[0].grid(True, linestyle=":", alpha=0.5)
    axes[0].legend(fontsize=8)
    axes[1].plot(t, y_test - y_pred, color=REG_FIT_COLOR, linewidth=1.0)
    axes[1].axhline(0, color="gray", linestyle="--", linewidth=0.8)
    axes[1].set_xlabel("Номер наблюдения в test")
    axes[1].set_ylabel("ошибка")
    axes[1].grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    path_16 = save_figure(fig, "task1_6_nonlinear")
    results["models"]["Нелинейная регрессия (curve_fit)"]["figure"] = path_16
    step_done(f"выбрана {best_fit['title']}, "
              f"MSE test = {metrics['test']['MSE']:.4f}, "
              f"corr = {metrics['test']['corr']:.4f}")

    # дополнительная корреляция для отчёта
    results["y_corr_test"] = corr_safe(y_test, preds["test"])
    return results
