"""Пункты 2 и 3 методички: одиночные методы и ансамбли.

    2.1  K ближайших соседей (KNN)
    2.2  машина опорных векторов (SVR)
    2.3  одиночное дерево решений (Decision Tree)
    3.1  случайный лес (Random Forest)
    3.2  бэггинг (Bagging)
    3.3  градиентный бустинг (Gradient Boosting)

Для каждого метода перебираются гиперпараметры (GridSearchCV,
хронологическая кросс-валидация TimeSeriesSplit). Лучшая модель обучается на
train и оценивается на val/test. Затем те же гиперпараметры повторно
обучаются БЕЗ масштабирования — это нужно для п. 5.4.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import (BaggingRegressor, GradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from src.config import SEED, banner, run_step, step_done, sub_step
from src.data_loader import regression_metrics
from src.io_utils import save_figure

MODEL_MSE_COLOR = "#C62828"
MODEL_TITLE_COLOR = "#1565C0"


def build_specs() -> list[dict]:
    """Список методов с сетками гиперпараметров и ключевым параметром для графика."""
    dt = DecisionTreeRegressor(random_state=SEED)
    return [
        {
            "key": "knn", "step": "2.1", "title": "KNN",
            "family": "одиночный",
            "estimator": KNeighborsRegressor(),
            "grid": {
                "n_neighbors": [3, 5, 7, 11, 15, 21],
                "weights": ["uniform", "distance"],
                "p": [1, 2],
            },
            "primary": "n_neighbors",
        },
        {
            "key": "svr", "step": "2.2", "title": "SVM (SVR)",
            "family": "одиночный",
            "estimator": SVR(),
            "grid": {
                "kernel": ["rbf", "linear"],
                "C": [0.1, 1.0, 10.0, 100.0],
                "epsilon": [0.01, 0.1, 0.5],
                "gamma": ["scale"],
            },
            "primary": "C",
        },
        {
            "key": "tree", "step": "2.3", "title": "Decision Tree",
            "family": "одиночный",
            "estimator": DecisionTreeRegressor(random_state=SEED),
            "grid": {
                "max_depth": [3, 5, 7, 10, None],
                "min_samples_leaf": [1, 2, 5],
                "min_samples_split": [2, 10],
            },
            "primary": "max_depth",
        },
        {
            "key": "rf", "step": "3.1", "title": "Random Forest",
            "family": "ансамбль",
            "estimator": RandomForestRegressor(random_state=SEED, n_jobs=1),
            "grid": {
                "n_estimators": [100, 200],
                "max_depth": [5, 10, None],
                "min_samples_leaf": [1, 2],
                "max_features": [1.0, "sqrt"],
            },
            "primary": "max_depth",
        },
        {
            "key": "bagging", "step": "3.2", "title": "Bagging",
            "family": "ансамбль",
            "estimator": BaggingRegressor(estimator=dt, random_state=SEED,
                                          n_jobs=1),
            "grid": {
                "n_estimators": [50, 100],
                "max_samples": [0.7, 1.0],
                "max_features": [0.7, 1.0],
            },
            "primary": "n_estimators",
        },
        {
            "key": "gb", "step": "3.3", "title": "Gradient Boosting",
            "family": "ансамбль",
            "estimator": GradientBoostingRegressor(random_state=SEED),
            "grid": {
                "n_estimators": [100, 200],
                "learning_rate": [0.05, 0.1],
                "max_depth": [2, 3],
                "subsample": [0.8, 1.0],
            },
            "primary": "n_estimators",
        },
    ]


def _metrics_for(model, prep: dict, scaled: bool) -> tuple[dict, dict]:
    """Прогнозы и метрики на train/val/test. scaled=False — сырые признаки."""
    key = "X" if scaled else "X_raw"
    preds, metrics = {}, {}
    for name, seg in prep["segments"].items():
        yp = np.asarray(model.predict(seg[key]), dtype=float).ravel()
        preds[name] = yp
        metrics[name] = regression_metrics(seg["Y_raw"], yp)
    return preds, metrics


def _format_params(params: dict) -> str:
    """Компактная строка гиперпараметров для таблицы."""
    parts = []
    for k, v in params.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:g}")
        else:
            parts.append(f"{k}={v}")
    return ", ".join(parts)


def _importance_frame(model, feature_names: list) -> pd.DataFrame | None:
    """Важности признаков, если модель их предоставляет."""
    if hasattr(model, "feature_importances_"):
        return pd.DataFrame({
            "признак": feature_names,
            "важность": np.asarray(model.feature_importances_, dtype=float),
        }).sort_values("важность", ascending=False)
    return None


def _tune_models(specs: list[dict], prep: dict) -> dict:
    """Перебирает гиперпараметры всех методов, обучает лучшие модели."""
    tr = prep["segments"]["train"]
    cv = TimeSeriesSplit(n_splits=3)
    results: dict = {}

    for spec in specs:
        if spec["family"] == "ансамбль" and spec["step"] == "3.1":
            banner("Пункт 3. Ансамблевые методы")
        run_step(spec["step"], f"{spec['title']}: подбор гиперпараметров")
        search = GridSearchCV(
            spec["estimator"], spec["grid"], scoring="neg_mean_squared_error",
            cv=cv, n_jobs=4, refit=True, error_score="raise",
        )
        search.fit(tr["X"], tr["Y_raw"].ravel())
        best = search.best_estimator_
        params = search.best_params_
        preds, metrics = _metrics_for(best, prep, scaled=True)

        # повторное обучение без масштабирования для п. 5.4
        raw_model = search.estimator
        raw_model.set_params(**params)
        raw_model.fit(tr["X_raw"], tr["Y_raw"].ravel())
        raw_preds, raw_metrics = _metrics_for(raw_model, prep, scaled=False)

        importance = _importance_frame(best, prep["feature_names"])
        results[spec["key"]] = {
            "title": spec["title"], "family": spec["family"],
            "step": spec["step"], "predictions": preds, "metrics": metrics,
            "predictions_unscaled": raw_preds,
            "metrics_unscaled": raw_metrics,
            "params": params, "params_text": _format_params(params),
            "best_cv_mse": float(-search.best_score_),
            "cv_results": search.cv_results_,
            "primary": spec["primary"],
            "importance": importance,
            "estimator": best,
            "scaled": True,
        }

        print()
        sub_step(f"лучшие параметры: {_format_params(params)}")
        print(f"      CV MSE (train): {results[spec['key']]['best_cv_mse']:.4f}")
        for name in ("train", "val", "test"):
            m = metrics[name]
            print(f"      {name:<6} MAE = {m['MAE']:.4f}   MSE = {m['MSE']:.4f}  "
                  f"corr = {m['corr']:.4f}")
        if importance is not None:
            top = importance.head(5)
            print("      важности признаков: "
                  + ", ".join(f"{r['признак']}={r['важность']:.3f}"
                              for _, r in top.iterrows()))
        m_test = metrics["test"]
        step_done(f"{_format_params(params)}; MSE test = {m_test['MSE']:.4f}, "
                  f"corr = {m_test['corr']:.4f}")
    return results


def _plot_tuning(results: dict, variant: int) -> str:
    """Зависимость CV-MSE от ключевого гиперпараметра для каждого метода."""
    items = list(results.items())
    ncols = 3
    nrows = int(np.ceil(len(items) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 3.8 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, (key, res) in zip(axes, items):
        cv = pd.DataFrame(res["cv_results"])
        primary = res["primary"]
        # В cv_results_ столбцы параметров называются param_<имя>.
        # sort=False — чтобы None и числа не сравнивались при группировке.
        grouped = cv.groupby(f"param_{primary}", sort=False)["mean_test_score"].max()
        x = [str(v) for v in grouped.index]
        y = -grouped.to_numpy(dtype=float)
        ax.plot(range(len(x)), y, "o-", color=MODEL_TITLE_COLOR, linewidth=1.8)
        ax.set_xticks(range(len(x)))
        ax.set_xticklabels(x, rotation=0, fontsize=8)
        ax.set_xlabel(primary)
        ax.set_ylabel("CV MSE (лучший)")
        ax.set_title(f"{res['title']}\nMSE test = "
                     f"{res['metrics']['test']['MSE']:.3f}", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.5)

    for ax in axes[len(items):]:
        ax.axis("off")
    fig.suptitle(f"Подбор гиперпараметров: хронологическая CV по train "
                 f"(вариант {variant})", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return save_figure(fig, "task2_3_hyperparameter_tuning")


def _plot_test_mse(results: dict, variant: int) -> str:
    """Столбчатая диаграмма MSE на тесте для всех методов пунктов 2-3."""
    items = sorted(results.items(), key=lambda kv: kv[1]["metrics"]["test"]["MSE"],
                   reverse=True)
    titles = [v["title"] for _, v in items]
    mse = [v["metrics"]["test"]["MSE"] for _, v in items]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars = ax.barh(titles, mse, color=MODEL_MSE_COLOR, alpha=0.8,
                   edgecolor="black")
    for bar, value in zip(bars, mse):
        ax.text(value, bar.get_y() + bar.get_height() / 2,
                f" {value:.3f}", va="center", fontsize=9)
    ax.set_xlabel("MSE на тестовой выборке")
    ax.set_title(f"Одиночные методы и ансамбли: MSE test (вариант {variant})")
    ax.grid(True, axis="x", linestyle=":", alpha=0.5)
    fig.tight_layout()
    return save_figure(fig, "task2_3_test_mse")


def run_ml_models(ds, prep: dict, variant: int) -> dict:
    """Выполняет пункты 2 и 3 методички."""
    specs = build_specs()
    results = _tune_models(specs, prep)

    banner("Сводка по одиночным методам и ансамблям")
    rows = []
    for res in results.values():
        m = res["metrics"]["test"]
        rows.append({
            "метод": res["title"], "семейство": res["family"],
            "MAE test": m["MAE"], "MSE test": m["MSE"],
            "RMSE test": m["RMSE"], "corr test": m["corr"],
            "параметры": res["params_text"],
        })
    table = pd.DataFrame(rows).sort_values("MSE test")
    print(table.round(4).to_string(index=False).replace("\n", "\n      "))

    fig_tuning = _plot_tuning(results, variant)
    fig_mse = _plot_test_mse(results, variant)
    results["_figures"] = {"tuning": fig_tuning, "test_mse": fig_mse}
    print()
    print(f"      графики: {os.path.basename(fig_tuning)}, "
          f"{os.path.basename(fig_mse)}")
    return results
