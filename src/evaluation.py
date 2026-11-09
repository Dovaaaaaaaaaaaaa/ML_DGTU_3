"""Пункт 5 методички: обработка результатов и итоговое сравнение методов.

    5.1  графики «реальные значения / прогноз» и «разность» для всех методов
    5.2  графики аппроксимирующих функций линейной, множественной и
         нелинейной регрессии
    5.3  дендрограммы (структуры деревьев) для методов на основе деревьев
    5.4  сравнение метрик при наличии и отсутствии масштабирования
    5.5  итоговая таблица -> output/results_Comparison.xlsx
    5.6  определение лучших/худших методов и интерпретация
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, plot_tree

from src.config import SEED, run_step, step_done
from src.io_utils import (save_figure, save_text, write_excel_sheets,
                          write_fixed_table)

ACTUAL_COLOR = "black"
PRED_COLOR = "#C62828"
RESID_COLOR = "#1565C0"


def _as_1d(a: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=float).ravel()


def _collect_1step(baseline: dict, ml: dict, nn: dict) -> tuple[list, dict]:
    """Собирает все методы с горизонтом 1 шаг в единый список."""
    entries, predictions = [], {}

    for name, res in baseline["models"].items():
        entries.append({
            "name": name, "family": res["family"], "metrics": res["metrics"],
            "params": res.get("params", ""), "scaled": res.get("scaled", True),
        })
        predictions[name] = _as_1d(res["predictions"]["test"])

    for key, res in ml.items():
        if key.startswith("_"):
            continue
        entries.append({
            "name": res["title"], "family": res["family"],
            "metrics": res["metrics"], "params": res["params_text"],
            "scaled": True,
        })
        predictions[res["title"]] = _as_1d(res["predictions"]["test"])

    for key, res in nn["models"].items():
        if res["horizon"] != 1:
            continue
        entries.append({
            "name": res["title"], "family": "нейросеть",
            "metrics": res["metrics"], "params": res["params_text"],
            "scaled": True,
        })
        predictions[res["title"]] = _as_1d(res["predictions"]["test"])

    return entries, predictions


# ----------------------------------------------------------------------
# 5.1
# ----------------------------------------------------------------------
def _plot_predictions_all(entries: list, predictions: dict,
                          y_true: np.ndarray, variant: int) -> str:
    """Сетка «прогноз / разность» для всех методов горизонт 1."""
    n = len(entries)
    nrows = int(np.ceil(n / 2))
    fig, axes = plt.subplots(nrows, 4, figsize=(20, 3.0 * nrows))
    axes = np.atleast_2d(axes)
    t = np.arange(len(y_true))

    for i, entry in enumerate(entries):
        row, pair = divmod(i, 2)
        ax_pred = axes[row, pair * 2]
        ax_res = axes[row, pair * 2 + 1]
        yp = predictions[entry["name"]]

        ax_pred.plot(t, y_true, color=ACTUAL_COLOR, linewidth=1.2,
                     label="реальные")
        ax_pred.plot(t, yp, color=PRED_COLOR, linewidth=1.0, alpha=0.9,
                     label="прогноз")
        m = entry["metrics"]["test"]
        ax_pred.set_title(f"{entry['name']}\nMSE={m['MSE']:.3f}, "
                          f"corr={m['corr']:.3f}", fontsize=8)
        ax_pred.grid(True, linestyle=":", alpha=0.4)
        ax_pred.legend(fontsize=6, loc="best")

        ax_res.plot(t, y_true - yp, color=RESID_COLOR, linewidth=0.9)
        ax_res.axhline(0, color="gray", linestyle="--", linewidth=0.7)
        ax_res.set_title(f"разность: реальное - прогноз\n"
                         f"MAE={m['MAE']:.3f}", fontsize=8)
        ax_res.grid(True, linestyle=":", alpha=0.4)

    for ax in axes.ravel()[2 * n:]:
        ax.axis("off")
    fig.suptitle(f"Пункт 5.1: прогноз и разность для всех методов "
                 f"(горизонт 1, test, вариант {variant})", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return save_figure(fig, "task5_1_predictions_all")


# ----------------------------------------------------------------------
# 5.2
# ----------------------------------------------------------------------
def _plot_regression_fits(baseline: dict, prep: dict, ds,
                          variant: int) -> str:
    """Аппроксимирующие функции, исходные данные и прогноз (три панели)."""
    y_test = _as_1d(prep["segments"]["test"]["Y_raw"])
    t = np.arange(len(y_test))
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- парная линейная ---
    simple = baseline["models"]["Парная линейная регрессия"]
    idx = baseline["best_feature_index"]
    x_test = prep["segments"]["test"]["X_raw"][:, idx]
    order = np.argsort(x_test)
    axes[0].scatter(x_test, y_test, s=16, alpha=0.5, color="#1565C0",
                    label="исходные данные (test)")
    axes[0].plot(x_test[order], simple["predictions"]["test"][order],
                 color="#2E7D32", linewidth=2.2,
                 label="аппроксимация (парная линейная)")
    axes[0].set_xlabel(baseline["best_feature"])
    axes[0].set_ylabel("y")
    axes[0].set_title("Парная линейная регрессия")
    axes[0].grid(True, linestyle=":", alpha=0.5)
    axes[0].legend(fontsize=8)

    # --- множественная линейная ---
    multiple = baseline["models"]["Множественная линейная регрессия"]
    axes[1].plot(t, y_test, color=ACTUAL_COLOR, linewidth=1.6,
                 label="исходные данные (test)")
    axes[1].plot(t, multiple["predictions"]["test"], color="#EF6C00",
                 linewidth=1.5, label="прогноз (множественная линейная)")
    axes[1].set_xlabel("Номер наблюдения в test")
    axes[1].set_ylabel("y")
    axes[1].set_title("Множественная линейная регрессия")
    axes[1].grid(True, linestyle=":", alpha=0.5)
    axes[1].legend(fontsize=8)

    # --- нелинейная ---
    nonlinear = baseline["models"]["Нелинейная регрессия (curve_fit)"]
    axes[2].plot(t, y_test, color=ACTUAL_COLOR, linewidth=1.6,
                 label="исходные данные (test)")
    axes[2].plot(t, nonlinear["predictions"]["test"], color="#6A1B9A",
                 linewidth=1.5,
                 label=f"прогноз (curve_fit: {nonlinear['params']})")
    axes[2].set_xlabel("Номер наблюдения в test")
    axes[2].set_ylabel("y")
    axes[2].set_title("Нелинейная регрессия curve_fit()")
    axes[2].grid(True, linestyle=":", alpha=0.5)
    axes[2].legend(fontsize=7)

    fig.suptitle(f"Пункт 5.2: аппроксимирующие функции и прогноз "
                 f"(вариант {variant})", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return save_figure(fig, "task5_2_regression_fits")


# ----------------------------------------------------------------------
# 5.3
# ----------------------------------------------------------------------
def _tree_for_display(estimator, prep: dict):
    """Достаёт дерево из модели; слишком глубокое заменяет мелким аналогом."""
    if isinstance(estimator, DecisionTreeRegressor):
        tree = estimator
    elif hasattr(estimator, "estimators_"):
        est = estimator.estimators_
        if isinstance(est, list):
            tree = est[0]
        else:
            # GradientBoosting: массив формы (n_estimators, 1)
            tree = est.ravel()[0]
    else:
        return None, ""

    note = ""
    if tree.get_depth() > 6:
        tr = prep["segments"]["train"]
        tree = DecisionTreeRegressor(max_depth=4, random_state=SEED)
        tree.fit(tr["X"], tr["Y_raw"].ravel())
        note = " (показано дерево глубины 4 для читаемости)"
    return tree, note


def _plot_tree_dendrograms(ml: dict, prep: dict, ds, variant: int) -> str:
    """Дендрограммы (структуры деревьев) Decision Tree, RF и GB."""
    specs = [("tree", "Decision Tree"), ("rf", "Random Forest"),
             ("gb", "Gradient Boosting")]
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    for ax, (key, title) in zip(axes, specs):
        res = ml.get(key)
        if res is None:
            ax.axis("off")
            continue
        tree, note = _tree_for_display(res["estimator"], prep)
        if tree is None:
            ax.axis("off")
            continue
        plot_tree(tree, feature_names=ds.feature_names, filled=True,
                  rounded=True, impurity=False, fontsize=7, ax=ax)
        ax.set_title(f"{title}{note}", fontsize=10)
    fig.suptitle(f"Пункт 5.3: дендрограммы решений методов на основе деревьев "
                 f"(вариант {variant})", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return save_figure(fig, "task5_3_tree_dendrogram")


# ----------------------------------------------------------------------
# 5.4
# ----------------------------------------------------------------------
def _scaling_table(ml: dict) -> pd.DataFrame:
    """Сравнение test-метрик при масштабировании и без него."""
    rows = []
    for key, res in ml.items():
        if key.startswith("_"):
            continue
        scaled = res["metrics"]["test"]
        raw = res["metrics_unscaled"]["test"]
        rows.append({
            "метод": res["title"],
            "MSE с масштабированием": scaled["MSE"],
            "MSE без масштабирования": raw["MSE"],
            "MAE с масштабированием": scaled["MAE"],
            "MAE без масштабирования": raw["MAE"],
            "corr с масштабированием": scaled["corr"],
            "corr без масштабирования": raw["corr"],
            "изменение MSE, %": (raw["MSE"] - scaled["MSE"])
                                / max(scaled["MSE"], 1e-12) * 100,
        })
    return pd.DataFrame(rows)


def _plot_scaling_effect(table: pd.DataFrame, variant: int) -> str:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(table))
    width = 0.38
    ax.bar(x - width / 2, table["MSE с масштабированием"], width,
           label="с масштабированием", color="#1565C0", edgecolor="black")
    ax.bar(x + width / 2, table["MSE без масштабирования"], width,
           label="без масштабирования", color="#C62828", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(table["метод"], rotation=20, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("MSE на test (log)")
    ax.set_title(f"Пункт 5.4: влияние масштабирования на MSE "
                 f"(вариант {variant})")
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    return save_figure(fig, "task5_4_scaling_effect")


# ----------------------------------------------------------------------
# 5.5
# ----------------------------------------------------------------------
def _summary_table(entries: list) -> pd.DataFrame:
    rows = []
    for entry in entries:
        m = entry["metrics"]
        rows.append({
            "Метод": entry["name"],
            "Семейство": entry["family"],
            "MAE train": m["train"]["MAE"],
            "MSE train": m["train"]["MSE"],
            "Corr train": m["train"]["corr"],
            "MAE val": m["val"]["MAE"],
            "MSE val": m["val"]["MSE"],
            "Corr val": m["val"]["corr"],
            "MAE test": m["test"]["MAE"],
            "MSE test": m["test"]["MSE"],
            "RMSE test": m["test"]["RMSE"],
            "Corr test": m["test"]["corr"],
            "Параметры": entry["params"],
            "Масштабирование": "да" if entry["scaled"] else "нет",
        })
    return pd.DataFrame(rows).sort_values("MSE test")


def _multistep_table(nn: dict) -> pd.DataFrame:
    rows = []
    for key, res in nn["models"].items():
        if res["horizon"] != 7:
            continue
        table = res["metrics_by_horizon"]
        for _, r in table.iterrows():
            rows.append({
                "модель": res["title"], "горизонт": int(r["горизонт"]),
                "MAE": r["MAE"], "MSE": r["MSE"], "RMSE": r["RMSE"],
                "R2": r["R2"], "corr": r["corr"],
            })
        mean = res["mean_metrics"]
        rows.append({
            "модель": res["title"], "горизонт": "среднее 1..7",
            "MAE": mean["MAE"], "MSE": mean["MSE"], "RMSE": mean["RMSE"],
            "R2": mean["R2"], "corr": mean["corr"],
        })
    return pd.DataFrame(rows)


def _parameters_table(nn: dict) -> pd.DataFrame:
    rows = []
    for key, res in nn["models"].items():
        rows.append({
            "модель": res["title"], "горизонт": res["horizon"],
            "параметров": res["n_parameters"],
            "конфигурация": res["params_text"],
            "val MSE": res["val_mse"],
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# 5.6
# ----------------------------------------------------------------------
def _conclusions(table: pd.DataFrame, multistep: pd.DataFrame,
                 scaling: pd.DataFrame, y_test: np.ndarray) -> str:
    best = table.iloc[0]
    worst = table.iloc[-1]
    top3 = table.head(3)
    bottom3 = table.tail(3)

    lines = []
    lines.append("ПУНКТ 5.6. Наилучшие и наихудшие методы, интерпретация")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Тестовая выборка: {len(y_test)} наблюдений; "
                 f"сравнение по MSE (меньше — лучше).")
    lines.append("")
    lines.append(f"ЛУЧШИЙ метод: {best['Метод']}")
    lines.append(f"  MAE = {best['MAE test']:.4f}, MSE = {best['MSE test']:.4f}, "
                 f"RMSE = {best['RMSE test']:.4f}, corr = {best['Corr test']:.4f}")
    lines.append(f"  параметры: {best['Параметры']}")
    lines.append("")
    lines.append(f"ХУДШИЙ метод: {worst['Метод']}")
    lines.append(f"  MAE = {worst['MAE test']:.4f}, MSE = {worst['MSE test']:.4f}, "
                 f"RMSE = {worst['RMSE test']:.4f}, corr = {worst['Corr test']:.4f}")
    lines.append(f"  параметры: {worst['Параметры']}")
    lines.append("")
    lines.append("Топ-3 метода по MSE test:")
    for _, r in top3.iterrows():
        lines.append(f"  {r['MSE test']:.4f}  {r['Метод']}")
    lines.append("")
    lines.append("Худшие 3 метода по MSE test:")
    for _, r in bottom3.iterrows():
        lines.append(f"  {r['MSE test']:.4f}  {r['Метод']}")
    lines.append("")

    spread = float(worst["MSE test"] / max(best["MSE test"], 1e-12))
    lines.append(f"Разрыв между лучшим и худшим по MSE: в {spread:.1f} раз.")
    lines.append("")
    def _delta_text(method: str) -> str:
        row = scaling[scaling["метод"] == method]
        if row.empty:
            return "нет данных"
        delta = float(row.iloc[0]["изменение MSE, %"])
        if abs(delta) < 0.05:
            return "не меняется"
        direction = "снижается" if delta < 0 else "возрастает"
        return f"{direction} на {abs(delta):.1f} %"

    lines.append("Интерпретация:")
    lines.append("  * В данном варианте y очень тесно связана с признаком a "
                 "(|r| > 0.99), поэтому")
    lines.append("    линейные модели дают низкую ошибку на train. На test "
                 "ошибка заметно выше:")
    lines.append("    в тестовой части ряда меняется уровень y, и методы, "
                 "ориентированные на")
    lines.append("    интерполяцию (KNN, деревья), переносят этот сдвиг хуже.")
    lines.append("  * Нелинейная аппроксимация не дала выигрыша по сравнению "
                 "с множественной")
    lines.append("    линейной: связь y с признаками близка к линейной, а "
                 "добавленные члены лишь")
    lines.append("    немного снижают ошибку на train.")
    lines.append(f"  * Влияние масштабирования на test: MSE KNN "
                 f"{_delta_text('KNN')}, MSE SVR {_delta_text('SVM (SVR)')}.")
    lines.append("    Деревья и ансамбли на деревьях инвариантны к "
                 "монотонному масштабированию,")
    lines.append("    поэтому для них разница нулевая. Для KNN и SVR масштаб "
                 "важен для")
    lines.append("    устойчивости, но итоговый эффект зависит от того, "
                 "насколько доминирует")
    lines.append("    один признак: при сильном доминировании a "
                 "стандартизация не всегда улучшает MSE.")
    lines.append("  * Деревья и ансамбли в этом варианте заметно уступают "
                 "линейным моделям и")
    lines.append("    нейросетям: они хорошо запоминают train (MSE близка к "
                 "нулю), но плохо")
    lines.append("    экстраполируют сдвиг уровня y в тестовой части ряда.")
    lines.append("  * Нейронные сети с масштабированным входом показали "
                 "лучший результат;")
    lines.append("    без стандартизации градиентный спуск становится "
                 "неустойчивым, поэтому")
    lines.append("    нейросети обучались только на масштабированных "
                 "признаках.")
    lines.append("")
    if not multistep.empty:
        lines.append("Многошаговый прогноз (нейронные сети, горизонт 7):")
        pivot = multistep[multistep["горизонт"] == "среднее 1..7"]
        for _, r in pivot.iterrows():
            lines.append(f"  {r['модель']}: MAE ср. = {r['MAE']:.4f}, "
                         f"MSE ср. = {r['MSE']:.4f}, corr ср. = {r['corr']:.4f}")
        lines.append("  * Ошибка на 7-м шаге заметно выше, чем на первых "
                     "шагах горизонта")
        lines.append("    (см. task4_multistep_errors): дальние значения y "
                     "зависят от большего")
        lines.append("    числа неизвестных будущих воздействий. При этом "
                     "средняя ошибка по всем")
        lines.append("    7 шагам может быть ниже одношаговой из-за "
                     "усреднения по горизонту.")
        lines.append("")
    lines.append("Сравнение масштабирования (п. 5.4):")
    for _, r in scaling.iterrows():
        lines.append(f"  {r['метод']:<22} MSE scaled = "
                     f"{r['MSE с масштабированием']:.4f}, без = "
                     f"{r['MSE без масштабирования']:.4f} "
                     f"({r['изменение MSE, %']:+.1f} %)")
    lines.append("")
    lines.append(f"Вывод: в данном варианте лучший результат показал "
                 f"{best['Метод']} "
                 f"(MSE test = {best['MSE test']:.4f}). Худшие результаты "
                 f"показали методы,")
    lines.append("переобучающиеся на train и плохо переносящие сдвиг уровня y "
                 "в тестовой части")
    lines.append("временного ряда; для них подбор гиперпараметров по "
                 "хронологической CV")
    lines.append("не компенсирует изменение режима ряда.")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Пункт 5 целиком
# ----------------------------------------------------------------------
def run_evaluation(ds, prep1: dict, baseline: dict, ml: dict, nn: dict,
                   variant: int) -> dict:
    """Выполняет пункт 5 методички."""
    results: dict = {}
    y_test = _as_1d(prep1["segments"]["test"]["Y_raw"])
    entries, predictions = _collect_1step(baseline, ml, nn)

    # ---------------- 5.1 ----------------
    run_step("5.1", "Графики прогноза и разности для всех методов")
    path_51 = _plot_predictions_all(entries, predictions, y_test, variant)
    results["figure_5_1"] = path_51
    step_done(f"{len(entries)} методов, {os.path.basename(path_51)}")

    # ---------------- 5.2 ----------------
    run_step("5.2", "Аппроксимирующие функции линейной/множественной/"
                    "нелинейной регрессии")
    path_52 = _plot_regression_fits(baseline, prep1, ds, variant)
    results["figure_5_2"] = path_52
    step_done(os.path.basename(path_52))

    # ---------------- 5.3 ----------------
    run_step("5.3", "Дендрограммы решений методов на основе деревьев")
    path_53 = _plot_tree_dendrograms(ml, prep1, ds, variant)
    results["figure_5_3"] = path_53
    step_done(os.path.basename(path_53))

    # ---------------- 5.4 ----------------
    run_step("5.4", "Сравнение метрик с масштабированием и без него")
    scaling = _scaling_table(ml)
    path_54 = _plot_scaling_effect(scaling, variant)
    results["scaling"] = scaling
    results["figure_5_4"] = path_54
    print()
    print(scaling.round(4).to_string(index=False).replace("\n", "\n      "))
    print("      примечание: линейные модели инвариантны к масштабированию; "
          "нейросети обучены")
    print("      с масштабированием, так как без него градиентный спуск "
          "неустойчив.")
    step_done(os.path.basename(path_54))

    # ---------------- 5.5 ----------------
    run_step("5.5", "Итоговая таблица метрик -> results_Comparison.xlsx")
    summary = _summary_table(entries)
    multistep = _multistep_table(nn)
    parameters = _parameters_table(nn)
    results["summary"] = summary
    results["multistep"] = multistep

    path_fixed = write_fixed_table(summary, "results_Comparison.xlsx",
                                   sheet_name="1-step")
    path_sheets = write_excel_sheets("results_Comparison_all", {
        "1-step": summary,
        "7-step": multistep,
        "scaling": scaling,
        "parameters": parameters,
    })
    results["xlsx_fixed"] = path_fixed
    results["xlsx_all"] = path_sheets
    print()
    print(summary.round(4).to_string(index=False).replace("\n", "\n      "))
    print()
    print(f"      фиксированный файл: {os.path.basename(path_fixed)}")
    print(f"      полный файл: {os.path.basename(path_sheets)}")
    step_done(f"{len(summary)} методов, {os.path.basename(path_fixed)}")

    # ---------------- 5.6 ----------------
    run_step("5.6", "Лучшие и худшие методы, интерпретация")
    text = _conclusions(summary, multistep, scaling, y_test)
    path_txt = save_text(text, "results_conclusions.txt")
    results["conclusions"] = text
    results["conclusions_path"] = path_txt
    print()
    print(text.replace("\n", "\n      "))
    step_done(os.path.basename(path_txt))

    return results
