"""Пункт 4 методички: нейронные сети (PyTorch).

    4.1  однослойный линейный персептрон с функцией активации
    4.2  многослойный персептрон (MLP)
    4.3  глубокая нейронная сеть произвольной архитектуры

Для каждого варианта строятся две модели:
    * прогноз на 1 шаг:  X[t] -> y[t+1]
    * прогноз на 7 шагов: X[t] -> (y[t+1], ..., y[t+7])

Входные признаки и целевая переменная масштабируются параметрами train
(StandardScaler). Прогнозы возвращаются в исходные единицы y. Для каждой
модели сохраняются кривые обучения (loss и MAE от эпохи) и число внутренних
параметров.
"""

import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.config import HORIZONS, SEED, run_step, step_done, sub_step
from src.data_loader import (inverse_y, mean_horizon_metric,
                             metrics_by_horizon, print_metrics,
                             regression_metrics)
from src.io_utils import save_figure

torch.set_num_threads(4)

TRAIN_COLOR = "#1565C0"
VAL_COLOR = "#C62828"


# ----------------------------------------------------------------------
# Архитектуры
# ----------------------------------------------------------------------
def get_activation(name: str):
    """Возвращает модуль активации или None для линейной функции."""
    name = name.lower()
    if name in ("linear", "none", "identity"):
        return None
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "gelu":
        return nn.GELU()
    if name == "leaky_relu":
        return nn.LeakyReLU(0.01)
    raise ValueError(f"Неизвестная активация: {name}")


class SingleLayerNet(nn.Module):
    """Однослойный персептрон: один линейный слой + активация."""

    def __init__(self, n_in: int, n_out: int, activation: str = "linear"):
        super().__init__()
        self.linear = nn.Linear(n_in, n_out)
        self.activation = get_activation(activation)

    def forward(self, x):
        z = self.linear(x)
        return self.activation(z) if self.activation is not None else z


class MLPNet(nn.Module):
    """Многослойный персептрон с настраиваемыми скрытыми слоями."""

    def __init__(self, n_in: int, n_out: int, hidden=(64, 32),
                 activation: str = "relu", dropout: float = 0.0):
        super().__init__()
        act = get_activation(activation)
        layers = []
        prev = n_in
        for width in hidden:
            layers.append(nn.Linear(prev, width))
            if act is not None:
                layers.append(copy.deepcopy(act))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = width
        layers.append(nn.Linear(prev, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    """Двухслойный residual-блок с BatchNorm и GELU."""

    def __init__(self, width: int, dropout: float = 0.1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(width, width), nn.BatchNorm1d(width), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, width), nn.BatchNorm1d(width), nn.GELU(),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.drop(self.body(x))


class DeepNet(nn.Module):
    """Глубокая сеть: входной блок, residual-блоки и голова регрессии."""

    def __init__(self, n_in: int, n_out: int, width: int = 128,
                 n_blocks: int = 2, dropout: float = 0.1):
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(n_in, width), nn.BatchNorm1d(width), nn.GELU())
        self.blocks = nn.Sequential(
            *[ResidualBlock(width, dropout) for _ in range(n_blocks)])
        self.head = nn.Sequential(
            nn.Linear(width, max(width // 2, 8)), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(width // 2, 8), n_out),
        )

    def forward(self, x):
        return self.head(self.blocks(self.input(x)))


# ----------------------------------------------------------------------
# Обучение
# ----------------------------------------------------------------------
def _make_loss(name: str):
    name = name.lower()
    if name == "mse":
        return nn.MSELoss()
    if name in ("mae", "l1"):
        return nn.L1Loss()
    if name == "huber":
        return nn.HuberLoss(delta=1.0)
    raise ValueError(f"Неизвестная функция потерь: {name}")


def _make_optimizer(name: str, params, lr: float, weight_decay: float):
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9,
                              weight_decay=weight_decay)
    raise ValueError(f"Неизвестный оптимизатор: {name}")


def _as_tensor(a: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(np.asarray(a, dtype=np.float32))


class TorchTrainer:
    """Мини-батчевое обучение с ранней остановкой по val-loss."""

    def __init__(self, model: nn.Module, lr: float = 1e-3, epochs: int = 400,
                 batch_size: int = 64, weight_decay: float = 0.0,
                 optimizer: str = "adam", loss: str = "mse",
                 patience: int = 40, grad_clip: float = 1.0,
                 seed: int = SEED):
        self.model = model
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer
        self.loss_name = loss
        self.patience = patience
        self.grad_clip = grad_clip
        self.seed = seed
        self.history: list[dict] = []

    def fit(self, Xtr, Ytr, Xval, Yval) -> tuple[list[dict], float]:
        torch.manual_seed(self.seed)
        Xtr_t, Ytr_t = _as_tensor(Xtr), _as_tensor(Ytr)
        Xval_t, Yval_t = _as_tensor(Xval), _as_tensor(Yval)

        criterion = _make_loss(self.loss_name)
        self.optimizer = _make_optimizer(
            self.optimizer_name, self.model.parameters(), self.lr,
            self.weight_decay)
        n = len(Xtr_t)
        best_state = copy.deepcopy(self.model.state_dict())
        best_val = float("inf")
        wait = 0
        self.history = []

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            perm = torch.randperm(n)
            for start in range(0, n, self.batch_size):
                idx = perm[start:start + self.batch_size]
                if len(idx) < 2:          # BatchNorm требует batch > 1
                    continue
                xb, yb = Xtr_t[idx], Ytr_t[idx]
                self.optimizer.zero_grad()
                out = self.model(xb)
                loss = criterion(out, yb)
                loss.backward()
                if self.grad_clip:
                    nn.utils.clip_grad_norm_(self.model.parameters(),
                                             self.grad_clip)
                self.optimizer.step()

            train_loss, train_mae = self._evaluate(Xtr_t, Ytr_t, criterion)
            val_loss, val_mae = self._evaluate(Xval_t, Yval_t, criterion)
            self.history.append({
                "epoch": epoch,
                "train_loss": train_loss, "val_loss": val_loss,
                "train_mae": train_mae, "val_mae": val_mae,
            })

            if val_loss < best_val - 1e-9:
                best_val = val_loss
                best_state = copy.deepcopy(self.model.state_dict())
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    break

        self.model.load_state_dict(best_state)
        return self.history, best_val

    def _evaluate(self, X: torch.Tensor, Y: torch.Tensor, criterion):
        self.model.eval()
        with torch.no_grad():
            out = self.model(X)
            loss = float(criterion(out, Y).item())
            mae = float(torch.mean(torch.abs(out - Y)).item())
        return loss, mae

    def predict(self, X) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            out = self.model(_as_tensor(X))
        return out.numpy()

    def n_parameters(self) -> int:
        return int(sum(p.numel() for p in self.model.parameters()
                       if p.requires_grad))


# ----------------------------------------------------------------------
# Наборы конфигураций и поиск
# ----------------------------------------------------------------------
TRAIN_KEYS = ("lr", "epochs", "batch_size", "weight_decay", "optimizer",
              "loss", "patience")


def _trainer_kwargs(config: dict) -> dict:
    return {k: config[k] for k in TRAIN_KEYS if k in config}


def perceptron_configs() -> list[dict]:
    base = {"epochs": 400, "batch_size": 64, "patience": 40,
            "weight_decay": 0.0, "optimizer": "adam", "loss": "mse"}
    variants = [
        {"activation": "linear", "lr": 1e-3},
        {"activation": "tanh", "lr": 1e-3},
        {"activation": "relu", "lr": 1e-3},
        {"activation": "sigmoid", "lr": 1e-3},
        {"activation": "tanh", "lr": 5e-3},
        {"activation": "linear", "lr": 5e-3, "optimizer": "rmsprop"},
        {"activation": "linear", "lr": 1e-3, "loss": "mae"},
    ]
    return [base | v for v in variants]


def mlp_configs() -> list[dict]:
    base = {"epochs": 400, "batch_size": 64, "patience": 40,
            "optimizer": "adam", "loss": "mse", "weight_decay": 0.0}
    variants = [
        {"hidden": (64, 32), "activation": "relu", "dropout": 0.0, "lr": 1e-3},
        {"hidden": (128, 64), "activation": "relu", "dropout": 0.1, "lr": 1e-3},
        {"hidden": (128, 128, 64), "activation": "relu", "dropout": 0.1,
         "lr": 1e-3},
        {"hidden": (128, 64), "activation": "tanh", "dropout": 0.1, "lr": 1e-3},
        {"hidden": (128, 64), "activation": "relu", "dropout": 0.2,
         "lr": 5e-4, "weight_decay": 1e-4},
        {"hidden": (256, 128), "activation": "relu", "dropout": 0.1,
         "lr": 1e-3, "loss": "huber"},
    ]
    return [base | v for v in variants]


def deep_configs() -> list[dict]:
    base = {"epochs": 500, "batch_size": 64, "patience": 50}
    variants = [
        {"width": 128, "n_blocks": 2, "dropout": 0.1, "lr": 1e-3,
         "weight_decay": 1e-4, "optimizer": "adam", "loss": "mse"},
        {"width": 256, "n_blocks": 3, "dropout": 0.15, "lr": 5e-4,
         "weight_decay": 1e-4, "optimizer": "adamw", "loss": "huber"},
        {"width": 192, "n_blocks": 2, "dropout": 0.2, "lr": 1e-3,
         "weight_decay": 1e-3, "optimizer": "adam", "loss": "mse"},
    ]
    return [base | v for v in variants]


def _build_model(arch: str, config: dict, n_in: int, n_out: int) -> nn.Module:
    if arch == "perceptron":
        return SingleLayerNet(n_in, n_out, config["activation"])
    if arch == "mlp":
        return MLPNet(n_in, n_out, config["hidden"],
                      config["activation"], config["dropout"])
    if arch == "deep":
        return DeepNet(n_in, n_out, config["width"],
                       config["n_blocks"], config["dropout"])
    raise ValueError(arch)


def _train_and_eval(arch: str, config: dict, prep: dict) -> dict:
    """Обучает одну конфигурацию и возвращает её метрики на val."""
    tr, val = prep["segments"]["train"], prep["segments"]["val"]
    torch.manual_seed(SEED)
    model = _build_model(arch, config, tr["X"].shape[1], tr["Y"].shape[1])
    trainer = TorchTrainer(model, seed=SEED, **_trainer_kwargs(config))
    history, best_val = trainer.fit(tr["X"], tr["Y"], val["X"], val["Y"])
    val_pred = inverse_y(prep, trainer.predict(val["X"]))
    metrics = regression_metrics(val["Y_raw"], val_pred)
    return {"trainer": trainer, "history": history,
            "best_val_loss": best_val, "config": config,
            "val_metrics": metrics}


def _search_best(arch: str, configs: list[dict], prep: dict,
                 title: str) -> dict:
    """Перебирает конфигурации, выбирает лучшую по MSE на валидации."""
    best = None
    for i, config in enumerate(configs, 1):
        sub_step(f"{title}: конфигурация {i}/{len(configs)} "
                 f"({_short_config(arch, config)})")
        try:
            trial = _train_and_eval(arch, config, prep)
        except Exception as exc:                       # noqa: BLE001
            print(f"        ошибка обучения: {exc}")
            continue
        m = trial["val_metrics"]
        print(f"        эпох: {len(trial['history'])}, "
              f"val MSE = {m['MSE']:.4f}, MAE = {m['MAE']:.4f}")
        if best is None or m["MSE"] < best["val_metrics"]["MSE"]:
            best = trial
    if best is None:
        raise RuntimeError(f"Ни одна конфигурация {arch} не обучилась")
    return best


def _short_config(arch: str, config: dict) -> str:
    if arch == "perceptron":
        return (f"act={config['activation']}, lr={config['lr']}, "
                f"loss={config['loss']}, opt={config['optimizer']}")
    if arch == "mlp":
        return (f"hidden={config['hidden']}, act={config['activation']}, "
                f"drop={config['dropout']}, lr={config['lr']}, "
                f"loss={config['loss']}")
    return (f"width={config['width']}, blocks={config['n_blocks']}, "
            f"drop={config['dropout']}, lr={config['lr']}, "
            f"loss={config['loss']}, opt={config['optimizer']}")


def _full_predictions(prep: dict, trainer: TorchTrainer) -> tuple[dict, dict]:
    """Прогнозы и метрики на train/val/test в исходных единицах y."""
    preds, metrics = {}, {}
    for name, seg in prep["segments"].items():
        yp = inverse_y(prep, trainer.predict(seg["X"]))
        preds[name] = yp
        metrics[name] = regression_metrics(seg["Y_raw"], yp)
    return preds, metrics


def _fit_architecture(arch: str, title: str, configs: list[dict],
                      prep1: dict, prep7: dict) -> dict:
    """Обучает архитектуру для горизонтов 1 и 7, возвращает обе модели."""
    step_number = {"perceptron": 1, "mlp": 2, "deep": 3}[arch]
    out = {}
    for horizon, prep in ((1, prep1), (7, prep7)):
        run_step(f"4.{step_number}", f"{title}: горизонт {horizon} шаг(ов)")
        print()
        best = _search_best(arch, configs, prep, title)
        trainer = best["trainer"]
        preds, metrics = _full_predictions(prep, trainer)

        if horizon > 1:
            per_h = metrics_by_horizon(
                prep["segments"]["test"]["Y_raw"], preds["test"])
            mean_m = mean_horizon_metric(per_h)
        else:
            per_h = None
            mean_m = metrics["test"]

        params = {k: v for k, v in best["config"].items()
                  if k not in TRAIN_KEYS}
        params.update(_trainer_kwargs(best["config"]))
        out[f"{arch}_h{horizon}"] = {
            "arch": arch, "title": title, "horizon": horizon,
            "predictions": preds, "metrics": metrics,
            "mean_metrics": mean_m, "metrics_by_horizon": per_h,
            "history": best["history"], "config": best["config"],
            "params_text": _short_config(arch, best["config"]),
            "n_parameters": trainer.n_parameters(),
            "val_mse": best["val_metrics"]["MSE"],
            "trainer": trainer,
            "prep": prep,
        }
        m = mean_m
        print(f"      лучшая конфигурация: {_short_config(arch, best['config'])}")
        print(f"      параметров: {trainer.n_parameters()}, "
              f"эпох: {len(best['history'])}")
        print_metrics("test", m)
        if per_h is not None:
            print("      метрики по шагам горизонта:")
            print(per_h.round(4).to_string(index=False)
                  .replace("\n", "\n      "))
        step_done(f"val MSE = {best['val_metrics']['MSE']:.4f}, "
                  f"test MAE = {m['MAE']:.4f}")
    return out


# ----------------------------------------------------------------------
# Графики
# ----------------------------------------------------------------------
def _plot_learning_curves(results: dict, arch: str, title: str,
                          variant: int) -> str:
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for col, horizon in enumerate(HORIZONS):
        res = results["models"][f"{arch}_h{horizon}"]
        hist = pd.DataFrame(res["history"])
        ax = axes[0, col]
        ax.plot(hist["epoch"], hist["train_loss"], color=TRAIN_COLOR,
                linewidth=1.5, label="train loss")
        ax.plot(hist["epoch"], hist["val_loss"], color=VAL_COLOR,
                linewidth=1.5, label="val loss")
        ax.set_title(f"{title}, горизонт {horizon}: функция потерь")
        ax.set_xlabel("Эпоха")
        ax.set_ylabel(f"loss ({res['config']['loss']})")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend()

        ax = axes[1, col]
        ax.plot(hist["epoch"], hist["train_mae"], color=TRAIN_COLOR,
                linewidth=1.5, label="train MAE")
        ax.plot(hist["epoch"], hist["val_mae"], color=VAL_COLOR,
                linewidth=1.5, label="val MAE")
        ax.set_title(f"{title}, горизонт {horizon}: MAE")
        ax.set_xlabel("Эпоха")
        ax.set_ylabel("MAE (масштабированные единицы)")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend()
    fig.suptitle(f"Кривые обучения: {title} (вариант {variant})", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return save_figure(fig, f"task4_{arch}_learning_curves")


def _plot_parameter_counts(results: dict, variant: int) -> str:
    archs = [("perceptron", "Персептрон"), ("mlp", "MLP"), ("deep", "Глубокая")]
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.35
    x = np.arange(len(archs))
    for offset, horizon in zip((-width / 2, width / 2), HORIZONS):
        counts = [results["models"][f"{a}_h{horizon}"]["n_parameters"]
                  for a, _ in archs]
        bars = ax.bar(x + offset, counts, width,
                      label=f"горизонт {horizon}")
        for bar, value in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, value,
                    f"{value}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([t for _, t in archs])
    ax.set_ylabel("Число обучаемых параметров")
    ax.set_title(f"Число внутренних параметров сетей (вариант {variant})")
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    return save_figure(fig, "task4_parameter_counts")


def _plot_multistep(results: dict, prep7: dict, variant: int) -> tuple[str, str]:
    """Прогноз на 7-м шаге и ошибки по всем шагам горизонта."""
    archs = [("perceptron", "Персептрон"), ("mlp", "MLP"), ("deep", "Глубокая")]
    fig, axes = plt.subplots(len(archs), 1, figsize=(13, 3.6 * len(archs)),
                             sharex=True)
    axes = np.atleast_1d(axes)
    y_raw = prep7["segments"]["test"]["Y_raw"]
    t = np.arange(len(y_raw))
    for ax, (arch, title) in zip(axes, archs):
        pred = results["models"][f"{arch}_h7"]["predictions"]["test"]
        ax.plot(t, y_raw[:, 6], color="black", linewidth=1.5,
                label="реальное y(t+7)")
        ax.plot(t, pred[:, 6], color=VAL_COLOR, linewidth=1.2,
                label="прогноз 7-го шага")
        m = results["models"][f"{arch}_h7"]["metrics_by_horizon"]
        ax.set_title(f"{title}: горизонт 7, MAE(7-й шаг) = "
                     f"{m.iloc[6]['MAE']:.3f}")
        ax.set_ylabel("y")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("Номер окна в тестовой выборке")
    fig.suptitle(f"Многошаговый прогноз на 7 шагов (вариант {variant})",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path_forecast = save_figure(fig, "task4_multistep_forecast")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for arch, title in archs:
        table = results["models"][f"{arch}_h7"]["metrics_by_horizon"]
        axes[0].plot(table["горизонт"], table["MAE"], "o-", label=title)
        axes[1].plot(table["горизонт"], table["MSE"], "o-", label=title)
    for ax, metric in zip(axes, ("MAE", "MSE")):
        ax.set_xlabel("Шаг горизонта")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} по шагам прогноза (test)")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend()
    fig.suptitle(f"Ошибки многошагового прогноза (вариант {variant})",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path_errors = save_figure(fig, "task4_multistep_errors")
    return path_forecast, path_errors


# ----------------------------------------------------------------------
# Пункт 4 целиком
# ----------------------------------------------------------------------
def run_neural_networks(ds, prep1: dict, prep7: dict, variant: int) -> dict:
    """Выполняет пункт 4 методички."""
    results: dict = {"models": {}, "variant": variant}

    print(f"      фреймворк: PyTorch {torch.__version__}, устройство: CPU")
    print(f"      train/val/test (горизонт 1): "
          f"{len(prep1['segments']['train']['Y_raw'])} / "
          f"{len(prep1['segments']['val']['Y_raw'])} / "
          f"{len(prep1['segments']['test']['Y_raw'])}")
    print(f"      train/val/test (горизонт 7): "
          f"{len(prep7['segments']['train']['Y_raw'])} / "
          f"{len(prep7['segments']['val']['Y_raw'])} / "
          f"{len(prep7['segments']['test']['Y_raw'])}")

    results["models"].update(_fit_architecture(
        "perceptron", "Однослойный персептрон", perceptron_configs(),
        prep1, prep7))
    results["models"].update(_fit_architecture(
        "mlp", "Многослойный персептрон", mlp_configs(), prep1, prep7))
    results["models"].update(_fit_architecture(
        "deep", "Глубокая нейронная сеть", deep_configs(), prep1, prep7))

    # --- графики ---
    figs = {}
    for arch, title in (("perceptron", "Однослойный персептрон"),
                        ("mlp", "Многослойный персептрон"),
                        ("deep", "Глубокая нейронная сеть")):
        figs[f"curves_{arch}"] = _plot_learning_curves(
            results, arch, title, variant)
    figs["params"] = _plot_parameter_counts(results, variant)
    figs["forecast"], figs["errors"] = _plot_multistep(results, prep7, variant)
    results["figures"] = figs

    # --- сводка по параметрам и метрикам ---
    rows = []
    for key, res in results["models"].items():
        m = res["mean_metrics"]
        rows.append({
            "модель": f"{res['title']} (h={res['horizon']})",
            "горизонт": res["horizon"],
            "параметров": res["n_parameters"],
            "MAE test": m["MAE"], "MSE test": m["MSE"],
            "corr test": m["corr"],
            "конфигурация": res["params_text"],
        })
    table = pd.DataFrame(rows)
    results["summary"] = table
    print()
    print(table.round(4).to_string(index=False).replace("\n", "\n      "))
    print()
    print(f"      графики: {', '.join(os.path.basename(p) for p in figs.values())}")
    return results
