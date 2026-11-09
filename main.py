"""Лабораторная работа №3: регрессия на временном ряду.

Точка входа. Последовательно выполняет пункты методички:
    1   предварительная обработка и базовые модели (линейная/нелинейная)
    2-3 одиночные методы и ансамбли с подбором гиперпараметров
    4   нейронные сети (1 шаг и 7 шагов)
    5   обработка результатов, сравнение и выводы

Примеры запуска:
    python main.py
    python main.py --variant 7
    python main.py --sections report
    python main.py --sections baseline,ml
    python main.py --list-variants
"""

import argparse
import os
import sys
import time

from src.config import (DATASET_PATH, HORIZONS, OUTPUT_DIR, SEED, VARIANT,
                        banner, force_utf8_stdio, set_seeds)
from src.data_loader import (load_dataset, prepare_dataset, print_preparation,
                             print_report, segment_sizes)
from src.io_utils import available_variants, output_files

SECTION_TITLES = {
    "baseline": "Пункт 1. Предварительная обработка, линейная и нелинейная "
                "регрессия",
    "ml": "Пункты 2-3. Одиночные методы и ансамбли, подбор гиперпараметров",
    "nn": "Пункт 4. Нейронные сети: 1 шаг и 7 шагов",
    "report": "Пункт 5. Обработка результатов и итоговое сравнение",
}
ALL_SECTIONS = ["baseline", "ml", "nn", "report"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Лабораторная работа №3: регрессия на временном ряду",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--variant", type=int, default=VARIANT,
                        help="номер варианта из датасета")
    parser.add_argument("--data", default=DATASET_PATH,
                        help="путь к xlsx с вариантами")
    parser.add_argument("--sections", default="all",
                        help="baseline,ml,nn,report или all; можно через запятую")
    parser.add_argument("--seed", type=int, default=SEED,
                        help="начальное значение генератора случайных чисел")
    parser.add_argument("--list-sections", action="store_true",
                        help="показать доступные разделы и выйти")
    parser.add_argument("--list-variants", action="store_true",
                        help="показать номера вариантов в датасете и выйти")
    return parser


def parse_sections(raw: str, parser: argparse.ArgumentParser) -> list[str]:
    raw = raw.strip().lower().replace(" ", "")
    if raw in ("all", "все"):
        return list(ALL_SECTIONS)

    aliases = {"1": "baseline", "2": "ml", "3": "ml", "4": "nn", "5": "report"}
    requested = []
    for part in raw.split(","):
        if not part:
            continue
        name = aliases.get(part, part)
        if name not in ALL_SECTIONS:
            parser.error(f"неизвестный раздел {part!r}; "
                         f"доступно: {', '.join(ALL_SECTIONS)} или all")
        if name not in requested:
            requested.append(name)
    if not requested:
        parser.error("не выбран ни один раздел")
    return requested


def _list_sections() -> None:
    print("Доступные разделы (--sections):")
    for name in ALL_SECTIONS:
        print(f"  {name:<9} {SECTION_TITLES[name]}")
    print(f"  {'all':<9} все разделы последовательно")


def _list_variants(path: str) -> int:
    if not os.path.exists(path):
        print(f"Файл не найден: {path}", file=sys.stderr)
        return 1
    variants = available_variants(path)
    print(f"В файле {path} найдено вариантов: {len(variants)}")
    print("  " + ", ".join(str(v) for v in variants))
    return 0


def _print_header(ds, variant: int, seed: int) -> None:
    banner("Лабораторная работа №3. Регрессия на временном ряду")
    print(f"      Python          : {sys.version.split()[0]}")
    print(f"      платформа       : {sys.platform}")
    print(f"      вариант         : {variant}")
    print(f"      seed            : {seed}")
    print(f"      датасет         : {ds.report['файл']}")
    print(f"      строк           : {ds.n_samples}")
    print(f"      признаки        : {', '.join(ds.feature_names)}")
    print(f"      горизонты       : {', '.join(str(h) for h in HORIZONS)} шаг(ов)")
    print(f"      каталог вывода  : {os.path.abspath(OUTPUT_DIR)}")


def _run_baseline(ds, prep1, variant):
    from src.baseline import run_baseline
    banner(SECTION_TITLES["baseline"])
    return run_baseline(ds, prep1, variant)


def _run_ml(ds, prep1, variant):
    from src.ml_models import run_ml_models
    banner(SECTION_TITLES["ml"])
    return run_ml_models(ds, prep1, variant)


def _run_nn(ds, prep1, prep7, variant):
    from src.neural_networks import run_neural_networks
    banner(SECTION_TITLES["nn"])
    return run_neural_networks(ds, prep1, prep7, variant)


def _run_report(ds, prep1, variant, results: dict):
    from src.evaluation import run_evaluation
    banner(SECTION_TITLES["report"])
    return run_evaluation(ds, prep1, results["baseline"], results["ml"],
                          results["nn"], variant)


def main(argv: list[str] | None = None) -> int:
    force_utf8_stdio()
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_sections:
        _list_sections()
        return 0
    if args.list_variants:
        return _list_variants(args.data)

    requested = parse_sections(args.sections, parser)
    if not os.path.exists(args.data):
        print(f"Файл с данными не найден: {args.data}", file=sys.stderr)
        print("Укажите путь флагом --data или положите Lab3_Dataset.xlsx "
              "в data/.", file=sys.stderr)
        return 1

    set_seeds(args.seed)
    try:
        ds = load_dataset(args.data, args.variant)
    except (KeyError, ValueError) as exc:
        print(f"Не удалось загрузить вариант {args.variant}: {exc}",
              file=sys.stderr)
        return 1

    _print_header(ds, args.variant, args.seed)
    print_report(ds)

    prep1 = prepare_dataset(ds, horizon=1)
    prep7 = prepare_dataset(ds, horizon=7)
    banner("Подготовка данных")
    print_preparation(prep1)
    sizes7 = segment_sizes(prep7)
    print(f"      горизонт 7: окна train/val/test = {sizes7['train']} / "
          f"{sizes7['val']} / {sizes7['test']}")

    results: dict = {}
    started = time.perf_counter()

    if "baseline" in requested:
        results["baseline"] = _run_baseline(ds, prep1, args.variant)
    if "ml" in requested:
        results["ml"] = _run_ml(ds, prep1, args.variant)
    if "nn" in requested:
        results["nn"] = _run_nn(ds, prep1, prep7, args.variant)

    if "report" in requested:
        for name, runner in (
            ("baseline", lambda: _run_baseline(ds, prep1, args.variant)),
            ("ml", lambda: _run_ml(ds, prep1, args.variant)),
            ("nn", lambda: _run_nn(ds, prep1, prep7, args.variant)),
        ):
            if name not in results:
                print(f"\nДля пункта 5 требуется раздел {name!r}: "
                      f"запускаем его автоматически.")
                results[name] = runner()
        results["report"] = _run_report(ds, prep1, args.variant, results)

    elapsed = time.perf_counter() - started
    banner("Готово")
    print(f"      вариант        : {args.variant}")
    print(f"      разделы         : {', '.join(requested)}")
    print(f"      время           : {elapsed:.1f} с")
    print(f"      каталог вывода  : {os.path.abspath(OUTPUT_DIR)}")
    n_png, n_xlsx, n_other = output_files()
    print(f"      файлов в output : {n_png} png, {n_xlsx} xlsx, "
          f"{n_other} прочих")

    if "report" in results:
        report = results["report"]
        summary = report["summary"]
        best = summary.iloc[0]
        worst = summary.iloc[-1]
        print(f"      лучший метод    : {best['Метод']} "
              f"(MSE test = {best['MSE test']:.4f})")
        print(f"      худший метод    : {worst['Метод']} "
              f"(MSE test = {worst['MSE test']:.4f})")
        print(f"      итоговая таблица: "
              f"{os.path.basename(report['xlsx_fixed'])}")
        print(f"      выводы          : "
              f"{os.path.basename(report['conclusions_path'])}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nВыполнение прервано пользователем.", file=sys.stderr)
        raise SystemExit(130)
