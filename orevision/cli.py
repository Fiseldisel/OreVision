"""Пакетная обработка изображений без участия пользователя.

Примеры:
  python -m orevision.cli --input "D:/шлифы/панорама.tif" --pdf
  python -m orevision.cli --input "D:/шлифы/партия_2026_07/" --out results/batch1 --pdf

Для каждого изображения в папке вывода создаётся подпапка с артефактами:
  overlay.jpg      — цветовая маска классов поверх изображения
  confidence.jpg   — карта уверенности модели
  report.pdf       — одностраничный отчёт (при --pdf)
Плюс сводные:  summary.csv, run_params.json, cli_log.txt
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from orevision.batch import process_batch
from orevision.config import load_config
from orevision.data import IMG_EXTS
from orevision.predict import Predictor

log = logging.getLogger("orevision.cli")


def collect_inputs(inp: Path) -> list[Path]:
    if inp.is_file():
        return [inp]
    if inp.is_dir():
        return sorted(
            p for p in inp.iterdir()
            if p.is_file() and p.suffix.lower() in IMG_EXTS
        )
    raise FileNotFoundError(inp)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", required=True, help="файл или папка с изображениями")
    parser.add_argument("--checkpoint", default=None, help="путь к .pt (по умолчанию models/base.pt)")
    parser.add_argument("--out", default=None, help="папка результатов (по умолчанию results/<дата>)")
    parser.add_argument("--mode", default="auto", choices=["auto", "photo", "panorama"])
    parser.add_argument("--config", default=None)
    parser.add_argument("--pdf", action="store_true", help="генерировать PDF-отчёты")
    parser.add_argument("--no-overlay", action="store_true", help="не сохранять оверлеи (быстрее)")
    parser.add_argument("--device", default=None, help="cuda | cpu")
    args = parser.parse_args()

    cfg = load_config(args.config)
    from orevision.model import default_checkpoint, migrate_legacy_checkpoint

    migrate_legacy_checkpoint(cfg["train"]["out_dir"])
    ckpt = Path(args.checkpoint) if args.checkpoint else default_checkpoint(cfg["train"]["out_dir"])
    if not ckpt.exists():
        sys.exit(f"Чекпойнт не найден: {ckpt}. Сначала обучите модель: python -m orevision.train")

    out_dir = (
        Path(args.out)
        if args.out
        else Path(cfg["report"]["out_dir"]) / time.strftime("run_%Y%m%d_%H%M%S")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(out_dir / "cli_log.txt", encoding="utf-8"),
        ],
    )

    files = collect_inputs(Path(args.input))
    if not files:
        sys.exit("Входных изображений не найдено")
    log.info("Файлов к обработке: %d, вывод: %s", len(files), out_dir)

    predictor = Predictor(ckpt, cfg, device=args.device)
    rows = process_batch(
        files, out_dir, predictor, cfg,
        mode=args.mode, overlays=not args.no_overlay, pdf=args.pdf,
    )
    ok = sum(1 for r in rows if r.get("verdict") != "ERROR")
    log.info("Готово: %d/%d успешно. Сводка: %s", ok, len(rows), out_dir / "summary.csv")


if __name__ == "__main__":
    main()
