"""Общий конвейер пакетной обработки — используется CLI и веб-интерфейсом."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from PIL import Image

from orevision.data import open_rgb
from orevision.predict import Predictor
from orevision.report import result_to_row, save_csv, save_pdf, save_run_params
from orevision.viz import EXPORT_MAX_SIDE, make_confidence_map, make_overlay

log = logging.getLogger("orevision.batch")


def process_one(
    predictor: Predictor,
    cfg: dict,
    path: Path,
    item_dir: Path | None = None,
    mode: str = "auto",
    overlays: bool = True,
    pdf: bool = True,
    progress_cb=None,
):
    """Анализ одного файла с сохранением артефактов. Возвращает (result, row)."""
    res = predictor.predict(path, mode=mode, progress_cb=progress_cb)
    if item_dir is not None:
        item_dir.mkdir(parents=True, exist_ok=True)
        if overlays or pdf:
            base = open_rgb(path)
            base.thumbnail((EXPORT_MAX_SIDE, EXPORT_MAX_SIDE), Image.LANCZOS)
            overlay = make_overlay(base, res, cfg, max_side=EXPORT_MAX_SIDE)
            conf = make_confidence_map(base, res, max_side=EXPORT_MAX_SIDE)
            if overlays:
                overlay.save(item_dir / "overlay.jpg", quality=90)
                conf.save(item_dir / "confidence.jpg", quality=90)
            if pdf:
                thumb = base.copy()
                thumb.thumbnail((1600, 1600))
                ov = overlay.copy()
                ov.thumbnail((1600, 1600))
                cf = conf.copy()
                cf.thumbnail((1600, 1600))
                save_pdf(res, cfg, item_dir / "report.pdf",
                         original=thumb, overlay=ov, confidence=cf)
    return res, result_to_row(res, cfg)


def process_batch(
    files: list[Path],
    out_dir: Path,
    predictor: Predictor,
    cfg: dict,
    mode: str = "auto",
    overlays: bool = True,
    pdf: bool = True,
    file_progress_cb=None,
) -> list[dict]:
    """Обрабатывает список файлов, пишет summary.csv и run_params.json."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_run_params(out_dir, cfg, predictor.info, [str(f) for f in files])

    rows = []
    used_dirs: set[str] = set()

    def unique_item_dir(stem: str) -> Path:
        # «10.jpg» и «10.tif» не должны перетирать артефакты друг друга
        name, k = stem, 2
        while name.lower() in used_dirs:
            name = f"{stem}__{k}"
            k += 1
        used_dirs.add(name.lower())
        return out_dir / name

    for i, f in enumerate(files, 1):
        t0 = time.time()
        log.info("[%d/%d] %s ...", i, len(files), f.name)
        if file_progress_cb:
            file_progress_cb(i - 1, len(files), f.name)
        try:
            res, row = process_one(
                predictor, cfg, f, item_dir=unique_item_dir(f.stem),
                mode=mode, overlays=overlays, pdf=pdf,
            )
            rows.append(row)
            log.info("    %s  (%.1f c)", res.conclusion, time.time() - t0)
        except Exception as e:
            log.exception("Ошибка на %s", f)
            rows.append({"file": f.name, "verdict": "ERROR", "conclusion": str(e)})
    if file_progress_cb:
        file_progress_cb(len(files), len(files), "готово")

    save_csv(rows, out_dir / "summary.csv")
    return rows
