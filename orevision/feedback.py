"""Набор дообучения (active learning): примеры, исправленные экспертом.

Геолог отмечает ошибочно классифицированные изображения/участки в интерфейсе —
они сохраняются в data/feedback/<класс>/ и автоматически подхватываются
аудитом данных (build_manifest) как дополнительный источник обучения.
Журнал data/feedback/feedback_log.csv хранит происхождение каждого примера.
"""

from __future__ import annotations

import csv
import hashlib
import io
import time
from pathlib import Path

from PIL import Image

FEEDBACK_DIRNAME = "feedback"
LOG_NAME = "feedback_log.csv"
LOG_FIELDS = [
    "saved_as",       # относительный путь внутри data/feedback
    "label",          # класс, назначенный экспертом
    "prev_pred",      # что предсказывала модель
    "source",         # исходный файл
    "mode",           # photo | panorama
    "tile",           # "r,c" для тайла панорамы, "" для целого фото
    "checkpoint",     # какая модель ошибалась
    "timestamp",
]


def feedback_root(cfg: dict) -> Path:
    """data/feedback рядом с манифестом (data/ в корне проекта)."""
    return Path(cfg["data"]["manifest"]).parent / FEEDBACK_DIRNAME


def save_feedback_sample(
    image: Image.Image,
    label: str,
    cfg: dict,
    *,
    prev_pred: str = "",
    source: str = "",
    mode: str = "",
    tile: str = "",
    checkpoint: str = "",
    max_side: int | None = None,
) -> Path:
    """Сохраняет пример в набор дообучения и пишет строку в журнал."""
    assert label in cfg["classes"]["order"], f"неизвестный класс: {label}"
    root = feedback_root(cfg)
    (root / label).mkdir(parents=True, exist_ok=True)

    im = image.convert("RGB")
    limit = max_side or int(cfg["data"]["cache_max_side"])
    if max(im.size) > limit:
        im = im.copy()
        im.thumbnail((limit, limit), Image.LANCZOS)

    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=92)
    digest = hashlib.md5(buf.getvalue()).hexdigest()[:10]
    name = f"fb_{time.strftime('%Y%m%d_%H%M%S')}_{digest}.jpg"
    dst = root / label / name
    dst.write_bytes(buf.getvalue())

    log = root / LOG_NAME
    new = not log.exists()
    with open(log, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if new:
            w.writeheader()
        w.writerow(
            {
                "saved_as": f"{label}/{name}",
                "label": label,
                "prev_pred": prev_pred,
                "source": source,
                "mode": mode,
                "tile": tile,
                "checkpoint": checkpoint,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
    return dst


def feedback_stats(cfg: dict) -> dict:
    """Количество собранных примеров по классам."""
    root = feedback_root(cfg)
    return {
        c: len(list((root / c).glob("*.jpg"))) if (root / c).is_dir() else 0
        for c in cfg["classes"]["order"]
    }


def feedback_sources(cfg: dict) -> list[dict]:
    """Источники для сканера датасета — только непустые классы."""
    root = feedback_root(cfg)
    out = []
    for c in cfg["classes"]["order"]:
        d = root / c
        if d.is_dir() and any(d.glob("*.jpg")):
            out.append({"dir": d, "class": c, "part": "feedback"})
    return out


def native_tile_crop(
    source, row: int, col: int, tile: int, stride: int, analysis_size: tuple
) -> Image.Image:
    """Вырезает тайл (row, col) из исходника в нативном разрешении.

    source: путь к файлу либо bytes загруженного изображения.
    Координаты восстанавливаются той же логикой, что при анализе.
    """
    from orevision.data import open_rgb
    from orevision.predict import _tile_origins

    if isinstance(source, (bytes, bytearray)):
        from PIL import ImageOps

        im = Image.open(io.BytesIO(source))
        im = ImageOps.exif_transpose(im).convert("RGB")
    else:
        im = open_rgb(source)

    W, H = analysis_size
    if im.size != (W, H):
        # фото анализировалось в уменьшенном масштабе — приводим к нему,
        # чтобы координаты тайлов совпали (пропорции идентичны исходнику)
        im = im.resize((W, H), Image.LANCZOS)
    xs = _tile_origins(W, tile, stride)
    ys = _tile_origins(H, tile, stride)
    x0, y0 = xs[col], ys[row]
    return im.crop((x0, y0, min(x0 + tile, W), min(y0 + tile, H)))
