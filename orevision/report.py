"""Отчёты: таблица метрик, CSV, PDF-заключение, лог параметров запуска."""

from __future__ import annotations

import json
import platform
from datetime import datetime
from pathlib import Path

import pandas as pd
from PIL import Image

from orevision.config import class_names, snapshot
from orevision.predict import AnalysisResult
from orevision.viz import legend_rows

# ------------------------------------------------------------------ таблицы


def metrics_table(result: AnalysisResult, cfg: dict) -> pd.DataFrame:
    """Таблица количественных метрик для интерфейса и экспорта."""
    rows = []
    for c in class_names(cfg):
        rows.append(
            {
                "Показатель": f"Доля площади: {cfg['classes']['display'][c].lower()}",
                "Значение": f"{100 * result.fractions[c]:.1f}%",
            }
        )
    sulfide = result.fractions["ordinary"] + result.fractions["hard"]
    rows.append({"Показатель": "Суммарно сульфидные срастания", "Значение": f"{100 * sulfide:.1f}%"})
    rows.append({"Показатель": "Доля талька", "Значение": f"{100 * result.talc_fraction:.1f}%"})
    for c in class_names(cfg):
        rows.append(
            {
                "Показатель": f"Средняя вероятность: {cfg['classes']['display'][c].lower()}",
                "Значение": f"{100 * result.probs[c]:.1f}%",
            }
        )
    rows.append({"Показатель": "Проанализировано тайлов", "Значение": str(result.n_tiles)})
    rows.append({"Показатель": "Тайлов фона исключено", "Значение": str(result.n_background)})
    rows.append({"Показатель": "Режим анализа", "Значение": result.mode})
    rows.append({"Показатель": "Время анализа, с", "Значение": f"{result.elapsed_sec:.1f}"})
    return pd.DataFrame(rows)


def result_to_row(result: AnalysisResult, cfg: dict) -> dict:
    """Строка сводного CSV по партии изображений."""
    row = {
        "file": result.source,
        "verdict": result.verdict,
        "verdict_ru": result.verdict_display,
        "mode": result.mode,
        "width": result.image_size[0],
        "height": result.image_size[1],
    }
    for c in class_names(cfg):
        row[f"frac_{c}"] = round(result.fractions[c], 4)
    for c in class_names(cfg):
        row[f"prob_{c}"] = round(result.probs[c], 4)
    row["talc_fraction"] = round(result.talc_fraction, 4)
    row["needs_review"] = bool(result.needs_review)
    row["n_tiles"] = result.n_tiles
    row["n_background"] = result.n_background
    row["elapsed_sec"] = result.elapsed_sec
    row["conclusion"] = result.conclusion
    return row


def save_csv(rows: list[dict], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    return path


# ------------------------------------------------------------------ PDF

def save_pdf(
    result: AnalysisResult,
    cfg: dict,
    path: str | Path,
    original: Image.Image | None = None,
    overlay: Image.Image | None = None,
    confidence: Image.Image | None = None,
) -> Path:
    """Одностраничный PDF-отчёт: изображения, метрики, заключение."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    if isinstance(path, (str, Path)):  # иначе file-like буфер (BytesIO)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(11.7, 8.3))  # A4 landscape
    gs = fig.add_gridspec(
        2, 3, height_ratios=[1.15, 1.0], hspace=0.28, wspace=0.12,
        left=0.04, right=0.97, top=0.88, bottom=0.05,
    )
    fig.suptitle(
        f"OreVision — отчёт по образцу «{result.source}»",
        fontsize=15, fontweight="bold",
    )
    review_note = (
        "   |   ⚠ пограничный случай — рекомендована проверка экспертом"
        if result.needs_review else ""
    )
    fig.text(
        0.04, 0.905,
        f"Вердикт: {result.verdict_display}   |   режим: {result.mode}{review_note}",
        fontsize=11,
    )
    fig.text(
        0.04, 0.015,
        f"Сформировано OreVision {datetime.now():%d.%m.%Y %H:%M} · "
        f"изображение {result.image_size[0]}×{result.image_size[1]} px · "
        f"тайл {result.tile} px · время анализа {result.elapsed_sec:.1f} с",
        fontsize=7.5, color="#555555",
    )

    def show(ax, img, title):
        if img is not None:
            ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    show(fig.add_subplot(gs[0, 0]), original, "Исходное изображение")
    show(fig.add_subplot(gs[0, 1]), overlay, "Карта классов (оверлей)")
    show(fig.add_subplot(gs[0, 2]), confidence, "Карта уверенности модели")

    # легенда цветов
    handles = [
        Patch(color=[v / 255 for v in color], label=name)
        for name, color in legend_rows(cfg)
    ]
    fig.legend(handles=handles, loc="upper right", fontsize=9, ncol=3,
               bbox_to_anchor=(0.97, 0.945))

    # таблица метрик
    ax_t = fig.add_subplot(gs[1, 0:2])
    ax_t.axis("off")
    df = metrics_table(result, cfg)
    table = ax_t.table(
        cellText=df.values, colLabels=df.columns,
        loc="center", cellLoc="left", colWidths=[0.62, 0.25],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.25)

    # заключение
    ax_c = fig.add_subplot(gs[1, 2])
    ax_c.axis("off")
    ax_c.set_title("Заключение", fontsize=10, loc="left")
    import textwrap

    ax_c.text(
        0.0, 0.95, "\n".join(textwrap.wrap(result.conclusion, 38)),
        fontsize=9.5, va="top", wrap=True,
    )

    fig.savefig(path, format="pdf")
    plt.close(fig)
    return path


# ------------------------------------------------------------- воспроизводимость

def save_run_params(
    out_dir: str | Path,
    cfg: dict,
    checkpoint_info: dict,
    files: list[str],
    extra: dict | None = None,
) -> Path:
    """Журнал параметров запуска — для воспроизводимости анализа."""
    import torch

    out = Path(out_dir) / "run_params.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "checkpoint": {
            "arch": checkpoint_info.get("arch"),
            "img_size": checkpoint_info.get("img_size"),
            "classes": checkpoint_info.get("classes"),
            "val_metrics": checkpoint_info.get("meta", {}).get("metrics", {}),
        },
        "config": snapshot(cfg),
        "files": files,
        **(extra or {}),
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out
