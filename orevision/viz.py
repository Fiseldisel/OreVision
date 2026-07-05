"""Визуализация: цветовой оверлей классов и карта уверенности.

Композиция выполняется в целевом (уменьшенном) размере — панорамы в сотни
мегапикселей не разворачиваются в полном разрешении.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from orevision.config import class_names
from orevision.predict import AnalysisResult, _tile_origins

DISPLAY_MAX_SIDE = 2600   # размер картинки для интерфейса
EXPORT_MAX_SIDE = 6000    # размер экспортируемого оверлея


def _class_color_map(cfg: dict) -> np.ndarray:
    """(n_classes, 3) uint8."""
    return np.array(
        [cfg["classes"]["colors"][c] for c in class_names(cfg)], dtype=np.uint8
    )


def _target_size(result: AnalysisResult, max_side: int) -> tuple[int, int]:
    W, H = result.analysis_size
    scale = min(1.0, max_side / max(W, H))
    return max(1, round(W * scale)), max(1, round(H * scale))


def _tile_map_image(
    result: AnalysisResult, values: np.ndarray, target: tuple[int, int]
) -> Image.Image:
    """Разворачивает карту тайлов (r, c, ch) в изображение размера target.

    Координаты тайлов масштабируются из analysis_size в target, поэтому
    границы совпадают с изображением без сдвига (включая прижатые к краю
    последние ряды/столбцы).
    """
    W, H = result.analysis_size
    tw, th = target
    sx, sy = tw / W, th / H
    xs = _tile_origins(W, result.tile, result.stride)
    ys = _tile_origins(H, result.tile, result.stride)
    canvas = np.zeros((th, tw, values.shape[-1]), dtype=values.dtype)
    for r, y in enumerate(ys):
        y0, y1 = int(y * sy), min(th, int(np.ceil((y + result.tile) * sy)))
        for c, x in enumerate(xs):
            x0, x1 = int(x * sx), min(tw, int(np.ceil((x + result.tile) * sx)))
            canvas[y0:y1, x0:x1] = values[r, c]
    return Image.fromarray(canvas)


def _composite(
    base: Image.Image,
    result: AnalysisResult,
    rgba_tiles: np.ndarray,
    max_side: int,
) -> Image.Image:
    target = _target_size(result, max_side)
    if base.size != target:
        base = base.resize(target, Image.BILINEAR)
    over = _tile_map_image(result, rgba_tiles, target)
    out = base.convert("RGBA")
    out.alpha_composite(over)
    return out.convert("RGB")


def make_overlay(
    base: Image.Image,
    result: AnalysisResult,
    cfg: dict,
    alpha: float = 0.45,
    max_side: int = DISPLAY_MAX_SIDE,
) -> Image.Image:
    """Цветовая маска классов поверх изображения (фон-тайлы не закрашиваются)."""
    colors = _class_color_map(cfg)
    tc = result.tile_classes
    rgb = np.zeros((*tc.shape, 3), dtype=np.uint8)
    mask = tc >= 0
    rgb[mask] = colors[tc[mask]]
    a = (mask.astype(np.uint8) * int(alpha * 255))[..., None]
    rgba = np.concatenate([rgb, a], axis=-1)
    return _composite(base, result, rgba, max_side)


def make_confidence_map(
    base: Image.Image,
    result: AnalysisResult,
    alpha: float = 0.55,
    max_side: int = DISPLAY_MAX_SIDE,
) -> Image.Image:
    """Тепловая карта уверенности (max softmax): жёлтое — уверенно, тёмное — спорно."""
    from matplotlib import colormaps

    conf = result.tile_conf.astype(np.float32)
    colored = (colormaps["viridis"](conf)[..., :3] * 255).astype(np.uint8)
    a = ((result.tile_classes >= 0).astype(np.uint8) * int(alpha * 255))[..., None]
    rgba = np.concatenate([colored, a], axis=-1)
    return _composite(base, result, rgba, max_side)


def legend_rows(cfg: dict) -> list[tuple[str, tuple[int, int, int]]]:
    """(русское имя, цвет) для легенды интерфейса/отчёта."""
    return [
        (cfg["classes"]["display"][c], tuple(cfg["classes"]["colors"][c]))
        for c in class_names(cfg)
    ]


def confusion_matrix_figure(cm: np.ndarray, labels: list[str]):
    """Фигура матрицы ошибок; используется обучением и вкладкой «О модели»."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Предсказано")
    ax.set_ylabel("Истина")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    return fig


def top_uncertain_tiles(
    base: Image.Image,
    result: AnalysisResult,
    cfg: dict,
    k: int = 6,
) -> list[dict]:
    """k участков с наименьшей уверенностью модели — «куда посмотреть глазами».

    Возвращает [{crop, caption, row, col, pred}] — координаты нужны для
    экспертной коррекции (вырезка тайла в нативном разрешении). base —
    изображение в масштабе display_thumb/анализа.
    """
    W, H = result.analysis_size
    sx, sy = base.width / W, base.height / H
    xs = _tile_origins(W, result.tile, result.stride)
    ys = _tile_origins(H, result.tile, result.stride)

    fg = result.tile_classes >= 0
    if not fg.any():
        return []
    conf = np.where(fg, result.tile_conf, np.inf)
    order = np.argsort(conf, axis=None)[: min(k, int(fg.sum()))]

    names = class_names(cfg)
    out = []
    for flat in order:
        r, c = np.unravel_index(flat, conf.shape)
        x0 = int(xs[c] * sx)
        y0 = int(ys[r] * sy)
        x1 = min(base.width, int((xs[c] + result.tile) * sx))
        y1 = min(base.height, int((ys[r] + result.tile) * sy))
        cls = names[result.tile_classes[r, c]]
        out.append(
            {
                "crop": base.crop((x0, y0, x1, y1)),
                "caption": (
                    f"{cfg['classes']['display'][cls]} — уверенность "
                    f"{100 * result.tile_conf[r, c]:.0f}% (ряд {r + 1}, столбец {c + 1})"
                ),
                "row": int(r),
                "col": int(c),
                "pred": cls,
            }
        )
    return out
