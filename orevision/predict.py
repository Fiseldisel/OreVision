"""Инференс-движок OreVision.

Один механизм — тайловый анализ — работает в двух режимах:

  * photo    — обычная микрофотография. Изображение приводится к масштабу
               обучения (max_side), разбивается на перекрывающиеся кропы;
               вердикт = усреднение вероятностей по кропам (TTA).
  * panorama — гигапиксельная панорама шлифа. Тайлы 1024 px в нативном
               разрешении; вердикт по экспертному правилу (ТЗ):
                 доля площади талька > 10%  -> оталькованная руда,
                 иначе преобладание обычных/тонких срастаний.

Для обоих режимов строится карта классов тайлов (интерпретация) и карта
уверенности модели.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from orevision.config import class_names, display_name
from orevision.data import open_rgb
from orevision.model import load_checkpoint, pick_device
from orevision.transforms import tensor_transform

log = logging.getLogger("orevision.predict")


@dataclass
class AnalysisResult:
    """Результат анализа одного изображения."""

    source: str
    mode: str                      # photo | panorama
    verdict: str                   # ключ класса
    verdict_display: str           # русское имя
    probs: dict                    # средние вероятности по классам
    fractions: dict                # доли площади по классам (по тайлам)
    talc_fraction: float
    talc_threshold: float
    image_size: tuple              # (W, H) исходного изображения
    analysis_size: tuple           # (W, H) на котором шёл анализ
    tile: int
    stride: int
    grid_shape: tuple              # (rows, cols)
    tile_classes: np.ndarray = field(repr=False)  # (r, c) int, -1 = фон
    tile_conf: np.ndarray = field(repr=False)     # (r, c) float, max prob
    tile_probs: np.ndarray = field(repr=False)    # (r, c, n_classes)
    n_tiles: int = 0
    n_background: int = 0
    elapsed_sec: float = 0.0
    conclusion: str = ""
    needs_review: bool = False   # пограничный случай — рекомендована проверка экспертом
    display_thumb: Image.Image | None = field(default=None, repr=False)


UNKNOWN_VERDICT = "unknown"
UNKNOWN_DISPLAY = "Не определено"


def _needs_review(r: "AnalysisResult") -> bool:
    """Пограничные случаи, где вердикту не стоит доверять слепо.

    Панорама: доли конкурирующих классов почти равны либо доля талька
    в пределах ±3 п.п. от порога. Фото: максимальная вероятность < 60%.
    """
    if r.verdict == UNKNOWN_VERDICT:
        return True
    if r.mode == "panorama":
        close_race = abs(
            r.fractions.get("ordinary", 0.0) - r.fractions.get("hard", 0.0)
        ) < 0.10 and r.verdict != "talc"
        near_threshold = abs(r.talc_fraction - r.talc_threshold) < 0.03
        return close_race or near_threshold
    return max(r.probs.values()) < 0.60


def _conclusion_text(r: "AnalysisResult") -> str:
    """Текстовое заключение в формате ТЗ."""
    if r.verdict == UNKNOWN_VERDICT:
        return (
            "Пригодных для анализа участков не найдено: изображение практически "
            "однотонное (фон/эпоксидка/пересвет). Требуется проверка образца вручную."
        )
    f = r.fractions
    pct = {c: 100.0 * v for c, v in f.items()}
    base = f"Руда классифицирована как {r.verdict_display.lower()}"
    if r.mode == "photo":
        # вердикт фото — по средним вероятностям; обоснование цитирует их же,
        # а не доли тайлов (агрегации могут расходиться на смешанных рудах)
        detail = f"вероятность класса — {100 * r.probs[r.verdict]:.0f}%"
    elif r.verdict == "talc":
        detail = (
            f"содержание талька — {pct['talc']:.0f}% "
            f"(порог {100 * r.talc_threshold:.0f}%)"
        )
    elif r.verdict == "hard":
        detail = f"преобладание тонких срастаний — {pct['hard']:.0f}%"
    else:
        detail = f"преобладание обычных срастаний — {pct['ordinary']:.0f}%"
    extra = (
        f"Доли по площади: обычные срастания {pct['ordinary']:.1f}%, "
        f"тонкие срастания {pct['hard']:.1f}%, тальк {pct['talc']:.1f}%."
    )
    return f"{base}: {detail}. {extra}"


def rescore(res: "AnalysisResult", cfg: dict, talc_threshold: float) -> "AnalysisResult":
    """Пересчитывает вердикт панорамы при изменении порога талька (без инференса).

    Для фото вердикт от порога не зависит — обновляется только текст заключения.
    """
    res.talc_threshold = float(talc_threshold)
    if res.mode == "panorama" and res.verdict != UNKNOWN_VERDICT:
        if res.fractions.get("talc", 0.0) > res.talc_threshold:
            res.verdict = "talc"
        elif res.fractions.get("ordinary", 0.0) >= res.fractions.get("hard", 0.0):
            res.verdict = "ordinary"
        else:
            res.verdict = "hard"
        res.verdict_display = display_name(cfg, res.verdict)
    res.needs_review = _needs_review(res)
    res.conclusion = _conclusion_text(res)
    return res


def _tile_origins(size: int, tile: int, stride: int) -> list[int]:
    """Координаты начал тайлов вдоль одной оси, последний прижат к краю."""
    if size <= tile:
        return [0]
    xs = list(range(0, size - tile + 1, stride))
    if xs[-1] != size - tile:
        xs.append(size - tile)
    return xs


class Predictor:
    def __init__(self, checkpoint: str | Path, cfg: dict, device=None):
        self.cfg = cfg
        self.device = torch.device(device) if device else pick_device()
        self.model, self.info = load_checkpoint(checkpoint, self.device)
        self.classes: list[str] = self.info["classes"]
        assert self.classes == class_names(cfg), (
            "Порядок классов чекпойнта не совпадает с config.yaml: "
            f"{self.classes} != {class_names(cfg)}"
        )
        self.img_size = self.info["img_size"]
        self.tf = tensor_transform(self.img_size)
        self.checkpoint_path = str(checkpoint)
        log.info(
            "Модель %s (%s), классы %s, device=%s",
            Path(checkpoint).name, self.info["arch"], self.classes, self.device,
        )

    # ------------------------------------------------------------- публичное API

    def predict(
        self,
        image: str | Path | Image.Image,
        mode: str = "auto",
        progress_cb=None,
    ) -> AnalysisResult:
        """Анализ одного изображения. mode: auto | photo | panorama."""
        t0 = time.time()
        if isinstance(image, (str, Path)):
            src_name = Path(image).name
            im = open_rgb(image)
        else:
            src_name = getattr(image, "filename", "") or "uploaded"
            im = image.convert("RGB") if image.mode != "RGB" else image

        orig_size = im.size
        icfg = self.cfg["infer"]
        if mode == "auto":
            mode = (
                "panorama"
                if im.width * im.height >= int(icfg["panorama_min_pixels"])
                else "photo"
            )

        if mode == "photo":
            # приводим к масштабу обучения
            max_side = int(icfg["photo_max_side"])
            if max(im.size) > max_side:
                if im is image:  # не трогаем объект вызывающего кода
                    im = im.copy()
                im.thumbnail((max_side, max_side), Image.LANCZOS)
            tile = self.img_size
            stride = int(tile * 0.75)
        else:
            tile = int(icfg["tile"])
            stride = tile - int(icfg.get("overlap", 0))

        result = self._analyze_tiles(im, tile, stride, progress_cb)
        probs, fractions = result["probs"], result["fractions"]

        talc_frac = fractions.get("talc", 0.0)
        thr = float(icfg["talc_threshold"])
        if result["n_tiles"] == 0:
            # всё изображение отфильтровано как фон — честно говорим
            # «не определено», а не выдаём уверенный вердикт первого класса
            verdict = UNKNOWN_VERDICT
        elif mode == "panorama":
            verdict = self._expert_rule(fractions, thr)
        else:
            verdict = self.classes[int(np.argmax([probs[c] for c in self.classes]))]

        res = AnalysisResult(
            source=src_name,
            mode=mode,
            verdict=verdict,
            verdict_display=(
                UNKNOWN_DISPLAY if verdict == UNKNOWN_VERDICT
                else display_name(self.cfg, verdict)
            ),
            probs=probs,
            fractions=fractions,
            talc_fraction=talc_frac,
            talc_threshold=thr,
            image_size=orig_size,
            analysis_size=im.size,
            tile=tile,
            stride=stride,
            grid_shape=result["grid_shape"],
            tile_classes=result["tile_classes"],
            tile_conf=result["tile_conf"],
            tile_probs=result["tile_probs"],
            n_tiles=result["n_tiles"],
            n_background=result["n_background"],
            elapsed_sec=round(time.time() - t0, 2),
        )
        res.needs_review = _needs_review(res)
        res.conclusion = _conclusion_text(res)
        # миниатюра для интерфейса — чтобы не декодировать панораму повторно
        thumb = im.copy()
        thumb.thumbnail((2600, 2600), Image.LANCZOS)
        res.display_thumb = thumb
        return res

    # ------------------------------------------------------------ внутренности

    @torch.inference_mode()
    def _analyze_tiles(self, im: Image.Image, tile: int, stride: int, progress_cb):
        icfg = self.cfg["infer"]
        xs = _tile_origins(im.width, tile, stride)
        ys = _tile_origins(im.height, tile, stride)
        rows, cols = len(ys), len(xs)
        n_classes = len(self.classes)

        tile_probs = np.zeros((rows, cols, n_classes), dtype=np.float32)
        tile_bg = np.zeros((rows, cols), dtype=bool)
        tile_area = np.zeros((rows, cols), dtype=np.float64)

        bg_std = float(icfg["background_std"])
        bg_mean = float(icfg["background_mean"])
        batch_size = int(icfg["batch_size"])

        batch, coords = [], []
        total = rows * cols
        done = 0

        def flush():
            nonlocal done
            if not batch:
                return
            x = torch.stack(batch).to(self.device, non_blocking=True)
            with torch.autocast(
                self.device.type, dtype=torch.bfloat16,
                enabled=self.device.type == "cuda",
            ):
                logits = self.model(x)
            p = torch.softmax(logits.float(), dim=1).cpu().numpy()
            for (r, c), pr in zip(coords, p):
                tile_probs[r, c] = pr
            done += len(batch)
            batch.clear()
            coords.clear()
            if progress_cb:
                progress_cb(done, total)

        for r, y in enumerate(ys):
            for c, x0 in enumerate(xs):
                # кроп строго в границах изображения: без чёрного паддинга PIL,
                # крайние тайлы меньшего размера учитываются с меньшим весом
                x1, y1 = min(x0 + tile, im.width), min(y + tile, im.height)
                crop = im.crop((x0, y, x1, y1))
                tile_area[r, c] = (x1 - x0) * (y1 - y)
                # фильтр фона: почти однотонные тёмные области (эпоксидка, край)
                small = np.asarray(
                    crop.convert("L").resize((64, 64), Image.BILINEAR), dtype=np.float32
                )
                if small.std() < bg_std and small.mean() < bg_mean:
                    tile_bg[r, c] = True
                    done += 1
                    if progress_cb:
                        progress_cb(done, total)
                    continue
                batch.append(self.tf(crop))
                coords.append((r, c))
                if len(batch) >= batch_size:
                    flush()
        flush()

        tile_classes = tile_probs.argmax(axis=2).astype(np.int16)
        tile_classes[tile_bg] = -1
        tile_conf = tile_probs.max(axis=2)
        tile_conf[tile_bg] = 0.0

        fg = ~tile_bg
        n_fg = int(fg.sum())
        if n_fg:
            # взвешивание по фактической площади тайла: крайние (обрезанные)
            # тайлы вносят пропорциональный вклад в доли и вероятности
            w = tile_area[fg]
            w = w / w.sum()
            mean_probs = (tile_probs[fg] * w[:, None]).sum(axis=0)
            fractions = {
                c: float(w[tile_classes[fg] == i].sum())
                for i, c in enumerate(self.classes)
            }
        else:  # изображение целиком «фон»
            mean_probs = np.full(n_classes, 1.0 / n_classes)
            fractions = {c: 0.0 for c in self.classes}

        return {
            "probs": {c: float(mean_probs[i]) for i, c in enumerate(self.classes)},
            "fractions": fractions,
            "tile_classes": tile_classes,
            "tile_conf": tile_conf,
            "tile_probs": tile_probs,
            "grid_shape": (rows, cols),
            "n_tiles": n_fg,
            "n_background": int(tile_bg.sum()),
        }

    def _expert_rule(self, fractions: dict, talc_threshold: float) -> str:
        """Экспертная логика классификации из ТЗ."""
        if fractions.get("talc", 0.0) > talc_threshold:
            return "talc"
        if fractions.get("ordinary", 0.0) >= fractions.get("hard", 0.0):
            return "ordinary"
        return "hard"
