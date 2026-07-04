"""Датасет: аудит, манифест, сплиты без утечки, кэш и torch Dataset.

Ключевые решения:
  * MD5-дедупликация. Точные дубликаты внутри класса -> остаётся один файл;
    точные дубликаты в РАЗНЫХ классах -> конфликт разметки, исключаются целиком.
  * Перцептивный хэш (dHash 16x16) ловит пересохранённые копии ("100.JPG" и
    "100_.jpg"). Похожие файлы объединяются в одну группу.
  * Групповой сплит: фото одной пробы (префикс "2550374-..." в имени файла,
    цепочки соседних кадров DSCN) не разносятся между train и val — иначе
    метрики завышаются из-за утечки.
  * Группы со смешанными классами никогда не попадают в val.
"""

from __future__ import annotations

import hashlib
import logging
import os
import random
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = None  # панорамы до ~200 Мпикс — отключаем защиту PIL

log = logging.getLogger("orevision.data")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

SAMPLE_ID_RE = re.compile(r"^(\d{6,8})[\s_-]")
DSCN_RE = re.compile(r"^DSCN(\d+)", re.IGNORECASE)
DSCN_CHAIN_GAP = 3   # кадры DSCN одного класса с номерами ближе gap — одна проба
DSCN_CHAIN_CAP = 6   # максимум файлов в одной DSCN-цепочке
DHASH_NEAR = 8       # порог Хэмминга для "почти дубликатов" (из 256 бит)


# ----------------------------------------------------------------- утилиты

def open_rgb(path: str | Path) -> Image.Image:
    """Открывает изображение с учётом EXIF-ориентации, в RGB."""
    im = Image.open(path)
    im = ImageOps.exif_transpose(im)
    if im.mode != "RGB":
        im = im.convert("RGB")
    return im


def _md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while data := f.read(chunk):
            h.update(data)
    return h.hexdigest()


def _dhash256(im: Image.Image) -> int:
    """dHash 16x16 (256 бит): устойчив к пересжатию/лёгкой коррекции."""
    g = im.convert("L").resize((17, 16), Image.LANCZOS)
    px = list(g.getdata())
    bits = 0
    for row in range(16):
        for col in range(16):
            left = px[row * 17 + col]
            right = px[row * 17 + col + 1]
            bits = (bits << 1) | (1 if left > right else 0)
    return bits


class _UnionFind:
    def __init__(self):
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


# ----------------------------------------------------------- построение манифеста

# Распознаваемые имена подпапок для «своего» датасета: <root>/<класс>/*.jpg
CLASS_DIR_ALIASES = {
    "ordinary": ("ordinary", "рядовая", "рядовые"),
    "hard": ("hard", "труднообогатимая", "труднообогатимые", "тонкие"),
    "talc": ("talc", "оталькованная", "оталькованные", "тальк"),
}


def _dir_class(name: str) -> str | None:
    n = name.strip().lower()
    for key, aliases in CLASS_DIR_ALIASES.items():
        for a in aliases:
            if n == a or n.startswith((a + " ", a + "_", a + "-")):
                return key
    return None


def _count_images(d: Path) -> int:
    return sum(1 for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)


def resolve_training_sources(cfg: dict) -> dict:
    """Гибко находит обучающий датасет.

    Порядок: (1) источники из config.yaml (структура датасета хакатона) —
    достаточно, чтобы присутствовали папки всех трёх классов; (2) «свой»
    датасет: подпапки root с именами классов (см. CLASS_DIR_ALIASES).

    Возвращает {"root", "layout": configured|generic|None, "sources",
    "counts": {класс: файлов}, "problems": [строки]}.
    """
    order = list(cfg["classes"]["order"])
    root = Path(cfg["data"]["root"])
    info = {"root": root, "layout": None, "sources": [], "counts": {}, "problems": []}
    if not root.is_dir():
        info["problems"].append(f"папка датасета не существует: {root}")
        return info

    # 1) структура из config.yaml
    configured = [s for s in cfg["data"]["sources"] if (root / s["dir"]).is_dir()]
    if configured and {s["class"] for s in configured} == set(order):
        missing = [s["dir"] for s in cfg["data"]["sources"] if s not in configured]
        if missing:
            log.warning("Часть источников отсутствует (используем найденные): %s", missing)
        info["layout"] = "configured"
        info["sources"] = configured
    else:
        # 2) свой датасет: папки по именам классов
        generic = []
        for d in sorted(root.iterdir()):
            if d.is_dir() and (cls := _dir_class(d.name)):
                generic.append({"dir": d.name, "class": cls, "part": "custom"})
        found = {s["class"] for s in generic}
        if found == set(order):
            info["layout"] = "generic"
            info["sources"] = generic
        else:
            lack = [c for c in order if c not in found]
            names = "; ".join(
                f"{cfg['classes']['display'][c]}: {', '.join(CLASS_DIR_ALIASES[c])}"
                for c in lack
            )
            info["problems"].append(
                "не найдены подпапки классов — ожидались имена ("
                + names + ") либо структура из config.yaml"
            )
            return info

    for cls in order:
        info["counts"][cls] = sum(
            _count_images(root / s["dir"]) for s in info["sources"] if s["class"] == cls
        )
    empty = [c for c in order if info["counts"][c] == 0]
    if empty:
        info["problems"].append(
            "нет изображений в классах: "
            + ", ".join(cfg["classes"]["display"][c] for c in empty)
        )
        info["layout"] = None
        info["sources"] = []
    return info


def scan_sources(cfg: dict) -> pd.DataFrame:
    """Обходит папки-источники и собирает базовую таблицу файлов.

    Структура датасета находится гибко (см. resolve_training_sources).
    Дополнительно подхватываются примеры, исправленные экспертом в интерфейсе
    (data/feedback/<класс>/, part="feedback") — механизм active learning.
    """
    root = Path(cfg["data"]["root"])
    resolved = resolve_training_sources(cfg)
    if not resolved["sources"]:
        raise FileNotFoundError(
            "Обучающий датасет не найден: " + "; ".join(resolved["problems"])
            + ". Путь настраивается в config.yaml -> data.root "
            "(или в config.local.yaml / через вкладку «Дообучение»)."
        )
    log.info(
        "Датасет: %s (структура: %s, классы: %s)",
        root, resolved["layout"],
        ", ".join(f"{c}={n}" for c, n in resolved["counts"].items()),
    )
    rows = []
    for src in resolved["sources"]:
        d = root / src["dir"]
        for p in sorted(d.iterdir()):
            # вложенные папки (например segmentation(...)) не сканируем
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                rows.append(
                    {
                        "path": str(p),
                        "relpath": str(p.relative_to(root)),
                        "filename": p.name,
                        "class": src["class"],
                        "part": src["part"],
                    }
                )
    from orevision.feedback import feedback_sources

    fb_total = 0
    for src in feedback_sources(cfg):
        for p in sorted(Path(src["dir"]).glob("*.jpg")):
            rows.append(
                {
                    "path": str(p),
                    "relpath": f"feedback/{src['class']}/{p.name}",
                    "filename": p.name,
                    "class": src["class"],
                    "part": "feedback",
                }
            )
            fb_total += 1
    if fb_total:
        log.info("Примеров дообучения (экспертный фидбэк): %d", fb_total)

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(
            "В папках-источниках не найдено ни одного изображения "
            f"({', '.join(sorted(IMG_EXTS))}). Проверьте data.root и data.sources "
            "в config.yaml."
        )
    log.info("Найдено файлов: %d", len(df))
    return df


def build_manifest(cfg: dict, progress: bool = True) -> pd.DataFrame:
    """Полный аудит: хэши, дубликаты, группы, сплит. Пишет manifest.csv."""
    df = scan_sources(cfg)

    md5s, dhashes, widths, heights, broken = [], [], [], [], []
    for i, path in enumerate(df["path"]):
        # значения собираются в локальные переменные и добавляются в списки
        # ровно один раз — сбой на любом шаге не рассинхронизирует колонки
        md5_v, dh_v, w_v, h_v, broken_v = "", 0, 0, 0, False
        try:
            md5_v = _md5(Path(path))
            with open_rgb(path) as im:
                w_v, h_v = im.width, im.height
                dh_v = _dhash256(im)
        except Exception as e:  # повреждённый файл не валит весь аудит
            log.warning("Не читается %s: %s", path, e)
            md5_v, dh_v, w_v, h_v, broken_v = "", 0, 0, 0, True
        md5s.append(md5_v)
        dhashes.append(dh_v)
        widths.append(w_v)
        heights.append(h_v)
        broken.append(broken_v)
        if progress and (i + 1) % 200 == 0:
            log.info("  хэшировано %d/%d", i + 1, len(df))
    df["md5"] = md5s
    df["dhash"] = [f"{h:064x}" for h in dhashes]
    df["width"] = widths
    df["height"] = heights
    df["broken"] = broken

    df["exclude"] = df["broken"]
    df["exclude_reason"] = ["broken" if b else "" for b in df["broken"]]

    # --- 1. Точные дубликаты (MD5)
    for md5, grp in df[~df["broken"]].groupby("md5"):
        if len(grp) < 2:
            continue
        classes = set(grp["class"])
        if len(classes) > 1:
            # один и тот же файл размечен в разные классы — доверять нельзя
            df.loc[grp.index, "exclude"] = True
            df.loc[grp.index, "exclude_reason"] = "md5-conflict-cross-class"
            log.warning(
                "Конфликт разметки (одинаковый файл в разных классах): %s",
                "; ".join(grp["relpath"]),
            )
        else:
            keep = grp.index[0]
            dup_idx = [i for i in grp.index if i != keep]
            df.loc[dup_idx, "exclude"] = True
            df.loc[dup_idx, "exclude_reason"] = "md5-duplicate"

    # --- 2. Группы для сплита (union-find)
    uf = _UnionFind()
    for idx in df.index:
        uf.find(f"f{idx}")

    # 2a. Префикс-ID пробы в имени файла
    for idx, name in zip(df.index, df["filename"]):
        m = SAMPLE_ID_RE.match(name)
        if m:
            uf.union(f"sample:{m.group(1)}", f"f{idx}")

    # 2b. Цепочки соседних DSCN-кадров одного класса и части
    dscn = []
    for idx, (name, cls, part) in enumerate(
        zip(df["filename"], df["class"], df["part"])
    ):
        m = DSCN_RE.match(name)
        if m:
            dscn.append((cls, part, int(m.group(1)), df.index[idx]))
    dscn.sort()
    chain_len = 1
    for (c1, p1, n1, i1), (c2, p2, n2, i2) in zip(dscn, dscn[1:]):
        if c1 == c2 and p1 == p2 and n2 - n1 <= DSCN_CHAIN_GAP and chain_len < DSCN_CHAIN_CAP:
            uf.union(f"f{i1}", f"f{i2}")
            chain_len += 1
        else:
            chain_len = 1

    # 2c. Почти-дубликаты по dHash (в т.ч. между классами и частями)
    live = df[~df["exclude"]]
    hashes = [(int(h, 16), idx) for h, idx in zip(live["dhash"], live.index)]
    near_pairs = []
    for a in range(len(hashes)):
        ha, ia = hashes[a]
        for b in range(a + 1, len(hashes)):
            hb, ib = hashes[b]
            if (ha ^ hb).bit_count() <= DHASH_NEAR:
                uf.union(f"f{ia}", f"f{ib}")
                near_pairs.append((ia, ib))
    if near_pairs:
        log.info("Почти-дубликатов (dHash<=%d): %d пар", DHASH_NEAR, len(near_pairs))
        for ia, ib in near_pairs[:20]:
            log.info("  ~ %s <-> %s", df.at[ia, "relpath"], df.at[ib, "relpath"])

    df["group"] = [uf.find(f"f{idx}") for idx in df.index]

    # --- 3. Групповой стратифицированный сплит
    df["split"] = ""
    _assign_split(df, cfg)

    out = Path(cfg["data"]["manifest"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    log.info("Манифест: %s (%d строк)", out, len(df))
    return df


def _assign_split(df: pd.DataFrame, cfg: dict) -> None:
    """Раскладывает группы в train/val так, чтобы val ≈ val_fraction по каждому классу.

    Группы со смешанными классами идут только в train. Экспертный фидбэк
    (part="feedback") тоже всегда в train: примеры отобраны по ошибкам модели
    и исказили бы валидационную метрику.
    """
    rng = random.Random(cfg["data"]["seed"])
    val_frac = float(cfg["data"]["val_fraction"])

    live = df[~df["exclude"]]
    feedback_groups = set(live.loc[live["part"] == "feedback", "group"])
    group_info = {}
    for g, grp in live.groupby("group"):
        classes = set(grp["class"])
        group_info[g] = {
            "classes": classes,
            "count": len(grp),
            "cls": (
                grp["class"].iloc[0]
                if len(classes) == 1 and g not in feedback_groups
                else None  # None = кандидат только в train
            ),
        }

    # целевой размер val считаем без фидбэка — он в val не попадает
    class_totals = live.loc[live["part"] != "feedback", "class"].value_counts().to_dict()
    val_target = {c: int(round(n * val_frac)) for c, n in class_totals.items()}
    val_current = defaultdict(int)

    pure = [(g, info) for g, info in group_info.items() if info["cls"] is not None]
    rng.shuffle(pure)
    # маленькие группы первыми заполняют val — точнее попадаем в целевую долю
    pure.sort(key=lambda x: x[1]["count"])

    val_groups = set()
    for g, info in pure:
        c = info["cls"]
        if val_current[c] + info["count"] <= val_target[c]:
            val_groups.add(g)
            val_current[c] += info["count"]

    df.loc[~df["exclude"], "split"] = [
        "val" if g in val_groups else "train" for g in live["group"]
    ]
    stats = (
        df[~df["exclude"]]
        .groupby(["class", "split"])
        .size()
        .unstack(fill_value=0)
        .to_string()
    )
    log.info("Сплит:\n%s", stats)


def load_manifest(cfg: dict) -> pd.DataFrame:
    df = pd.read_csv(cfg["data"]["manifest"], encoding="utf-8-sig", keep_default_na=False)
    # CSV хранит булевы как строки "True"/"False" — приводим явно
    for col in ("exclude", "broken"):
        df[col] = df[col].astype(str).str.lower().isin(("true", "1"))
    return df


# ------------------------------------------------------------------- кэш

def cache_path(cfg: dict, md5: str) -> Path:
    return Path(cfg["data"]["cache_dir"]) / f"{md5}.jpg"


def build_cache(cfg: dict, df: pd.DataFrame | None = None) -> None:
    """Складывает уменьшенные копии (max_side) в cache_dir. Идемпотентно."""
    if df is None:
        df = load_manifest(cfg)
    df = df[~df["exclude"].astype(bool)]
    max_side = int(cfg["data"]["cache_max_side"])
    cdir = Path(cfg["data"]["cache_dir"])
    cdir.mkdir(parents=True, exist_ok=True)
    made = 0
    for i, row in enumerate(df.itertuples()):
        dst = cache_path(cfg, row.md5)
        if dst.exists():
            continue
        with open_rgb(row.path) as im:
            im.thumbnail((max_side, max_side), Image.LANCZOS)
            # атомарная запись: прерванный запуск не оставит усечённый JPEG,
            # который exists()-проверка навсегда приняла бы за готовый
            tmp = dst.with_suffix(".tmp")
            im.save(tmp, "JPEG", quality=92)
            tmp.replace(dst)
        made += 1
        if made % 200 == 0:
            log.info("  кэш: %d новых", made)
    log.info("Кэш готов: %s (+%d файлов)", cdir, made)


# ------------------------------------------------------------- torch Dataset

class OreDataset:
    """Дообёртка манифеста для обучения. Читает из кэша уменьшенных копий."""

    def __init__(self, cfg: dict, split: str, transform=None):
        df = load_manifest(cfg)
        df = df[(~df["exclude"].astype(bool)) & (df["split"] == split)]
        self.rows = df.reset_index(drop=True)
        self.cfg = cfg
        self.transform = transform
        self.classes = list(cfg["classes"]["order"])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self) -> int:
        return len(self.rows)

    def labels(self) -> list[int]:
        return [self.class_to_idx[c] for c in self.rows["class"]]

    def __getitem__(self, i: int):
        row = self.rows.iloc[i]
        p = cache_path(self.cfg, row["md5"])
        if not p.exists():  # страховка: кэш не построен — читаем оригинал
            im = open_rgb(row["path"])
            im.thumbnail(
                (self.cfg["data"]["cache_max_side"],) * 2, Image.LANCZOS
            )
        else:
            im = open_rgb(p)
        y = self.class_to_idx[row["class"]]
        if self.transform is not None:
            im = self.transform(im)
        return im, y
