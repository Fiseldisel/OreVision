"""Загрузка и валидация конфигурации OreVision."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config(path: str | Path | None = None) -> dict:
    """Читает config.yaml и нормализует пути.

    Все относительные пути в конфиге трактуются относительно корня проекта.
    """
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["_config_path"] = str(cfg_path)
    cfg["_project_root"] = str(PROJECT_ROOT)

    # Абсолютные пути
    cfg["data"]["root"] = str(Path(cfg["data"]["root"]))
    for key in ("manifest", "cache_dir"):
        p = Path(cfg["data"][key])
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        cfg["data"][key] = str(p)
    for section, key in (("train", "out_dir"), ("report", "out_dir")):
        p = Path(cfg[section][key])
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        cfg[section][key] = str(p)

    order = cfg["classes"]["order"]
    assert len(order) == len(set(order)) == 3, "Ожидаются 3 уникальных класса"
    for c in order:
        assert c in cfg["classes"]["display"], f"Нет display-имени для класса {c}"
        assert c in cfg["classes"]["colors"], f"Нет цвета для класса {c}"
    return cfg


def class_names(cfg: dict) -> list[str]:
    return list(cfg["classes"]["order"])


def display_name(cfg: dict, cls: str) -> str:
    return cfg["classes"]["display"][cls]


def class_color(cfg: dict, cls: str) -> tuple[int, int, int]:
    return tuple(cfg["classes"]["colors"][cls])


def snapshot(cfg: dict) -> dict:
    """Копия конфига без служебных ключей — для логов воспроизводимости."""
    snap = copy.deepcopy({k: v for k, v in cfg.items() if not k.startswith("_")})
    return snap
