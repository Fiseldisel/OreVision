"""Модель классификации: transfer learning на torchvision-бэкбонах."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

log = logging.getLogger("orevision.model")


def build_model(arch: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """Создаёт бэкбон с новой классификационной головой."""
    if arch == "efficientnet_v2_s":
        weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.efficientnet_v2_s(weights=weights)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif arch == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.convnext_tiny(weights=weights)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, num_classes)
    elif arch == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        m = models.resnet50(weights=weights)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    else:
        raise ValueError(f"Неизвестная архитектура: {arch}")
    return m


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    arch: str,
    classes: list[str],
    img_size: int,
    meta: dict | None = None,
) -> None:
    payload = {
        "state_dict": model.state_dict(),
        "arch": arch,
        "classes": classes,
        "img_size": img_size,
        "meta": meta or {},
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    log.info("Чекпойнт сохранён: %s", path)


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu"):
    """Возвращает (model.eval(), info-словарь). Само определяет архитектуру."""
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        # старые чекпойнты содержат TorchVersion (подкласс str) в мета —
        # разрешаем только этот безопасный тип, режим weights_only сохраняется
        from torch.torch_version import TorchVersion

        with torch.serialization.safe_globals([TorchVersion]):
            payload = torch.load(path, map_location="cpu", weights_only=True)
    arch = payload["arch"]
    classes = payload["classes"]
    model = build_model(arch, num_classes=len(classes), pretrained=False)
    model.load_state_dict(payload["state_dict"])
    model.eval().to(device)
    info = {
        "arch": arch,
        "classes": classes,
        "img_size": int(payload.get("img_size", 384)),
        "meta": payload.get("meta", {}),
    }
    return model, info


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ------------------------------------------------- версии модели (архив/откат)

MODEL_ARTIFACTS = [
    "best.pt",
    "metrics.json",
    "history.json",
    "classification_report.txt",
    "confusion_matrix.png",
]


def archive_model(models_dir: str | Path) -> Path | None:
    """Складывает текущую модель СО ВСЕМИ артефактами в models/archive/<ts>/.

    Возвращает путь архива либо None, если архивировать нечего.
    """
    import shutil
    import time as _t

    models_dir = Path(models_dir)
    if not (models_dir / "best.pt").exists():
        return None
    dst = models_dir / "archive" / _t.strftime("%Y%m%d_%H%M%S")
    dst.mkdir(parents=True, exist_ok=True)
    for name in MODEL_ARTIFACTS:
        src = models_dir / name
        if src.exists():
            shutil.copy2(src, dst / name)
    log.info("Модель заархивирована: %s", dst)
    return dst


def list_archives(models_dir: str | Path) -> list[Path]:
    """Архивные версии, новые в конце. Поддерживает и старый плоский формат."""
    arch = Path(models_dir) / "archive"
    if not arch.is_dir():
        return []
    dirs = [d for d in arch.iterdir() if d.is_dir() and (d / "best.pt").exists()]
    flat = [f for f in arch.glob("best_*.pt")]  # старый формат: только веса
    return sorted(dirs + flat, key=lambda p: p.name)


def restore_model(models_dir: str | Path, archive: str | Path) -> Path | None:
    """Откат к архивной версии. Текущая модель СНАЧАЛА архивируется —
    откат всегда обратим. Возвращает путь архива текущей модели."""
    import shutil

    models_dir = Path(models_dir)
    archive = Path(archive)
    backup = archive_model(models_dir)
    if archive.is_dir():
        for name in MODEL_ARTIFACTS:
            src = archive / name
            if src.exists():
                shutil.copy2(src, models_dir / name)
    else:  # плоский .pt старого формата — восстанавливаются только веса
        shutil.copy2(archive, models_dir / "best.pt")
    log.info("Восстановлена версия %s (текущая сохранена в %s)", archive.name, backup)
    return backup
