"""Модель классификации: transfer learning на torchvision-бэкбонах."""

from __future__ import annotations

import logging
import re
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

BASE_CHECKPOINT = "base.pt"          # базовая (заводская) модель
# суффиксы производных моделей: base_дообученная.pt, base_переобученная.pt, ...
SUFFIX_FINETUNE = "_дообученная"      # быстрое дообучение (тёплый старт)
SUFFIX_RETRAIN = "_переобученная"     # полное переобучение с нуля
_ANY_SUFFIX = re.compile(rf"({SUFFIX_FINETUNE}|{SUFFIX_RETRAIN})(_\d+)?$")


def migrate_legacy_checkpoint(models_dir: str | Path) -> Path | None:
    """Совместимость: если базовой base.pt нет, но есть старый best.pt —
    копируем его в base.pt (best.pt не трогаем). Возвращает путь base.pt."""
    import shutil

    models_dir = Path(models_dir)
    base = models_dir / BASE_CHECKPOINT
    legacy = models_dir / "best.pt"
    if not base.exists() and legacy.exists():
        shutil.copy2(legacy, base)
        log.info("Базовая модель создана из best.pt: %s", base)
    return base if base.exists() else None


def default_checkpoint(models_dir: str | Path) -> Path:
    """Путь к базовой модели; понимает старое имя best.pt для совместимости."""
    models_dir = Path(models_dir)
    base = models_dir / BASE_CHECKPOINT
    legacy = models_dir / "best.pt"
    return base if base.exists() or not legacy.exists() else legacy


def next_output_name(models_dir: str | Path, base_ckpt: str | Path, suffix: str) -> Path:
    """Имя нового производного чекпойнта: <корень>_<суффикс>.pt, далее _2, _3…

    Корень берётся без уже имеющегося суффикса — повторное дообучение даёт
    base_дообученная_2, а не base_дообученная_дообученная.
    """
    stem = _ANY_SUFFIX.sub("", Path(base_ckpt).stem)
    models_dir = Path(models_dir)
    cand = models_dir / f"{stem}{suffix}.pt"
    n = 2
    while cand.exists():
        cand = models_dir / f"{stem}{suffix}_{n}.pt"
        n += 1
    return cand


def list_checkpoints(models_dir: str | Path) -> list[Path]:
    """Все чекпойнты в models/, базовая base.pt — первой."""
    models_dir = Path(models_dir)
    pts = [p for p in models_dir.glob("*.pt")]
    return sorted(pts, key=lambda p: (p.name != BASE_CHECKPOINT, p.name.lower()))


def is_base_checkpoint(path: str | Path) -> bool:
    return Path(path).name in (BASE_CHECKPOINT, "best.pt")


def delete_checkpoint(path: str | Path) -> bool:
    """Удаляет производный чекпойнт. Базовую модель удалить нельзя."""
    path = Path(path)
    if is_base_checkpoint(path) or not path.exists():
        return False
    path.unlink()
    log.info("Чекпойнт удалён: %s", path)
    return True
