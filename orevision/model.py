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
