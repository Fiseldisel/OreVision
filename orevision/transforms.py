"""Аугментации и препроцессинг.

Части датасета сняты на разном оборудовании (ч1 — жёлто-зелёная тональность,
ч2 и панорамы — тёмные), поэтому упор на цветовые аугментации: модель должна
опираться на морфологию срастаний, а не на баланс белого конкретного микроскопа.
"""

from __future__ import annotations

import torch
from torchvision.transforms import v2

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def train_transform(img_size: int) -> v2.Compose:
    return v2.Compose(
        [
            v2.RandomResizedCrop(
                img_size, scale=(0.2, 1.0), ratio=(0.8, 1.25), antialias=True
            ),
            v2.RandomHorizontalFlip(0.5),
            v2.RandomVerticalFlip(0.5),
            # шлиф не имеет "верха" — повороты на 90° легальны
            v2.RandomApply([v2.RandomRotation((90, 90))], p=0.5),
            v2.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.4, hue=0.06),
            v2.RandomGrayscale(p=0.15),
            v2.RandomApply([v2.GaussianBlur(5, sigma=(0.1, 1.5))], p=0.15),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def eval_transform(img_size: int) -> v2.Compose:
    """Для валидации: центральный кроп с сохранением масштаба текстуры."""
    return v2.Compose(
        [
            v2.Resize(int(img_size * 1.15), antialias=True),
            v2.CenterCrop(img_size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def tensor_transform(img_size: int) -> v2.Compose:
    """Для инференса тайлов/кропов: вход уже PIL нужного размера или больше."""
    return v2.Compose(
        [
            v2.Resize((img_size, img_size), antialias=True),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
