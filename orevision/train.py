"""Обучение классификатора руд.

Запуск:  python -m orevision.train [--config config.yaml] [--epochs N]

Артефакты в models/:
  best.pt                — лучший чекпойнт (по val macro-F1)
  metrics.json           — метрики лучшей эпохи + разбивка по классам и частям
  history.json           — кривые обучения
  confusion_matrix.png   — матрица ошибок на валидации
  classification_report.txt
  train_log.txt          — полный лог запуска (воспроизводимость)
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, WeightedRandomSampler

from orevision.config import class_names, load_config, snapshot
from orevision.data import OreDataset
from orevision.model import build_model, pick_device, save_checkpoint
from orevision.transforms import eval_transform, train_transform

log = logging.getLogger("orevision.train")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    ys, ps, probs = [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(x)
        p = torch.softmax(logits.float(), dim=1).cpu().numpy()
        probs.append(p)
        ps.append(p.argmax(1))
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps), np.concatenate(probs)


def plot_confusion(cm: np.ndarray, labels: list[str], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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
    ax.set_title("Матрица ошибок (val)")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--arch", default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument(
        "--init-from", default=None,
        help="чекпойнт для тёплого старта (дообучение вместо обучения с нуля)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    tcfg = cfg["train"]
    if args.epochs:
        tcfg["epochs"] = args.epochs
    if args.arch:
        tcfg["arch"] = args.arch
    if args.lr:
        tcfg["lr"] = args.lr

    out_dir = Path(tcfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(out_dir / "train_log.txt", encoding="utf-8"),
        ],
    )
    set_seed(cfg["data"]["seed"])
    device = pick_device()
    classes = class_names(cfg)
    display = [cfg["classes"]["display"][c] for c in classes]
    log.info("device=%s torch=%s classes=%s", device, torch.__version__, classes)

    img_size = int(tcfg["img_size"])
    ds_train = OreDataset(cfg, "train", transform=train_transform(img_size))
    ds_val = OreDataset(cfg, "val", transform=eval_transform(img_size))
    log.info("train=%d val=%d", len(ds_train), len(ds_val))

    if tcfg.get("balance_sampler", True):
        labels = np.array(ds_train.labels())
        counts = np.bincount(labels, minlength=len(classes)).astype(float)
        weights = (1.0 / counts)[labels]
        sampler = WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double), num_samples=len(labels)
        )
        shuffle = None
    else:
        sampler, shuffle = None, True

    nw = int(tcfg["num_workers"])
    dl_train = DataLoader(
        ds_train, batch_size=int(tcfg["batch_size"]), sampler=sampler, shuffle=shuffle,
        num_workers=nw, pin_memory=True, persistent_workers=nw > 0, drop_last=True,
    )
    dl_val = DataLoader(
        ds_val, batch_size=int(tcfg["batch_size"]), shuffle=False,
        num_workers=nw, pin_memory=True, persistent_workers=nw > 0,
    )

    if args.init_from:
        # дообучение: тёплый старт от существующего чекпойнта (active learning)
        import orevision.model as _m

        model, init_info = _m.load_checkpoint(args.init_from, device)
        assert init_info["classes"] == classes, "классы чекпойнта не совпадают с конфигом"
        assert init_info["arch"] == tcfg["arch"] or not args.arch, (
            "архитектура чекпойнта не совпадает с --arch"
        )
        tcfg["arch"] = init_info["arch"]
        model.train()
        log.info("Тёплый старт от %s (%s)", args.init_from, init_info["arch"])
    else:
        model = build_model(tcfg["arch"], num_classes=len(classes), pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=float(tcfg["label_smoothing"]))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(tcfg["lr"]), weight_decay=float(tcfg["weight_decay"])
    )
    epochs = int(tcfg["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    use_amp = bool(tcfg.get("amp", True)) and device.type == "cuda"

    best_f1, best_state, best_epoch = -1.0, None, -1
    patience = int(tcfg.get("patience", 10))
    bad_epochs = 0
    history = []
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for x, y in dl_train:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item() * x.size(0)
            seen += x.size(0)
        scheduler.step()

        ys, ps, _ = evaluate(model, dl_val, device)
        macro = f1_score(ys, ps, average="macro")
        per_class = f1_score(ys, ps, average=None, labels=range(len(classes)))
        history.append(
            {
                "epoch": epoch,
                "train_loss": running / max(seen, 1),
                "val_macro_f1": float(macro),
                **{f"f1_{c}": float(v) for c, v in zip(classes, per_class)},
                "lr": scheduler.get_last_lr()[0],
            }
        )
        log.info(
            "epoch %02d/%d loss=%.4f val macro-F1=%.4f (%s)",
            epoch, epochs, running / max(seen, 1), macro,
            " ".join(f"{c}={v:.3f}" for c, v in zip(classes, per_class)),
        )

        if macro > best_f1:
            best_f1, best_epoch = float(macro), epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                log.info("Ранняя остановка: %d эпох без улучшения", patience)
                break

    # --- финальная оценка лучшей модели
    model.load_state_dict(best_state)
    ys, ps, probs = evaluate(model, dl_val, device)
    macro = f1_score(ys, ps, average="macro")
    auc = roc_auc_score(ys, probs, multi_class="ovr", average="macro")
    cm = confusion_matrix(ys, ps, labels=range(len(classes)))
    report_txt = classification_report(
        ys, ps, labels=range(len(classes)), target_names=display, digits=3
    )
    log.info("ЛУЧШАЯ эпоха %d: val macro-F1=%.4f\n%s", best_epoch, macro, report_txt)

    # разбивка по частям съёмки (домены ч1/ч2)
    parts = ds_val.rows["part"].to_numpy()
    per_part = {}
    for part in sorted(set(parts)):
        m = parts == part
        per_part[part] = {
            "n": int(m.sum()),
            "macro_f1": float(f1_score(ys[m], ps[m], average="macro")),
            "accuracy": float((ys[m] == ps[m]).mean()),
        }
        log.info("part=%s n=%d macro-F1=%.4f acc=%.4f",
                 part, per_part[part]["n"], per_part[part]["macro_f1"],
                 per_part[part]["accuracy"])

    metrics = {
        "best_epoch": best_epoch,
        "val_macro_f1": float(macro),
        "val_auc_ovr": float(auc),
        "per_class_f1": {
            c: float(v)
            for c, v in zip(classes, f1_score(ys, ps, average=None, labels=range(len(classes))))
        },
        "per_part": per_part,
        "confusion_matrix": cm.tolist(),
        "train_time_sec": round(time.time() - t0, 1),
        "n_train": len(ds_train),
        "n_val": len(ds_val),
    }

    # предыдущая модель (с метриками) уходит в архив — обучение всегда обратимо
    from orevision.model import archive_model

    archive_model(out_dir)

    save_checkpoint(
        out_dir / "best.pt", model, tcfg["arch"], classes, img_size,
        meta={"metrics": metrics, "config": snapshot(cfg), "torch": str(torch.__version__)},
    )
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "classification_report.txt").write_text(report_txt, encoding="utf-8")
    plot_confusion(cm, display, out_dir / "confusion_matrix.png")
    log.info("Готово за %.1f мин. Артефакты: %s", (time.time() - t0) / 60, out_dir)


if __name__ == "__main__":
    main()
