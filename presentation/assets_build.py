"""Ассеты презентации: графики, миниатюры, рендер PDF-отчёта."""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
ROOT = Path(__file__).parent.parent
OUT = Path(__file__).parent / "assets"
OUT.mkdir(exist_ok=True)
DATA = Path(r"C:\Users\user\Documents\my_project\data\Задача 3. Скажи мне, кто твой шлиф")

ACCENT = "#C77B3F"
DARK = "#23272E"
MUTED = "#6B7280"


def thumb(src, name, max_side=1400, quality=88):
    with Image.open(src) as im:
        im = im.convert("RGB")
        im.thumbnail((max_side, max_side), Image.LANCZOS)
        im.save(OUT / name, quality=quality)
    print("thumb", name)


def thumb43(src, name, max_side=1100, quality=88):
    """Центр-кроп к 4:3 — одинаковая геометрия карточек с фото классов."""
    with Image.open(src) as im:
        im = im.convert("RGB")
        w, h = im.size
        if w / h > 4 / 3:
            nw = int(h * 4 / 3)
            im = im.crop(((w - nw) // 2, 0, (w - nw) // 2 + nw, h))
        else:
            nh = int(w * 3 / 4)
            im = im.crop((0, (h - nh) // 2, w, (h - nh) // 2 + nh))
        im.thumbnail((max_side, max_side), Image.LANCZOS)
        im.save(OUT / name, quality=quality)
    print("thumb43", name)


# --- кривые обучения
hist = json.loads((ROOT / "models" / "history.json").read_text(encoding="utf-8"))
epochs = [h["epoch"] for h in hist]
loss = [h["train_loss"] for h in hist]
f1 = [h["val_macro_f1"] for h in hist]
fig, ax1 = plt.subplots(figsize=(8.2, 4.0), dpi=170)
ax1.plot(epochs, loss, color=MUTED, lw=2, label="train loss")
ax1.set_xlabel("Эпоха", fontsize=11)
ax1.set_ylabel("Train loss", color=MUTED, fontsize=11)
ax1.tick_params(axis="y", labelcolor=MUTED)
ax2 = ax1.twinx()
ax2.plot(epochs, f1, color=ACCENT, lw=2.5, label="val macro-F1")
ax2.set_ylabel("Val macro-F1", color=ACCENT, fontsize=11)
ax2.tick_params(axis="y", labelcolor=ACCENT)
best_i = max(range(len(f1)), key=lambda i: f1[i])
# десятичная запятая — по-русски
best_label = f"лучшая эпоха {epochs[best_i]}: F1 {f1[best_i]:.3f}".replace(".", ",")
ax2.scatter([epochs[best_i]], [f1[best_i]], color=ACCENT, zorder=5, s=70, label=best_label)
# сводная легенда обеих осей — не пересекается с кривыми в отличие от аннотации
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax2.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=10, framealpha=0.95)
for ax in (ax1, ax2):
    ax.spines["top"].set_visible(False)
ax1.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(OUT / "training_curves.png")
plt.close(fig)
print("training_curves.png")

# --- примеры классов (из demo/), одинаковая геометрия 4:3
thumb43(ROOT / "demo" / "фото_рядовая.jpg", "sample_ordinary.jpg", 1100)
thumb43(ROOT / "demo" / "фото_труднообогатимая.jpg", "sample_hard.jpg", 1100)
thumb43(ROOT / "demo" / "фото_оталькованная.jpg", "sample_talc.jpg", 1100)

# --- домен-шифт ч1 vs ч2: для ч2 выбираем реально ТЁМНЫЙ снимок
thumb43(DATA / "Фото руд по сортам. ч1" / "Рядовые руды" / "2539589-1.JPG", "domain_ch1.jpg", 1000)
# характерный тёмный снимок ч2 (тот же стиль, что панорамы)
thumb43(DATA / "Фото руд по сортам. ч2" / "тонкие" / "-4.jpg", "domain_ch2.jpg", 1000)

# --- пример ошибки санити-теста (кадр «оталькованных» с массивным сульфидом)
thumb43(
    DATA / "Фото руд по сортам. ч1" / "Оталькованные руды" / "DSCN5032.JPG",
    "error_case.jpg", 900,
)

# --- панорама: оригинал + оверлей + уверенность
thumb(DATA / "Панорамы" / "10.jpg", "pano_original.jpg", 1500, 82)
thumb(ROOT / "results" / "panoramas_full" / "10" / "overlay.jpg", "pano_overlay.jpg", 1500, 82)
thumb(ROOT / "results" / "panoramas_full" / "10" / "confidence.jpg", "pano_confidence.jpg", 1500, 82)
thumb(ROOT / "results" / "panoramas_full" / "15" / "overlay.jpg", "pano15_overlay.jpg", 1500, 82)

# --- confusion matrix (копия)
import shutil

shutil.copy(ROOT / "models" / "confusion_matrix.png", OUT / "confusion_matrix.png")
print("confusion_matrix.png")

# --- первая страница PDF-отчёта как изображение (свежий отчёт с исправленным макетом)
import fitz

pdf_src = ROOT / "results" / "pres_report" / "15" / "report.pdf"
if not pdf_src.exists():
    pdf_src = ROOT / "results" / "panoramas_full" / "15" / "report.pdf"
doc = fitz.open(pdf_src)
pix = doc[0].get_pixmap(dpi=170)
pix.save(OUT / "report_page.png")
doc.close()
with Image.open(OUT / "report_page.png") as im:  # подрезаем поля страницы
    w, h = im.size
    im.crop((int(w * 0.02), int(h * 0.02), int(w * 0.98), int(h * 0.985))).save(
        OUT / "report_page.png"
    )
print("report_page.png (", pdf_src.parent.parent.name, ")")

# --- кропы скриншотов UI: без сайдбара/Deploy и без разрезанных элементов снизу
with Image.open(OUT / "ui_verdict.png") as im:
    w, h = im.size  # 2481x1515 при масштабе 1.5
    im.crop((int(w * 0.157), int(h * 0.055), w, int(h * 0.812))).save(OUT / "ui_verdict_crop.png")
print("ui_verdict_crop.png")
with Image.open(OUT / "ui_doubt.png") as im:
    w, h = im.size
    im.crop((int(w * 0.157), int(h * 0.03), w, int(h * 0.80))).save(OUT / "ui_doubt_crop.png")
print("ui_doubt_crop.png")

print("ASSETS DONE")
