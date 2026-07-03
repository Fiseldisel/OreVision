"""OreVision — веб-интерфейс классификации руд по OM-изображениям шлифов.

Запуск:  streamlit run app.py
"""

from __future__ import annotations

import io
import json
import time
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

from orevision.batch import process_batch
from orevision.cli import collect_inputs
from orevision.config import load_config
from orevision.data import open_rgb
from orevision.predict import Predictor, _tile_origins, rescore
from orevision.report import metrics_table, result_to_row, save_pdf
from orevision.viz import (
    legend_rows,
    make_confidence_map,
    make_overlay,
    top_uncertain_tiles,
)

st.set_page_config(page_title="OreVision — классификация руд", page_icon="🪨", layout="wide")

PROJECT_ROOT = Path(__file__).resolve().parent
DEMO_DIR = PROJECT_ROOT / "demo"
VERDICT_BANNER = {"ordinary": st.success, "hard": st.error, "talc": st.info}
VERDICT_EMOJI = {"ordinary": "🟩", "hard": "🟥", "talc": "🟦", "unknown": "⬜", "ERROR": "⚠️"}


@st.cache_resource(show_spinner="Загружаю модель...")
def get_predictor(ckpt: str) -> Predictor:
    return Predictor(ckpt, load_config())


def jpeg_bytes(img: Image.Image, quality: int = 90) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def html_legend(cfg: dict) -> str:
    """Легенда с точными цветами маски из config.yaml."""
    parts = [
        f'<span style="display:inline-block;width:13px;height:13px;background:rgb{color};'
        f'border-radius:3px;margin:0 6px 0 14px;vertical-align:-2px"></span>{name}'
        for name, color in legend_rows(cfg)
    ]
    return '<div style="margin-bottom:6px">' + "".join(parts) + "</div>"


def composition_bar(res, cfg: dict) -> str:
    """Stacked-бар долей площади с риской порога талька."""
    colors = {c: tuple(cfg["classes"]["colors"][c]) for c in cfg["classes"]["order"]}
    segs = []
    for c in cfg["classes"]["order"]:
        w = 100 * res.fractions.get(c, 0.0)
        if w > 0.2:
            segs.append(
                f'<div style="width:{w:.1f}%;background:rgb{colors[c]};height:100%" '
                f'title="{cfg["classes"]["display"][c]}: {w:.1f}%"></div>'
            )
    thr = 100 * res.talc_threshold
    # риска порога отсчитывается от правого края: тальк — последний сегмент
    tick = (
        f'<div style="position:absolute;right:{thr:.1f}%;top:-3px;bottom:-3px;'
        f'width:2px;background:#222" title="порог талька {thr:.0f}%"></div>'
    )
    return (
        '<div style="position:relative;height:16px;border-radius:4px;overflow:visible;'
        'background:#e8e8e8;margin:4px 0 2px 0">'
        f'<div style="display:flex;height:100%;border-radius:4px;overflow:hidden">{"".join(segs)}</div>'
        f"{tick}</div>"
        '<div style="font-size:0.8em;color:#666;margin-bottom:8px">'
        "Состав по площади (риска — порог талька, отсчёт от правого края)</div>"
    )


def interactive_class_map(overlay: Image.Image, res, cfg: dict):
    """Карта классов с hover-подсказками по каждому тайлу (plotly)."""
    import numpy as np
    import plotly.graph_objects as go

    W, H = overlay.size
    aW, aH = res.analysis_size
    sx, sy = W / aW, H / aH
    xs = [(x + res.tile / 2) * sx for x in _tile_origins(aW, res.tile, res.stride)]
    ys = [(y + res.tile / 2) * sy for y in _tile_origins(aH, res.tile, res.stride)]

    names = [cfg["classes"]["display"][c] for c in cfg["classes"]["order"]]
    text = []
    for r in range(res.tile_probs.shape[0]):
        row = []
        for c in range(res.tile_probs.shape[1]):
            if res.tile_classes[r, c] < 0:
                row.append("фон — исключён из анализа")
            else:
                p = res.tile_probs[r, c]
                cls = names[int(res.tile_classes[r, c])]
                row.append(
                    f"<b>{cls}</b> · уверенность {100 * res.tile_conf[r, c]:.0f}%<br>"
                    + " · ".join(f"{n}: {100 * v:.0f}%" for n, v in zip(names, p))
                )
        text.append(row)

    fig = go.Figure()
    fig.add_trace(go.Image(z=np.asarray(overlay)))
    fig.add_trace(
        go.Heatmap(
            x=xs, y=ys, z=res.tile_conf, opacity=0.0,
            hoverinfo="text", text=text, showscale=False,
        )
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=min(720, int(720 * H / W) if W >= H else 720),
        hoverlabel=dict(bgcolor="white", font_size=13),
        dragmode="pan",
    )
    return fig


def render_result(key: str, entry: dict, cfg: dict, alpha: float) -> None:
    res = entry["result"]
    base = res.display_thumb

    VERDICT_BANNER.get(res.verdict, st.warning)(f"**Вердикт: {res.verdict_display}**")
    if res.needs_review:
        st.warning(
            "⚠ **Пограничный случай** — доли конкурирующих классов близки или доля "
            "талька рядом с порогом. Рекомендована проверка экспертом "
            "(см. карту уверенности и спорные участки ниже)."
        )
    st.markdown(composition_bar(res, cfg), unsafe_allow_html=True)
    st.markdown(f"*{res.conclusion}*")

    overlay = make_overlay(base, res, cfg, alpha=alpha)
    conf_map = make_confidence_map(base, res, alpha=alpha)

    tab_ov, tab_conf, tab_orig, tab_doubt = st.tabs(
        ["Карта классов", "Карта уверенности", "Исходное", "Спорные участки"]
    )
    with tab_ov:
        st.markdown(html_legend(cfg), unsafe_allow_html=True)
        try:
            st.plotly_chart(
                interactive_class_map(overlay, res, cfg),
                config={"scrollZoom": True, "displaylogo": False},
            )
            st.caption(
                "Наведите курсор на участок — вероятности классов по тайлу; "
                "колесо мыши — зум, перетаскивание — панорамирование."
            )
        except ImportError:
            st.image(overlay, width="stretch")
    with tab_conf:
        st.caption("Жёлтые участки — модель уверена; тёмно-синие — спорные, стоит проверить глазами.")
        st.image(conf_map, width="stretch")
    with tab_orig:
        st.image(base, width="stretch")
    with tab_doubt:
        st.caption("Участки с наименьшей уверенностью модели — с них стоит начать ручную проверку шлифа.")
        tiles = top_uncertain_tiles(base, res, cfg, k=6)
        if tiles:
            cols = st.columns(3)
            for i, (crop, cap) in enumerate(tiles):
                with cols[i % 3]:
                    st.image(crop, caption=cap, width="stretch")
        else:
            st.info("Нет проанализированных участков.")

    col_t, col_d = st.columns([3, 2])
    with col_t:
        st.dataframe(metrics_table(res, cfg), width="stretch", hide_index=True)
    with col_d:
        stem = Path(res.source).stem
        ov_bytes = jpeg_bytes(overlay)
        cf_bytes = jpeg_bytes(conf_map)
        pdf_buf = io.BytesIO()
        thumb = base.copy()
        thumb.thumbnail((1600, 1600))
        ov_s = overlay.copy()
        ov_s.thumbnail((1600, 1600))
        cf_s = conf_map.copy()
        cf_s.thumbnail((1600, 1600))
        save_pdf(res, cfg, pdf_buf, original=thumb, overlay=ov_s, confidence=cf_s)
        csv_bytes = pd.DataFrame([result_to_row(res, cfg)]).to_csv(index=False).encode("utf-8-sig")

        st.download_button("⬇ Оверлей (JPG)", ov_bytes, f"{stem}_overlay.jpg",
                           "image/jpeg", key=f"ov_{key}")
        st.download_button("⬇ Карта уверенности (JPG)", cf_bytes,
                           f"{stem}_confidence.jpg", "image/jpeg", key=f"cf_{key}")
        st.download_button("⬇ Отчёт (PDF)", pdf_buf.getvalue(), f"{stem}_report.pdf",
                           "application/pdf", key=f"pdf_{key}")
        st.download_button("⬇ Метрики (CSV)", csv_bytes, f"{stem}_metrics.csv",
                           "text/csv", key=f"csv_{key}")

        zbuf = io.BytesIO()
        with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr(f"{stem}_overlay.jpg", ov_bytes)
            z.writestr(f"{stem}_confidence.jpg", cf_bytes)
            z.writestr(f"{stem}_report.pdf", pdf_buf.getvalue())
            z.writestr(f"{stem}_metrics.csv", csv_bytes)
        st.download_button("📦 Всё одним архивом (ZIP)", zbuf.getvalue(),
                           f"{stem}_orevision.zip", "application/zip", key=f"zip_{key}")


def summary_dataframe(rows: list[dict]) -> tuple[pd.DataFrame, dict]:
    """Сводная таблица с прогресс-барами долей и цветными вердиктами."""
    view = pd.DataFrame(
        {
            "Файл": [r.get("file", "?") for r in rows],
            "Вердикт": [
                f"{VERDICT_EMOJI.get(r.get('verdict', ''), '')} {r.get('verdict_ru', r.get('verdict', ''))}"
                + (" ⚠" if r.get("needs_review") else "")
                for r in rows
            ],
            "Обычные, %": [100 * r.get("frac_ordinary", 0) for r in rows],
            "Тонкие, %": [100 * r.get("frac_hard", 0) for r in rows],
            "Тальк, %": [100 * r.get("frac_talc", 0) for r in rows],
            "Время, с": [r.get("elapsed_sec", "") for r in rows],
        }
    )
    col_cfg = {
        name: st.column_config.ProgressColumn(name, min_value=0, max_value=100, format="%.0f%%")
        for name in ("Обычные, %", "Тонкие, %", "Тальк, %")
    }
    return view, col_cfg


def run_jobs(jobs, predictor: Predictor, cfg: dict, mode: str, ckpt_name: str, talc_thr: float) -> None:
    """Анализ списка (имя, источник, сигнатура) с прогрессом и кэшем в сессии."""
    for name, src, sig in jobs:
        cache_key = f"{name}|{sig}|{mode}|{ckpt_name}"
        if cache_key in st.session_state["results"]:
            continue
        bar = st.progress(0.0, text=f"Анализ {name}...")
        t0 = time.time()

        def cb(done, total, _bar=bar, _n=name, _t0=t0):
            elapsed = max(time.time() - _t0, 1e-6)
            speed = done / elapsed
            eta = int((total - done) / speed) if speed > 0 else 0
            _bar.progress(
                done / max(total, 1),
                text=f"Анализ {_n}: тайл {done}/{total} · {speed:.1f} тайл/с · осталось ~{eta} с",
            )

        try:
            if isinstance(src, Path):
                im = open_rgb(src)
            else:
                im = Image.open(io.BytesIO(src.getvalue()))
                im = ImageOps.exif_transpose(im)
                im = im.convert("RGB")
            res = predictor.predict(im, mode=mode, progress_cb=cb)
            res.source = name
            rescore(res, cfg, talc_thr)
            st.session_state["results"][cache_key] = {"result": res}
            bar.progress(1.0, text=f"{name}: готово за {time.time() - t0:.1f} c")
        except Exception as e:
            bar.empty()
            st.error(f"Ошибка на {name}: {e}")


def main() -> None:
    cfg = load_config()

    st.title("🪨 OreVision")
    st.caption(
        "Автоматическая классификация руд по панорамным OM-изображениям полированных шлифов: "
        "рядовые / труднообогатимые (тонкие срастания) / оталькованные."
    )

    # ---------------------------------------------------------------- sidebar
    with st.sidebar:
        st.header("Модель")
        models_dir = Path(cfg["train"]["out_dir"])
        ckpts = sorted(models_dir.glob("*.pt"))
        if not ckpts:
            st.error(f"Нет чекпойнтов в {models_dir}. Обучите модель: `python -m orevision.train`")
            st.stop()
        ckpt = st.selectbox("Чекпойнт", ckpts, format_func=lambda p: p.name)
        predictor = get_predictor(str(ckpt))
        meta_metrics = predictor.info.get("meta", {}).get("metrics", {})
        if meta_metrics:
            st.caption(
                f"val macro-F1: **{meta_metrics.get('val_macro_f1', 0):.3f}** · "
                f"устройство: `{predictor.device}`"
            )

        st.header("Параметры анализа")
        mode = st.radio(
            "Режим", ["auto", "photo", "panorama"],
            help="auto: изображения крупнее 30 Мпикс анализируются как панорамы (тайлы в нативном разрешении)",
        )
        talc_thr = st.slider(
            "Порог доли талька, %", 1, 50, int(100 * cfg["infer"]["talc_threshold"]),
            help="Экспертное правило: доля площади талька выше порога → оталькованная руда",
        ) / 100.0
        alpha = st.slider("Прозрачность маски", 0.0, 1.0, 0.45, 0.05)
        cfg["infer"]["talc_threshold"] = talc_thr
        # предиктор кэширован и держит свою копию конфига — синхронизируем порог,
        # чтобы пакетный режим и свежие анализы считались с актуальным значением
        predictor.cfg["infer"]["talc_threshold"] = talc_thr

    tab_an, tab_batch, tab_about = st.tabs(
        ["🔬 Анализ изображений", "📁 Пакетная обработка", "ℹ️ О модели и данных"]
    )

    # ------------------------------------------------------------- анализ
    with tab_an:
        st.session_state.setdefault("results", {})

        # --- демо-примеры в один клик
        demo_files = {
            "🟩 Пример: рядовая": DEMO_DIR / "фото_рядовая.jpg",
            "🟥 Пример: труднообогатимая": DEMO_DIR / "фото_труднообогатимая.jpg",
            "🟦 Пример: оталькованная": DEMO_DIR / "фото_оталькованная.jpg",
            "🗺 Пример: панорама (50 Мпикс)": DEMO_DIR / "панорама_фрагмент.jpg",
        }
        demo_files = {k: v for k, v in demo_files.items() if v.exists()}
        if demo_files:
            st.caption("Быстрый старт — встроенные примеры:")
            demo_cols = st.columns(len(demo_files))
            for col, (label, p) in zip(demo_cols, demo_files.items()):
                with col:
                    if st.button(label, key=f"demo_{p.stem}", width="stretch"):
                        pst = p.stat()
                        run_jobs(
                            [(p.name, p, f"{p}:{pst.st_mtime_ns}:{pst.st_size}")],
                            predictor, cfg, mode, ckpt.name, talc_thr,
                        )

        up_files = st.file_uploader(
            "Изображения шлифов (TIFF/PNG/JPEG/BMP, поддерживаются панорамы до ~1 ГБ)",
            type=["jpg", "jpeg", "png", "tif", "tiff", "bmp"],
            accept_multiple_files=True,
        )
        disk_path = st.text_input(
            "…или путь к файлу/папке на диске (без загрузки через браузер)",
            placeholder=r"C:\данные\панорамы\10.jpg",
        )

        if st.button("▶ Анализировать", type="primary"):
            # ключ кэша включает сигнатуру содержимого: одноимённые, но разные
            # файлы не подменяют друг друга, обновлённый файл переанализируется
            jobs: list[tuple[str, object, str]] = []
            for f in up_files or []:
                sig = f"{getattr(f, 'file_id', '')}:{f.size}"
                jobs.append((f.name, f, sig))
            if disk_path.strip():
                try:
                    for p in collect_inputs(Path(disk_path.strip())):
                        pst = p.stat()
                        jobs.append((p.name, p, f"{p}:{pst.st_mtime_ns}:{pst.st_size}"))
                except FileNotFoundError:
                    st.error(f"Путь не найден: {disk_path}")
            if not jobs:
                st.warning("Добавьте изображения или укажите путь.")
            run_jobs(jobs, predictor, cfg, mode, ckpt.name, talc_thr)

        results = {
            k: v for k, v in st.session_state["results"].items()
            if k.endswith(f"|{mode}|{ckpt.name}")
        }
        if results:
            # порог могли подвинуть после анализа — вердикт пересчитывается без инференса
            for entry in results.values():
                rescore(entry["result"], cfg, talc_thr)

            st.divider()
            names = list(results.keys())
            if len(names) > 1:
                rows = [result_to_row(v["result"], cfg) for v in results.values()]
                st.subheader("Сводка по загруженным изображениям")
                view, col_cfg = summary_dataframe(rows)
                st.dataframe(view, width="stretch", hide_index=True, column_config=col_cfg)
                st.download_button(
                    "⬇ Сводный CSV",
                    pd.DataFrame(rows).to_csv(index=False).encode("utf-8-sig"),
                    "orevision_summary.csv", "text/csv",
                )
                st.divider()

            sel = st.selectbox(
                "Изображение", names,
                format_func=lambda k: results[k]["result"].source,
            ) if len(names) > 1 else names[0]
            render_result(sel, results[sel], cfg, alpha)

            if st.button("🗑 Очистить результаты"):
                st.session_state["results"].clear()
                st.rerun()

    # ------------------------------------------------------- пакетный режим
    with tab_batch:
        st.markdown(
            "Обработка папки целиком **без участия пользователя**: для каждого файла — "
            "оверлей, карта уверенности, PDF-отчёт; в конце — сводный `summary.csv` "
            "и `run_params.json` (лог параметров для воспроизводимости)."
        )
        in_dir = st.text_input("Папка с изображениями", placeholder=r"C:\данные\партия_2026_07")
        # default фиксируется на сессию: меняющийся value у виджета без key
        # сбрасывал бы введённый пользователем путь при каждом rerun
        if "batch_out_default" not in st.session_state:
            st.session_state["batch_out_default"] = str(
                Path(cfg["report"]["out_dir"]) / time.strftime("run_%Y%m%d_%H%M%S")
            )
        out_dir = st.text_input(
            "Папка результатов",
            value=st.session_state["batch_out_default"],
            key="batch_out_dir",
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            do_pdf = st.checkbox("PDF-отчёты", value=True)
        with c2:
            do_overlay = st.checkbox("Оверлеи (JPG)", value=True)
        with c3:
            batch_mode = st.selectbox("Режим", ["auto", "photo", "panorama"], key="batch_mode")

        if st.button("▶ Запустить пакетную обработку", type="primary", key="run_batch"):
            try:
                files = collect_inputs(Path(in_dir.strip()))
            except FileNotFoundError:
                st.error(f"Папка не найдена: {in_dir}")
                files = []
            if not files:
                st.warning("В папке нет поддерживаемых изображений.")
            else:
                bar = st.progress(0.0, text=f"0/{len(files)}")
                t0 = time.time()

                def fcb(i, n, name, _bar=bar, _t0=t0):
                    elapsed = time.time() - _t0
                    eta = int(elapsed / i * (n - i)) if i else 0
                    _bar.progress(
                        i / max(n, 1),
                        text=f"{i}/{n}  {name}" + (f" · осталось ~{eta} с" if i else ""),
                    )

                rows = process_batch(
                    files, Path(out_dir), predictor, cfg,
                    mode=batch_mode, overlays=do_overlay, pdf=do_pdf,
                    file_progress_cb=fcb,
                )
                st.success(f"Готово: {len(rows)} файлов → {out_dir}")
                view, col_cfg = summary_dataframe(rows)
                st.dataframe(view, width="stretch", hide_index=True, column_config=col_cfg)
                st.download_button(
                    "⬇ summary.csv", pd.DataFrame(rows).to_csv(index=False).encode("utf-8-sig"),
                    "summary.csv", "text/csv", key="batch_csv",
                )

    # ------------------------------------------------------------- о модели
    with tab_about:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Качество модели (валидация)")
            mfile = Path(cfg["train"]["out_dir"]) / "metrics.json"
            if mfile.exists():
                m = json.loads(mfile.read_text(encoding="utf-8"))
                mc1, mc2 = st.columns(2)
                mc1.metric("Macro-F1", f"{m['val_macro_f1']:.3f}")
                if "val_auc_ovr" in m:
                    mc2.metric("ROC-AUC (ovr)", f"{m['val_auc_ovr']:.3f}")
                per_class = m.get("per_class_f1", {})
                st.table(
                    pd.DataFrame(
                        {
                            "Класс": [cfg["classes"]["display"][c] for c in per_class],
                            "F1": [f"{v:.3f}" for v in per_class.values()],
                        }
                    )
                )
                per_part = m.get("per_part", {})
                if per_part:
                    st.caption("По частям датасета (разные условия съёмки):")
                    st.table(
                        pd.DataFrame(
                            {
                                "Часть": list(per_part),
                                "n": [p["n"] for p in per_part.values()],
                                "Macro-F1": [f"{p['macro_f1']:.3f}" for p in per_part.values()],
                            }
                        )
                    )
            cm = Path(cfg["train"]["out_dir"]) / "confusion_matrix.png"
            if cm.exists():
                st.image(str(cm), caption="Матрица ошибок (валидация)")
        with col2:
            st.subheader("Как это работает")
            st.markdown(
                """
1. **Тайловый анализ.** Изображение разбивается на тайлы; панорамы — в нативном
   разрешении (1024 px), фото — в масштабе обучения. Каждый тайл классифицирует
   дообученная сеть **EfficientNetV2-S** (transfer learning с ImageNet).
2. **Экспертное правило (панорамы).** Доля площади талька > порога (по умолчанию 10%)
   → *оталькованная*; иначе — преобладание обычных либо тонких срастаний.
3. **Интерпретируемость.** Карта классов с hover-подсказками по каждому участку,
   карта уверенности и галерея спорных участков — геолог видит, «почему» модель
   так решила, и с чего начать ручную проверку.
4. **Устойчивость.** Части датасета сняты на разном оборудовании; обучение с
   агрессивными цветовыми аугментациями заставляет модель опираться на
   морфологию срастаний, а не на баланс белого.
                """
            )
            st.subheader("Легенда")
            st.markdown(html_legend(cfg), unsafe_allow_html=True)


main()
