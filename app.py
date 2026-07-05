"""OreVision — веб-интерфейс классификации руд по OM-изображениям шлифов.

Запуск:  streamlit run app.py
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
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
from orevision.feedback import (
    feedback_stats,
    native_tile_crop,
    save_feedback_sample,
)
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


CLASS_PICK_HELP = "Укажите верный класс — участок попадёт в набор дообучения"


def _display_to_key(cfg: dict, display: str) -> str:
    for c in cfg["classes"]["order"]:
        if cfg["classes"]["display"][c] == display:
            return c
    raise KeyError(display)


def _save_tile_feedback(entry: dict, cfg: dict, t: dict, label_key: str, ckpt_name: str) -> None:
    """Сохраняет исправленный тайл; при наличии исходника — в нативном разрешении."""
    res = entry["result"]
    img = t["crop"]
    src = entry.get("src")
    if src is not None:
        try:
            img = native_tile_crop(
                src, t["row"], t["col"], res.tile, res.stride, tuple(res.analysis_size)
            )
        except Exception:
            pass  # запасной вариант — кроп с display-миниатюры
    save_feedback_sample(
        img, label_key, cfg,
        prev_pred=t["pred"], source=res.source, mode=res.mode,
        tile=f"{t['row']},{t['col']}", checkpoint=ckpt_name,
    )


def render_result(key: str, entry: dict, cfg: dict, alpha: float, ckpt_name: str = "") -> None:
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
            import plotly  # noqa: F401

            has_plotly = True
        except ImportError:
            has_plotly = False
        if has_plotly:
            try:
                st.plotly_chart(
                    interactive_class_map(overlay, res, cfg),
                    config={"scrollZoom": True, "displaylogo": False},
                )
                st.caption(
                    "Наведите курсор на участок — вероятности классов по тайлу; "
                    "колесо мыши — зум, перетаскивание — панорамирование."
                )
            except Exception as e:  # runtime-сбой plotly не должен ронять вкладку
                st.image(overlay, width="stretch")
                st.warning(f"Интерактивная карта недоступна ({e}); показана статичная.")
        else:
            # без plotly подсказки при наведении не работают — сообщаем явно,
            # а не выдаём статичную картинку за «так и задумано»
            st.image(overlay, width="stretch")
            st.info(
                "Подсказки при наведении и зум требуют пакета **plotly**. "
                "Установите его и перезапустите приложение:  `pip install plotly`"
            )
    with tab_conf:
        st.caption("Жёлтые участки — модель уверена; тёмно-синие — спорные, стоит проверить глазами.")
        st.image(conf_map, width="stretch")
    with tab_orig:
        st.image(base, width="stretch")
    with tab_doubt:
        st.caption(
            "Участки с наименьшей уверенностью модели. Если модель ошиблась — "
            "укажите верный класс и нажмите «В набор дообучения»: это режим "
            "экспертной проверки (active learning)."
        )
        tiles = top_uncertain_tiles(base, res, cfg, k=6)
        if tiles:
            st.session_state.setdefault("fb_saved", set())
            display_names = [cfg["classes"]["display"][c] for c in cfg["classes"]["order"]]
            cols = st.columns(3)
            for i, t in enumerate(tiles):
                tile_id = f"{key}|{t['row']}|{t['col']}"
                with cols[i % 3]:
                    st.image(t["crop"], caption=t["caption"], width="stretch")
                    if tile_id in st.session_state["fb_saved"]:
                        st.success("✓ в наборе дообучения")
                        continue
                    pick = st.selectbox(
                        "Верный класс", display_names,
                        index=cfg["classes"]["order"].index(t["pred"]),
                        key=f"fbsel_{tile_id}", help=CLASS_PICK_HELP,
                        label_visibility="collapsed",
                    )
                    if st.button("💾 В набор дообучения", key=f"fbbtn_{tile_id}"):
                        _save_tile_feedback(entry, cfg, t, _display_to_key(cfg, pick), ckpt_name)
                        st.session_state["fb_saved"].add(tile_id)
                        st.rerun()
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

    with st.expander("✏️ Экспертная коррекция вердикта (дообучение)"):
        if res.mode == "photo":
            st.caption(
                "Если модель ошиблась с классом всего снимка — укажите верный, "
                "снимок попадёт в набор дообучения."
            )
            display_names = [cfg["classes"]["display"][c] for c in cfg["classes"]["order"]]
            default_i = (
                cfg["classes"]["order"].index(res.verdict)
                if res.verdict in cfg["classes"]["order"] else 0
            )
            cor_col, btn_col = st.columns([3, 2])
            with cor_col:
                pick = st.selectbox(
                    "Верный класс снимка", display_names, index=default_i,
                    key=f"fbphoto_sel_{key}", help=CLASS_PICK_HELP,
                )
            with btn_col:
                st.write("")
                saved_flag = f"photo|{key}"
                if saved_flag in st.session_state.get("fb_saved", set()):
                    st.success("✓ в наборе дообучения")
                elif st.button("💾 Сохранить пример", key=f"fbphoto_btn_{key}"):
                    src = entry.get("src")
                    if isinstance(src, Path):
                        img = open_rgb(src)
                    elif isinstance(src, (bytes, bytearray)):
                        img = ImageOps.exif_transpose(
                            Image.open(io.BytesIO(src))
                        ).convert("RGB")
                    else:
                        img = base
                    save_feedback_sample(
                        img, _display_to_key(cfg, pick), cfg,
                        prev_pred=res.verdict, source=res.source, mode=res.mode,
                        checkpoint=ckpt_name,
                    )
                    st.session_state.setdefault("fb_saved", set()).add(saved_flag)
                    st.rerun()
        else:
            st.caption(
                "Для панорамы исправляются отдельные участки — вкладка "
                "«Спорные участки» выше. Целая панорама как один пример "
                "обучению не помогает."
            )
        stats = feedback_stats(cfg)
        st.caption(
            "Собрано примеров: " + " · ".join(
                f"{cfg['classes']['display'][c]}: **{n}**" for c, n in stats.items()
            ) + " — дообучение во вкладке «🎓 Дообучение»."
        )


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
                src_keep = src  # путь: нативная вырезка тайлов при коррекции
            else:
                raw = src.getvalue()
                im = Image.open(io.BytesIO(raw))
                im = ImageOps.exif_transpose(im)
                im = im.convert("RGB")
                # байты храним для экспертной коррекции (кроме гигантских файлов)
                src_keep = raw if len(raw) < 300 * 1024 * 1024 else None
            res = predictor.predict(im, mode=mode, progress_cb=cb)
            res.source = name
            rescore(res, cfg, talc_thr)
            st.session_state["results"][cache_key] = {"result": res, "src": src_keep}
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
        from orevision.model import (
            BASE_CHECKPOINT,
            list_checkpoints,
            migrate_legacy_checkpoint,
        )

        st.header("Модель")
        models_dir = Path(cfg["train"]["out_dir"])
        migrate_legacy_checkpoint(models_dir)  # best.pt -> base.pt (совместимость)
        ckpts = list_checkpoints(models_dir)   # base.pt первой
        if not ckpts:
            st.error(f"Нет чекпойнтов в {models_dir}. Обучите модель: `python -m orevision.train`")
            st.stop()
        names = [p.name for p in ckpts]
        # после дообучения новая модель выбирается автоматически
        want = st.session_state.pop("select_ckpt", None)
        if "ckpt_select" not in st.session_state or st.session_state["ckpt_select"] not in names:
            st.session_state["ckpt_select"] = names[0]
        if want in names:
            st.session_state["ckpt_select"] = want
        ckpt_name = st.selectbox(
            "Чекпойнт", names, key="ckpt_select",
            help=f"«{BASE_CHECKPOINT}» — базовая модель; «…_дообученная» / "
                 "«…_переобученная» — обученные вами версии. Выбор здесь и есть "
                 "переключение активной модели (в т.ч. возврат к базовой).",
        )
        ckpt = models_dir / ckpt_name
        predictor = get_predictor(str(ckpt))
        kind = "базовая" if ckpt_name in (BASE_CHECKPOINT, "best.pt") else "дообученная"
        meta_metrics = predictor.info.get("meta", {}).get("metrics", {})
        if meta_metrics:
            st.caption(
                f"{kind} · val macro-F1: **{meta_metrics.get('val_macro_f1', 0):.3f}** · "
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

    tab_an, tab_batch, tab_learn, tab_about = st.tabs(
        ["🔬 Анализ изображений", "📁 Пакетная обработка", "🎓 Дообучение", "ℹ️ О модели и данных"]
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
            render_result(sel, results[sel], cfg, alpha, ckpt_name=ckpt.name)

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

    # ----------------------------------------------------------- дообучение
    with tab_learn:
        if done_msg := st.session_state.pop("train_done_msg", None):
            st.success(done_msg)
        st.markdown(
            "**Режим экспертной проверки (active learning).** Исправленные вами "
            "примеры (вкладки «Спорные участки» и «Экспертная коррекция») копятся "
            "в наборе дообучения и включаются в обучение как отдельный источник — "
            "всегда в train, никогда в валидацию."
        )
        stats = feedback_stats(cfg)
        sc = st.columns(4)
        for i, (c, n) in enumerate(stats.items()):
            sc[i].metric(cfg["classes"]["display"][c], n)
        sc[3].metric("Всего", sum(stats.values()))

        from orevision.feedback import clear_feedback, delete_feedback_samples

        log_path = Path(cfg["data"]["manifest"]).parent / "feedback" / "feedback_log.csv"
        if log_path.exists() and sum(stats.values()) > 0:
            with st.expander("Журнал исправлений (здесь же можно удалять)"):
                fb_df = pd.read_csv(log_path, encoding="utf-8-sig")
                view = fb_df.iloc[::-1].reset_index(drop=True)  # свежие сверху
                event = st.dataframe(
                    view, width="stretch", hide_index=True,
                    on_select="rerun", selection_mode="multi-row",
                    key="fb_log_table",
                )
                picked = list(getattr(event.selection, "rows", []) or [])
                dc1, dc2, _ = st.columns([2, 2, 3])
                if dc1.button(
                    f"🗑 Удалить выбранные ({len(picked)})",
                    disabled=not picked, key="fb_del_sel",
                ):
                    n = delete_feedback_samples(
                        cfg, view.iloc[picked]["saved_as"].tolist()
                    )
                    # снять отметки «✓ в наборе», чтобы участок можно было
                    # сохранить заново с правильным классом
                    st.session_state.pop("fb_saved", None)
                    st.toast(f"Удалено примеров: {n}")
                    st.rerun()
                confirm = dc2.checkbox("подтверждаю", key="fb_clear_confirm")
                if dc2.button(
                    "🗑 Очистить весь набор", disabled=not confirm, key="fb_clear_all",
                ):
                    n = clear_feedback(cfg)
                    st.session_state.pop("fb_saved", None)
                    st.toast(f"Набор дообучения очищен ({n} примеров)")
                    st.rerun()

        st.divider()

        # обучение возможно только там, где лежит обучающий датасет: фидбэк
        # подмешивается к нему, на 1-5 исправлениях модель не дообучить
        from orevision.config import save_local_override
        from orevision.data import resolve_training_sources

        ds = resolve_training_sources(cfg)
        can_train = bool(ds["sources"])
        if can_train:
            st.caption(
                f"Датасет: `{ds['root']}` · " + " · ".join(
                    f"{cfg['classes']['display'][c].split()[0].lower()}: {n}"
                    for c, n in ds["counts"].items()
                ) + (" · структура: своя (папки-классы)" if ds["layout"] == "generic" else "")
            )
        else:
            st.error(
                "**Дообучение на этой машине пока недоступно** — "
                + "; ".join(ds["problems"]) + ".\n\n"
                "Обучение использует обучающий датасет плюс ваши исправления "
                "(на нескольких исправлениях в отрыве от датасета модель не дообучить). "
                "Сбор исправлений при этом работает — они копятся в `data/feedback/`, "
                "и папку можно перенести на машину с датасетом."
            )
            with st.expander("📂 Указать путь к датасету", expanded=True):
                st.markdown(
                    "Подойдёт **датасет хакатона** (папка «Задача 3. Скажи мне, кто твой "
                    "шлиф») **или свой набор** — папка с тремя подпапками по классам "
                    "(`рядовые`/`ordinary`, `тонкие`/`hard`, `оталькованные`/`talc`; "
                    "внутри JPG/PNG/TIFF/BMP, рекомендуем от 30 фото на класс)."
                )
                new_root = st.text_input(
                    "Путь к папке датасета", key="ds_root_input",
                    placeholder=r"D:\данные\Задача 3. Скажи мне, кто твой шлиф",
                )
                if st.button("Проверить и сохранить", key="ds_root_save"):
                    probe_cfg = {**cfg, "data": {**cfg["data"], "root": new_root.strip()}}
                    probe = resolve_training_sources(probe_cfg)
                    if probe["sources"]:
                        save_local_override(
                            cfg, {"data": {"root": new_root.strip()}}
                        )
                        st.success(
                            "Датасет найден ("
                            + ", ".join(f"{c}: {n}" for c, n in probe["counts"].items())
                            + "). Путь сохранён в config.local.yaml."
                        )
                        st.rerun()
                    else:
                        st.error("Не похоже на датасет: " + "; ".join(probe["problems"]))
        from orevision.model import (
            BASE_CHECKPOINT,
            SUFFIX_FINETUNE,
            SUFFIX_RETRAIN,
            delete_checkpoint,
            is_base_checkpoint,
            next_output_name,
        )

        st.caption(
            f"Активная модель: **{ckpt.name}**. Обучение создаёт **новый** чекпойнт "
            f"(«…{SUFFIX_FINETUNE}» / «…{SUFFIX_RETRAIN}») — базовая `{BASE_CHECKPOINT}` "
            "не меняется. Чтобы вернуться к базовой, просто выберите её в списке "
            "чекпойнтов слева."
        )
        c1, c2, c3 = st.columns(3)
        quick = c1.button(
            "⚡ Быстрое дообучение", type="primary", disabled=not can_train,
            help="Тёплый старт от активной модели, несколько эпох с малым lr — "
                 "быстро подхватывает исправления. Результат: новый файл …_дообученная.pt",
        )
        full = c2.button(
            "🔁 Полное переобучение", disabled=not can_train,
            help="Обучение с нуля от ImageNet-весов на всех данных + исправления. "
                 "Результат: новый файл …_переобученная.pt",
        )
        # удаление лишней производной версии (базовую удалить нельзя)
        if not is_base_checkpoint(ckpt) and c3.button(
            "🗑 Удалить эту версию",
            help=f"Удалить активную модель «{ckpt.name}» из списка. "
                 f"Базовая `{BASE_CHECKPOINT}` не затрагивается.",
        ):
            delete_checkpoint(ckpt)
            get_predictor.clear()
            st.session_state["select_ckpt"] = BASE_CHECKPOINT
            st.session_state.get("results", {}).clear()
            st.toast(f"Удалено: {ckpt.name}. Активна базовая модель.")
            st.rerun()

        if quick and sum(stats.values()) == 0:
            st.warning(
                "Набор дообучения пуст — быстрому дообучению не на чем учиться. "
                "Сначала исправьте хотя бы несколько примеров (вкладка "
                "«Спорные участки» или «Экспертная коррекция»)."
            )
        elif quick or full:
            models_dir = Path(cfg["train"]["out_dir"])
            if quick:
                out_path = next_output_name(models_dir, ckpt, SUFFIX_FINETUNE)
                train_step = [
                    "orevision.train", "--init-from", str(ckpt),
                    "--out-name", out_path.name, "--epochs", "6", "--lr", "2e-5",
                ]
                label = "Быстрое дообучение"
            else:
                out_path = next_output_name(models_dir, BASE_CHECKPOINT, SUFFIX_RETRAIN)
                train_step = ["orevision.train", "--out-name", out_path.name]
                label = "Полное переобучение"
            steps = [["orevision.tools.build_manifest", "--cache"], train_step]
            with st.status(f"{label}: не закрывайте вкладку…", expanded=True) as status:
                log_area = st.empty()
                ok = True
                from collections import deque

                for step in steps:
                    st.write("`python -m " + " ".join(step) + "`")
                    proc = subprocess.Popen(
                        [sys.executable, "-u", "-m", *step],
                        cwd=str(PROJECT_ROOT),
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, encoding="utf-8", errors="replace",
                        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                    )
                    tail: deque[str] = deque(maxlen=14)
                    for line in proc.stdout:
                        tail.append(line.rstrip())
                        log_area.code("\n".join(tail))
                    if proc.wait() != 0:
                        ok = False
                        break
                if ok and out_path.exists():
                    status.update(label=f"{label}: готово ✅", state="complete")
                    get_predictor.clear()
                    st.session_state["results"].clear()
                    st.session_state["select_ckpt"] = out_path.name  # авто-выбор новой
                    try:
                        m = json.loads(
                            (models_dir / "metrics.json").read_text(encoding="utf-8")
                        )
                        quality = (
                            f"Val macro-F1: **{m.get('val_macro_f1', 0):.3f}**, "
                            f"AUC: **{m.get('val_auc_ovr', 0):.3f}**. "
                        )
                    except Exception:
                        quality = ""
                    # сообщение показывается ПОСЛЕ rerun — иначе rerun его съедает
                    st.session_state["train_done_msg"] = (
                        f"Готово: создана модель **{out_path.name}** и выбрана "
                        f"активной. {quality}Базовая `{BASE_CHECKPOINT}` осталась "
                        "в списке — переключайтесь в сайдбаре в любой момент. "
                        "Полный лог обучения: models/train_log.txt"
                    )
                    st.rerun()
                else:
                    status.update(label=f"{label}: ошибка ❌", state="error")
                    st.error("Обучение завершилось с ошибкой — см. лог выше.")

    # ------------------------------------------------------------- о модели
    with tab_about:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Качество модели (валидация)")
            st.caption(f"Активный чекпойнт: **{ckpt.name}** ({kind})")
            # ВСЁ (числа и матрица ошибок) — из meta выбранной модели; глобальные
            # metrics.json/confusion_matrix.png описывают лишь последнее обучение
            # и после дообучения не совпадали бы с выбором в сайдбаре
            m = predictor.info.get("meta", {}).get("metrics", {})
            if m:
                mc1, mc2 = st.columns(2)
                mc1.metric("Macro-F1", f"{m.get('val_macro_f1', 0):.3f}")
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
                                "n": [p.get("n", 0) for p in per_part.values()],
                                "Macro-F1": [f"{p.get('macro_f1', 0):.3f}" for p in per_part.values()],
                            }
                        )
                    )
                cm_data = m.get("confusion_matrix")
                if cm_data:
                    from orevision.viz import confusion_matrix_figure

                    display_names = [
                        cfg["classes"]["display"][c] for c in cfg["classes"]["order"]
                    ]
                    st.pyplot(confusion_matrix_figure(cm_data, display_names))
                    st.caption(f"Матрица ошибок модели «{ckpt.name}» на валидации")
            else:
                st.info(
                    "В этом чекпойнте не сохранены метрики валидации "
                    "(вероятно, он обучен старой версией). Переобучите модель — "
                    "новые чекпойнты хранят метрики внутри себя."
                )
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
