"""Скриншоты UI для презентации (selenium + headless Edge)."""

import sys
import time
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.options import Options

OUT = Path(__file__).parent / "assets"
OUT.mkdir(parents=True, exist_ok=True)
URL = "http://localhost:8513"

opts = Options()
opts.add_argument("--headless=new")
opts.add_argument("--window-size=1680,1150")
opts.add_argument("--force-device-scale-factor=1.5")  # чётче для слайдов
driver = webdriver.Edge(options=opts)
driver.set_page_load_timeout(60)

try:
    driver.get(URL)
    time.sleep(8)  # инициализация streamlit + загрузка модели

    def click_button_containing(text):
        for b in driver.find_elements(By.TAG_NAME, "button"):
            if text in (b.text or ""):
                driver.execute_script("arguments[0].scrollIntoView({block:'center'})", b)
                time.sleep(0.4)
                b.click()
                return True
        return False

    def shot(name, scroll_to=None, selector=None):
        if selector is not None:
            el = driver.find_element(By.CSS_SELECTOR, selector)
            driver.execute_script("arguments[0].scrollIntoView({block:'start'})", el)
            time.sleep(0.8)
        elif scroll_to is not None:
            driver.execute_script(
                "(document.querySelector('[data-testid=\"stMain\"]')||document.scrollingElement).scrollTo(0, arguments[0])",
                scroll_to,
            )
            time.sleep(0.8)
        driver.save_screenshot(str(OUT / name))
        print("saved", name)

    # 1) главный экран с демо-кнопками
    shot("ui_home.png", scroll_to=0)

    # 2) анализ демо-панорамы
    assert click_button_containing("панорама"), "кнопка демо-панорамы не найдена"
    time.sleep(16)  # анализ ~2-4 c + рендер plotly
    shot("ui_verdict.png", selector='[data-testid="stAlertContainer"]')

    # 3) интерактивная карта классов (plotly)
    for t in driver.find_elements(By.CSS_SELECTOR, 'button[role="tab"]'):
        if t.text.strip() == "Карта классов":
            t.click()
            break
    time.sleep(3)
    shot("ui_classmap.png", selector=".js-plotly-plot")

    # 4) спорные участки
    for t in driver.find_elements(By.CSS_SELECTOR, 'button[role="tab"]'):
        if "Спорные" in t.text:
            t.click()
            break
    time.sleep(2)
    shot("ui_doubt.png", selector='button[role="tab"]')

    print("OK")
finally:
    driver.quit()
