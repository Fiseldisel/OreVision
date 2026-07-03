"""Дострел: скриншот вкладки «Спорные участки»."""

import time
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.options import Options

OUT = Path(__file__).parent / "assets"
opts = Options()
opts.add_argument("--headless=new")
opts.add_argument("--window-size=1680,1150")
opts.add_argument("--force-device-scale-factor=1.5")
driver = webdriver.Edge(options=opts)

try:
    driver.get("http://localhost:8513")
    time.sleep(8)
    btn = next(b for b in driver.find_elements(By.TAG_NAME, "button") if "панорама" in (b.text or ""))
    driver.execute_script("arguments[0].click()", btn)
    time.sleep(16)

    tab = next(t for t in driver.find_elements(By.CSS_SELECTOR, 'button[role="tab"]') if "Спорные" in t.text)
    driver.execute_script("arguments[0].click()", tab)
    time.sleep(3)
    driver.execute_script("arguments[0].scrollIntoView({block:'start'})", tab)
    time.sleep(1)
    driver.save_screenshot(str(OUT / "ui_doubt.png"))
    print("saved ui_doubt.png")
finally:
    driver.quit()
