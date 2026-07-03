"""OreVision — автоматическая классификация руд по OM-изображениям полированных шлифов.

Модули:
  config     — загрузка config.yaml
  data       — манифест датасета, кэш, Dataset для обучения
  transforms — аугментации и препроцессинг
  model      — построение/загрузка модели (transfer learning)
  train      — обучение:            python -m orevision.train
  predict    — инференс-движок (фото + панорамы)
  report     — отчёты: CSV / PDF / текстовое заключение
  viz        — оверлеи и карты уверенности
  cli        — пакетная обработка:  python -m orevision.cli
"""

__version__ = "0.1.0"
