"""Аудит датасета и построение манифеста со сплитами.

Запуск:  python -m orevision.tools.build_manifest [--config config.yaml] [--cache]
"""

from __future__ import annotations

import argparse
import logging

from orevision.config import load_config
from orevision.data import build_cache, build_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="путь к config.yaml")
    parser.add_argument(
        "--cache", action="store_true", help="сразу построить кэш уменьшенных копий"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    df = build_manifest(cfg)
    if args.cache:
        build_cache(cfg, df)

    excluded = df[df["exclude"]]
    if len(excluded):
        print("\nИсключено файлов:", len(excluded))
        print(excluded.groupby("exclude_reason").size().to_string())


if __name__ == "__main__":
    main()
