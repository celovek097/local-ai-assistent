from __future__ import annotations

import argparse
import logging
import sys

from assistant import Assistant, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Локальный голосовой ИИ-ассистент (Ollama + Vosk + Piper)"
    )
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        help="Путь к YAML-конфигу (по-умолчанию: config.yaml в корне проекта).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Подробный лог.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    cfg = load_config(args.config)
    try:
        Assistant(cfg).run()
    except KeyboardInterrupt:
        logging.info("Прервано пользователем.")
    return 0


if name == "__main__":
    sys.exit(main())