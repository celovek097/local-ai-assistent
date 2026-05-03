from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


class History:
    """Хранилище истории диалога в JSONL-файле (по одному сообщению на строку)."""

    def init(self, path: Path):
        self.path = Path(path)

    def clear(self) -> None:
        self.path.write_text("", encoding="utf-8")

    def append(self, message: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(message, ensure_ascii=False) + "\n")

    def load(self) -> list[dict]:
        if not self.path.exists():
            return []
        messages: list[dict] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    messages.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return messages

    def tail(self, window_pairs: int) -> list[dict]:
        """Вернуть последние window_pairs пар user+assistant."""
        if window_pairs <= 0:
            return []
        msgs = self.load()
        return msgs[-window_pairs * 2 :]

    def extend(self, messages: Iterable[dict]) -> None:
        for m in messages:
            self.append(m)