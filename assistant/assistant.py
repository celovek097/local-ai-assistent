from __future__ import annotations

import logging

from .config import Config
from .history import History
from .llm import LLM
from .stt import STT
from .tts import build_tts


log = logging.getLogger(name)


class Assistant:
    """Цикл: слушаем микрофон → распознаём → спрашиваем LLM → озвучиваем."""

    def init(self, cfg: Config):
        self.cfg = cfg
        self.history = History(cfg.history.file)
        if cfg.history.clear_on_start:
            self.history.clear()
        log.info("Загружаю STT-модель Vosk...")
        self.stt = STT(cfg.stt)
        log.info("Загружаю TTS-голос Piper...")
        self.tts = build_tts(cfg.tts)
        self.llm = LLM(cfg.llm, self.history)

    def _is_exit(self, text: str) -> bool:
        phrase = self.cfg.stt.exit_phrase
        return bool(phrase) and phrase in text

    def run(self) -> None:
        log.info("Готов слушать. Скажите «%s» для выхода.", self.cfg.stt.exit_phrase)
        for text in self.stt.listen():
            log.info("Вы: %s", text)
            if self._is_exit(text):
                log.info("Завершаю работу.")
                return
            try:
                answer = self.llm.respond(text)
            except Exception:
                log.exception("Ошибка обращения к Ollama")
                continue
            log.info("Ассистент: %s", answer)
            try:
                self.tts.speak(answer)
            except Exception:
                log.exception("Ошибка синтеза речи")