from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Iterator

import pyaudio
from vosk import KaldiRecognizer, Model, SetLogLevel

from .config import STTConfig


SetLogLevel(-1)  # глушим вывод нативной библиотеки Vosk


class STT:
    """Распознавание речи через Vosk + микрофон через PyAudio."""

    def init(self, cfg: STTConfig):
        if not cfg.model_path.exists():
            raise FileNotFoundError(
                f"Модель Vosk не найдена: {cfg.model_path}. "
                "Скачайте её на https://alphacephei.com/vosk/models и распакуйте."
            )
        self.cfg = cfg
        self.model = Model(str(cfg.model_path))
        self.recognizer = KaldiRecognizer(self.model, cfg.sample_rate)

    @contextmanager
    def _audio_stream(self):
        pa = pyaudio.PyAudio()
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.cfg.sample_rate,
            input=True,
            frames_per_buffer=self.cfg.block_size // 2,
        )
        try:
            yield stream
        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()

    def listen(self) -> Iterator[str]:
        """Бесконечный генератор распознанных непустых фраз."""
        with self._audio_stream() as stream:
            while True:
                data = stream.read(self.cfg.block_size, exception_on_overflow=False)
                if self.recognizer.AcceptWaveform(data):
                    result = json.loads(self.recognizer.Result())
                    text = (result.get("text") or "").strip().lower()
                    if text:
                        yield text