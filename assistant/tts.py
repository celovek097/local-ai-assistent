from __future__ import annotations

from typing import Protocol

import numpy as np
import sounddevice as sd
from piper import PiperVoice

from .config import TTSConfig


class TTSEngine(Protocol):
    def speak(self, text: str) -> None: ...


class PiperTTS:
    """TTS на базе Piper. Поддерживает выбор голоса и тонкую настройку синтеза."""

    def init(self, cfg: TTSConfig):
        if not cfg.voice.exists():
            raise FileNotFoundError(
                f"Голос Piper не найден: {cfg.voice}. "
                "Скачайте .onnx (и .onnx.json) на "
                "https://huggingface.co/rhasspy/piper-voices"
            )
        self.cfg = cfg
        self.voice = PiperVoice.load(str(cfg.voice))
        self._sample_rate = self.voice.config.sample_rate
        # Параметры синтеза, передаваемые в Piper.
        self._synth_kwargs = self._build_synth_kwargs()

    def _build_synth_kwargs(self) -> dict:
        """Не все версии Piper принимают одинаковые kwargs — поэтому мягко пробуем."""
        return {
            "length_scale": self.cfg.length_scale,
            "noise_scale": self.cfg.noise_scale,
            "noise_w": self.cfg.noise_w,
        }

    def _synthesize(self, text: str):
        try:
            return self.voice.synthesize_stream_raw(text, **self._synth_kwargs)
        except TypeError:
            # Старые/новые версии могут не поддерживать эти kwargs.
            return self.voice.synthesize_stream_raw(text)

    def _apply_volume(self, block: np.ndarray) -> np.ndarray:
        if self.cfg.volume == 1.0:
            return block
        scaled = block.astype(np.int32) * self.cfg.volume
        return np.clip(scaled, np.iinfo(np.int16).min, np.iinfo(np.int16).max).astype(
            np.int16
        )

    def speak(self, text: str) -> None:
        if not text.strip():
            return
        with sd.RawOutputStream(
            samplerate=self._sample_rate,
            channels=1,
            dtype="int16",
            device=self.cfg.output_device,
        ) as stream:
            for audio_bytes in self._synthesize(text):
                block = np.frombuffer(audio_bytes, dtype=np.int16)
                block = self._apply_volume(block)
                stream.write(block.tobytes())


def build_tts(cfg: TTSConfig) -> TTSEngine:
    engine = (cfg.engine or "piper").lower()
    if engine == "piper":
        return PiperTTS(cfg)
    raise ValueError(f"Неизвестный TTS-движок: {cfg.engine}")