from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class LLMConfig:
    model: str = "gemma3:latest"
    system_prompt: str = ""
    history_window: int = 10
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class STTConfig:
    model_path: Path = PROJECT_ROOT / "models" / "vosk-model-small-ru-0.22"
    sample_rate: int = 16000
    block_size: int = 8000
    exit_phrase: str = "выход"


@dataclass
class TTSConfig:
    engine: str = "piper"
    voice: Path = PROJECT_ROOT / "models" / "ru_RU-irina-medium.onnx"
    length_scale: float = 1.0
    noise_scale: float = 0.667
    noise_w: float = 0.8
    volume: float = 1.0
    output_device: int | str | None = None


@dataclass
class HistoryConfig:
    file: Path = PROJECT_ROOT / "history.jsonl"
    clear_on_start: bool = True


@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    history: HistoryConfig = field(default_factory=HistoryConfig)


def _resolve_path(value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_config(path: str | Path | None = None) -> Config:
    """Загрузить YAML-конфиг. Если путь не задан — берётся config.yaml в корне."""
    cfg_path = Path(path) if path else PROJECT_ROOT / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Файл конфигурации не найден: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    llm_raw = raw.get("llm", {})
    stt_raw = raw.get("stt", {})
    tts_raw = raw.get("tts", {})
    hist_raw = raw.get("history", {})

    llm = LLMConfig(
        model=llm_raw.get("model", LLMConfig.model),
        system_prompt=llm_raw.get("system_prompt", ""),
        history_window=int(llm_raw.get("history_window", 10)),
        options=dict(llm_raw.get("options") or {}),
    )

    stt = STTConfig(
        model_path=_resolve_path(stt_raw.get("model_path", STTConfig.model_path)),
        sample_rate=int(stt_raw.get("sample_rate", 16000)),
        block_size=int(stt_raw.get("block_size", 8000)),
        exit_phrase=str(stt_raw.get("exit_phrase", "выход")).lower(),
    )

    tts = TTSConfig(
        engine=tts_raw.get("engine", "piper"),
        voice=_resolve_path(tts_raw.get("voice", TTSConfig.voice)),
        length_scale=float(tts_raw.get("length_scale", 1.0)),
        noise_scale=float(tts_raw.get("noise_scale", 0.667)),
        noise_w=float(tts_raw.get("noise_w", 0.8)),
        volume=float(tts_raw.get("volume", 1.0)),
        output_device=tts_raw.get("output_device"),
    )

    history = HistoryConfig(
        file=_resolve_path(hist_raw.get("file", HistoryConfig.file)),
        clear_on_start=bool(hist_raw.get("clear_on_start", True)),
    )

    return Config(llm=llm, stt=stt, tts=tts, history=history)