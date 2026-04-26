"""Configuration models and YAML loader."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel


class AudioConfig(BaseModel):
    input_device: int | str | None = None
    sample_rate: int = 16000
    channels: int = 1
    chunk_ms: int = 32
    noise_gate_db: float = -40.0
    low_cut_hz: float = 80.0
    high_cut_hz: float = 7500.0
    chime_path: str = "assets/chime.aiff"
    bargein_rms_threshold: float = 2000.0
    bargein_consecutive_chunks: int = 3


class VADConfig(BaseModel):
    threshold: float = 0.35
    min_speech_ms: int = 200
    min_silence_ms: int = 600


class WakeWordConfig(BaseModel):
    phrases: list[str] = ["チチクサ", "ちちくさ"]
    max_duration_ms: int = 3000
    fuzzy_threshold: int = 2


class STTConfig(BaseModel):
    model: str = "holotherapper/lilfugu-8bit"
    language: str = "Japanese"


class LLMConfig(BaseModel):
    model: str = "prism-ml/Ternary-Bonsai-8B-mlx-2bit"
    max_tokens_short: int = 150
    max_tokens_long: int = 500
    temperature: float = 0.7
    top_p: float = 0.9
    port: int = 8080
    base_url: str | None = None


class TTSConfig(BaseModel):
    voice: str | None = None
    rate: int | None = None


class SearchConfig(BaseModel):
    max_results: int = 3


class AppConfig(BaseModel):
    audio: AudioConfig = AudioConfig()
    vad: VADConfig = VADConfig()
    wakeword: WakeWordConfig = WakeWordConfig()
    stt: STTConfig = STTConfig()
    llm: LLMConfig = LLMConfig()
    tts: TTSConfig = TTSConfig()
    search: SearchConfig = SearchConfig()
    conversation_timeout: int = 30
    goodbye_phrases: list[str] = ["さようなら", "終わり", "バイバイ", "おやすみ"]

    @classmethod
    def load(cls, path: str = "config.yaml") -> AppConfig:
        config_path = Path(path)
        if config_path.exists():
            with open(config_path) as f:
                data = yaml.safe_load(f) or {}
            return cls(**data)
        return cls()
