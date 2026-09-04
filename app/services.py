from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from app.audio import Audio


QUAIL_MODEL_ID = "quail-vf-2.2-l-16khz"
TYTO_MODEL_ID = "tyto-1.1-l-16khz"
MODEL_DIR = Path("models")

_model_lock = threading.Lock()
_models: dict[str, Any] = {}


def _aic_model(model_id: str):
    import aic_sdk as aic

    with _model_lock:
        if model_id not in _models:
            path = aic.Model.download(model_id, MODEL_DIR)
            _models[model_id] = aic.Model.from_file(path)
        return _models[model_id]


class LiveQuailProcessor:
    def __init__(self, sample_rate: int, block_size: int, level: float):
        import aic_sdk as aic

        license_key = os.environ.get("AIC_SDK_LICENSE")
        if not license_key:
            raise RuntimeError("AIC_SDK_LICENSE is not configured")
        model = _aic_model(QUAIL_MODEL_ID)
        config = aic.ProcessorConfig(sample_rate, block_size, False)
        self.processor = aic.Processor(model, license_key, config)
        self.context = self.processor.get_context()
        self.context.set_parameter(aic.ProcessorParameter.EnhancementLevel, level)
        self.sample_rate = sample_rate

    @property
    def delay_ms(self) -> float:
        return round(self.context.get_audio_delay() * 1000 / self.sample_rate, 1)

    def process(self, samples: np.ndarray) -> np.ndarray:
        return self.processor.process(np.ascontiguousarray(samples, dtype=np.float32))

    def close(self) -> None:
        self.processor.terminate_session()


class LiveTytoAnalyzer:
    dimensions = (
        "risk_score",
        "noise",
        "interfering_speech",
        "speaker_reverb",
        "speaker_loudness",
        "packet_loss",
        "codec_degradation",
    )

    def __init__(self, sample_rate: int, block_size: int):
        import aic_sdk as aic

        license_key = os.environ.get("AIC_SDK_LICENSE")
        if not license_key:
            raise RuntimeError("AIC_SDK_LICENSE is not configured")
        model = _aic_model(TYTO_MODEL_ID)
        self.collector, self.analyzer = aic.analyzer_pair(model, license_key)
        self.collector.initialize(aic.ProcessorConfig(sample_rate, block_size, False))
        self.window_samples = 5 * sample_rate
        self.buffered_samples = 0
        self.smoothed: dict[str, float] = {}

    @property
    def ready(self) -> bool:
        return self.buffered_samples >= self.window_samples

    def buffer(self, samples: np.ndarray) -> None:
        block = np.ascontiguousarray(samples, dtype=np.float32)
        self.collector.buffer(block)
        self.buffered_samples += len(block)

    def analyze(self) -> dict[str, Any] | None:
        if not self.ready:
            return None
        result = self.analyzer.analyze_buffered()
        values = {name: float(getattr(result, name)) for name in self.dimensions}
        for name, value in values.items():
            prior = self.smoothed.get(name)
            self.smoothed[name] = value if prior is None else 0.3 * value + 0.7 * prior
        return {
            "model": TYTO_MODEL_ID,
            "window_seconds": 5,
            **{name: round(value, 3) for name, value in self.smoothed.items()},
        }

    def close(self) -> None:
        self.analyzer.terminate_session()


def enhance_with_quail(audio: Audio, level: float) -> tuple[Audio, dict[str, Any]]:
    import aic_sdk as aic

    started = time.perf_counter()
    license_key = os.environ.get("AIC_SDK_LICENSE")
    if not license_key:
        raise RuntimeError("AIC_SDK_LICENSE is not configured")

    model = _aic_model(QUAIL_MODEL_ID)
    config = aic.ProcessorConfig.optimal(model)
    processor = aic.Processor(model, license_key, config)
    context = processor.get_context()
    context.set_parameter(aic.ProcessorParameter.EnhancementLevel, level)

    block_size = config.block_size
    delay = context.get_audio_delay()
    input_length = len(audio.samples)
    padded_length = ((input_length + delay + block_size - 1) // block_size) * block_size
    padded = np.pad(audio.samples, (0, padded_length - input_length))
    try:
        blocks = [
            processor.process(padded[i : i + block_size])
            for i in range(0, padded_length, block_size)
        ]
        enhanced = np.concatenate(blocks)[delay : delay + input_length]
    finally:
        processor.terminate_session()

    return Audio(np.ascontiguousarray(enhanced, dtype=np.float32)), {
        "model": QUAIL_MODEL_ID,
        "enhancement_level": level,
        "audio_delay_ms": round(delay * 1000 / audio.sample_rate, 1),
        "processing_ms": round((time.perf_counter() - started) * 1000),
    }


def analyze_with_tyto(audio: Audio) -> dict[str, Any]:
    import aic_sdk as aic

    license_key = os.environ.get("AIC_SDK_LICENSE")
    if not license_key:
        raise RuntimeError("AIC_SDK_LICENSE is not configured")
    analyzer = aic.FileAnalyzer(_aic_model(TYTO_MODEL_ID), license_key)
    results = analyzer.analyze(audio.samples, audio.sample_rate, audio.sample_rate)
    dimensions = (
        "risk_score",
        "noise",
        "interfering_speech",
        "speaker_reverb",
        "speaker_loudness",
        "packet_loss",
        "codec_degradation",
    )
    return {
        "model": TYTO_MODEL_ID,
        "windows": len(results),
        "window_seconds": 5,
        "step_seconds": 1,
        **{name: round(float(np.mean([getattr(item, name) for item in results])), 3) for name in dimensions},
    }
