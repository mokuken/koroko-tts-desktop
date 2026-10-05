"""WAV/MP3 writing, chunk joining and lightweight post-processing."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import soundfile as sf

log = logging.getLogger(__name__)

#: Never let concatenated chunks clip.
PEAK_CEILING = 0.99

_mp3_supported: bool | None = None


def silence(seconds: float, sample_rate: int) -> np.ndarray:
    count = max(0, int(round(seconds * sample_rate)))
    return np.zeros(count, dtype=np.float32)


def join_parts(
    parts: list[tuple[np.ndarray, float]], sample_rate: int
) -> np.ndarray:
    """Concatenate ``(samples, pause_after_seconds)`` pairs into one track."""
    if not parts:
        return np.zeros(0, dtype=np.float32)
    pieces: list[np.ndarray] = []
    have_audio = False
    for samples, pause_after in parts:
        chunk = np.asarray(samples, dtype=np.float32).reshape(-1)
        if chunk.size:
            pieces.append(chunk)
            have_audio = True
        if pause_after > 0 and have_audio:
            pieces.append(silence(pause_after, sample_rate))
    if not have_audio:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(pieces).astype(np.float32, copy=False)


def limit_peak(samples: np.ndarray, ceiling: float = PEAK_CEILING) -> np.ndarray:
    """Scale the whole track down only if it would clip (never boosts quiet audio)."""
    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    if samples.size == 0:
        return samples
    peak = float(np.abs(samples).max())
    if peak > ceiling and peak > 0:
        log.info("Limiting peak %.3f -> %.3f", peak, ceiling)
        samples = samples * (ceiling / peak)
    return np.nan_to_num(samples, nan=0.0, posinf=ceiling, neginf=-ceiling)


def trim_silence(
    samples: np.ndarray,
    sample_rate: int,
    threshold: float = 0.0035,
    keep: float = 0.05,
) -> np.ndarray:
    """Trim leading/trailing near-silence introduced by chunk joins."""
    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    if samples.size == 0:
        return samples
    window = max(1, int(0.01 * sample_rate))
    loud = np.abs(samples[: window * ((samples.size // window) or 1)]).max()
    if loud < threshold:
        return samples
    quiet = np.abs(samples) < threshold
    keep_samples = int(keep * sample_rate)
    start = 0
    while start < samples.size and quiet[start]:
        start += 1
    end = samples.size
    while end > start and quiet[end - 1]:
        end -= 1
    start = max(0, start - keep_samples)
    end = min(samples.size, end + keep_samples)
    return samples[start:end]


def write_wav(path: str | Path, samples: np.ndarray, sample_rate: int) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(samples, dtype=np.float32).reshape(-1), sample_rate)
    log.info("Wrote %s (%.1f s)", path, samples.size / float(sample_rate))
    return path


def mp3_supported() -> bool:
    """True when the bundled libsndfile can encode MPEG audio."""
    global _mp3_supported
    if _mp3_supported is not None:
        return _mp3_supported
    try:
        formats = sf.available_formats()
        _mp3_supported = "MP3" in formats
    except Exception:  # pragma: no cover - defensive
        _mp3_supported = False
    log.info("MP3 encoding available: %s", _mp3_supported)
    return _mp3_supported


def write_mp3(path: str | Path, samples: np.ndarray, sample_rate: int) -> Path:
    if not mp3_supported():
        raise RuntimeError("MP3 export is not available in this build.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(samples, dtype=np.float32).reshape(-1), sample_rate)
    log.info("Wrote %s", path)
    return path


def probe_duration(path: str | Path) -> float:
    """Duration in seconds, or 0.0 when the file cannot be read."""
    try:
        return float(sf.info(str(path)).duration)
    except Exception:
        return 0.0