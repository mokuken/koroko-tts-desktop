"""High-level TTS orchestration: load once, chunk, synthesise, join.

The GUI owns no synthesis logic; it calls :meth:`TTSEngine.synthesize` from a
worker thread and receives human-readable progress callbacks.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from . import audio_io, chunker, model_manager
from .supertonic_onnx import ModelError, TextToSpeech, load_voice_style

log = logging.getLogger(__name__)

#: Fixed base seed -> identical audio for identical input (repeatable renders).
BASE_SEED = 20260101

StatusCallback = Callable[[str], None]
ProgressCallback = Callable[[int, int], None]

_OOM_MARKERS = (
    "out of memory",
    "bad_alloc",
    "allocat",
    "insufficient memory",
    "memoryaccess",
    "ran out of memory",
)


class TTSError(RuntimeError):
    """A failure that can be shown to the user verbatim."""


class GenerationCancelled(TTSError):
    """Raised when the user presses Cancel."""


@dataclass
class SynthesisResult:
    samples: np.ndarray
    sample_rate: int
    chunk_count: int
    duration: float
    elapsed: float
    voice: str
    speed: float
    provider: str

    @property
    def real_time_factor(self) -> float:
        return self.elapsed / self.duration if self.duration > 0 else 0.0


def _is_out_of_memory(exc: BaseException) -> bool:
    message = f"{type(exc).__name__}: {exc}".lower()
    return isinstance(exc, MemoryError) or any(m in message for m in _OOM_MARKERS)


def _friendly(exc: BaseException) -> str:
    """Turn an exception into something a non-technical user can act on."""
    if isinstance(exc, ModelError):
        return str(exc)
    if isinstance(exc, FileNotFoundError):
        return f"A required file is missing:\n{exc.filename or exc}"
    if isinstance(exc, PermissionError):
        return (
            "Windows denied access to a file.\n\n"
            "The app folder is probably read-only. Move the app to a "
            "writable folder (for example Desktop) and try again."
        )
    if _is_out_of_memory(exc):
        return (
            "Ran out of memory while generating speech.\n\n"
            "Close other applications, or try a shorter script / a lower speed, "
            "then try again."
        )
    if isinstance(exc, ValueError):
        return f"Invalid input:\n{exc}"
    return f"{type(exc).__name__}: {exc}"


class TTSEngine:
    """Owns the ONNX sessions. Safe to use from one worker thread at a time."""

    def __init__(
        self,
        onnx_dir: Path | str | None = None,
        prefer_gpu: bool = False,
        total_steps: int = 8,
    ):
        self._onnx_dir = Path(onnx_dir) if onnx_dir else None
        self._prefer_gpu = bool(prefer_gpu)
        self._total_steps = int(total_steps)
        self._tts: TextToSpeech | None = None
        self._style_cache: dict[str, object] = {}
        self._lock = threading.Lock()

    # -- state -------------------------------------------------------------- #
    @property
    def is_loaded(self) -> bool:
        return self._tts is not None

    @property
    def sample_rate(self) -> int:
        return self._tts.sample_rate if self._tts else 44100

    @property
    def provider_label(self) -> str:
        return self._tts.provider_label if self._tts else "CPU"

    def unload(self) -> None:
        with self._lock:
            self._tts = None
            self._style_cache.clear()

    # -- loading ------------------------------------------------------------ #
    def load(self, on_status: StatusCallback | None = None) -> TextToSpeech:
        """Load the ONNX graph. Cheap enough (~2 s) but kept off the GUI thread."""
        with self._lock:
            if self._tts is not None:
                return self._tts

            status = model_manager.model_status()
            onnx_dir = Path(self._onnx_dir) if self._onnx_dir else status.onnx_dir
            if onnx_dir is None or not onnx_dir.is_dir():
                raise TTSError(
                    "The Supertonic model could not be loaded.\n\n"
                    + model_manager.INSTALL_HINT.format(
                        folder=model_manager.supertonic_dir() / "onnx"
                    )
                )

            if on_status:
                on_status("Loading model...")
            try:
                self._tts = TextToSpeech(
                    str(onnx_dir), prefer_gpu=self._prefer_gpu, on_progress=on_status
                )
            except ModelError as exc:
                raise TTSError(f"The Supertonic model could not be loaded.\n\n{exc}") from exc
            except Exception as exc:
                log.exception("Model load failed")
                raise TTSError(_friendly(exc)) from exc
            return self._tts

    # -- voices ------------------------------------------------------------- #
    def list_voices(self) -> list[model_manager.VoiceInfo]:
        return model_manager.list_voices()

    def _style_for(self, voice_id: str, on_status: StatusCallback | None = None):
        cached = self._style_cache.get(voice_id)
        if cached is not None:
            return cached
        info = model_manager.find_voice(voice_id)
        if info is None:
            available = ", ".join(v.id for v in self.list_voices()) or "none"
            raise TTSError(
                f"Voice '{voice_id}' is not available.\n\nInstalled voices: {available}"
            )
        problem = model_manager.validate_voice_file(info.path)
        if problem:
            raise TTSError(f"The voice '{info.id}' could not be loaded.\n\n{problem}")
        if on_status:
            on_status(f"Preparing voice {info.id}...")
        try:
            style = load_voice_style([str(info.path)])
        except Exception as exc:
            log.exception("Voice load failed for %s", info.path)
            raise TTSError(_friendly(exc)) from exc
        self._style_cache[voice_id] = style
        return style

    # -- synthesis ---------------------------------------------------------- #
    def synthesize(
        self,
        text: str,
        voice_id: str,
        speed: float = 1.0,
        language: str = "en",
        cancel: threading.Event | None = None,
        on_status: StatusCallback | None = None,
        on_chunk: ProgressCallback | None = None,
        trim: bool = True,
    ) -> SynthesisResult:
        """Render narration audio. Blocks the calling thread until finished."""
        if chunker.is_blank(text):
            raise TTSError("There is no text to speak.\n\nPaste or type a script first.")

        cleaned = chunker.clean_text(text)
        tts = self.load(on_status)
        style = self._style_for(voice_id, on_status)
        speed = float(speed)
        max_chars = tts.max_chars_for(language)

        chunks = chunker.chunk_text(cleaned, max_chars=max_chars)
        if not chunks:
            raise TTSError("There is no speakable text in the input.")

        total = len(chunks)
        parts: list[tuple[np.ndarray, float]] = []
        started = time.perf_counter()
        for index, chunk in enumerate(chunks, start=1):
            if cancel is not None and cancel.is_set():
                raise GenerationCancelled("Generation cancelled.")
            if on_status:
                on_status(
                    f"Generating chunk {index} / {total}..."
                    if total > 1
                    else f"Generating... (~{chunker.estimate_duration_seconds(chunk.text, speed):.0f}s)"
                )
            rng = np.random.default_rng(BASE_SEED * 1_000_003 + index)
            try:
                samples, _ = tts.infer(
                    chunk.text,
                    language,
                    style,
                    total_step=self._total_steps,
                    speed=speed,
                    rng=rng,
                )
            except Exception as exc:
                log.exception("Synthesis failed on chunk %d/%d", index, total)
                raise TTSError(
                    "Could not generate speech.\n\nReason:\n" + _friendly(exc)
                ) from exc
            parts.append((samples, chunk.pause_after))
            if on_chunk:
                on_chunk(index, total)

        if on_status:
            on_status("Combining audio...")

        joined = audio_io.join_parts(parts, tts.sample_rate)
        if trim:
            joined = audio_io.trim_silence(joined, tts.sample_rate)
        joined = audio_io.limit_peak(joined)
        elapsed = time.perf_counter() - started

        if joined.size == 0:
            raise TTSError("The model returned empty audio. Try a different script.")

        result = SynthesisResult(
            samples=joined,
            sample_rate=tts.sample_rate,
            chunk_count=total,
            duration=joined.size / float(tts.sample_rate),
            elapsed=elapsed,
            voice=voice_id,
            speed=speed,
            provider=tts.provider_label,
        )
        log.info(
            "Rendered %d chunk(s) -> %.1fs audio in %.1fs (%.2fx realtime, %s)",
            total,
            result.duration,
            elapsed,
            result.real_time_factor,
            result.provider,
        )
        return result


def render_to_file(
    result: SynthesisResult, path: Path, extension: str = ".wav"
) -> Path:
    """Write a finished render to disk (WAV mandatory, MP3 optional)."""
    path = Path(path)
    if extension.lower() == ".mp3" and audio_io.mp3_supported():
        return audio_io.write_mp3(path, result.samples, result.sample_rate)
    if extension.lower() == ".mp3":
        log.info("MP3 unsupported here; writing WAV instead")
    return audio_io.write_wav(path.with_suffix(".wav"), result.samples, result.sample_rate)


def describe_languages() -> Sequence[str]:
    from .supertonic_onnx import AVAILABLE_LANGS

    return AVAILABLE_LANGS