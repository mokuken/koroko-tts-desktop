"""Locating the Supertonic model and the available voice styles.

The app never downloads anything by itself. If the model is missing it shows
the exact folder to fill and offers a button to open it in Explorer.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from .supertonic_onnx import ONNX_CONFIG_FILES, ONNX_FILES
from .utils import models_dir, supertonic_dir, user_voices_dir

log = logging.getLogger(__name__)

VOICE_RE = re.compile(r"^[A-Za-z]{1,3}[-_]?\d{1,3}$")

INSTALL_HINT = (
    "Place the Supertonic model files in:\n\n"
    "{folder}\n\n"
    "Expected layout:\n"
    "  models\\supertonic\\onnx\\vocoder.onnx\n"
    "  models\\supertonic\\onnx\\vector_estimator.onnx\n"
    "  models\\supertonic\\onnx\\text_encoder.onnx\n"
    "  models\\supertonic\\onnx\\duration_predictor.onnx\n"
    "  models\\supertonic\\onnx\\tts.json\n"
    "  models\\supertonic\\onnx\\unicode_indexer.json\n"
    "  models\\supertonic\\voice_styles\\F3.json   (and the other voices)\n\n"
    "You can get them from the official model repository:\n"
    "  huggingface.co/Supertone/supertonic-3\n\n"
    "or run:  python download_model.py"
)


@dataclass(frozen=True)
class VoiceInfo:
    id: str
    path: Path


@dataclass
class ModelStatus:
    ok: bool = False
    onnx_dir: Path | None = None
    voice_dirs: list[Path] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    sample_rate: int | None = None
    detail: str = ""

    @property
    def display_dir(self) -> Path:
        return self.onnx_dir or supertonic_dir() / "onnx"


# --------------------------------------------------------------------------- #
# discovery
# --------------------------------------------------------------------------- #
def _is_onnx_dir(path: Path) -> bool:
    return path.is_dir() and all(
        (path / name).is_file() for name in ONNX_FILES.values()
    )


def _is_voice_dir(path: Path) -> bool:
    return path.is_dir() and any(_voice_files(path))


def _voice_files(path: Path) -> list[Path]:
    if not path.is_dir():
        return []
    return [
        p
        for p in sorted(path.glob("*.json"))
        if VOICE_RE.match(p.stem) and not p.stem.lower().startswith("tts")
    ]


def find_onnx_dir() -> Path | None:
    """Search the usual spots for a complete ONNX folder."""
    primary = supertonic_dir()
    candidates: list[Path] = [
        primary / "onnx",
        primary,
        models_dir() / "onnx",
    ]
    candidates.extend(sorted(p / "onnx" for p in _subdirs(models_dir())))
    candidates.extend(sorted(p for p in _subdirs(models_dir())))
    candidates.extend(sorted(_subdirs(primary)))

    for path in candidates:
        if _is_onnx_dir(path):
            log.info("Found ONNX model in %s", path)
            return path

    # Last resort: shallow search so a renamed subfolder still works.
    for found in sorted(models_dir().rglob(ONNX_FILES["vocoder"])):
        if _is_onnx_dir(found.parent):
            log.info("Found ONNX model via search in %s", found.parent)
            return found.parent
    return None


def find_voice_dirs() -> list[Path]:
    """Voice folders, canonical model folder first, then user extras."""
    primary = supertonic_dir()
    candidates: list[Path] = [
        primary / "voice_styles",
        primary,
        user_voices_dir(),
    ]
    candidates.extend(sorted(p / "voice_styles" for p in _subdirs(models_dir())))
    result: list[Path] = []
    for path in candidates:
        if _is_voice_dir(path) and path not in result:
            result.append(path)
    return result


def _subdirs(path: Path) -> list[Path]:
    if not path.is_dir():
        return []
    try:
        return [p for p in path.iterdir() if p.is_dir() and not p.name.startswith(".")]
    except OSError:
        return []


# --------------------------------------------------------------------------- #
# voices
# --------------------------------------------------------------------------- #
def list_voices() -> list[VoiceInfo]:
    """Discover voices from disk - never hard-coded."""
    found: dict[str, VoiceInfo] = {}
    for folder in find_voice_dirs():
        for file in _voice_files(folder):
            info = VoiceInfo(id=file.stem.upper(), path=file)
            found.setdefault(info.id, info)  # first folder wins
    return [found[key] for key in sorted(found)]


def find_voice(voice_id: str) -> VoiceInfo | None:
    if not voice_id:
        return None
    target = voice_id.strip().upper()
    for info in list_voices():
        if info.id == target:
            return info
    return None


def default_voice() -> str | None:
    """Prefer F3 (English female), then any F*, then anything at all."""
    voices = [v.id for v in list_voices()]
    for candidate in ("F3", "F1", "F2"):
        if candidate in voices:
            return candidate
    for voice in voices:
        if voice.startswith("F"):
            return voice
    return voices[0] if voices else None


def validate_voice_file(path: Path) -> str | None:
    """Return an error message when a voice file is unusable, else None."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        for key in ("style_ttl", "style_dp"):
            if key not in payload or "data" not in payload[key] or "dims" not in payload[key]:
                return f"'{path.name}' is not a valid Supertonic voice style (no {key})."
    except (OSError, json.JSONDecodeError) as exc:
        return f"'{path.name}' could not be read: {exc}"
    return None


# --------------------------------------------------------------------------- #
# status
# --------------------------------------------------------------------------- #
def model_status() -> ModelStatus:
    """Check the installation and describe exactly what is wrong if it is."""
    onnx_dir = find_onnx_dir()
    voices = list_voices()
    status = ModelStatus(onnx_dir=onnx_dir)
    status.voice_dirs = find_voice_dirs()

    if onnx_dir is None:
        expected = supertonic_dir() / "onnx"
        present = [name for name in ONNX_FILES.values() if (expected / name).is_file()]
        if present:
            status.missing = [
                name
                for name in (*ONNX_FILES.values(), *ONNX_CONFIG_FILES)
                if not (expected / name).is_file()
            ]
            status.detail = (
                "The model folder exists but is incomplete.\n\n"
                f"Missing: {', '.join(status.missing)}"
            )
        else:
            status.detail = "No Supertonic model files were found."
        return status

    missing = [
        name
        for name in (*ONNX_CONFIG_FILES, *ONNX_FILES.values())
        if not (onnx_dir / name).is_file()
    ]
    if missing:
        status.missing = missing
        status.detail = "The model folder is incomplete.\n\nMissing: " + ", ".join(missing)
        return status

    if not voices:
        status.detail = (
            "The ONNX model is present but no voice styles were found.\n\n"
            "Add the voice_styles folder (for example F3.json) to:\n"
            f"{supertonic_dir() / 'voice_styles'}"
        )
        return status

    try:
        with open(onnx_dir / "tts.json", "r", encoding="utf-8") as handle:
            status.sample_rate = int(json.load(handle)["ae"]["sample_rate"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        status.detail = f"tts.json is corrupt or unreadable: {exc}"
        return status

    status.ok = True
    status.detail = f"Model ready at {onnx_dir}"
    return status