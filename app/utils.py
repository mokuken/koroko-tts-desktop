"""Paths, logging and small formatting helpers.

Everything is resolved relative to the application folder so the build stays
portable: no Program Files, no %USERPROFILE% dependency, no admin rights.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
import tempfile
from pathlib import Path

APP_NAME = "Supertonic TTS"
APP_SLUG = "SupertonicTTS"

_dirs_cache: dict[str, Path] = {}
_logging_ready = False


# --------------------------------------------------------------------------- #
# paths
# --------------------------------------------------------------------------- #
def is_frozen() -> bool:
    """True when running from a PyInstaller bundle."""
    return bool(getattr(sys, "frozen", False))


def base_dir() -> Path:
    """Folder that contains the app (read-only source of models)."""
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def _writable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


def _writable_root() -> Path:
    """Where config.json / output / logs live.

    Normally the app folder itself (portable). If the app was copied somewhere
    read-only (e.g. Program Files) we fall back to LOCALAPPDATA so the app
    still runs instead of crashing on startup.
    """
    candidate = base_dir()
    if _writable(candidate):
        return candidate
    fallback = Path(os.environ.get("LOCALAPPDATA") or tempfile.gettempdir()) / APP_SLUG
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def _dirs() -> dict[str, Path]:
    if _dirs_cache:
        return _dirs_cache
    base = base_dir()
    root = _writable_root()
    _dirs_cache.update(
        root=root,
        base=base,
        models=base / "models",
        supertonic=base / "models" / "supertonic",
        voices=base / "voices",
        output=root / "output",
        logs=root / "logs",
        config=root / "config.json",
    )
    return _dirs_cache


def app_root() -> Path:
    """Writable app folder (config.json, output, logs)."""
    return _dirs()["root"]


def models_dir() -> Path:
    """Folder searched for Supertonic model files."""
    return _dirs()["models"]


def supertonic_dir() -> Path:
    """Canonical model location: <app>/models/supertonic."""
    return _dirs()["supertonic"]


def user_voices_dir() -> Path:
    """Folder where the user can drop extra voice styles."""
    return _dirs()["voices"]


def output_dir() -> Path:
    d = _dirs()["output"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def logs_dir() -> Path:
    d = _dirs()["logs"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def config_path() -> Path:
    return _dirs()["config"]


def portable() -> bool:
    """True when settings live next to the executable (normal portable case)."""
    return app_root() == base_dir()


# --------------------------------------------------------------------------- #
# logging
# --------------------------------------------------------------------------- #
def setup_logging(verbose: bool = False) -> Path:
    """Configure rotating file logging to logs/app.log. Idempotent."""
    global _logging_ready
    log_file = logs_dir() / "app.log"
    if _logging_ready:
        return log_file

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    for h in list(root.handlers):
        root.removeHandler(h)

    try:
        handler: logging.Handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=2 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
    except OSError:
        handler = logging.StreamHandler(sys.stderr)

    handler.setFormatter(
        logging.Formatter("%(asctime)s  %(levelname)-7s  %(name)s: %(message)s")
    )
    root.addHandler(handler)

    if verbose:
        stream = logging.StreamHandler(sys.stdout)
        stream.setFormatter(logging.Formatter("%(levelname)-7s %(name)s: %(message)s"))
        root.addHandler(stream)

    _logging_ready = True
    logging.getLogger(__name__).info("Logging to %s", log_file)
    return log_file


def log_exception(message: str, exc: BaseException | None = None) -> None:
    """Write a full traceback to the log; the GUI only shows ``message``."""
    log = logging.getLogger("supertonic_tts")
    if exc is not None:
        log.error("%s\n%s", message, "".join(_format_tb(exc)), exc_info=True)
    else:
        log.exception(message)


def _format_tb(exc: BaseException) -> str:
    import traceback

    return "".join(
        traceback.format_exception(type(exc), exc, exc.__traceback__)
    )


# --------------------------------------------------------------------------- #
# formatting
# --------------------------------------------------------------------------- #
def format_time(seconds: float) -> str:
    """Seconds -> MM:SS (or H:MM:SS past an hour)."""
    if seconds is None or seconds < 0 or seconds != seconds:  # NaN guard
        seconds = 0.0
    total = int(round(float(seconds)))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def default_filename(extension: str = ".wav") -> str:
    """narration_2026-10-05_2330.wav"""
    from datetime import datetime

    return "narration_" + datetime.now().strftime("%Y-%m-%d_%H%M") + extension


def format_size(num_bytes: float) -> str:
    step = 1024.0
    for unit in ("B", "KB", "MB", "GB"):
        if abs(num_bytes) < step or unit == "GB":
            return f"{num_bytes:.0f} {unit}" if unit == "B" else f"{num_bytes:.1f} {unit}"
        num_bytes /= step
    return f"{num_bytes:.1f} GB"


# --------------------------------------------------------------------------- #
# misc
# --------------------------------------------------------------------------- #
def open_in_file_manager(path: Path) -> bool:
    """Reveal a path in Explorer. Returns False if it could not be opened."""
    path = Path(path)
    target = str(path if path.exists() else path.parent)
    try:
        if path.exists():
            os.startfile(target)  # noqa: S606 - Windows shell open
        else:
            os.startfile(target)  # noqa: S606
        return True
    except OSError:
        return False