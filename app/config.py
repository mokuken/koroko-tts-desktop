"""Persistent settings stored in ``config.json`` next to the app.

Everything is local. No network, no registry, no telemetry.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from .utils import config_path, default_filename, output_dir

log = logging.getLogger(__name__)

APP_VERSION = 1

#: Speech speeds offered in the UI (multiplier; 1.00 == Supertonic reference).
SPEED_CHOICES: tuple[float, ...] = (
    0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20,
)
DEFAULT_SPEED = 1.00
DEFAULT_VOICE = "F3"
MIN_SPEED, MAX_SPEED = 0.5, 2.0

DEFAULT_WIDTH = 960
DEFAULT_HEIGHT = 800
MIN_WIDTH, MIN_HEIGHT = 620, 560


DEFAULTS: dict[str, Any] = {
    "version": APP_VERSION,
    "voice": DEFAULT_VOICE,
    "speed": DEFAULT_SPEED,
    "language": "en",
    "total_steps": 8,
    "prefer_gpu": False,
    "trim_silence": True,
    "last_output_dir": "",
    "window": {"width": DEFAULT_WIDTH, "height": DEFAULT_HEIGHT, "x": None, "y": None},
}


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _safe_int(value: Any, fallback: int, low: int, high: int) -> int:
    try:
        return max(low, min(high, int(value)))
    except (TypeError, ValueError):
        return fallback


def _safe_bool(value: Any, fallback: bool) -> bool:
    return value if isinstance(value, bool) else fallback


def _sanitise(raw: Any) -> dict[str, Any]:
    """Coerce arbitrary JSON into a valid settings dict."""
    data = raw if isinstance(raw, dict) else {}
    out = dict(DEFAULTS)
    out["window"] = dict(DEFAULTS["window"])

    voice = data.get("voice")
    if isinstance(voice, str) and 0 < len(voice.strip()) <= 32:
        out["voice"] = voice.strip()

    try:
        speed = float(data.get("speed", DEFAULT_SPEED))
    except (TypeError, ValueError):
        speed = DEFAULT_SPEED
    out["speed"] = round(_clamp(speed, MIN_SPEED, MAX_SPEED), 3)

    language = data.get("language")
    if isinstance(language, str) and len(language) <= 8:
        out["language"] = language.strip() or "en"

    out["total_steps"] = _safe_int(data.get("total_steps"), 8, 1, 32)
    out["prefer_gpu"] = _safe_bool(data.get("prefer_gpu"), False)
    out["trim_silence"] = _safe_bool(data.get("trim_silence"), True)

    last_dir = data.get("last_output_dir")
    if isinstance(last_dir, str) and len(last_dir) <= 4096:
        out["last_output_dir"] = last_dir

    window = data.get("window")
    if isinstance(window, dict):
        out["window"] = {
            "width": _safe_int(
                window.get("width"), DEFAULT_WIDTH, MIN_WIDTH, 10000
            ),
            "height": _safe_int(
                window.get("height"), DEFAULT_HEIGHT, MIN_HEIGHT, 10000
            ),
            "x": _safe_int(window.get("x"), 0, -32000, 32000)
            if isinstance(window.get("x"), (int, float))
            else None,
            "y": _safe_int(window.get("y"), 0, -32000, 32000)
            if isinstance(window.get("y"), (int, float))
            else None,
        }
    return out


class Config:
    """Tiny JSON-backed settings object with corruption recovery."""

    def __init__(self, path: Path | None = None, data: dict[str, Any] | None = None):
        self.path = Path(path) if path is not None else config_path()
        self.data: dict[str, Any] = _sanitise(data if data is not None else {})

    # -- io ----------------------------------------------------------------- #
    @classmethod
    def load(cls, path: Path | None = None) -> "Config":
        target = Path(path) if path is not None else config_path()
        if not target.exists():
            log.info("No config at %s; using defaults", target)
            return cls(target)
        try:
            with open(target, "r", encoding="utf-8") as handle:
                return cls(target, json.load(handle))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
            log.warning("config.json unusable (%s); falling back to defaults", exc)
            cfg = cls(target)
            try:
                backup = target.with_name(target.name + ".corrupt")
                os.replace(target, backup)
                log.warning("Moved unusable config to %s", backup)
            except OSError:
                pass
            return cfg

    def save(self) -> bool:
        """Atomically persist. Returns False (and logs) if the disk says no."""
        self.data["version"] = APP_VERSION
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            handle = tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=str(self.path.parent),
                prefix=".config-",
                suffix=".tmp",
                delete=False,
            )
            try:
                json.dump(self.data, handle, indent=2, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            finally:
                handle.close()
            os.replace(handle.name, self.path)
            return True
        except (OSError, TypeError, ValueError) as exc:
            log.error("Could not save config to %s: %s", self.path, exc)
            return False

    # -- typed accessors ---------------------------------------------------- #
    @property
    def voice(self) -> str:
        return self.data["voice"]

    @voice.setter
    def voice(self, value: str) -> None:
        self.data["voice"] = str(value)

    @property
    def speed(self) -> float:
        return float(self.data["speed"])

    @speed.setter
    def speed(self, value: float) -> None:
        self.data["speed"] = round(_clamp(float(value), MIN_SPEED, MAX_SPEED), 3)

    @property
    def language(self) -> str:
        return str(self.data["language"])

    @language.setter
    def language(self, value: str) -> None:
        self.data["language"] = str(value)

    @property
    def total_steps(self) -> int:
        return int(self.data["total_steps"])

    @total_steps.setter
    def total_steps(self, value: int) -> None:
        self.data["total_steps"] = _safe_int(value, 8, 1, 32)

    @property
    def prefer_gpu(self) -> bool:
        return bool(self.data["prefer_gpu"])

    @prefer_gpu.setter
    def prefer_gpu(self, value: bool) -> None:
        self.data["prefer_gpu"] = _safe_bool(bool(value), False)

    @property
    def trim_silence(self) -> bool:
        return bool(self.data["trim_silence"])

    @trim_silence.setter
    def trim_silence(self, value: bool) -> None:
        self.data["trim_silence"] = _safe_bool(bool(value), True)

    @property
    def last_output_dir(self) -> Path:
        raw = self.data.get("last_output_dir") or ""
        path = Path(raw) if raw else output_dir()
        return path

    @last_output_dir.setter
    def last_output_dir(self, value: Path | str) -> None:
        self.data["last_output_dir"] = str(value)

    @property
    def window(self) -> dict[str, Any]:
        return self.data["window"]

    def window_size(self) -> tuple[int, int]:
        win = self.window
        return int(win["width"]), int(win["height"])

    def window_position(self) -> tuple[int | None, int | None]:
        win = self.window
        return win.get("x"), win.get("y")

    def set_window(self, width: int, height: int, x: int | None, y: int | None) -> None:
        self.data["window"] = {
            "width": _safe_int(width, DEFAULT_WIDTH, MIN_WIDTH, 10000),
            "height": _safe_int(height, DEFAULT_HEIGHT, MIN_HEIGHT, 10000),
            "x": int(x) if isinstance(x, (int, float)) else None,
            "y": int(y) if isinstance(y, (int, float)) else None,
        }

    # -- misc --------------------------------------------------------------- #
    def suggested_filename(self, extension: str = ".wav") -> str:
        return default_filename(extension)