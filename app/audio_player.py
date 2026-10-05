"""Minimal WAV playback: play / pause / stop / seek with a position clock.

Primary backend is ``pygame.mixer.music`` (ships its own SDL DLLs, so the build
stays portable). It is used instead of ``pygame.mixer.Sound`` because the music
interface is the only one exposing pause/unpause/seek.

Note on the backend's quirks, verified against pygame 2.6:

* ``get_pos()`` reports time elapsed since ``play()``, *not* absolute position,
  so an absolute clock is kept here instead.
* ``get_busy()`` is also False while paused, so callers must track their own
  playing flag.
* ``get_pos()`` returns ``-1`` once stopped.

If no output device can be opened we fall back to the stdlib ``winsound``
player; if that also fails, playback is disabled but saving still works.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)

MIXER_RATE = 44100
MIXER_SIZE = -16
MIXER_BUFFER = 1024

#: Never declare "finished" before the mixer has had a moment to start.
STARTUP_GRACE_S = 0.15


class AudioPlayer:
    """Drive every method from the GUI thread; safe to poll from a timer."""

    def __init__(self, on_finished: Callable[[], None] | None = None):
        self.on_finished = on_finished
        self.backend = "none"
        self.error: str = ""
        self.available = False

        self._pygame = None
        self._path: Path | None = None
        self._duration = 0.0
        self._offset = 0.0
        self._anchor = 0.0
        self._playing = False
        self._winsound_until = 0.0
        self._lock = threading.Lock()

        self._init_backends()

    # -- setup -------------------------------------------------------------- #
    def _init_backends(self) -> None:
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        try:
            import pygame

            pygame.mixer.init(
                frequency=MIXER_RATE, size=MIXER_SIZE, channels=2, buffer=MIXER_BUFFER
            )
            if not pygame.mixer.get_init():
                raise RuntimeError("mixer init returned no configuration")
            pygame.mixer.music.set_endevent  # ensure the music module is linked
            self._pygame = pygame
            self.backend = "pygame"
            self.available = True
            log.info("Audio backend: pygame %s mixer %s", pygame.version.ver, pygame.mixer.get_init())
            return
        except Exception as exc:
            log.warning("pygame mixer unavailable (%s)", exc)
            self.error = str(exc)

        try:  # pragma: no cover - Windows only
            import winsound  # noqa: F401

            self.backend = "winsound"
            self.available = True
            log.info("Audio backend: winsound fallback (no seek / pause)")
            return
        except Exception as exc:  # pragma: no cover - non-Windows dev box
            log.warning("winsound unavailable (%s)", exc)

        self.backend = "none"
        self.error = "No audio output device could be opened."

    # -- library ------------------------------------------------------------ #
    def load(self, path: str | Path, duration: float | None = None) -> float:
        """Point the player at a WAV file. Returns the duration in seconds."""
        self.stop()
        path = Path(path)
        if not path.is_file():
            self._clear()
            raise FileNotFoundError(path)

        self._path = path
        self._duration = float(duration) if duration and duration > 0 else 0.0
        self._offset = 0.0

        if self.backend == "pygame":
            if not self._duration:
                from .audio_io import probe_duration

                self._duration = probe_duration(path)
            try:
                self._pygame.mixer.music.load(str(path))
            except Exception as exc:
                log.exception("Failed to load %s", path)
                self._clear()
                raise RuntimeError(f"Could not open the audio file: {exc}") from exc
            log.info("Loaded %s (%.1fs)", path.name, self._duration)
        return self._duration

    def _clear(self) -> None:
        self._path = None
        self._duration = 0.0
        self._offset = 0.0
        self._playing = False

    def unload(self) -> None:
        self.stop()
        self._clear()
        if self.backend == "pygame":
            try:
                self._pygame.mixer.music.unload()
            except Exception:  # pragma: no cover - defensive
                pass

    # -- state -------------------------------------------------------------- #
    @property
    def has_track(self) -> bool:
        return self._path is not None

    @property
    def duration(self) -> float:
        return self._duration

    @property
    def is_playing(self) -> bool:
        return self._playing

    @property
    def supports_seek(self) -> bool:
        return self.backend == "pygame"

    @property
    def supports_pause(self) -> bool:
        return self.backend == "pygame"

    @property
    def position(self) -> float:
        if not self._playing:
            return min(self._offset, self._duration) if self._duration else 0.0
        elapsed = time.monotonic() - self._anchor
        if self._duration:
            elapsed = min(elapsed, self._duration)
        return max(0.0, elapsed)

    # -- transport ---------------------------------------------------------- #
    def play(self) -> bool:
        """Start or resume. Returns True when audio is actually running."""
        if not self._path:
            return False
        if self.backend == "pygame":
            if self._duration and self._offset >= self._duration - 0.05:
                self._offset = 0.0
            if self._pygame.mixer.music.get_busy():
                try:
                    self._pygame.mixer.music.pause()
                except Exception:  # pragma: no cover - defensive
                    pass
            try:
                self._pygame.mixer.music.play(start=self._offset)
            except Exception as exc:
                log.warning("Playback failed: %s", exc)
                self._playing = False
                self.available = False
                self.error = f"Audio playback failed: {exc}"
                return False
            self._anchor = time.monotonic()
            self._playing = True
            return True
        return self._play_winsound()  # pragma: no cover - Windows only

    def pause(self) -> bool:
        if not self._playing or self.backend != "pygame":
            return False
        self._offset = self.position
        try:
            self._pygame.mixer.music.pause()
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("Pause failed: %s", exc)
        self._playing = False
        return True

    def toggle(self) -> bool:
        return self.pause() if self._playing else self.play()

    def stop(self) -> None:
        if self.backend == "pygame":
            try:
                self._pygame.mixer.music.stop()
            except Exception:  # pragma: no cover - defensive
                pass
        elif self._playing:  # pragma: no cover - Windows only
            self._stop_winsound()
        self._playing = False
        self._offset = 0.0

    def seek(self, fraction: float) -> None:
        """Jump to a 0..1 position, staying in the current play state."""
        if not self._duration:
            return
        target = max(0.0, min(1.0, float(fraction))) * self._duration
        self._offset = target
        if not self._playing:
            return
        try:
            self._pygame.mixer.music.play(start=target)
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("Seek failed: %s", exc)
            return
        self._anchor = time.monotonic()

    def seek_seconds(self, seconds: float) -> None:
        if self._duration:
            self.seek(max(0.0, float(seconds)) / self._duration)

    # -- polling ------------------------------------------------------------ #
    def poll(self) -> None:
        """Call from a timer. Raises the finished callback at the end of a track."""
        if not self._playing or self.backend != "pygame":
            return
        elapsed = time.monotonic() - self._anchor
        try:
            busy = self._pygame.mixer.music.get_busy()
        except Exception:  # pragma: no cover - defensive
            busy = True
        if busy or elapsed < STARTUP_GRACE_S:
            return
        if self._duration and elapsed < self._duration - 0.02:
            # mixer went quiet early (device grab, another app) - do not rewind
            return
        self._offset = 0.0
        self._playing = False
        if self.on_finished:
            try:
                self.on_finished()
            except Exception:  # pragma: no cover - defensive
                log.exception("on_finished callback failed")

    # -- winsound fallback -------------------------------------------------- #
    def _play_winsound(self) -> bool:  # pragma: no cover - Windows only
        try:
            import winsound

            winsound.PlaySound(
                str(self._path), winsound.SND_FILENAME | winsound.SND_ASYNC
            )
            self._playing = True
            self._winsound_until = time.monotonic() + max(
                0.0, self._duration - self._offset
            )
            return True
        except Exception as exc:
            log.warning("winsound playback failed: %s", exc)
            self.available = False
            self.error = f"Audio playback failed: {exc}"
            return False

    def _stop_winsound(self) -> None:  # pragma: no cover - Windows only
        try:
            import winsound

            winsound.PlaySound(None, winsound.SND_PURGE)
        except Exception:
            pass

    # -- lifecycle ---------------------------------------------------------- #
    def shutdown(self) -> None:
        self.stop()
        if self.backend == "pygame":
            try:
                self._pygame.mixer.quit()
            except Exception:  # pragma: no cover - defensive
                pass
        self.available = False
