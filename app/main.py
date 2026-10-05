"""Entry point for Supertonic TTS.

Usage:
    SupertonicTTS                      normal launch
    SupertonicTTS --verbose            also log to the console
    SupertonicTTS --check              report the installation and exit
    SupertonicTTS --selftest           render a short WAV headlessly and exit
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading


def _console():
    """A writable stdout even in a windowed (--noconsole) build.

    PyInstaller sets ``sys.stdout`` to None for GUI executables. When such an
    exe is started from a terminal we can still reach the real console through
    the CONOUT$ device; otherwise output is dropped silently (the log file is
    always written regardless).
    """
    if sys.stdout is not None:
        return sys.stdout
    try:
        return open("CONOUT$", "w", encoding="utf-8", buffering=1)
    except OSError:
        return None


def say(*parts) -> None:
    stream = _console()
    if stream is not None:
        print(*parts, file=stream)
        try:
            stream.flush()
        except (OSError, ValueError):
            pass


def _check() -> int:
    """Print what the app found on disk. No model loading."""
    from . import model_manager, utils
    from .supertonic_onnx import ONNX_CONFIG_FILES, ONNX_FILES

    say(f"Supertonic TTS {__import__('app').__version__}")
    say(f"  app folder     : {utils.base_dir()}")
    say(f"  settings folder: {utils.app_root()}")
    say(f"  log file       : {utils.setup_logging()}")

    status = model_manager.model_status()
    say(f"  model folder   : {status.display_dir}")
    if status.ok:
        voices = model_manager.list_voices()
        say(f"  model          : OK ({status.sample_rate} Hz, {len(voices)} voices)")
        say(f"  voices         : {', '.join(v.id for v in voices)}")
        say(f"  default voice  : {model_manager.default_voice()}")
        return 0
    say("  model          : NOT READY")
    say(f"  detail         : {status.detail}")
    if status.missing:
        say(f"  missing        : {', '.join(status.missing)}")
    say("")
    say(model_manager.INSTALL_HINT.format(folder=status.display_dir))
    return 1


def _selftest(voice: str, text: str, speed: float) -> int:
    """Render one clip without opening a window. Verifies a portable build."""
    from . import audio_io, utils
    from .config import Config
    from .tts_engine import TTSEngine, TTSError

    log = logging.getLogger("supertonic_tts")
    say(f"App folder : {utils.base_dir()}")
    say(f"Log file   : {utils.setup_logging()}")
    say("Loading the model, please wait ...")
    settings = Config.load()
    engine = TTSEngine(
        prefer_gpu=settings.prefer_gpu, total_steps=settings.total_steps
    )
    try:
        result = engine.synthesize(
            text,
            voice,
            speed=speed,
            on_status=lambda m: say(f"  {m}"),
            on_chunk=lambda i, n: say(f"  chunk {i}/{n}"),
        )
    except TTSError as exc:
        say(f"\nFAILED\n{exc}")
        log.error("Self-test failed: %s", exc)
        return 1

    target = utils.output_dir() / "selftest.wav"
    audio_io.write_wav(target, result.samples, result.sample_rate)
    say(
        f"\nOK  {result.duration:.2f}s of {result.sample_rate} Hz audio "
        f"in {result.elapsed:.2f}s ({result.real_time_factor:.2f}x realtime)"
    )
    say(f"    voice    : {result.voice}   speed: {result.speed:.2f}")
    say(f"    runtime  : {result.provider}")
    say(f"    written  : {target}")
    return 0


def _install_excepthook() -> None:
    """Never let a traceback kill the app without a word in the log."""
    from .utils import logs_dir, log_exception

    def hook(exc_type, exc_value, exc_tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        log_exception("Unhandled exception", exc_value)
        try:
            import tkinter.messagebox as mb

            mb.showerror(
                "Supertonic TTS",
                "Something went wrong.\n\n"
                f"{exc_type.__name__}: {exc_value}\n\n"
                f"Details were written to:\n{logs_dir() / 'app.log'}",
            )
        except Exception:
            pass

    sys.excepthook = hook


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="SupertonicTTS", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--verbose", action="store_true", help="also log to the console")
    parser.add_argument("--version", action="store_true", help="print the version and exit")
    parser.add_argument("--check", action="store_true", help="report the installation and exit")
    parser.add_argument(
        "--selftest", action="store_true",
        help="render a short WAV without a window, then exit",
    )
    parser.add_argument("--voice", default="F3", help="voice used by --selftest")
    parser.add_argument("--text", default=None, help="text used by --selftest")
    args = parser.parse_args(argv)

    from . import __version__, utils

    if args.version:
        say(f"Supertonic TTS {__version__}")
        return 0

    if args.check:
        return _check()

    if args.selftest:
        utils.setup_logging(verbose=args.verbose)
        from .tts_engine import TTSError

        text = args.text or (
            "Supertonic TTS is running locally on this computer. "
            "Nothing is uploaded anywhere."
        )
        try:
            return _selftest(args.voice, text, 1.0)
        except TTSError as exc:
            say(f"\nFAILED\n{exc}")
            return 1

    log_file = utils.setup_logging(verbose=args.verbose)
    log = logging.getLogger("supertonic_tts")
    _install_excepthook()

    log.info("=" * 66)
    log.info("Supertonic TTS %s starting", __version__)
    log.info("Portable folder: %s", utils.base_dir())
    log.info("Settings folder: %s%s", utils.app_root(), "" if utils.portable() else " (read-only app folder)")
    log.info("Log file: %s", log_file)

    # Keep a hard ceiling on worker threads so chunk loops stay predictable.
    threading.stack_size(4 * 1024 * 1024)

    from .gui import run

    run()
    log.info("Supertonic TTS closed cleanly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())