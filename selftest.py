"""Headless end-to-end check of the engine (no GUI)."""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app import audio_io, chunker, model_manager, utils
from app.config import Config
from app.tts_engine import TTSEngine, TTSError


def show(title: str) -> None:
    print("\n" + "=" * 66)
    print(title)
    print("=" * 66)


def main() -> int:
    utils.setup_logging(verbose=False)
    show("Model status")
    status = model_manager.model_status()
    print("ok:", status.ok, "| onnx:", status.onnx_dir)
    print("detail:", status.detail)
    print("sample_rate:", status.sample_rate)
    if not status.ok:
        print(status.detail)
        return 1

    voices = model_manager.list_voices()
    print("voices:", [v.id for v in voices])
    print("default:", model_manager.default_voice())

    show("Config round-trip")
    cfg = Config.load()
    print("loaded:", cfg.data)
    cfg.voice = "F3"
    cfg.speed = 1.0
    print("save:", cfg.save())
    print("reload:", Config.load().data["voice"], Config.load().speed)

    show("Validation")
    engine = TTSEngine(prefer_gpu=False, total_steps=8)
    try:
        engine.synthesize("   ", "F3")
        print("FAIL: empty text accepted")
        return 1
    except TTSError as exc:
        print("empty text ->", str(exc).splitlines()[0])
    try:
        engine.synthesize("Hello there.", "ZZ9")
        print("FAIL: bad voice accepted")
        return 1
    except TTSError as exc:
        print("bad voice ->", str(exc).splitlines()[0])

    show("Short sentence (F3, 1.00)")
    statuses: list[str] = []
    chunks: list[tuple[int, int]] = []
    result = engine.synthesize(
        "Supertonic runs entirely on your own machine. No internet required.",
        "F3",
        speed=1.0,
        on_status=statuses.append,
        on_chunk=lambda i, n: chunks.append((i, n)),
    )
    print("statuses:", statuses)
    print("chunks:", chunks)
    print(
        f"audio={result.duration:.2f}s elapsed={result.elapsed:.2f}s "
        f"rtf={result.real_time_factor:.2f}x voice={result.voice} provider={result.provider}"
    )
    out = utils.output_dir() / "selftest_short.wav"
    audio_io.write_wav(out, result.samples, result.sample_rate)
    print("wrote", out, f"({utils.format_size(out.stat().st_size)})")

    show("Speed control")
    for speed in (0.85, 1.20):
        r = engine.synthesize(
            "The same sentence, rendered at a different speed.",
            "F3",
            speed=speed,
        )
        print(f"speed={speed:.2f} -> {r.duration:.2f}s audio, {r.elapsed:.2f}s compute")

    show("Repeatability (same input twice)")
    a = engine.synthesize("Repeatability check.", "F3", speed=1.0)
    b = engine.synthesize("Repeatability check.", "F3", speed=1.0)
    print("identical:", bool((a.samples == b.samples).all()))

    show("Voice variation")
    for voice in ("M1", "F1", "F5"):
        r = engine.synthesize("Same words, different voice.", voice, speed=1.0)
        print(f"{voice}: {r.duration:.2f}s")

    show("Long-form narration (multi-chunk)")
    script = (
        "Welcome back to the channel. Today we are looking at how a small local "
        "tool can turn a script into narration without ever sending it to a server.\n\n"
        "First, paste your script into the box. Then choose a voice, pick a speed, "
        "and press Generate Speech. The app splits long text at paragraph and "
        "sentence boundaries so the result sounds continuous.\n\n"
        "Dr. Alvarez measured the change in 2024, and the numbers were clear: "
        "e.g. latency dropped from 4.5 s to 1.2 s. Finally, save the file as WAV "
        "and drop it straight into your editor."
    )
    progress: list[tuple[int, int]] = []
    result = engine.synthesize(
        script,
        "F3",
        speed=1.05,
        on_status=statuses.append,
        on_chunk=lambda i, n: progress.append((i, n)),
    )
    print("progress:", progress)
    print(
        f"chunks={result.chunk_count} audio={result.duration:.2f}s "
        f"elapsed={result.elapsed:.2f}s rtf={result.real_time_factor:.2f}x"
    )
    long_out = utils.output_dir() / "selftest_long.wav"
    audio_io.write_wav(long_out, result.samples, result.sample_rate)
    print("wrote", long_out, f"({utils.format_size(long_out.stat().st_size)})")

    show("Cancellation")
    cancel = threading.Event()
    cancel.set()
    try:
        engine.synthesize("This should not render at all.", "F3", cancel=cancel)
        print("FAIL: cancel ignored")
        return 1
    except TTSError as exc:
        print("cancel ->", str(exc))

    show("MP3 export")
    if audio_io.mp3_supported():
        mp3 = utils.output_dir() / "selftest_short.mp3"
        audio_io.write_mp3(mp3, result.samples, result.sample_rate)
        print("wrote", mp3, f"({utils.format_size(mp3.stat().st_size)})")
    else:
        print("MP3 not supported in this soundfile build")

    show("Timing")
    t0 = time.perf_counter()
    engine.unload()
    engine.load()
    print(f"reload took {time.perf_counter() - t0:.2f}s")

    print("\nALL CHECKS COMPLETED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())