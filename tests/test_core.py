"""Unit tests for Supertonic TTS.

Run with:  python -m unittest discover -s tests -v
No third-party test runner required.
"""

from __future__ import annotations

import datetime
import json
import sys
import tempfile
import threading
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from app import audio_io, chunker, model_manager, utils
from app.config import DEFAULT_SPEED, SPEED_CHOICES, Config
from app.supertonic_onnx import ONNX_FILES, ModelError, TextToSpeech, select_providers
from app.tts_engine import TTSEngine, TTSError

MODEL_PRESENT = model_manager.find_onnx_dir() is not None
needs_model = unittest.skipUnless(MODEL_PRESENT, "Supertonic model is not installed")


# --------------------------------------------------------------------------- #
class TestTextValidation(unittest.TestCase):
    def test_blank_inputs_are_rejected(self):
        for value in ("", "   ", "\n\n", "\t \r\n ", "  ", None):
            with self.subTest(value=repr(value)):
                self.assertTrue(chunker.is_blank(value or ""))

    def test_real_text_passes(self):
        for value in ("Hi", "Hello there.", "3.14", "a", "DON'T"):
            with self.subTest(value=value):
                self.assertFalse(chunker.is_blank(value))

    def test_word_and_char_counts(self):
        self.assertEqual(chunker.word_count("one two three"), 3)
        self.assertEqual(chunker.word_count(""), 0)
        self.assertEqual(chunker.char_count("abcd"), 4)
        self.assertEqual(chunker.word_count("line one\nline two"), 4)

    def test_empty_text_produces_no_chunks(self):
        self.assertEqual(chunker.chunk_text(""), [])
        self.assertEqual(chunker.chunk_text("   \n\n  "), [])

    def test_engine_rejects_blank_text_before_touching_the_model(self):
        engine = TTSEngine()
        with self.assertRaises(TTSError) as ctx:
            engine.synthesize("   \n ", "F3")
        self.assertIn("no text", str(ctx.exception).lower())
        self.assertFalse(engine.is_loaded, "model must not load for invalid input")


# --------------------------------------------------------------------------- #
class TestChunking(unittest.TestCase):
    def test_short_text_is_one_chunk(self):
        chunks = chunker.chunk_text("Hello there, this is short.")
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, "Hello there, this is short.")
        self.assertEqual(chunks[0].pause_after, 0.0)

    def test_no_chunk_exceeds_the_model_limit(self):
        text = " ".join(f"This is sentence number {i} of a long script." for i in range(120))
        chunks = chunker.chunk_text(text, max_chars=300)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk.text), 300, chunk.text)

    def test_paragraph_breaks_always_become_chunk_boundaries(self):
        # A pause can only be inserted between chunks, so a paragraph break in
        # the middle of a long paragraph must still be honoured.
        text = (
            "First paragraph runs long enough to fill most of a single chunk on "
            "its own, so a blank line arrives after a substantial amount of text.\n\n"
            "Second paragraph begins here and continues for a little while longer."
        )
        chunks = chunker.chunk_text(text, max_chars=300)
        self.assertEqual(len(chunks), 2, [c.text for c in chunks])
        self.assertAlmostEqual(chunks[0].pause_after, chunker.PAUSES["paragraph"])
        self.assertAlmostEqual(chunks[-1].pause_after, 0.0)

    def test_line_breaks_split_only_substantial_chunks(self):
        long_line = "This line is deliberately long so that it fills a chunk by itself."
        text = "\n".join([long_line, long_line, "tiny", "tiny2", "tiny3"])
        chunks = chunker.chunk_text(text, max_chars=300)
        # The two long lines are separated; the short trailing lines stay together.
        self.assertGreaterEqual(len(chunks), 2)
        self.assertIn("tiny2 tiny3", chunks[-1].text)

    def test_single_newline_breaks_are_honoured(self):
        text = "Line one is here.\nLine two is here.\n\nParagraph two starts."
        chunks = chunker.chunk_text(text, max_chars=45)
        self.assertGreater(len(chunks), 1)

    def test_abbreviations_do_not_split_sentences(self):
        text = "Dr. Smith met Mr. Jones at 3.5 p.m. on Jan. 3, 2024 to discuss the plan."
        chunks = chunker.chunk_text(text, max_chars=300)
        self.assertEqual(len(chunks), 1)
        self.assertIn("Dr. Smith", chunks[0].text)
        self.assertIn("3.5 p.m.", chunks[0].text)
        self.assertIn("Jan. 3, 2024", chunks[0].text)

    def test_initials_do_not_split(self):
        text = "The author J. R. R. Tolkien wrote it long ago indeed."
        chunks = chunker.chunk_text(text, max_chars=300)
        self.assertEqual(len(chunks), 1)

    def test_oversized_sentence_is_split_at_word_boundaries(self):
        words = " ".join(f"word{i}" for i in range(200))
        chunks = chunker.chunk_text(words, max_chars=100)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk.text), 100)
        # No word should be lost.
        rebuilt = " ".join(c.text for c in chunks).split()
        self.assertEqual(rebuilt, words.split())

    def test_no_lost_characters(self):
        text = (
            "First paragraph with two sentences. Here is the second one.\n\n"
            "Second paragraph! Is it ok? Yes.\n\n"
            "Third paragraph, " + "with a very long tail " * 30 + "and an end."
        )
        chunks = chunker.chunk_text(text, max_chars=200)
        original = " ".join(chunker.clean_text(text).split())
        rebuilt = " ".join(" ".join(c.text for c in chunks).split())
        # Punctuation may be trimmed at hard splits; letters must all survive.
        strip = lambda s: "".join(ch for ch in s if ch.isalnum())  # noqa: E731
        self.assertEqual(strip(original), strip(rebuilt))

    def test_unicode_and_punctuation_survive(self):
        text = 'He said "hello" — then left. Numbers: 1, 2, 3. Acronyms: NASA, FBI, AI.'
        chunks = chunker.chunk_text(text, max_chars=60)
        joined = " ".join(c.text for c in chunks)
        self.assertIn("NASA", joined)
        self.assertIn('"hello"', joined)

    def test_clean_text_normalises_newlines(self):
        self.assertEqual(chunker.clean_text("a\r\nb\rc"), "a\nb\nc")
        self.assertEqual(chunker.clean_text("  padded  "), "padded")

    def test_estimate_scales_with_speed(self):
        text = "word " * 200
        slow = chunker.estimate_duration_seconds(text, 0.75)
        fast = chunker.estimate_duration_seconds(text, 1.20)
        self.assertGreater(slow, fast)
        self.assertAlmostEqual(slow / fast, 1.20 / 0.75, places=3)


# --------------------------------------------------------------------------- #
class TestConfig(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "config.json"

    def tearDown(self):
        self.tmp.cleanup()

    def test_defaults_when_file_missing(self):
        cfg = Config.load(self.path)
        self.assertEqual(cfg.speed, DEFAULT_SPEED)
        self.assertEqual(cfg.window_size()[0] > 0, True)
        self.assertFalse(self.path.exists(), "loading must not create the file")

    def test_round_trip(self):
        cfg = Config.load(self.path)
        cfg.voice = "F4"
        cfg.speed = 1.15
        cfg.language = "de"
        cfg.last_output_dir = Path(self.tmp.name)
        cfg.set_window(1234, 789, 55, 66)
        self.assertTrue(cfg.save())

        again = Config.load(self.path)
        self.assertEqual(again.voice, "F4")
        self.assertEqual(again.speed, 1.15)
        self.assertEqual(again.language, "de")
        self.assertEqual(again.window_size(), (1234, 789))
        self.assertEqual(again.window_position(), (55, 66))
        self.assertEqual(again.last_output_dir, Path(self.tmp.name))

    def test_corrupt_file_falls_back_and_backs_up(self):
        self.path.write_text("{not json at all", encoding="utf-8")
        cfg = Config.load(self.path)
        self.assertEqual(cfg.speed, DEFAULT_SPEED)
        self.assertTrue((self.path.parent / "config.json.corrupt").exists())

    def test_garbage_values_are_clamped(self):
        self.path.write_text(
            json.dumps(
                {
                    "voice": 12345,
                    "speed": "fast",
                    "total_steps": 9999,
                    "window": {"width": "wide", "height": -20, "x": "left"},
                    "prefer_gpu": "yes please",
                }
            ),
            encoding="utf-8",
        )
        cfg = Config.load(self.path)
        self.assertEqual(cfg.speed, DEFAULT_SPEED)
        self.assertEqual(cfg.total_steps, 32)
        self.assertEqual(cfg.voice, "F3")
        self.assertGreaterEqual(cfg.window_size()[1], 560)
        self.assertEqual(cfg.window_position(), (None, None))
        self.assertFalse(cfg.prefer_gpu)

    def test_speed_setter_clamps(self):
        cfg = Config.load(self.path)
        cfg.speed = 99.0
        self.assertLessEqual(cfg.speed, 2.0)
        cfg.speed = 0.01
        self.assertGreaterEqual(cfg.speed, 0.5)

    def test_saved_file_is_valid_json(self):
        cfg = Config.load(self.path)
        cfg.save()
        json.loads(self.path.read_text(encoding="utf-8"))

    def test_offers_ten_speeds(self):
        self.assertEqual(len(SPEED_CHOICES), 10)
        self.assertEqual(SPEED_CHOICES[0], 0.75)
        self.assertEqual(SPEED_CHOICES[-1], 1.20)
        self.assertIn(1.00, SPEED_CHOICES)


# --------------------------------------------------------------------------- #
class TestModelDiscovery(unittest.TestCase):
    @needs_model
    def test_onnx_dir_found(self):
        found = model_manager.find_onnx_dir()
        self.assertIsNotNone(found)
        for name in ONNX_FILES.values():
            self.assertTrue((found / name).is_file(), f"missing {name}")
        self.assertTrue((found / "tts.json").is_file())
        self.assertTrue((found / "unicode_indexer.json").is_file())

    def test_missing_model_is_reported_not_raised(self):
        status = model_manager.ModelStatus()
        self.assertFalse(status.ok)
        self.assertEqual(status.voice_dirs, [])

    def test_incomplete_folder_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            onnx = Path(tmp) / "models" / "supertonic" / "onnx"
            onnx.mkdir(parents=True)
            (onnx / "vocoder.onnx").write_bytes(b"not a real model")
            saved = utils._dirs_cache.copy()
            try:
                utils._dirs_cache.clear()
                utils._dirs_cache.update(
                    root=Path(tmp),
                    base=Path(tmp),
                    models=Path(tmp) / "models",
                    supertonic=onnx.parent,
                    voices=Path(tmp) / "voices",
                    output=Path(tmp) / "output",
                    logs=Path(tmp) / "logs",
                    config=Path(tmp) / "config.json",
                )
                status = model_manager.model_status()
                self.assertFalse(status.ok)
                self.assertIn("incomplete", status.detail.lower())
                self.assertTrue(status.missing)
            finally:
                utils._dirs_cache.clear()
                utils._dirs_cache.update(saved)

    def test_engine_load_error_is_human_readable(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = TTSEngine(onnx_dir=Path(tmp) / "nope")
            with self.assertRaises(TTSError) as ctx:
                engine.load()
            message = str(ctx.exception)
            self.assertIn("model", message.lower())
            self.assertNotIn("Traceback", message)

    def test_text_to_speech_reports_incomplete_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ModelError) as ctx:
                TextToSpeech(tmp)
            self.assertIn("incomplete", str(ctx.exception).lower())


# --------------------------------------------------------------------------- #
class TestVoiceDiscovery(unittest.TestCase):
    @needs_model
    def test_voices_are_discovered_not_hardcoded(self):
        voices = model_manager.list_voices()
        self.assertTrue(voices, "no voices found")
        ids = [v.id for v in voices]
        self.assertEqual(ids, sorted(ids), "voices should be sorted")
        self.assertTrue(all(v.path.is_file() for v in voices))
        self.assertTrue(all(i[0] in "FMmf" for i in ids), ids)

    @needs_model
    def test_default_voice_is_a_female_english_voice(self):
        default = model_manager.default_voice()
        self.assertIsNotNone(default)
        self.assertTrue(default.startswith("F"), default)

    @needs_model
    def test_f3_is_present_and_findable(self):
        self.assertIsNotNone(model_manager.find_voice("F3"))
        self.assertIsNotNone(model_manager.find_voice("f3"), "lookup is case-insensitive")
        self.assertIsNone(model_manager.find_voice("NOPE"))

    @needs_model
    def test_voice_files_are_valid(self):
        for voice in model_manager.list_voices():
            with self.subTest(voice=voice.id):
                self.assertIsNone(model_manager.validate_voice_file(voice.path))

    def test_corrupt_voice_file_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "F9.json"
            bad.write_text("{}", encoding="utf-8")
            message = model_manager.validate_voice_file(bad)
            self.assertIsNotNone(message)
            self.assertIn("F9.json", message)

    @needs_model
    def test_unknown_voice_gives_a_helpful_error(self):
        engine = TTSEngine()
        with self.assertRaises(TTSError) as ctx:
            engine.synthesize("Hello there.", "ZZ9")
        self.assertIn("ZZ9", str(ctx.exception))
        self.assertIn("F3", str(ctx.exception))


# --------------------------------------------------------------------------- #
class TestAudioOutput(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_wav_is_written_and_readable(self):
        samples = np.sin(np.linspace(0, 2 * np.pi * 440, 44100)).astype(np.float32) * 0.5
        target = self.dir / "out.wav"
        audio_io.write_wav(target, samples, 44100)
        self.assertTrue(target.is_file())
        self.assertGreater(target.stat().st_size, 44, "WAV needs a data chunk")
        import soundfile as sf

        data, rate = sf.read(str(target))
        self.assertEqual(rate, 44100)
        self.assertEqual(data.shape[0], samples.size)
        self.assertAlmostEqual(float(np.abs(data).max()), 0.5, delta=0.01)

    def test_chunks_are_joined_with_their_pauses(self):
        tone = np.ones(100, dtype=np.float32)
        joined = audio_io.join_parts([(tone, 0.5), (tone, 0.0)], 100)
        # 100 samples of tone, 50 samples of silence (0.5 s at 100 Hz), 100 more.
        self.assertEqual(joined.size, 250)
        self.assertAlmostEqual(float(joined[50]), 1.0, places=6)
        self.assertAlmostEqual(float(joined[125]), 0.0, places=6)
        self.assertAlmostEqual(float(joined[200]), 1.0, places=6)

    def test_writing_is_atomic_and_leaves_nothing_behind_when_locked(self):
        """Regression: the second Generate used to fail to cache its audio.

        libsndfile opens with plain ``fopen``, and so does the audio player, so
        Windows blocks rewriting a render the mixer still holds - surfacing as
        an opaque ``LibsndfileError: System error``. Caching therefore goes
        through a temp file (never corrupting the previous render) and the GUI
        releases the player first; if the destination is still locked the write
        must fail cleanly, with no scratch file left over.
        """
        import soundfile as sf

        first = np.full(1000, 0.25, dtype=np.float32)
        second = np.full(1000, -0.75, dtype=np.float32)
        target = self.dir / "cached.wav"
        audio_io.write_wav(target, first, 44100)

        # Case 1: destination is released (the normal path after unload()).
        audio_io.write_wav(target, second, 44100)
        data, _ = sf.read(str(target))
        self.assertAlmostEqual(float(data[0]), -0.75, delta=0.01)

        # Case 2: destination still held open -> clean failure, no debris.
        held = open(target, "rb")
        try:
            with self.assertRaises((OSError, RuntimeError)):
                audio_io.write_wav(target, first, 44100)
        finally:
            held.close()
        self.assertEqual(list(self.dir.glob("*.tmp")), [], "temp file was not cleaned up")

        # The previously written render must be intact and still readable.
        data, _ = sf.read(str(target))
        self.assertAlmostEqual(float(data[0]), -0.75, delta=0.01)

    def test_failed_write_leaves_no_temp_file_behind(self):
        target = self.dir / "cached.wav"
        with patch.object(audio_io.sf, "write", side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                audio_io.write_wav(target, np.zeros(10, dtype=np.float32), 44100)
        self.assertEqual(list(self.dir.glob("*.tmp")), [])
        self.assertFalse(target.exists())

    def test_generated_filenames_are_second_resolution(self):
        """Regression: minute resolution made every render in a minute collide.

        A second Generate rewrote the first render's file, which is what broke
        voice switching (and then froze the app on "Preparing...").
        """
        stamp = utils.default_filename(".wav")
        self.assertRegex(stamp, r"^narration_\d{4}-\d{2}-\d{2}_\d{6}\.wav$")

        # Freezing the clock proves the whole minute stays distinguishable.
        frozen = datetime.datetime(2026, 1, 1, 9, 30, 0)
        names = {
            (frozen + datetime.timedelta(seconds=s)).strftime("%Y-%m-%d_%H%M%S")
            for s in range(60)
        }
        self.assertEqual(len(names), 60, "timestamps must not repeat within a minute")

    def test_join_of_nothing_is_empty(self):
        self.assertEqual(audio_io.join_parts([], 44100).size, 0)
        self.assertEqual(audio_io.join_parts([(np.zeros(0, np.float32), 0.2)], 44100).size, 0)

    def test_peak_limiter_prevents_clipping(self):
        loud = np.array([1.5, -1.5, 0.5], dtype=np.float32)
        limited = audio_io.limit_peak(loud)
        self.assertLessEqual(
            float(np.abs(limited).max()), audio_io.PEAK_CEILING + 1e-6
        )
        self.assertAlmostEqual(
            float(limited[0]), audio_io.PEAK_CEILING, delta=1e-6
        )
        # Proportions are preserved.
        self.assertAlmostEqual(float(limited[2]) / 0.5, float(limited[0]) / 1.5, places=5)

    def test_limiter_does_not_boost_quiet_audio(self):
        quiet = np.array([0.1, -0.1], dtype=np.float32)
        self.assertTrue(np.array_equal(audio_io.limit_peak(quiet), quiet))

    def test_nans_are_removed(self):
        dirty = np.array([np.nan, 1.0], dtype=np.float32)
        clean = audio_io.limit_peak(dirty)
        self.assertFalse(np.isnan(clean).any())

    def test_duration_probe(self):
        samples = np.zeros(44100, dtype=np.float32)
        target = self.dir / "probe.wav"
        audio_io.write_wav(target, samples, 44100)
        self.assertAlmostEqual(audio_io.probe_duration(target), 1.0, places=3)
        self.assertEqual(audio_io.probe_duration(self.dir / "missing.wav"), 0.0)

    @unittest.skipUnless(audio_io.mp3_supported(), "no MP3 encoder available")
    def test_mp3_export(self):
        samples = np.sin(np.linspace(0, 2 * np.pi * 220, 44100)).astype(np.float32) * 0.4
        target = self.dir / "out.mp3"
        audio_io.write_mp3(target, samples, 44100)
        self.assertTrue(target.is_file())
        self.assertGreater(target.stat().st_size, 1000)

    def test_save_directory_is_created_on_demand(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "deep" / "nested" / "file.wav"
            audio_io.write_wav(target, np.zeros(100, dtype=np.float32), 44100)
            self.assertTrue(target.is_file())


# --------------------------------------------------------------------------- #
class TestProviderSelection(unittest.TestCase):
    def test_cpu_is_always_available(self):
        providers, label = select_providers(prefer_gpu=False)
        self.assertIn("CPUExecutionProvider", providers)
        self.assertEqual(providers[0], "CPUExecutionProvider")
        self.assertEqual(label, "CPU")

    def test_gpu_is_never_mandatory(self):
        for prefer in (False, True):
            providers, label = select_profiles = select_providers(prefer_gpu=prefer)
            self.assertIn("CPUExecutionProvider", providers)
            self.assertIn(label, ("CPU", "DirectML", "CUDA", "TensorRT", "ROCm"))

    def test_label_is_human_readable(self):
        self.assertEqual(select_providers(False)[1], "CPU")


# --------------------------------------------------------------------------- #
class TestUtils(unittest.TestCase):
    def test_format_time(self):
        self.assertEqual(utils.format_time(0), "00:00")
        self.assertEqual(utils.format_time(9.4), "00:09")
        self.assertEqual(utils.format_time(65), "01:05")
        self.assertEqual(utils.format_time(3725), "1:02:05")
        self.assertEqual(utils.format_time(-5), "00:00")
        self.assertEqual(utils.format_time(float("nan")), "00:00")

    def test_default_filename_shape(self):
        name = utils.default_filename(".wav")
        self.assertTrue(name.startswith("narration_"), name)
        self.assertTrue(name.endswith(".wav"), name)
        self.assertRegex(name, r"^narration_\d{4}-\d{2}-\d{2}_\d{6}\.wav$")

    def test_format_size(self):
        self.assertEqual(utils.format_size(512), "512 B")
        self.assertTrue(utils.format_size(2048).endswith("KB"))

    def test_base_dir_is_the_project_folder_in_dev(self):
        self.assertTrue((utils.base_dir() / "app").is_dir())
        self.assertTrue((utils.base_dir() / "models").is_dir())

    def test_log_file_is_created(self):
        log_file = utils.setup_logging()
        self.assertTrue(log_file.exists())
        self.assertEqual(log_file.name, "app.log")


# --------------------------------------------------------------------------- #
class TestEndToEnd(unittest.TestCase):
    @needs_model
    def test_generates_playable_audio(self):
        engine = TTSEngine(total_steps=4)
        progress: list[tuple[int, int]] = []
        result = engine.synthesize(
            "Supertonic TTS works entirely on this computer.",
            "F3",
            speed=1.0,
            on_chunk=lambda i, n: progress.append((i, n)),
        )
        self.assertTrue(progress and progress[0] == (1, 1), progress)
        self.assertEqual(result.sample_rate, 44100)
        self.assertGreater(result.duration, 0.5)
        self.assertEqual(result.samples.ndim, 1)
        self.assertEqual(result.provider, "CPU")
        self.assertGreater(float(np.abs(result.samples).max()), 0.05, "audio is not silent")

    @needs_model
    def test_output_is_repeatable(self):
        engine = TTSEngine(total_steps=4)
        a = engine.synthesize("Repeatable narration please.", "F3", speed=1.0)
        b = engine.synthesize("Repeatable narration please.", "F3", speed=1.0)
        self.assertTrue(
            np.array_equal(a.samples, b.samples), "identical input must give identical audio"
        )

    @needs_model
    def test_speed_changes_duration(self):
        engine = TTSEngine(total_steps=4)
        text = "Speed control should shorten the audio."
        slow = engine.synthesize(text, "F3", speed=0.80)
        fast = engine.synthesize(text, "F3", speed=1.20)
        self.assertGreater(slow.duration, fast.duration)

    @needs_model
    def test_long_text_is_chunked_and_joined(self):
        engine = TTSEngine(total_steps=4)
        text = (
            "First paragraph explains the idea in a calm and steady voice. "
            "It is meant for narration.\n\nSecond paragraph adds a little more "
            "detail about how the pieces fit together in practice."
        )
        progress: list[tuple[int, int]] = []
        result = engine.synthesize(
            text, "F3", speed=1.0, on_chunk=lambda i, n: progress.append((i, n))
        )
        self.assertEqual(len(progress), result.chunk_count)
        self.assertEqual([i for i, _ in progress], list(range(1, result.chunk_count + 1)))
        self.assertGreater(result.duration, 3.0)

    @needs_model
    def test_cancel_stops_before_the_first_chunk(self):
        engine = TTSEngine(total_steps=4)
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(TTSError) as ctx:
            engine.synthesize("This should never be rendered at all.", "F3", cancel=cancel)
        self.assertIn("cancel", str(ctx.exception).lower())

    @needs_model
    def test_result_survives_a_write_read_cycle(self):
        engine = TTSEngine(total_steps=4)
        result = engine.synthesize("Write me to disk please.", "F3", speed=1.0)
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "final.wav"
            audio_io.write_wav(target, result.samples, result.sample_rate)
            self.assertTrue(target.is_file())
            self.assertAlmostEqual(
                audio_io.probe_duration(target), result.duration, delta=0.05
            )


class TestCommandLine(unittest.TestCase):
    """The --check / --selftest modes are how a portable build gets verified."""

    def setUp(self):
        self.out = StringIO()
        patcher = patch.object(sys, "stdout", self.out)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_version_and_help(self):
        from app import main as app_main

        self.assertEqual(app_main.main(["--version"]), 0)
        self.assertIn("Supertonic TTS", self.out.getvalue())

    @needs_model
    def test_check_reports_a_healthy_install(self):
        from app import main as app_main

        self.assertEqual(app_main.main(["--check"]), 0)
        report = self.out.getvalue()
        self.assertIn("OK", report)
        self.assertIn("44100 Hz", report)
        self.assertIn("F3", report)

    def test_check_reports_a_missing_model(self):
        from app import main as app_main

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(utils, "base_dir", lambda: Path(tmp)):
                with patch.object(model_manager, "find_onnx_dir", lambda: None):
                    with patch.object(model_manager, "find_voice_dirs", lambda: []):
                        self.assertEqual(app_main.main(["--check"]), 1)
        report = self.out.getvalue()
        self.assertIn("NOT READY", report)
        self.assertIn("huggingface.co", report)

    def test_selftest_reports_engine_failures_instead_of_crashing(self):
        """Regression: _selftest catches TTSError but must import the name.

        A NameError here used to crash the whole command instead of printing a
        readable failure, which is the exact path a user hits with a corrupt
        or half-downloaded model.
        """
        from app import main as app_main

        boom = TTSError("The Supertonic model could not be loaded.")
        with patch.object(TTSEngine, "synthesize", side_effect=boom):
            with patch.object(TTSEngine, "load", return_value=object()):
                code = app_main._selftest("F3", "hello", 1.0)
        self.assertEqual(code, 1)
        self.assertIn("FAILED", self.out.getvalue())
        self.assertIn("could not be loaded", self.out.getvalue())

    def test_main_selftest_propagates_a_tts_error(self):
        from app import main as app_main

        with patch.object(app_main, "_selftest", side_effect=TTSError("nope")):
            self.assertEqual(app_main.main(["--selftest"]), 1)
        self.assertIn("FAILED", self.out.getvalue())

    @needs_model
    def test_selftest_writes_a_playable_wav(self):
        from app import main as app_main

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(utils, "output_dir", lambda: Path(tmp)):
                self.assertEqual(app_main._selftest("F3", "Portable check.", 1.0), 0)
            wavs = list(Path(tmp).glob("*.wav"))
            self.assertEqual(len(wavs), 1)
            info = audio_io.probe_duration(wavs[0])
            self.assertGreater(info, 0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
