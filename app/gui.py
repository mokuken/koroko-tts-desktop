"""The Supertonic TTS desktop window (Tkinter, dark theme).

Kept deliberately small: a text box, two settings, one primary button, a
player and two save buttons. All heavy work happens on a worker thread and
arrives back through a queue, so the window never freezes.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from . import audio_io, chunker, model_manager, theme, utils
from .audio_player import AudioPlayer
from .config import SPEED_CHOICES, Config
from .supertonic_onnx import AVAILABLE_LANGS
from .tts_engine import TTSEngine, TTSError, SynthesisResult

log = logging.getLogger(__name__)

TICK_MS = 80
META_DEBOUNCE_MS = 150


# --------------------------------------------------------------------------- #
# small widgets
# --------------------------------------------------------------------------- #
class FooterProgress(tk.Canvas):
    """A hairline progress rail for the status footer.

    ttk's progressbar cannot be made this thin: clam's trough element ignores
    ``thickness``, so it reserves 18px regardless. Drawing two rectangles gives
    exact control over the height and matches the borderless footer.
    """

    MIN_VISIBLE = 2.0  # px of fill shown for any non-zero progress

    def __init__(self, master, thickness: int = 3, **kwargs):
        kwargs.setdefault("bg", theme.SURFACE)
        kwargs.setdefault("height", thickness)
        kwargs.setdefault("highlightthickness", 0)
        kwargs.setdefault("borderwidth", 0)
        kwargs.setdefault("takefocus", 0)
        super().__init__(master, width=110, **kwargs)
        self._thickness = thickness
        self._value = 0.0
        self._trough = self.create_rectangle(
            0, 0, 1, thickness, fill=theme.SURFACE_ALT, outline=""
        )
        self._fill = self.create_rectangle(0, 0, 0, thickness, fill=theme.ACCENT, outline="")
        self.bind("<Configure>", lambda _e: self._redraw())

    @property
    def value(self) -> float:
        return self._value

    def set_value(self, percent: float) -> None:
        self._value = max(0.0, min(100.0, float(percent)))
        self._redraw()

    def _redraw(self) -> None:
        width = max(1, self.winfo_width())
        height = self._thickness
        self.coords(self._trough, 0, 0, width, height)
        filled = width * (self._value / 100.0)
        if 0.0 < self._value < 100.0:
            filled = max(self.MIN_VISIBLE, filled)
        self.coords(self._fill, 0, 0, filled, height)


class TransportIcon(tk.Canvas):
    """Font-independent play/pause/stop icon that doubles as a button."""

    SIZE = 14

    def __init__(self, master, kind: str, command, **kwargs):
        bg = kwargs.pop("bg", theme.SURFACE)
        super().__init__(
            master,
            width=self.SIZE,
            height=self.SIZE,
            bg=bg,
            highlightthickness=0,
            borderwidth=0,
            takefocus=0,
            cursor="hand2",
            **kwargs,
        )
        self._kind = kind
        self._command = command
        self._hover = False
        self._enabled = True
        self.bind("<Button-1>", self._click)
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self._draw()

    def _click(self, _event=None):
        if self._enabled:
            self._command()

    def _on_enter(self, _event=None):
        self._hover = True
        self._draw()

    def _on_leave(self, _event=None):
        self._hover = False
        self._draw()

    def set_kind(self, kind: str) -> None:
        if kind != self._kind:
            self._kind = kind
            self._draw()

    def set_enabled(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if enabled != self._enabled:
            self._enabled = enabled
            self.configure(cursor="hand2" if enabled else "arrow")
            self._draw()

    def _color(self) -> str:
        if not self._enabled:
            return theme.TEXT_FAINT
        return theme.ACCENT_HI if self._hover else theme.TEXT_DIM

    def _draw(self) -> None:
        self.delete("all")
        color = self._color()
        pad, size = 3, self.SIZE
        if self._kind == "play":
            self.create_polygon(
                pad, pad, pad, size - pad, size - pad - 1, size / 2,
                fill=color, outline="",
            )
        elif self._kind == "pause":
            w = 3
            self.create_rectangle(pad, pad, pad + w, size - pad, fill=color, outline="")
            self.create_rectangle(
                size - pad - w, pad, size - pad, size - pad, fill=color, outline=""
            )
        else:
            self.create_rectangle(
                pad, pad, size - pad, size - pad, fill=color, outline=""
            )


# --------------------------------------------------------------------------- #
# application
# --------------------------------------------------------------------------- #
class TTSApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.config = Config.load()
        self.engine = TTSEngine(
            prefer_gpu=self.config.prefer_gpu, total_steps=self.config.total_steps
        )
        self.player = AudioPlayer(on_finished=self._on_playback_finished)

        self._events: queue.Queue = queue.Queue()
        self._cancel = threading.Event()
        self._worker: threading.Thread | None = None
        self._busy = False
        self._result: SynthesisResult | None = None
        self._current_wav: Path | None = None
        self._suppress_scale = False
        self._meta_job: str | None = None
        self._speed_values = self._speed_choices()
        self._tick_count = 0

        root.title(utils.APP_NAME)
        root.minsize(640, 620)
        self.style = theme.apply_style(root)
        theme.configure_root(root)
        self._restore_geometry()
        root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build()
        self._refresh_model_view()
        self.root.after(TICK_MS, self._tick)

    # -- geometry ------------------------------------------------------------ #
    @property
    def tick_count(self) -> int:
        """Timer frames served. Stops advancing if the event loop ever dies."""
        return self._tick_count

    def _speed_choices(self) -> list[float]:
        values = list(SPEED_CHOICES)
        try:
            custom = float(self.config.speed)
        except (TypeError, ValueError):
            return values
        if not any(abs(custom - v) < 1e-6 for v in values):
            values.append(custom)
            values.sort()
        return values

    def _restore_geometry(self) -> None:
        width, height = self.config.window_size()
        x, y = self.config.window_position()
        screen_w, screen_h = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        usable = (
            isinstance(x, int)
            and isinstance(y, int)
            and -screen_w < x < screen_w - 120
            and -screen_h < y < screen_h - 80
        )
        if usable:
            self.root.geometry(f"{width}x{height}+{x}+{y}")
        else:
            self.root.geometry(f"{width}x{height}")
            self.root.update_idletasks()
            cx, cy = self.root.winfo_width(), self.root.winfo_height()
            self.root.geometry(
                f"{width}x{height}+{(screen_w - cx) // 2}+{max(0, (screen_h - cy) // 3)}"
            )

    # -- layout -------------------------------------------------------------- #
    def _build(self) -> None:
        root = self.root
        fonts = theme.fonts()

        self._header = tk.Frame(root, bg=theme.BG)
        self._header.pack(fill="x", padx=theme.PAD + 4, pady=(12, 8))
        tk.Label(
            self._header, text=utils.APP_NAME, bg=theme.BG, fg=theme.TEXT,
            font=fonts["title"],
        ).pack(side="left")
        self._runtime_label = tk.Label(
            self._header, text="Runtime: --", bg=theme.BG, fg=theme.TEXT_FAINT,
            font=fonts["tiny"],
        )
        self._runtime_label.pack(side="right")

        tk.Frame(root, bg=theme.BORDER, height=1).pack(fill="x")

        self._content = tk.Frame(root, bg=theme.BG)
        self._content.pack(fill="both", expand=True)

        self._build_main_view()
        self._build_status_bar(fonts)

    # -- main view ----------------------------------------------------------- #
    def _build_main_view(self) -> None:
        for child in self._content.winfo_children():
            child.destroy()

        fonts = theme.fonts()
        pad = theme.PAD

        tk.Label(
            self._content, text="Narration", bg=theme.BG, fg=theme.TEXT,
            font=fonts["label"],
        ).pack(anchor="w", padx=pad, pady=(pad, 6))

        editor_border = tk.Frame(self._content, bg=theme.BORDER, padx=1, pady=1)
        editor_border.pack(fill="both", expand=True, padx=pad)
        editor = tk.Frame(editor_border, bg=theme.ENTRY_BG)
        editor.pack(fill="both", expand=True)

        self._scroll = ttk.Scrollbar(
            editor, orient="vertical", style="Dark.Vertical.TScrollbar"
        )
        self._scroll.pack(side="right", fill="y", padx=(0, 1), pady=1)

        self._text = tk.Text(
            editor,
            wrap="word",
            yscrollcommand=self._scroll.set,
            height=10,
            width=60,
            insertwidth=1,
        )
        theme.text_widget(self._text)
        self._text.pack(side="left", fill="both", expand=True)
        self._scroll.configure(command=self._text.yview)

        self._hint = tk.Label(
            editor, text="Paste your narration here\u2026", bg=theme.ENTRY_BG,
            fg=theme.TEXT_FAINT, font=fonts["body"], anchor="w",
        )
        self._hint.place(x=13, y=11)

        self._meta = tk.Label(
            self._content, text="", bg=theme.BG, fg=theme.TEXT_FAINT, font=fonts["tiny"]
        )
        self._meta.pack(anchor="w", padx=pad, pady=(6, 0))

        # -- settings row -- #
        settings = tk.Frame(self._content, bg=theme.BG)
        settings.pack(fill="x", padx=pad, pady=(12, 0))
        settings.columnconfigure(1, weight=0)
        settings.columnconfigure(3, weight=1)

        tk.Label(
            settings, text="Voice", bg=theme.BG, fg=theme.TEXT_DIM, font=fonts["small"]
        ).grid(row=0, column=0, sticky="w", padx=(0, 8))
        self._voice_combo = ttk.Combobox(
            settings, state="readonly", width=11, style="Dark.TCombobox"
        )
        self._voice_combo.grid(row=0, column=1, sticky="w")
        self._voice_combo.bind("<<ComboboxSelected>>", self._on_voice_selected)

        tk.Label(
            settings, text="Speed", bg=theme.BG, fg=theme.TEXT_DIM, font=fonts["small"]
        ).grid(row=0, column=2, sticky="w", padx=(24, 8))
        self._speed_combo = ttk.Combobox(
            settings, state="readonly", width=7, style="Dark.TCombobox",
            values=[f"{v:.2f}" for v in self._speed_values],
        )
        self._speed_combo.grid(row=0, column=3, sticky="w")
        self._speed_combo.bind("<<ComboboxSelected>>", self._on_speed_selected)
        self._speed_combo.set(f"{self.config.speed:.2f}")

        self._language_combo = ttk.Combobox(
            settings, state="readonly", width=7, style="Dark.TCombobox",
            values=list(AVAILABLE_LANGS),
        )
        tk.Label(
            settings, text="Language", bg=theme.BG, fg=theme.TEXT_DIM, font=fonts["small"]
        ).grid(row=0, column=4, sticky="w", padx=(24, 8))
        self._language_combo.grid(row=0, column=5, sticky="w")
        self._language_combo.set(self.config.language)
        self._language_combo.bind("<<ComboboxSelected>>", self._on_language_selected)

        # -- generate row -- #
        actions = tk.Frame(self._content, bg=theme.BG)
        actions.pack(fill="x", padx=pad, pady=(14, 0))

        self._generate_btn = ttk.Button(
            actions, text="Generate Speech", style="Accent.TButton",
            command=self._on_generate,
        )
        self._generate_btn.pack(side="left", fill="x", expand=True)

        self._cancel_btn = ttk.Button(
            actions, text="Cancel", style="Ghost.TButton", command=self._on_cancel
        )

        # -- player -- #
        self._build_player(fonts)

        # -- save row -- #
        save_row = tk.Frame(self._content, bg=theme.BG)
        save_row.pack(fill="x", padx=pad, pady=(12, pad))
        self._save_wav_btn = ttk.Button(
            save_row, text="Save WAV", style="Dark.TButton", command=lambda: self._save("wav"),
            state="disabled",
        )
        self._save_wav_btn.pack(side="left", fill="x", expand=True, padx=(0, 6))
        self._mp3_supported = audio_io.mp3_supported()
        self._save_mp3_btn = ttk.Button(
            save_row, text="Save MP3", style="Dark.TButton", command=lambda: self._save("mp3"),
            state="disabled",
        )
        if self._mp3_supported:
            self._save_mp3_btn.pack(side="left", fill="x", expand=True)
        else:
            self._save_mp3_btn.pack_forget()

        self._bind_editor_keys()
        self._text.bind("<<Modified>>", self._on_modified)
        self._update_meta()
        self.root.after(80, lambda: self._text.focus_set())

    def _build_player(self, fonts) -> None:
        pad = theme.PAD
        card = tk.Frame(
            self._content, bg=theme.SURFACE, highlightthickness=1,
            highlightbackground=theme.BORDER, highlightcolor=theme.BORDER, padx=12, pady=10,
        )
        card.pack(fill="x", padx=pad, pady=(14, 0))

        self._play_icon = TransportIcon(
            card, "play", self._on_play_toggle, bg=theme.SURFACE
        )
        self._play_icon.pack(side="left", padx=(0, 8))
        self._stop_icon = TransportIcon(
            card, "stop", self._on_stop, bg=theme.SURFACE
        )
        self._stop_icon.pack(side="left", padx=(0, 12))

        self._time_label = tk.Label(
            card, text="00:00 / 00:00", bg=theme.SURFACE, fg=theme.TEXT_DIM,
            font=fonts["mono"],
        )
        self._time_label.pack(side="right")

        self._seek = ttk.Scale(
            card, orient="horizontal", style="Dark.Horizontal.TScale", from_=0.0, to=1000.0,
            command=self._on_seek, state="disabled",
        )
        self._seek.pack(side="left", fill="x", expand=True, padx=(0, 12))

        self._now_playing = tk.Label(
            self._content, text="No audio yet - generate speech to preview it.",
            bg=theme.BG, fg=theme.TEXT_FAINT, font=fonts["tiny"],
        )
        self._now_playing.pack(anchor="w", padx=pad, pady=(6, 0))

    def _build_status_bar(self, fonts) -> None:
        tk.Frame(self.root, bg=theme.BORDER, height=1).pack(fill="x")
        bar = tk.Frame(self.root, bg=theme.SURFACE, padx=theme.PAD, pady=7)
        bar.pack(fill="x")
        self._status_label = tk.Label(
            bar, text="Ready", bg=theme.SURFACE, fg=theme.TEXT_DIM, font=fonts["small"],
            anchor="w",
        )
        self._status_label.pack(side="left")
        self._engine_label = tk.Label(
            bar, text="", bg=theme.SURFACE, fg=theme.TEXT_FAINT, font=fonts["tiny"],
        )
        self._engine_label.pack(side="right")

        # Thin progress rail between the status text and the engine label.
        # Only visible while generating; packed away the moment it finishes so
        # the footer always reads as a plain status line when idle.
        self._progress = FooterProgress(bar, thickness=3)
        self._progress.pack_forget()  # revealed by _show_progress only

    def _show_progress(self, value: float) -> None:
        """Reveal the footer rail and move it to ``value`` percent."""
        self._progress.set_value(value)
        self._progress.pack(side="left", fill="x", expand=True, padx=(12, 12))

    def _hide_progress(self) -> None:
        if self._progress.winfo_manager():
            self._progress.pack_forget()
        self._progress.set_value(0)

    # -- model-missing view -------------------------------------------------- #
    def _build_missing_view(self, status: model_manager.ModelStatus) -> None:
        for child in self._content.winfo_children():
            child.destroy()
        fonts = theme.fonts()
        pad = theme.PAD

        card = tk.Frame(
            self._content, bg=theme.SURFACE, highlightthickness=1,
            highlightbackground=theme.BORDER, highlightcolor=theme.BORDER,
            padx=22, pady=20,
        )
        card.pack(fill="both", expand=True, padx=pad, pady=pad)

        tk.Label(
            card, text="Model not found", bg=theme.SURFACE, fg=theme.TEXT,
            font=fonts["title"],
        ).pack(anchor="w")
        tk.Label(
            card,
            text=model_manager.INSTALL_HINT.format(folder=status.display_dir),
            bg=theme.SURFACE, fg=theme.TEXT_DIM, font=fonts["small"],
            justify="left", anchor="nw", wraplength=560,
        ).pack(fill="x", pady=(10, 16))

        row = tk.Frame(card, bg=theme.SURFACE)
        row.pack(fill="x")
        ttk.Button(
            row, text="Open Models Folder", style="Accent.TButton",
            command=lambda: utils.open_in_file_manager(status.display_dir.parent),
        ).pack(side="left")
        ttk.Button(
            row, text="Refresh", style="Dark.TButton", command=self._refresh_model_view
        ).pack(side="left", padx=(10, 0))

        hint = tk.Label(
            card,
            text=(
                "Already have the files? Put them in the folder above and press Refresh.\n"
                "Downloaded with the model script? The onnx\\ folder should sit inside "
                "models\\supertonic\\."
            ),
            bg=theme.SURFACE, fg=theme.TEXT_FAINT, font=fonts["tiny"],
            justify="left", anchor="w",
        )
        hint.pack(fill="x", pady=(16, 0))

    def _refresh_model_view(self) -> None:
        status = model_manager.model_status()
        self._model_status = status
        if status.ok:
            self._build_main_view()
            self._populate_voices()
            self._set_status("Ready", theme.TEXT_DIM)
            log.info(
                "Model ready: %s (voices: %s)", status.onnx_dir,
                ", ".join(v.id for v in self.engine.list_voices()),
            )
        else:
            self._build_missing_view(status)
            self._set_status("Model not found", theme.WARN)
            log.warning("Model check failed: %s", status.detail)

    def _populate_voices(self) -> None:
        voices = [v.id for v in self.engine.list_voices()]
        self._voice_combo.configure(values=voices)
        current = self.config.voice
        if current not in voices:
            current = model_manager.default_voice() or (voices[0] if voices else "")
            if current:
                self.config.voice = current
        if current:
            self._voice_combo.set(current)
        if voices:
            self._engine_label.configure(text=f"{len(voices)} voices")
        else:
            self._engine_label.configure(text="no voices")

    # -- text events --------------------------------------------------------- #
    def _bind_editor_keys(self) -> None:
        self._text.bind("<Control-a>", self._select_all)
        self._text.bind("<Control-A>", self._select_all)
        self._text.bind("<Control-y>", self._redo)
        self._text.bind("<Control-Y>", self._redo)
        self._text.bind("<Control-Shift-Z>", self._redo)
        self._text.bind("<Control-Return>", lambda _e: self._on_generate())
        self._text.bind("<Control-KP_Enter>", lambda _e: self._on_generate())

    def _select_all(self, _event=None) -> str:
        self._text.tag_add("sel", "1.0", "end-1c")
        return "break"

    def _redo(self, _event=None) -> str:
        try:
            self._text.edit_redo()
        except tk.TclError:
            pass
        return "break"

    def _on_modified(self, _event=None) -> None:
        if not self._text.edit_modified():
            return
        self._text.edit_modified(False)
        if self._meta_job is not None:
            try:
                self.root.after_cancel(self._meta_job)
            except tk.TclError:
                pass
        self._meta_job = self.root.after(META_DEBOUNCE_MS, self._update_meta)

    def _update_meta(self) -> None:
        self._meta_job = None
        try:
            text = self._text.get("1.0", "end-1c")
        except tk.TclError:  # widget destroyed
            return
        chars = chunker.char_count(text)
        words = chunker.word_count(text)
        if self._hint.winfo_exists():
            if chars:
                self._hint.place_forget()
            else:
                self._hint.place(x=13, y=11)
        try:
            speed = float(self.config.speed)
        except (TypeError, ValueError):
            speed = 1.0
        estimate = chunker.estimate_duration_seconds(text, speed)
        pieces = [f"{chars:,} characters", f"{words:,} words"]
        if chars:
            chunks = len(chunker.chunk_text(text, max_chars=300))
            pieces.append(f"{chunks} chunk{'s' if chunks != 1 else ''}")
            pieces.append(f"~{utils.format_time(estimate)} audio")
        self._meta.configure(text="  \u00b7  ".join(pieces))

        if not self._busy:
            self._generate_btn.configure(
                state="normal" if chars and not chunker.is_blank(text) else "disabled"
            )

    def _on_voice_selected(self, _event=None) -> None:
        value = self._voice_combo.get().strip()
        if value:
            self.config.voice = value
            self._save_config()

    def _on_speed_selected(self, _event=None) -> None:
        try:
            self.config.speed = float(self._speed_combo.get())
        except (TypeError, ValueError):
            return
        self.config.language = self._language_combo.get() or "en"
        self._save_config()
        self._update_meta()

    def _on_language_selected(self, _event=None) -> None:
        value = self._language_combo.get().strip()
        if value:
            self.config.language = value
            self._save_config()

    # -- status -------------------------------------------------------------- #
    def _set_status(self, text: str, color: str = theme.TEXT_DIM) -> None:
        self._status_label.configure(text=text, fg=color)

    def _set_runtime(self, text: str) -> None:
        if "Runtime:" in text:
            self._runtime_label.configure(
                text=text.replace("Runtime: ", "Runtime: "), fg=theme.TEXT_DIM
            )

    # -- generation ---------------------------------------------------------- #
    def _on_generate(self) -> None:
        if self._busy:
            return
        text = self._text.get("1.0", "end-1c")
        if chunker.is_blank(text):
            self._set_status("Nothing to speak - paste or type some text first.", theme.WARN)
            self._text.focus_set()
            return
        if not model_manager.model_status().ok:
            self._set_status("Model not found.", theme.DANGER)
            self._refresh_model_view()
            return

        self._player_reset()
        self._cancel.clear()
        self._set_busy(True)
        self._show_progress(0)
        self._set_status("Preparing...", theme.ACCENT)

        # Prefer the live widget values over config: they are what the user is
        # looking at, and they cannot drift from the visible selection.
        voice = (self._voice_combo.get() or "").strip() or self.config.voice
        voice = voice or (model_manager.default_voice() or "")
        language = self.config.language
        if language not in AVAILABLE_LANGS:
            language = "en"

        def worker() -> None:
            try:
                result = self.engine.synthesize(
                    text,
                    voice,
                    speed=float(self.config.speed),
                    language=language,
                    cancel=self._cancel,
                    on_status=lambda msg: self._events.put(("status", msg)),
                    on_chunk=lambda i, n: self._events.put(("chunk", (i, n))),
                    trim=self.config.trim_silence,
                )
                self._events.put(("done", result))
            except TTSError as exc:
                self._events.put(("error", str(exc)))
            except Exception as exc:  # pragma: no cover - safety net
                utils.log_exception("Unhandled generation error", exc)
                self._events.put(
                    ("error", f"Could not generate speech.\n\nReason:\n{exc}")
                )

        self._worker = threading.Thread(target=worker, name="tts", daemon=True)
        self._worker.start()

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        self._generate_btn.configure(
            state="disabled" if busy else ("normal" if self._text.get("1.0", "end-1c").strip() else "disabled")
        )
        if busy:
            self._cancel_btn.pack(side="left", padx=(10, 0))
            self._save_wav_btn.configure(state="disabled")
            self._save_mp3_btn.configure(state="disabled")
            self._voice_combo.configure(state="disabled")
            self._speed_combo.configure(state="disabled")
            self._language_combo.configure(state="disabled")
            self.root.config(cursor="watch")
        else:
            self._cancel_btn.pack_forget()
            self._set_save_buttons(self._current_wav is not None)
            self._voice_combo.configure(state="readonly")
            self._speed_combo.configure(state="readonly")
            self._language_combo.configure(state="readonly")
            self.root.config(cursor="")
            self._hide_progress()
            self._update_meta()

    def _on_cancel(self) -> None:
        if self._busy:
            self._cancel.set()
            self._cancel_btn.configure(state="disabled")
            self._set_status("Cancelling after the current chunk\u2026", theme.WARN)

    # -- results ------------------------------------------------------------- #
    def _finish(self, result: SynthesisResult) -> None:
        self._result = result
        self._set_busy(False)
        self._set_runtime(f"Runtime: {result.provider}")

        # Release the previous render before writing the new one. The mixer
        # keeps a handle on the file it is playing, and on Windows that blocks
        # overwriting it - which is what made the *second* Generate silently
        # fail and the third one freeze on "Preparing...".
        self.player.unload()

        target = utils.output_dir() / utils.default_filename(".wav")
        try:
            audio_io.write_wav(target, result.samples, result.sample_rate)
            self._current_wav = target
        except (OSError, RuntimeError) as exc:
            # soundfile.LibsndfileError is a RuntimeError, not an OSError, so
            # catching only OSError let this escape into the timer callback.
            utils.log_exception("Could not write the preview file", exc)
            self._current_wav = None
            self._set_save_buttons(False)
            self._set_status(
                f"Speech generated, but it could not be cached to disk.\n\n"
                f"Close anything holding files in {utils.output_dir()} and try again."
                f"\n\nTechnical detail: {exc}",
                theme.WARN,
            )
            return

        self._load_player(result)
        self._set_save_buttons(True)
        self._set_status(
            f"Done \u2014 {utils.format_time(result.duration)} of audio "
            f"in {result.elapsed:.1f}s ({result.chunk_count} chunk"
            f"{'s' if result.chunk_count != 1 else ''}, voice {result.voice}).",
            theme.SUCCESS,
        )

    def _load_player(self, result: SynthesisResult) -> None:
        if self._current_wav is None:
            self._set_save_buttons(False)
            return
        try:
            self.player.load(self._current_wav, result.duration)
        except (OSError, RuntimeError) as exc:
            utils.log_exception("Could not load audio into the player", exc)
            self._set_save_buttons(True)
            self._now_playing.configure(
                text="Audio is ready, but playback is unavailable - use Save WAV."
            )
            self._play_icon.set_enabled(False)
            self._stop_icon.set_enabled(False)
            self._seek.configure(state="disabled")
            self._set_time(0.0, 0.0)
            return
        self._now_playing.configure(
            text=f"{self._current_wav.name}  \u00b7  {result.sample_rate} Hz  \u00b7  "
                 f"voice {result.voice}  \u00b7  speed {result.speed:.2f}"
        )
        self._play_icon.set_kind("play")
        self._play_icon.set_enabled(self.player.available)
        self._stop_icon.set_enabled(self.player.available)
        self._seek.configure(state="normal" if self.player.supports_seek else "disabled")
        self._set_time(0.0, self.player.duration)

    def _fail(self, message: str) -> None:
        self._set_busy(False)
        self._set_status("Error", theme.DANGER)
        detail = message or "An unknown error occurred."
        log.error("Generation failed: %s", detail)
        log.error("Technical detail written to %s", utils.logs_dir() / "app.log")
        messagebox.showerror(utils.APP_NAME, detail, parent=self.root)

    def _set_save_buttons(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        self._save_wav_btn.configure(state=state)
        if self._mp3_supported:
            self._save_mp3_btn.configure(state=state)

    # -- player -------------------------------------------------------------- #
    def _player_reset(self) -> None:
        self.player.stop()
        self._play_icon.set_kind("play")
        self._stop_icon.set_enabled(False)
        self._seek.configure(value=0, state="disabled")
        self._set_time(0.0, 0.0)
        self._set_save_buttons(False)
        self._current_wav = None
        self._now_playing.configure(text="Generating\u2026")

    def _on_play_toggle(self) -> None:
        if not self.player.has_track:
            return
        if not self.player.available:
            messagebox.showinfo(
                utils.APP_NAME,
                "No audio output device is available.\n\n"
                "You can still save the generated audio with Save WAV.",
                parent=self.root,
            )
            return
        self.player.toggle()
        self._play_icon.set_kind("pause" if self.player.is_playing else "play")

    def _on_stop(self) -> None:
        self.player.stop()
        self._play_icon.set_kind("play")
        self._suppress_scale = True
        self._seek.configure(value=0)
        self._suppress_scale = False
        self._set_time(0.0, self.player.duration)

    def _on_seek(self, _value=None) -> None:
        if self._suppress_scale or not self.player.supports_seek:
            return
        try:
            self.player.seek(self._seek.get() / 1000.0)
        except (tk.TclError, ValueError):
            return
        self._set_time(self.player.position, self.player.duration)

    def _on_playback_finished(self) -> None:
        self._play_icon.set_kind("play")

    def _set_time(self, position: float, duration: float) -> None:
        self._time_label.configure(
            text=f"{utils.format_time(position)} / {utils.format_time(duration)}"
        )

    # -- saving -------------------------------------------------------------- #
    def _save(self, fmt: str) -> None:
        if self._result is None:
            return
        target_dir = self.config.last_output_dir
        if not target_dir.is_dir():
            target_dir = utils.output_dir()
        initial = self.config.suggested_filename(f".{fmt}")
        chosen = filedialog.asksaveasfilename(
            parent=self.root,
            title=f"Save narration as {fmt.upper()}",
            initialdir=str(target_dir),
            initialfile=initial,
            defaultextension=f".{fmt}",
            filetypes=[(f"{fmt.upper()} audio", f"*.{fmt}"), ("All files", "*.*")],
        )
        if not chosen:
            return
        destination = Path(chosen)
        if fmt == "mp3" and not self._mp3_supported:
            messagebox.showerror(
                utils.APP_NAME,
                "MP3 export is not available in this build.\n\nPlease save as WAV.",
                parent=self.root,
            )
            return
        try:
            if fmt == "mp3":
                audio_io.write_mp3(destination, self._result.samples, self._result.sample_rate)
            else:
                audio_io.write_wav(destination, self._result.samples, self._result.sample_rate)
        except (OSError, RuntimeError, ValueError) as exc:
            utils.log_exception("Save failed", exc)
            messagebox.showerror(
                utils.APP_NAME,
                f"Could not save the audio.\n\n{exc}",
                parent=self.root,
            )
            return
        self.config.last_output_dir = destination.parent
        self._set_status(f"Saved {destination.name}", theme.SUCCESS)
        self._save_config()

    # -- main loop ----------------------------------------------------------- #
    def _tick(self) -> None:
        """One timer frame: drain worker events, then advance the player.

        The ``finally`` is load-bearing. This callback is scheduled with
        ``root.after``; if an exception escapes, the whole event loop dies and
        the window freezes on whatever status was last painted. A failure while
        caching audio therefore used to brick the app mid-session. Re-arming the
        timer unconditionally keeps a single bad frame from ending the run.
        """
        try:
            self._tick_count += 1
            self._drain_events()
            self.player.poll()
            if self.player.has_track:
                self._set_time(self.player.position, self.player.duration)
                if self.player.is_playing:
                    self._suppress_scale = True
                    if self.player.duration:
                        self._seek.configure(
                            value=1000.0 * self.player.position / self.player.duration
                        )
                    self._suppress_scale = False
        except Exception as exc:  # pragma: no cover - last-resort guard
            utils.log_exception("Timer frame failed", exc)
            if self._busy:
                # The worker may still be alive and about to queue a result;
                # do not leave the UI stuck on "Preparing...".
                self._cancel.set()
        finally:
            try:
                self.root.after(TICK_MS, self._tick)
            except tk.TclError:  # pragma: no cover - window is closing
                pass

    def _drain_events(self) -> None:
        while True:
            try:
                kind, payload = self._events.get_nowait()
            except queue.Empty:
                return
            if kind == "status":
                self._set_status(str(payload), theme.ACCENT)
                self._set_runtime(str(payload))
            elif kind == "chunk":
                index, total = payload
                self._show_progress(100.0 * index / max(1, total))
            elif kind == "done":
                self._finish(payload)
            elif kind == "error":
                if "cancelled" in str(payload).lower():
                    self._set_busy(False)
                    self._set_status("Cancelled.", theme.WARN)
                else:
                    self._fail(str(payload))

    # -- shutdown ------------------------------------------------------------ #
    def _save_config(self) -> None:
        try:
            width, height = self.root.winfo_width(), self.root.winfo_height()
            x, y = self.root.winfo_x(), self.root.winfo_y()
            if width > 1 and height > 1:
                self.config.set_window(width, height, x, y)
            # Read the widgets rather than trusting config, so a selection made
            # while the controls were disabled is still persisted.
            voice = self._voice_combo.get().strip()
            if voice:
                self.config.voice = voice
            self.config.save()
        except Exception as exc:  # pragma: no cover - never block exit
            utils.log_exception("Could not save settings", exc)

    def _on_close(self) -> None:
        self._cancel.set()
        self.player.shutdown()
        self._save_config()
        self.root.destroy()


def run() -> None:
    root = tk.Tk()
    try:
        TTSApp(root)
    except Exception as exc:  # pragma: no cover - last-resort guard
        utils.log_exception("Fatal error while starting the UI", exc)
        messagebox.showerror(
            utils.APP_NAME,
            "Supertonic TTS could not start.\n\n"
            f"{type(exc).__name__}: {exc}\n\n"
            f"Technical details: {utils.logs_dir() / 'app.log'}",
        )
        raise
    root.mainloop()