"""GUI smoke test: build the real window and verify the layout numerically.

We cannot rely on screenshots, so instead we walk the live widget tree and
assert that every interactive element exists, is visible, has a sensible size,
and that the primary workflow (generate -> player -> save) actually runs.
"""

from __future__ import annotations

import sys
import time
import tkinter as tk
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import audio_io, chunker, model_manager, theme, utils
from app.gui import TTSApp

failures: list[str] = []
checks = 0


def check(condition: bool, message: str) -> None:
    global checks
    checks += 1
    if not condition:
        failures.append(message)
        print(f"  FAIL  {message}")
    else:
        print(f"  ok    {message}")


def dump(widget: tk.Misc, depth: int = 0) -> None:
    for child in widget.winfo_children():
        try:
            geo = (
                f"{child.winfo_width()}x{child.winfo_height()} "
                f"@ {child.winfo_x()},{child.winfo_y()}"
            )
            label = ""
            for attr in ("text", "cget"):
                pass
            if child.winfo_class() in ("TLabel", "Label", "TButton", "Button", "TCombobox"):
                try:
                    label = f"  text={child.cget('text')!r}"
                except tk.TclError:
                    pass
            print("    " + "  " * depth + f"{child.winfo_class():16s} {geo}{label}")
            dump(child, depth + 1)
        except tk.TclError:
            continue


def settle(app: TTSApp, ms: int = 320) -> None:
    """Let debounced work (e.g. the character counter) run."""
    root = app.root
    root.update()
    time.sleep(ms / 1000.0)
    root.update()


def main() -> int:
    utils.setup_logging()
    # Start from a clean slate so first-launch defaults are actually tested.
    cfg_file = utils.config_path()
    if cfg_file.exists():
        cfg_file.unlink()
    root = tk.Tk()
    app = TTSApp(root)
    root.update_idletasks()
    root.update()

    print("\n--- widget tree ---")
    dump(root)

    print("\n--- layout checks ---")
    check(app._text.winfo_exists(), "narration text box exists")
    check(app._text.winfo_height() > 120, f"text box is tall ({app._text.winfo_height()}px)")
    check(
        not utils.config_path().exists(),
        "first launch writes no config until something is saved",
    )
    check(
        str(app._voice_combo["state"]) == "readonly",
        f"voice dropdown is readonly ({app._voice_combo['state']!r})",
    )
    check(
        len(app._voice_combo.cget("values")) == len(model_manager.list_voices()),
        f"voice dropdown lists all {len(model_manager.list_voices())} discovered voices",
    )
    check(
        app._voice_combo.get() in [v.id for v in model_manager.list_voices()],
        f"selected voice '{app._voice_combo.get()}' is a real voice",
    )
    check(
        app._speed_combo.get() == "1.00", f"speed defaults to 1.00 (got {app._speed_combo.get()})"
    )
    check(
        root.title() == "Supertonic TTS", f"window title is 'Supertonic TTS' ({root.title()})"
    )
    check(
        app._generate_btn["text"] == "Generate Speech",
        "primary button reads 'Generate Speech'",
    )
    check(
        str(app._generate_btn["state"]) == "disabled",
        "Generate is disabled with empty text",
    )
    check(
        root.winfo_width() >= 640 and root.winfo_height() >= 620,
        f"window honours minimum size ({root.winfo_width()}x{root.winfo_height()})",
    )

    print("\n--- text entry + counters ---")
    app._text.insert("1.0", "Hello world. This is a narration test.")
    app._text.edit_modified(True)
    app._on_modified()
    settle(app)
    meta = app._meta.cget("text")
    check("characters" in meta, f"counter shows characters ({meta!r})")
    check("words" in meta, f"counter shows words ({meta!r})")
    check("chunk" in meta, f"counter shows chunk count ({meta!r})")
    check(str(app._generate_btn["state"]) == "normal", "Generate enables once text exists")
    check(not app._hint.winfo_ismapped(), "placeholder hint hides when text is present")

    print("\n--- keyboard shortcuts ---")
    app._text.tag_remove("sel", "1.0", "end")
    app._select_all()
    check(
        app._text.tag_ranges("sel") != (),
        "Ctrl+A selects all text",
    )
    app._redo()  # must not raise when there is nothing to redo

    print("\n--- generation through the real GUI path ---")
    app._voice_combo.set("F3")
    app._on_voice_selected()
    app._speed_combo.set("1.20")
    app._on_speed_selected()  # simulates the user picking a speed
    app._text.delete("1.0", "end")
    app._text.insert(
        "1.0",
        "Supertonic TTS builds narration locally. "
        "Nothing is uploaded anywhere.",
    )
    app._text.edit_modified(True)
    app._on_modified()
    root.update()
    app._on_generate()
    check(app._busy, "generation switches the UI into busy state")
    check(
        str(app._cancel_btn.winfo_manager()) == "pack",
        "Cancel button appears during generation",
    )

    deadline = time.time() + 180
    while app._busy and time.time() < deadline:
        root.update()
        time.sleep(0.05)
    root.update()

    check(not app._busy, "generation finished")
    check(app._result is not None, "a result was produced")
    if app._result is not None:
        check(app._result.sample_rate == 44100, "audio is 44.1 kHz")
        check(app._result.duration > 1.0, f"audio has length ({app._result.duration:.1f}s)")
        status = app._status_label.cget("text")
        check("Done" in status, f"status reports Done ({status!r})")
    check(
        str(app._save_wav_btn["state"]) == "normal", "Save WAV enables after generating"
    )
    check(
        str(app._save_mp3_btn["state"]) == "normal" if app._mp3_supported else True,
        "Save MP3 enables after generating",
    )
    check(utils.config_path().exists(), "settings file is created on the first save")
    check(app.player.has_track, "generated audio loaded into the player")
    check(app.player.duration > 1.0, f"player knows the duration ({app.player.duration:.1f}s)")

    print("\n--- player transport ---")
    if app.player.available:
        app._on_play_toggle()
        root.update()
        time.sleep(0.6)
        root.update()
        playing = app.player.is_playing
        pos = app.player.position
        check(playing, "play starts")
        check(pos > 0.0, f"position advances while playing ({pos:.2f}s)")
        check("pause" == app._play_icon._kind, "icon switches to pause")
        app._on_play_toggle()
        root.update()
        check(not app.player.is_playing, "pause stops playback")
        check("play" == app._play_icon._kind, "icon switches back to play")
        if app.player.supports_seek:
            app._on_seek()
            app._seek.configure(value=500)
            app._on_seek()
            check(
                abs(app.player.position - app.player.duration / 2) < 1.0,
                f"seek jumps to the middle ({app.player.position:.1f}s)",
            )
        app._on_stop()
        root.update()
        check(app.player.position == 0.0, "stop rewinds to zero")
        check("00:00 /" in app._time_label.cget("text"), "time label updates")
    else:
        print(f"  skip  no audio device ({app.player.error})")

    print("\n--- persistence ---")
    app._speed_combo.set("0.95")
    app._on_speed_selected()
    app._on_voice_selected()
    app._save_config()
    from app.config import Config

    reloaded = Config.load()
    check(abs(reloaded.speed - 0.95) < 1e-6, f"speed persisted ({reloaded.speed})")
    check(reloaded.voice == "F3", f"voice persisted ({reloaded.voice})")
    check(
        reloaded.window["width"] > 0 and reloaded.window["height"] > 0,
        f"window size persisted ({reloaded.window})",
    )

    print("\n--- settings are restored on relaunch ---")
    app._speed_combo.set("1.15")
    app._on_speed_selected()
    app._voice_combo.set("M2")
    app._on_voice_selected()
    app._save_config()
    app._on_close()
    relaunched = tk.Tk()
    app2 = TTSApp(relaunched)
    relaunched.update()
    check(
        app2._speed_combo.get() == "1.15",
        f"speed restored after relaunch ({app2._speed_combo.get()})",
    )
    check(
        app2._voice_combo.get() == "M2",
        f"voice restored after relaunch ({app2._voice_combo.get()})",
    )
    check(
        relaunched.winfo_width() == reloaded.window["width"]
        and relaunched.winfo_height() == reloaded.window["height"],
        f"window size restored ({relaunched.winfo_width()}x{relaunched.winfo_height()})",
    )
    check(
        relaunched.winfo_x() == reloaded.window["x"]
        and relaunched.winfo_y() == reloaded.window["y"],
        f"window position restored ({relaunched.winfo_x()},{relaunched.winfo_y()})",
    )
    app = app2
    root = relaunched

    print("\n--- cancel path (needs several chunks to be interruptible) ---")
    app._text.delete("1.0", "end")
    app._text.insert(
        "1.0",
        (
            "Once upon a time, in a small village nestled between rolling hills, "
            "there lived a young artist named Clara who painted the sunrise. "
            "Every morning she woke before dawn to catch the first light. "
            "Her gallery was known throughout the region for vivid colour.\n\n"
            "People travelled from far away to see her work, and many said the "
            "paintings told stories that words never could. The critics agreed.\n\n"
            "And so the story continues, chapter after chapter, without end."
        ),
    )
    app._on_modified()
    settle(app)
    app._on_generate()
    root.update()
    check(app._busy, "long-form generation started")
    time.sleep(1.2)
    root.update()
    app._on_cancel()
    cancel_deadline = time.time() + 180
    while app._busy and time.time() < cancel_deadline:
        root.update()
        time.sleep(0.05)
    root.update()
    check(not app._busy, "cancel returns the UI to idle")
    check(
        "cancel" in app._status_label.cget("text").lower(),
        f"status reflects cancellation ({app._status_label.cget('text')!r})",
    )

    print("\n--- missing-model view ---")
    app._build_missing_view(model_manager.ModelStatus())
    root.update()
    labels = []

    def collect(w: tk.Misc) -> None:
        for child in w.winfo_children():
            try:
                if child.winfo_class() in ("Label", "TLabel"):
                    labels.append(child.cget("text"))
            except tk.TclError:
                pass
            collect(child)

    collect(app._content)
    joined = " ".join(labels)
    check(
        any("Model not found" in text for text in labels),
        "first-launch screen says 'Model not found'",
    )
    check("supertonic" in joined.lower(), "first-launch screen names the models folder")
    check(
        any("Open Models Folder" in t for w in [root] for t in _all_button_texts(w)),
        "first-launch screen offers 'Open Models Folder'",
    )

    print("\n--- rebuild back to normal view ---")
    app._refresh_model_view()
    root.update()
    check(app._text.winfo_exists(), "app returns to the main view after Refresh")
    check(app._voice_combo.get() in [v.id for v in model_manager.list_voices()], "voice reselected")

    app._on_close()
    print(f"{checks - len(failures)}/{checks} checks passed")
    if failures:
        print("FAILURES:")
        for item in failures:
            print("  -", item)
        return 1
    print("GUI SMOKE TEST PASSED")
    return 0


def _all_button_texts(widget: tk.Misc) -> list[str]:
    found: list[str] = []
    for child in widget.winfo_children():
        try:
            if child.winfo_class() in ("TButton", "Button"):
                found.append(str(child.cget("text")))
        except tk.TclError:
            pass
        found.extend(_all_button_texts(child))
    return found


if __name__ == "__main__":
    raise SystemExit(main())
