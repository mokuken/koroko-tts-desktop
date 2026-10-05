"""Screenshot the window so the footer rail can be eyeballed.

Writes output/shots/idle.png and output/shots/generating.png. Requires Pillow
(ImageGrab), which is dev-only and not a runtime dependency.
"""

from __future__ import annotations

import sys
import time
import tkinter as tk
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import utils
from app.gui import TTSApp

SHOTS = utils.output_dir() / "shots"


def grab(root: tk.Tk, name: str) -> None:
    from PIL import ImageGrab

    SHOTS.mkdir(parents=True, exist_ok=True)
    x, y = root.winfo_rootx(), root.winfo_rooty()
    w, h = root.winfo_width(), root.winfo_height()
    img = ImageGrab.grab(bbox=(x, y, x + w, y + h))
    path = SHOTS / f"{name}.png"
    img.save(path)
    print(f"  wrote {path}  ({w}x{h})")


def footer_of(app: TTSApp) -> tuple[int, int, int, int]:
    bar = app._status_label.master
    bar.update_idletasks()
    x = bar.winfo_rootx() - app.root.winfo_rootx()
    y = bar.winfo_rooty() - app.root.winfo_rooty()
    return x, y, bar.winfo_width(), bar.winfo_height()


def main() -> int:
    utils.setup_logging()
    root = tk.Tk()
    app = TTSApp(root)
    root.update_idletasks()
    root.update()
    app._text.insert("1.0", "Hello world. This is a narration test of the footer rail.")
    app._on_modified()
    root.update()
    time.sleep(0.4)
    root.update()

    print("footer bar geometry (relative to window):", footer_of(app))

    print("idle")
    grab(root, "idle")

    app._show_progress(45.0)
    root.update()
    print("showing 45%")
    grab(root, "generating")

    app._show_progress(100.0)
    root.update()
    grab(root, "complete")

    app._hide_progress()
    root.update()
    grab(root, "hidden")

    app._on_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())