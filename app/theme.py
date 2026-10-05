"""Dark theme: palette, fonts and ttk widget styling.

Tk has no rounded corners, so the look is achieved with flat surfaces, a
1 px border, generous padding and one accent colour used only for the primary
action.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import font as tkfont
from tkinter import ttk

BG = "#14161a"
SURFACE = "#1b1e25"
SURFACE_ALT = "#22262f"
SURFACE_HI = "#2a2f3a"
BORDER = "#2d323d"
TEXT = "#e7e9ee"
TEXT_DIM = "#8b93a1"
TEXT_FAINT = "#666e7c"
ACCENT = "#4d86f7"
ACCENT_HI = "#6a9bff"
ACCENT_LO = "#3a6bd6"
ACCENT_DIM = "#23375e"
SUCCESS = "#3fb950"
DANGER = "#f0584f"
WARN = "#d9a441"
SELECT = "#2f4d80"
ENTRY_BG = "#171a20"

UI_FAMILY = "Segoe UI"
MONO_FAMILY = "Consolas"

PAD = 14


def font(size: int = 10, weight: str = "normal", family: str = UI_FAMILY):
    return (family, size, weight)


def fonts() -> dict[str, tuple]:
    return {
        "title": font(15, "bold"),
        "subtitle": font(9, "normal"),
        "label": font(9, "bold"),
        "body": font(10),
        "small": font(9),
        "tiny": font(8),
        "mono": font(9, "normal", MONO_FAMILY),
        "button": font(10, "semibold"),
    }


def configure_root(root: tk.Misc) -> None:
    root.configure(bg=BG)


def apply_style(root: tk.Misc) -> ttk.Style:
    """Install the dark ``clam`` theme."""
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:  # pragma: no cover - defensive
        pass

    style.configure(".", background=BG, foreground=TEXT, bordercolor=BORDER,
                    lightcolor=BORDER, darkcolor=BORDER, focuscolor=BG)

    style.configure("Dark.TFrame", background=BG)
    style.configure("Surface.TFrame", background=SURFACE)
    style.configure("Alt.TFrame", background=SURFACE_ALT)

    style.configure(
        "Dark.TLabel",
        background=BG,
        foreground=TEXT,
        font=font(10),
    )
    style.configure(
        "Dim.TLabel", background=BG, foreground=TEXT_DIM, font=font(9)
    )
    style.configure(
        "Surface.TLabel", background=SURFACE, foreground=TEXT, font=font(10)
    )
    style.configure(
        "Section.TLabel", background=BG, foreground=TEXT, font=font(10, "bold")
    )
    style.configure(
        "Faint.TLabel", background=BG, foreground=TEXT_FAINT, font=font(8)
    )

    style.configure(
        "Dark.TButton",
        background=SURFACE_ALT,
        foreground=TEXT,
        bordercolor=BORDER,
        focuscolor=SURFACE_ALT,
        font=font(10),
        padding=(14, 8),
        relief="flat",
    )
    style.map(
        "Dark.TButton",
        background=[("active", SURFACE_HI), ("pressed", SURFACE_HI), ("disabled", SURFACE)],
        foreground=[("disabled", TEXT_FAINT)],
        bordercolor=[("active", SURFACE_HI)],
    )

    style.configure(
        "Accent.TButton",
        background=ACCENT,
        foreground="#ffffff",
        bordercolor=ACCENT,
        focuscolor=ACCENT,
        font=font(10, "bold"),
        padding=(10, 11),
        relief="flat",
    )
    style.map(
        "Accent.TButton",
        background=[("active", ACCENT_HI), ("pressed", ACCENT_LO), ("disabled", ACCENT_DIM)],
        bordercolor=[("active", ACCENT_HI), ("disabled", ACCENT_DIM)],
        foreground=[("disabled", "#8b93a1")],
    )

    style.configure(
        "Ghost.TButton",
        background=BG,
        foreground=TEXT_DIM,
        bordercolor=BORDER,
        focuscolor=BG,
        font=font(9),
        padding=(10, 6),
        relief="flat",
    )
    style.map(
        "Ghost.TButton",
        background=[("active", SURFACE_ALT)],
        foreground=[("active", TEXT)],
    )

    style.configure(
        "Dark.TCombobox",
        fieldbackground=ENTRY_BG,
        background=ENTRY_BG,
        foreground=TEXT,
        arrowcolor=TEXT_DIM,
        bordercolor=BORDER,
        lightcolor=ENTRY_BG,
        darkcolor=ENTRY_BG,
        insertcolor=TEXT,
        font=font(10),
        padding=(8, 6),
        relief="flat",
    )
    style.map(
        "Dark.TCombobox",
        fieldbackground=[("readonly", ENTRY_BG), ("hover", ENTRY_BG), ("focus", ENTRY_BG)],
        background=[("readonly", ENTRY_BG)],
        foreground=[("readonly", TEXT), ("disabled", TEXT_FAINT)],
        arrowcolor=[("active", TEXT), ("hover", TEXT)],
        bordercolor=[("focus", ACCENT), ("hover", BORDER)],
        selectbackground=[("readonly", ENTRY_BG)],
        selectforeground=[("readonly", TEXT)],
    )
    root.option_add("*TCombobox*Listbox.background", ENTRY_BG)
    root.option_add("*TCombobox*Listbox.foreground", TEXT)
    root.option_add("*TCombobox*Listbox.selectBackground", SELECT)
    root.option_add("*TCombobox*Listbox.selectForeground", "#ffffff")

    # The footer progress rail is drawn on a Canvas rather than styled here:
    # clam's progressbar trough element ignores `thickness`, so ttk always
    # reserves 18px no matter how thin it is told to be. See gui.FooterProgress.
    style.configure(
        "Dark.Horizontal.TScale",
        background=BG,
        troughcolor=SURFACE_HI,
        bordercolor=BORDER,
        lightcolor=BG,
        darkcolor=BG,
    )
    style.map("Dark.Horizontal.TScale", background=[("active", ACCENT)])

    style.configure(
        "Dark.Vertical.TScrollbar",
        background=SURFACE_HI,
        troughcolor=SURFACE,
        bordercolor=SURFACE,
        arrowcolor=TEXT_DIM,
        lightcolor=SURFACE_HI,
        darkcolor=SURFACE_HI,
        arrowsize=11,
        width=11,
    )
    style.map(
        "Dark.Vertical.TScrollbar",
        background=[("active", ACCENT), ("pressed", ACCENT)],
    )

    style.configure("TNotebook", background=BG, bordercolor=BORDER)
    style.configure("TNotebook.Tab", background=SURFACE, foreground=TEXT_DIM,
                    padding=(12, 6), font=font(9))
    style.map("TNotebook.Tab", background=[("selected", SURFACE_ALT)], foreground=[("selected", TEXT)])

    return style


def text_widget(text: tk.Text) -> None:
    """Apply dark colours to a plain ``tk.Text``."""
    text.configure(
        background=ENTRY_BG,
        foreground=TEXT,
        insertbackground=TEXT,
        selectbackground=SELECT,
        selectforeground="#ffffff",
        highlightbackground=BORDER,
        highlightcolor=ACCENT,
        highlightthickness=1,
        borderwidth=0,
        relief="flat",
        padx=12,
        pady=10,
        wrap="word",
        font=font(11),
        undo=True,
        maxundo=1000,
        tabs="1c",
    )


def check_available_fonts(root: tk.Misc) -> dict[str, bool]:
    available = set(tkfont.families(root))
    return {
        "ui": UI_FAMILY in available or "Tahoma" in available,
        "mono": MONO_FAMILY in available,
    }