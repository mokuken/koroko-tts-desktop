# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the portable Supertonic TTS build.

Produces dist\\SupertonicTTS\\ containing SupertonicTTS.exe plus everything it
needs. Model files are NOT bundled: the app locates models/supertonic/ next to
the executable at runtime, which keeps the exe small and the model swappable.
"""

import sys
from pathlib import Path

PROJECT = Path(SPECPATH).resolve()
ICON = PROJECT / "assets" / "SupertonicTTS.ico"

block_cipher = None

hidden = [
    "app",
    "app.main",
    "app.gui",
    "app.tts_engine",
    "app.supertonic_onnx",
    "app.audio_io",
    "app.audio_player",
    "app.chunker",
    "app.config",
    "app.model_manager",
    "app.theme",
    "app.utils",
    "soundfile",
    "numpy",
    "onnxruntime",
    # pygame ships a PyInstaller hook that collects the SDL DLLs, so only the
# mixer needs naming here. pygame.mixer.music is an attribute, not a module.
"pygame",
"pygame.mixer",
"cffi",
]

excludes = [
    # Keep the build lean; none of these are used at runtime.
    "tkinter.test",
    "test",
    "unittest",
    "pydoc_data",
    "librosa",
    "sklearn",
    "scipy",
    "pandas",
    "matplotlib",
    "PIL",
    "IPython",
    "notebook",
    "torch",
    "tensorflow",
    "setuptools",
    "pip",
    "huggingface_hub",
    "requests",
    "urllib3",
]

a = Analysis(
    ["launcher.py"],
    pathex=[str(PROJECT)],
    binaries=[],
    datas=[],
    hiddenimports=hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="SupertonicTTS",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,  # GUI app: no console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ICON) if ICON.exists() else None,
    version=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="SupertonicTTS",
)
