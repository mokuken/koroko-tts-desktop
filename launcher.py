"""Frozen-app entry point.

PyInstaller executes the entry script as ``__main__`` without a package
context, which breaks the relative imports used throughout ``app/``. Keeping a
tiny launcher at the repository root makes ``app`` import as a real package in
both cases:

    python launcher.py            # development
    SupertonicTTS.exe            # portable build

Everything else lives in ``app/main.py``.
"""

from app.main import main

if __name__ == "__main__":
    raise SystemExit(main())