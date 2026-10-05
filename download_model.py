"""Optional helper: download the Supertonic model into models/supertonic/.

This is a separate script on purpose. The application never downloads anything
on its own - if the model is missing it tells you exactly where to put it.

Usage:
    python download_model.py                 # download / resume, then verify
    python download_model.py --check         # verify an existing install only
    python download_model.py --force         # re-download even if present
    python download_model.py --repo owner/name

Files come from the official Hugging Face repository. ~400 MB.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app import model_manager, utils
from app.supertonic_onnx import ONNX_CONFIG_FILES, ONNX_FILES

DEFAULT_REPO = "Supertone/supertonic-3"
WANTED = [
    "onnx/*",
    "voice_styles/*",
    "LICENSE",
    "README.md",
    "config.json",
]


def report() -> bool:
    status = model_manager.model_status()
    folder = status.display_dir
    print(f"\nModel folder : {folder}")
    print(f"Voices folder: {model_manager.supertonic_dir() / 'voice_styles'}")
    if status.ok:
        voices = model_manager.list_voices()
        print(f"Status       : OK ({len(voices)} voices, {status.sample_rate} Hz)")
        print(f"Voices       : {', '.join(v.id for v in voices)}")
        return True
    print(f"Status       : INCOMPLETE\n{status.detail}")
    if status.missing:
        print(f"Missing      : {', '.join(status.missing)}")
    return False


def download(repo: str) -> int:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(
            "huggingface_hub is not installed.\n"
            "Install it with:  pip install huggingface_hub\n"
            "Or download the model manually from:\n"
            f"  https://huggingface.co/{repo}\n"
            f"and place it in: {model_manager.supertonic_dir()}"
        )
        return 2

    target = model_manager.supertonic_dir()
    target.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {repo} -> {target}")
    print("This is about 400 MB and needs an internet connection.")
    print("Cancel with Ctrl+C at any time; a partial download can be resumed.\n")
    try:
        snapshot_download(
            repo_id=repo, allow_patterns=WANTED, local_dir=str(target)
        )
    except KeyboardInterrupt:
        print("\nCancelled. Run the script again to resume.")
        return 130
    except Exception as exc:
        print(f"\nDownload failed: {exc}")
        print(f"Check your connection, then retry. Target: {target}")
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="only verify, do not download")
    parser.add_argument("--force", action="store_true", help="re-download if already present")
    parser.add_argument("--repo", default=DEFAULT_REPO, help="Hugging Face repo id")
    args = parser.parse_args(argv)

    utils.setup_logging()
    print("=" * 66)
    print(" Supertonic TTS - model setup")
    print("=" * 66)
    print(f"Source: https://huggingface.co/{args.repo}")
    print("License: BigScience Open RAIL-M (see models/supertonic/LICENSE)\n")

    if args.check:
        return 0 if report() else 1

    if not args.force and report():
        print("\nModel already installed. Use --force to re-download.")
        return 0

    code = download(args.repo)
    if code != 0:
        return code
    print()
    return 0 if report() else 1


if __name__ == "__main__":
    raise SystemExit(main())
