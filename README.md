# koroko-tts-desktop

A small offline narration app for Windows 11. Paste a script, pick a voice and
a speed, press one button, get a WAV you can listen to and save.

It uses the [Supertonic 3](https://github.com/supertone-inc/supertonic) model
from Supertone, running locally through ONNX Runtime. There is no server, no
account, no API key and no network traffic at runtime. Nothing you type leaves
the machine.

* **Runtime:** Python 3.11 / 3.12, Tkinter, ONNX Runtime, soundfile, pygame
* **Packaging:** PyInstaller, one folder, no installer, no admin rights
* **Model:** ~400 MB, downloaded once, kept separate from the executable

---

## Quick start (portable build)

```
1. Unzip the folder anywhere (no install, no admin).
2. Double-click  SupertonicTTS.exe
3. Paste a script, choose F3 and 1.00, click  Generate Speech
4. Press play, then  Save WAV
```

`README.txt` inside the built folder is the end-user guide.

## Quick start (from source)

```bat
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python download_model.py     :: ~400 MB, one time
.venv\Scripts\python -m app.main
```

Or on Windows, `run_dev.bat` picks the interpreter and starts the app.

---

## What it does

| | |
|---|---|
| **Generate** | Paste text, click once, listen. Long scripts are split and rejoined automatically. |
| **Voices** | F1-F5 and M1-M5, discovered from the `voice_styles` folder. Drop in a new `.json` and it appears in the list. |
| **Speeds** | 0.75 to 1.20 in 0.05 steps. 0.90-1.10 sounds most natural. |
| **Languages** | English, Korean and Japanese. The chunker and the model's own text preprocessing handle each. |
| **Cancel** | Long scripts render chunk by chunk; cancel at any point and keep whatever finished. |
| **Play** | Play, pause, stop, scrub. Built on SDL via pygame. |
| **Save** | WAV. MP3 if libsndfile supports it. |
| **Settings** | Remembered in `config.json` next to the app. |

Nothing is uploaded. There is no telemetry, no update check and no phone-home.

## Project layout

```
koroko-tts-desktop/
  app/
    main.py            entry point, --check / --selftest / --verbose
    gui.py             the window: compose, generate, play, save
    tts_engine.py      model lifecycle, chunking, cancellation, progress
    supertonic_onnx.py ONNX graph, voice styles, text normalisation
    chunker.py         splits narration at natural boundaries
    audio_io.py        WAV/MP3 writing, joining, silence, peak limiting
    audio_player.py    playback transport
    config.py          validated, atomically saved settings
    model_manager.py   finds the model and the voices
    theme.py           dark theme and ttk styling
    utils.py           portable paths, logging, formatting
  tests/
    test_core.py       64 unit and end-to-end tests
    gui_smoke.py       52 scripted GUI workflow checks
  download_model.py    optional model fetcher
  build_windows.bat    produces the portable build
  SupertonicTTS.spec   PyInstaller definition
  models/supertonic/   model files (not in git, see LICENSES)
```

## Building the portable app

```bat
build_windows.bat                :: app only, ~110 MB
build_windows.bat --with-model   :: app + model, ~490 MB, ready to ship
build_windows.bat --test         :: run the tests first
```

Output is `dist\SupertonicTTS\`. Copy the whole folder to any Windows 10/11 PC.
The model is deliberately *not* embedded in the executable: it keeps the app
small, lets you swap models, and keeps the separately-licensed weights in one
obvious place.

Verify a build before shipping it:

```powershell
.\dist\SupertonicTTS\SupertonicTTS.exe --check      # model found, voices listed
.\dist\SupertonicTTS\SupertonicTTS.exe --selftest   # renders output\selftest.wav
```

`--check` exits 0 when the model is usable and 1 otherwise, so it works as a
deployment gate.

## Tests

```bat
.venv\Scripts\python -m unittest discover -s tests -p "test_*.py"
.venv\Scripts\python tests\gui_smoke.py
```

`test_core.py` covers chunking, config, discovery, audio IO, provider
selection, the CLI and real model synthesis. Tests that need the model skip
themselves cleanly when it is absent, so the suite still runs on a fresh
checkout.

`gui_smoke.py` drives the real widgets without a mouse: generate, play, seek,
cancel, save, persist settings, relaunch, and the missing-model screen.

## Performance

Measured on this machine (CPU only, model load ~1.8 s, default 8 steps):

| Text | Chunks | Audio produced | Time taken | Faster than playback by |
|---|---|---|---|---|
| 8 words | 1 | 3.6 s | 3.2 s | 1.1x |
| 300 words | 6 | 115.2 s | 60.7 s | 1.9x |

Rule of thumb: roughly **2 minutes of audio per minute of waiting**, and the gap
widens on longer texts because per-chunk overhead is paid once. A full paragraph
page lands around 4-6 minutes of audio.

The model generates faster than playback on CPU. A GPU is used automatically if
`onnxruntime-gpu` or `onnxruntime-directml` is installed and `prefer_gpu` is set;
otherwise it falls back to CPU silently. Supertonic ships CPU reference
weights, so CPU is the supported path.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| "Model not found" on launch | Copy `models\supertonic` next to the exe. Press **Open Models Folder** in the app. |
| "could not be loaded" | A model file is corrupt, usually a half-finished download. Re-run `download_model.py --force`. |
| No voices listed | `models\supertonic\voice_styles\` is empty or missing. |
| No sound, but export works | Check Windows volume and the default output device. Playback failure never blocks saving. |
| "Ran out of memory" | Close other apps, or generate in smaller pieces. |
| Settings not remembered | The app folder is read-only, so settings go to `%LOCALAPPDATA%\SupertonicTTS`. |
| Anything else | `logs\app.log`. |

## Licence

The **model** is Supertone's, under the BigScience Open RAIL-M licence. See
`LICENSES/README.md` for what that means and where the full text lives.

The Python wrapper in `app/supertonic_onnx.py` is adapted from Supertonic's
MIT-licensed reference implementation, with the original notice preserved.

If you redistribute the build commercially, review the obligations for the
LGPL-licensed native libraries it contains (libsndfile, SDL2).