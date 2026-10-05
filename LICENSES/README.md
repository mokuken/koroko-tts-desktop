# Third-party licenses

## The Supertonic model (code and weights)

The Supertonic TTS model is published by **Supertone**.

* Model repository: <https://huggingface.co/Supertone/supertonic-3>
* Reference implementation: <https://github.com/supertone-inc/supertonic>
* License: **BigScience Open RAIL-M** (`Supertonic-MODEL-LICENSE.txt`)

The full license text is bundled in two places so it travels with any copy of
this app:

* `models/supertonic/LICENSE` (source tree)
* `LICENSES/Supertonic-MODEL-LICENSE.txt` (portable build)

### What this means for you

The Open RAIL-M license grants broad rights to use, reproduce and redistribute
the model, but it adds use-based restrictions. Do not use the model for
surveillance, biometric identification, or to impersonate a real person, and do
not use it in ways that discriminate. Commercial narration work is permitted
under this license.

Because the license travels with the weights rather than with the code, the
`models/` folder is **not** bundled into the git repository. Fetch it with
`python download_model.py` or download it from the model card.

### Our code

`app/supertonic_onnx.py` is adapted from the official reference implementation
in `supertone-inc/supertonic` (`py/helper.py`), which is distributed under the
**MIT license**. The original copyright notice is preserved in that project's
repository. The changes are documented in the module docstring.

All other files in this repository are provided as-is for personal use.

## Python packages used at runtime

| Package | License | Role |
|---|---|---|
| onnxruntime | MIT | runs the ONNX graphs |
| numpy | BSD-3-Clause | array maths |
| soundfile | BSD-3-Clause | WAV/MP3 I/O (bundles libsndfile, LGPL-2.1-or-later) |
| pygame | LGPL-2.1-or-later | audio playback (bundles SDL2, zlib, libpng) |

Build-only dependencies (`pyinstaller`, `huggingface_hub`) are not part of the
portable build.

Note on LGPL components (libsndfile, SDL2): they are dynamically linked inside
the packaged app. If you redistribute this build commercially, review your
obligations to offer the corresponding source or a relinked version.
