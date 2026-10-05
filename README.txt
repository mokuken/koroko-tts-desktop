================================================================
 Supertonic TTS  -  quick start
================================================================

WHAT THIS IS
  A small offline narration tool. Paste a script, pick a voice and
  a speed, press Generate Speech, listen, save a WAV. Nothing is
  uploaded anywhere. No account, no API key, no internet needed.

FOLDER LAYOUT
  SupertonicTTS.exe   the application
  models\supertonic\  the model files (~400 MB)
  voices\             optional extra voice styles you add yourself
  output\             every render is cached here
  logs\app.log        technical details when something goes wrong
  config.json         your saved settings (created on first close)

FIRST RUN
  1. Double-click SupertonicTTS.exe
  2. If it says "Model not found", press "Open Models Folder"
     and put the model here:
        models\supertonic\onnx\        (vocoder.onnx, vector_estimator.onnx,
                                        text_encoder.onnx, duration_predictor.onnx,
                                        tts.json, unicode_indexer.json)
        models\supertonic\voice_styles\  (F1..F5.json, M1..M5.json)
  3. Press "Refresh"

  Get the model from https://huggingface.co/Supertone/supertonic-3
  (only onlines are used - the app itself never downloads anything).

MAKING A NARRATION
  1. Click the big text box and paste your script (Ctrl+V)
  2. Choose a voice, e.g. F3
  3. Choose a speed; 1.00 is the model's natural pace
  4. Click "Generate Speech"
  5. Press the play button to listen
  6. Click "Save WAV" and pick a folder

  Long scripts are fine. The app splits them at paragraph and
  sentence boundaries, shows "Generating chunk 3 / 9..." and joins
  the pieces with natural pauses.

SPEEDS
  0.75  0.80  0.85  0.90  0.95  1.00  1.05  1.10  1.15  1.20
  Higher is faster. 0.90 - 1.10 sounds most natural.

VOICES
  F1 F2 F3 F4 F5   female   (F3 is the default)
  M1 M2 M3 M4 M5   male
  The list is read from the voice_styles folder, so if you add a
  new .json file it simply appears in the dropdown.

TROUBLESHOOTING
  "Model not found"      the models folder is missing or incomplete
  "could not be loaded"  a model file is corrupt - re-download it
  "Ran out of memory"    close other apps or use a shorter script
  No sound when playing  check Windows volume / the right output device
  Something else?        look in logs\app.log

  You can still save audio even if playback is unavailable.

CHECKING THE INSTALL FROM A TERMINAL
  Open PowerShell in this folder and try:

    .\SupertonicTTS.exe --check       report model/voice status
    .\SupertonicTTS.exe --selftest    render a test WAV, no window
    .\SupertonicTTS.exe --version     print the version

  --check and --selftest print nothing on double-click, so run them
  from PowerShell or Command Prompt if you want to see the output.

LICENSES
  The model is published by Supertone under the BigScience
  Open RAIL-M license. See LICENSES\.

Everything runs on your own machine. No telemetry, no network calls.
================================================================
