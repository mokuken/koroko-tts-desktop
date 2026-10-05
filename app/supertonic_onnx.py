"""Supertonic ONNX inference.

Adapted from the official reference implementation
(github.com/supertone-inc/supertonic, ``py/helper.py``) with three changes
that matter for a desktop app:

* provider selection with graceful CPU fallback,
* an injectable RNG so the same text/voice/speed always yields the same audio,
* direct return of trimmed mono float32 samples.

The text pre-processing is kept byte-for-byte identical to upstream because it
is what makes the model pronounce punctuation, numbers and acronyms correctly.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Callable, Sequence
from unicodedata import normalize

import numpy as np
import onnxruntime as ort

log = logging.getLogger(__name__)

AVAILABLE_LANGS = (
    "en ko ja ar bg cs da de el es et fi fr hi hr hu id it lt lv nl pl pt ro "
    "ru sk sl sv tr uk vi na"
).split()

DEFAULT_LANG = "en"

#: Supertonic's own per-language chunk ceilings.
LANG_MAX_CHARS = {"ko": 120, "ja": 120}

ONNX_FILES = {
    "dp": "duration_predictor.onnx",
    "text_enc": "text_encoder.onnx",
    "vector_est": "vector_estimator.onnx",
    "vocoder": "vocoder.onnx",
}
ONNX_CONFIG_FILES = ("tts.json", "unicode_indexer.json")

_EMOJI = re.compile(
    "[\U0001f600-\U0001f64f\U0001f300-\U0001f5ff\U0001f680-\U0001f6ff"
    "\U0001f700-\U0001f77f\U0001f780-\U0001f7ff\U0001f800-\U0001f8ff"
    "\U0001f900-\U0001f9ff\U0001fa00-\U0001fa6f\U0001fa70-\U0001faff"
    "☀-⛿✀-➿\U0001f1e6-\U0001f1ff]+",
    flags=re.UNICODE,
)
_REPLACEMENTS = {
    "–": "-", "‑": "-", "—": "-", "_": " ",
    "“": '"', "”": '"', "‘": "'", "’": "'",
    "´": "'", "`": "'", "[": " ", "]": " ", "|": " ", "/": " ",
    "#": " ", "→": " ", "←": " ",
}
_EXPR_REPLACEMENTS = {"@": " at ", "e.g.,": "for example, ", "i.e.,": "that is, "}


class ModelError(RuntimeError):
    """Raised when the ONNX assets are missing, incomplete or unreadable."""


# --------------------------------------------------------------------------- #
# text front-end (identical to upstream)
# --------------------------------------------------------------------------- #
class UnicodeProcessor:
    def __init__(self, unicode_indexer_path: str):
        with open(unicode_indexer_path, "r", encoding="utf-8") as handle:
            self.indexer = json.load(handle)
        log.debug("unicode indexer entries: %d", len(self.indexer))

    def _preprocess_text(self, text: str, lang: str) -> str:
        text = normalize("NFKD", text)
        text = _EMOJI.sub("", text)
        for key, value in _REPLACEMENTS.items():
            text = text.replace(key, value)
        text = re.sub(r"[♥☆♡©\\]", "", text)
        for key, value in _EXPR_REPLACEMENTS.items():
            text = text.replace(key, value)

        text = re.sub(r" ,", ",", text)
        text = re.sub(r" \.", ".", text)
        text = re.sub(r" !", "!", text)
        text = re.sub(r" \?", "?", text)
        text = re.sub(r" ;", ";", text)
        text = re.sub(r" :", ":", text)
        text = re.sub(r" '", "'", text)

        while '""' in text:
            text = text.replace('""', '"')
        while "''" in text:
            text = text.replace("''", "'")
        while "``" in text:
            text = text.replace("``", "`")

        text = re.sub(r"\s+", " ", text).strip()
        if not re.search(r"[.!?;:,'\"')\]}…。」』】〉》›»]$", text):
            text += "."
        if lang not in AVAILABLE_LANGS:
            raise ValueError(
                f"Language '{lang}' is not supported by Supertonic. "
                f"Choose one of: {', '.join(AVAILABLE_LANGS)}"
            )
        return f"<{lang}>" + text + f"</{lang}>"

    @staticmethod
    def _get_text_mask(text_ids_lengths: np.ndarray) -> np.ndarray:
        return length_to_mask(text_ids_lengths)

    def __call__(
        self, text_list: Sequence[str], lang_list: Sequence[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        processed = [
            self._preprocess_text(t, lang) for t, lang in zip(text_list, lang_list)
        ]
        lengths = np.array([len(t) for t in processed], dtype=np.int64)
        text_ids = np.zeros((len(processed), int(lengths.max())), dtype=np.int64)
        for i, text in enumerate(processed):
            values = [self.indexer[ord(char)] for char in text]
            text_ids[i, : len(values)] = np.array(values, dtype=np.int64)
        return text_ids, self._get_text_mask(lengths)


def length_to_mask(lengths: np.ndarray, max_len: int | None = None) -> np.ndarray:
    max_len = max_len or int(lengths.max())
    mask = (np.arange(0, max_len) < np.expand_dims(lengths, axis=1)).astype(np.float32)
    return mask.reshape(-1, 1, max_len)


def get_latent_mask(
    wav_lengths: np.ndarray, base_chunk_size: int, chunk_compress_factor: int
) -> np.ndarray:
    latent_size = base_chunk_size * chunk_compress_factor
    latent_lengths = (wav_lengths + latent_size - 1) // latent_size
    return length_to_mask(latent_lengths)


# --------------------------------------------------------------------------- #
# voice styles
# --------------------------------------------------------------------------- #
@dataclass
class Style:
    ttl: np.ndarray  # [1, latent_dim, seq]
    dp: np.ndarray  # [1, latent_dim, seq]


def load_voice_style(paths: Sequence[str]) -> Style:
    if not paths:
        raise ModelError("No voice style file was provided.")
    first_path = str(paths[0])
    try:
        with open(first_path, "r", encoding="utf-8") as handle:
            first = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelError(f"Could not read voice style '{first_path}': {exc}") from exc

    ttl_dims = first["style_ttl"]["dims"]
    dp_dims = first["style_dp"]["dims"]
    ttl_style = np.zeros([1, ttl_dims[1], ttl_dims[2]], dtype=np.float32)
    dp_style = np.zeros([1, dp_dims[1], dp_dims[2]], dtype=np.float32)

    for i, voice_style_path in enumerate(paths):
        try:
            with open(str(voice_style_path), "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise ModelError(
                f"Voice style '{voice_style_path}' is missing or corrupt: {exc}"
            ) from exc
        ttl_style[i] = np.asarray(payload["style_ttl"]["data"], dtype=np.float32).reshape(
            ttl_dims[1], ttl_dims[2]
        )
        dp_style[i] = np.asarray(payload["style_dp"]["data"], dtype=np.float32).reshape(
            dp_dims[1], dp_dims[2]
        )
    return Style(ttl_style, dp_style)


# --------------------------------------------------------------------------- #
# execution providers
# --------------------------------------------------------------------------- #
PROVIDER_LABELS = {
    "DmlExecutionProvider": "DirectML",
    "CUDAExecutionProvider": "CUDA",
    "TensorrtExecutionProvider": "TensorRT",
    "ROCMExecutionProvider": "ROCm",
    "CPUExecutionProvider": "CPU",
}


def available_providers() -> list[str]:
    try:
        return list(ort.get_available_providers())
    except Exception:  # pragma: no cover - defensive
        return ["CPUExecutionProvider"]


def select_providers(prefer_gpu: bool) -> tuple[list[str], str]:
    """Pick the best available provider, always keeping CPU as a fallback."""
    available = available_providers()
    log.info("Available ONNX providers: %s", available)
    order = (
        ["DmlExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
        if prefer_gpu
        else ["CPUExecutionProvider"]
    )
    chosen = [p for p in order if p in available]
    if "CPUExecutionProvider" in available and "CPUExecutionProvider" not in chosen:
        chosen.append("CPUExecutionProvider")
    if not chosen:
        chosen = ["CPUExecutionProvider"]
    label = PROVIDER_LABELS.get(chosen[0], chosen[0])
    return chosen, label


def _session(path: str, providers: list[str]) -> ort.InferenceSession:
    if not os.path.isfile(path):
        raise ModelError(f"Missing model file: {os.path.basename(path)}")
    options = ort.SessionOptions()
    options.log_severity_level = 3  # errors only; we log details ourselves
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, sess_options=options, providers=providers)


# --------------------------------------------------------------------------- #
# engine
# --------------------------------------------------------------------------- #
class TextToSpeech:
    """Loads the Supertonic ONNX graph and synthesises single chunks."""

    def __init__(
        self,
        onnx_dir: str,
        prefer_gpu: bool = False,
        on_progress: Callable[[str], None] | None = None,
    ):
        self.onnx_dir = str(onnx_dir)
        missing = [
            name
            for name in (*ONNX_CONFIG_FILES, *ONNX_FILES.values())
            if not os.path.isfile(os.path.join(self.onnx_dir, name))
        ]
        if missing:
            raise ModelError(
                "The Supertonic model folder is incomplete.\n\n"
                "Missing file(s):\n  " + "\n  ".join(missing)
            )

        try:
            with open(os.path.join(self.onnx_dir, "tts.json"), "r", encoding="utf-8") as h:
                self.cfgs = json.load(h)
        except (OSError, json.JSONDecodeError) as exc:
            raise ModelError(f"tts.json could not be read: {exc}") from exc

        self.sample_rate = int(self.cfgs["ae"]["sample_rate"])
        self.base_chunk_size = int(self.cfgs["ae"]["base_chunk_size"])
        self.chunk_compress_factor = int(self.cfgs["ttl"]["chunk_compress_factor"])
        self.ldim = int(self.cfgs["ttl"]["latent_dim"])

        self.providers, self.provider_label = select_providers(prefer_gpu)
        if on_progress:
            on_progress(f"Runtime: {self.provider_label}")

        try:
            self._load_sessions()
        except Exception as exc:
            if self.providers != ["CPUExecutionProvider"]:
                log.warning("GPU provider failed (%s); retrying on CPU", exc)
                self.providers, self.provider_label = ["CPUExecutionProvider"], "CPU"
                if on_progress:
                    on_progress("Runtime: CPU")
                self._load_sessions()
            else:
                raise ModelError(
                    f"ONNX Runtime could not load the model.\n\nReason: {exc}"
                ) from exc

        try:
            self.text_processor = UnicodeProcessor(
                os.path.join(self.onnx_dir, "unicode_indexer.json")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise ModelError(f"unicode_indexer.json is missing or corrupt: {exc}") from exc

        log.info(
            "Supertonic ready: %d Hz, provider=%s, models=%s",
            self.sample_rate,
            self.provider_label,
            ", ".join(ONNX_FILES.values()),
        )

    def _load_sessions(self) -> None:
        self.dp_ort = _session(
            os.path.join(self.onnx_dir, ONNX_FILES["dp"]), self.providers
        )
        self.text_enc_ort = _session(
            os.path.join(self.onnx_dir, ONNX_FILES["text_enc"]), self.providers
        )
        self.vector_est_ort = _session(
            os.path.join(self.onnx_dir, ONNX_FILES["vector_est"]), self.providers
        )
        self.vocoder_ort = _session(
            os.path.join(self.onnx_dir, ONNX_FILES["vocoder"]), self.providers
        )

    # -- latent sampling ---------------------------------------------------- #
    def _sample_noisy_latent(
        self, duration: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        batch = len(duration)
        wav_len_max = duration.max() * self.sample_rate
        wav_lengths = (duration * self.sample_rate).astype(np.int64)
        chunk_size = self.base_chunk_size * self.chunk_compress_factor
        latent_len = int(((wav_len_max + chunk_size - 1) / chunk_size))
        latent_dim = self.ldim * self.chunk_compress_factor
        noisy = rng.standard_normal((batch, latent_dim, latent_len)).astype(np.float32)
        latent_mask = get_latent_mask(
            wav_lengths, self.base_chunk_size, self.chunk_compress_factor
        )
        return noisy * latent_mask, latent_mask

    # -- inference ---------------------------------------------------------- #
    def infer(
        self,
        text: str,
        lang: str,
        style: Style,
        total_step: int = 8,
        speed: float = 1.0,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, float]:
        """Synthesise one chunk. Returns (mono float32 samples, duration sec)."""
        rng = rng or np.random.default_rng()
        text_ids, text_mask = self.text_processor([text], [lang])

        dur_onnx, *_ = self.dp_ort.run(
            None, {"text_ids": text_ids, "style_dp": style.dp, "text_mask": text_mask}
        )
        dur_onnx = dur_onnx / float(speed)
        duration = float(np.asarray(dur_onnx).reshape(-1)[0])
        if not np.isfinite(duration) or duration <= 0:
            raise ModelError("The model predicted an invalid audio length.")

        text_emb, *_ = self.text_enc_ort.run(
            None,
            {"text_ids": text_ids, "style_ttl": style.ttl, "text_mask": text_mask},
        )
        latent, latent_mask = self._sample_noisy_latent(dur_onnx, rng)
        total_step_np = np.array([total_step], dtype=np.float32)
        for step in range(total_step):
            latent, *_ = self.vector_est_ort.run(
                None,
                {
                    "noisy_latent": latent,
                    "text_emb": text_emb,
                    "style_ttl": style.ttl,
                    "text_mask": text_mask,
                    "latent_mask": latent_mask,
                    "current_step": np.array([step], dtype=np.float32),
                    "total_step": total_step_np,
                },
            )
        wav, *_ = self.vocoder_ort.run(None, {"latent": latent})

        wav = np.asarray(wav, dtype=np.float32).reshape(-1)
        wanted = min(wav.size, max(1, int(round(duration * self.sample_rate))))
        return wav[:wanted], wav.size / float(self.sample_rate)

    def max_chars_for(self, lang: str) -> int:
        return LANG_MAX_CHARS.get(lang, 300)