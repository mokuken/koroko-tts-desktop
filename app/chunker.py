"""Long-text handling: validation, counting and narration-aware chunking.

Supertonic's reference implementation accepts roughly 300 characters per
forward pass. Blindly cutting every N characters produces clipped words and
robotic narration, so we split on real structure instead:

    paragraph -> line -> sentence -> (comma / clause) -> word

Every chunk records how it was separated from the next one so the audio join
can insert a natural pause (comma < sentence < line break < paragraph break).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: Maximum characters per synthesis pass (Supertonic reference limit).
MAX_CHARS = 300

#: Preferred floor so we do not emit a chunk for every tiny sentence.
MIN_CHARS = 90

#: Pause inserted after a chunk, keyed by the break that followed it.
PAUSES = {
    "sentence": 0.14,
    "clause": 0.22,
    "line": 0.28,
    "paragraph": 0.46,
    "end": 0.0,
}

# Words that end in a period without ending a sentence.
_ABBREVIATIONS = frozenset(
    """
    mr mrs ms dr prof sr jr st mt ft rev gen col sgt capt lt hon
    inc ltd co corp llc plc dept est fig no vol nos pp ed eds
    ave blvd rd ln hwy apt ste
    jan feb mar apr jun jul aug sep sept oct nov dec
    mon tue tues wed thu thur thurs fri sat sun
    vs etc al approx dept univ
    eg ie cf ibid viz ca approx est
    am pm a m p m
    usa us uk eu un nato
    e g i e
    u s u k
    """.split()
)

_ZERO_WIDTH = re.compile(r"[‌‍﻿]")
_NON_BREAKING = {
    " ": " ", " ": " ", " ": " ", " ": " ",
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "…": "...",
}
_TRANS = str.maketrans(_NON_BREAKING)
_SENTENCE_END = re.compile(r"[.!?]+[\"'’”)\]]*")
_TOKEN_BEFORE = re.compile(r"([A-Za-z][A-Za-z0-9]*|\d+)$")


@dataclass(frozen=True)
class Chunk:
    """One synthesis unit plus the pause that should follow it."""

    text: str
    pause_after: float

    def __len__(self) -> int:  # pragma: no cover - convenience
        return len(self.text)


# --------------------------------------------------------------------------- #
# validation / counting
# --------------------------------------------------------------------------- #
def clean_text(text: str) -> str:
    """Normalise pasted text without destroying paragraph structure."""
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _ZERO_WIDTH.sub("", text)
    text = text.translate(_TRANS)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_blank(text: str) -> bool:
    return not re.sub(r"[\s\W_]+", "", text or "", flags=re.UNICODE)


def word_count(text: str) -> int:
    return len((text or "").split())


def char_count(text: str) -> int:
    return len(text or "")


#: Measured Supertonic pace on a 10-core desktop CPU, averaged over short
#: sentences (131-145 wpm) and multi-sentence passages (~180 wpm).
WORDS_PER_MINUTE = 165.0


def estimate_duration_seconds(text: str, speed: float = 1.0) -> float:
    """Approximate narration length in seconds (preview only, never exact)."""
    speed = max(0.25, float(speed))
    return (word_count(text) / WORDS_PER_MINUTE) * 60.0 / speed


# --------------------------------------------------------------------------- #
# sentence splitting
# --------------------------------------------------------------------------- #
def _ends_sentence_before(text: str, dot_index: int) -> bool:
    """False when the '.' at ``dot_index`` belongs to an abbreviation/initial."""
    token_match = _TOKEN_BEFORE.search(text[:dot_index])
    if not token_match:
        return True
    token = token_match.group(1)
    if token.isdigit():  # 3.14, 1.5, 2024.
        return False
    if len(token) == 1 and token.isupper():  # "J." in "J. R. R. Tolkien"
        return False
    return token.lower() not in _ABBREVIATIONS


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    """Character spans of each sentence inside one line of text."""
    spans: list[tuple[int, int]] = []
    start = 0
    for match in _SENTENCE_END.finditer(text):
        end = match.end()
        nxt = end
        while nxt < len(text) and text[nxt].isspace():
            nxt += 1
        if nxt == end:
            continue  # no whitespace after the punctuation -> not a boundary
        if not _ends_sentence_before(text, match.start()):
            continue
        spans.append((start, end))
        start = nxt
    if start < len(text):
        spans.append((start, len(text)))
    return spans


def _break_kind(text: str, index: int) -> str:
    """Classify the whitespace gap that follows ``text[index]``."""
    tail = text[index:]
    stripped = tail.lstrip(" \t")
    if not stripped.startswith("\n"):
        return "sentence"
    after_newlines = stripped.lstrip("\n")
    if after_newlines.startswith(("\n", "")):
        return "paragraph"
    return "line"


@dataclass
class _Item:
    text: str
    break_before: str
    break_after: str


def _split_items(text: str) -> list[_Item]:
    items: list[_Item] = []
    blocks = [p.strip() for p in re.split(r"\n[ \t]*\n+", text)]
    blocks = [p for p in blocks if p]
    for pi, paragraph in enumerate(blocks):
        lines = [line.strip() for line in paragraph.split("\n") if line.strip()]
        last_para = pi == len(blocks) - 1
        for li, line in enumerate(lines):
            spans = _sentence_spans(line)
            last_line = li == len(lines) - 1
            for si, (start, end) in enumerate(spans):
                sentence = line[start:end].strip()
                if not sentence:
                    continue
                if not items:
                    before = "start"
                elif si == 0:
                    # First sentence of this line: the break that introduced it.
                    before = "paragraph" if li == 0 else "line"
                else:
                    before = "sentence"
                if si < len(spans) - 1:
                    after = "sentence"
                elif not last_line:
                    after = "line"
                elif not last_para:
                    after = "paragraph"
                else:
                    after = "end"
                items.append(_Item(sentence, before, after))
    return items


# --------------------------------------------------------------------------- #
# oversized sentences
# --------------------------------------------------------------------------- #
_LEAD_JUNK = re.compile(r'^[\s,;:.!?)\]}"“”‘’]+')


def _best_cut(window: str, min_chars: int) -> int:
    """Earliest pleasant break at/after ``min_chars`` inside ``window``."""
    for pattern in (r"[,;:]\s", r"[)\]”\"']\s", r"\s"):
        match = re.search(pattern, window[min_chars:])
        if match:
            cut = min_chars + match.end()
            while cut > 0 and window[cut - 1] in "”’\"')]":
                cut -= 1
            return max(1, cut)
    return len(window)


def _hard_split(sentence: str, max_chars: int, min_chars: int = 40) -> list[str]:
    pieces: list[str] = []
    rest = sentence.strip()
    guard = 0
    while len(rest) > max_chars and guard < 10_000:
        guard += 1
        cut = _best_cut(rest[:max_chars], min_chars)
        head = rest[:cut].strip().rstrip(",;:- ")
        tail = _LEAD_JUNK.sub("", rest[cut:])
        if not head:
            head, tail = rest[:cut].strip(), rest[cut:].strip()
        pieces.append(head)
        if not tail or tail == rest:
            break
        rest = tail
    if rest:
        pieces.append(rest)
    return [p for p in pieces if p]


def _pause_for(kind: str, sentence: str) -> float:
    if kind == "sentence" and sentence.endswith((",", ";", ":", "...", "-")):
        return PAUSES["clause"]
    return PAUSES.get(kind, 0.0)


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def chunk_text(
    text: str,
    max_chars: int = MAX_CHARS,
    min_chars: int = MIN_CHARS,
) -> list[Chunk]:
    """Split narration text into chunks the model can synthesise.

    Splitting happens at the strongest available boundary. A blank line always
    starts a new chunk so paragraph pacing survives; a single line break starts
    a new chunk once the current chunk is already substantial, which keeps
    hard-wrapped scripts from turning into one pause per line.

    Returns a list of :class:`Chunk`. Empty/whitespace-only input yields an
    empty list rather than raising, so callers can validate first if they wish.
    """
    if max_chars < 40:
        max_chars = 40
    cleaned = clean_text(text)
    if not cleaned:
        return []

    units: list[tuple[str, str, str]] = []
    for item in _split_items(cleaned):
        pieces = _hard_split(item.text, max_chars)
        for i, piece in enumerate(pieces):
            is_last = i == len(pieces) - 1
            units.append(
                (
                    piece,
                    item.break_before if i == 0 else "sentence",
                    item.break_after if is_last else "sentence",
                )
            )

    chunks: list[Chunk] = []
    buffer: list[str] = []
    length = 0
    previous_break = "sentence"

    for piece, break_before, break_after in units:
        addition = len(piece) + (1 if buffer else 0)
        if buffer:
            too_long = length + addition > max_chars
            structural = break_before == "paragraph" or (
                break_before == "line" and length >= min_chars
            )
            if too_long or structural:
                chunks.append(
                    Chunk(" ".join(buffer), _pause_for(previous_break, buffer[-1]))
                )
                buffer, length = [], 0
                addition = len(piece)
        buffer.append(piece)
        length += addition
        previous_break = break_after

    if buffer:
        chunks.append(Chunk(" ".join(buffer), _pause_for(previous_break, buffer[-1])))
    return chunks


def chunk_summary(chunks: list[Chunk]) -> str:
    if not chunks:
        return "0 chunks"
    if len(chunks) == 1:
        return "1 chunk"
    return f"{len(chunks)} chunks"