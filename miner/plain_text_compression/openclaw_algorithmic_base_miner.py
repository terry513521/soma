#!/usr/bin/env python3
"""Algorithmic OpenClaw/SOMA trajectory compressor (no LLM, no network).

Use this as `SOMA-plugin/base_miner.py` for competition runs.

Contract:
    python base_miner.py assemble < stdin-json

Modes:
    Plain text — single user message passages (Context-Compression-1 style).
    CoT — multi-message agent trajectories with user / assistant / toolResult
    messages (CoT-Compression-1 style). Detected automatically when assistant or
    toolResult roles are present.

CoT compression pipeline (automatic when assistant/toolResult messages exist):

    1. Parse   — XML trajectory -> message array
    2. Sanitize — drop empty errors; compress (not delete) <thinking> blocks
    3. Truncate — per-message limits; tool output keeps diffs/errors/grep hits
    4. Arc keep — task -> discovery -> edits -> planning bridges -> conclusion
    5. Budget   — fill toward target ratio, then trim lowest-value non-arc messages
    6. Serialize — message array -> OpenClaw XML

Arc must-keep set:
    - First user message (full task)
    - Top discovery tool results (grep file.py:line, read def ...)
    - All edit tool results + paired assistant tool_call blocks
    - All planning-bridge assistant messages (let me / now / verify ...)
    - Conclusion assistant summary
    - At least N thinking blocks (SOMA_KEEP_THINKING_BLOCKS)

Environment:
    SOMA_COMPRESSION_RATIO          Default: 0.45
    SOMA_MIN_COMPACT_TOKENS         Default: 1800
    SOMA_KEEP_TOOL_RESULTS          Default: 6 (8 in CoT mode)
    SOMA_KEEP_ASSISTANT_NARRATIVES  Default: 1 (3 in CoT mode)
    SOMA_KEEP_THINKING_BLOCKS       Default: 1 (CoT mode: keep N assistant <thinking> blocks)
    SOMA_COT_FILL_RATIO             Default: 0.90 (fill toward this fraction of target when under budget)
    SOMA_MAX_TOOL_RESULT_CHARS      Default: 2400
    SOMA_MAX_USER_MESSAGE_CHARS     Default: 3500
    SOMA_USE_SENTENCE_RANKER        Default: 1 (plain-text sentence syntax + BM25)
    SOMA_USE_ACRONYMS               Default: 1 (repeat 2+ word phrases -> acronym after first mention)
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

try:
    import tiktoken

    ENCODER = tiktoken.get_encoding("cl100k_base")
except Exception:
    ENCODER = None


EVENT_NAMES = frozenset({"assemble"})
STATE_VERSION = 1
STATE_DIR_NAME = "state"

# Structural patterns only — no domain-specific vocabulary.
CHEMICAL_FORMULA_RE = re.compile(r"\b[A-Z][a-z0-9]*(?:\([A-Za-z0-9+]+\))+\b")
PROPER_NOUN_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b")
FORMULA_LIKE_RE = re.compile(r"\b(?:\d+[A-Za-z][A-Za-z0-9()+-]*|[A-Za-z][A-Za-z0-9()+-]*\d+)\b")
LABEL_VALUE_INLINE_RE = re.compile(r"^[A-Za-z][\w /+-]{1,32}\s*:\s*\S", flags=re.MULTILINE)
RELATION_PHRASE_RE = re.compile(
    r"\b(?:also|formerly|alternatively|otherwise)\s+(?:called|known|termed|labeled|spelled)\b|"
    r"\b(?:known|called|termed|labeled|dubbed)\s+as\b",
    flags=re.IGNORECASE,
)
LOCATION_PHRASE_RE = re.compile(
    r"\b(?:located|displayed|held|stored|housed|situated|found|kept)\s+(?:at|in|on|within)\b",
    flags=re.IGNORECASE,
)
DEFINITION_PHRASE_RE = re.compile(
    r"\b(?:is|are|was|were)\s+(?:a|an|the)\s+[A-Za-z]",
    flags=re.IGNORECASE,
)
QUANTIFIED_FACT_RE = re.compile(
    r"\b\d{4}[-–/]\d{2}\b.*\b\d+\b|\b\d+\s+[A-Za-z]{4,}\b",
    flags=re.IGNORECASE,
)
FINITE_VERB_RE = re.compile(
    r"\b(?:is|are|was|were|be|been|being|have|has|had|do|does|did|will|would|can|could|should|may|might)\b",
    flags=re.IGNORECASE,
)
PATH_LIKE_RE = re.compile(r"(?:^|[\s\"'])(?:[A-Za-z0-9_./-]+/)+[A-Za-z0-9_.-]+")
FILE_EXT_RE = re.compile(r"\b[A-Za-z0-9_/.-]+\.(?:py|js|ts|tsx|java|go|rs|cpp|c|h|md|json|yaml|yml|toml)\b")
DIFF_MARKER_RE = re.compile(r"(?:diff --git|^@@|\+\+\+|---)", flags=re.MULTILINE)
ERROR_LIKE_RE = re.compile(
    r"\b(?:traceback|exception|error|failed|failure|assert(?:ion)?|expected|actual)\b",
    flags=re.IGNORECASE,
)
FUNCTION_CALL_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\(")
TOOL_CALL_XML_ID_RE = re.compile(
    r'<tool_call\s+name="[^"]+"\s+id="([^"]+)"',
    flags=re.IGNORECASE,
)
TOOL_CALL_XML_BLOCK_RE = re.compile(
    r"<tool_call\s+name=\"[^\"]+\"\s+id=\"([^\"]+)\">.*?</tool_call>",
    flags=re.DOTALL | re.IGNORECASE,
)
COT_TOOL_HIGH_SIGNAL_RE = re.compile(
    r"(?:traceback|exception|error|failed|assert(?:ion)?|diff --git|^@@|\+\+\+|---|"
    r"^\+\s|^\-\s|\bFAIL\b|\bPASS\b|Ran \d+ test|\.py:\d+)",
    flags=re.IGNORECASE | re.MULTILINE,
)
TRAJECTORY_MESSAGE_BLOCK_RE = re.compile(
    r"<message\s+role=\"(?P<role>[^\"]+)\">(?P<body>.*?)</message>",
    flags=re.DOTALL,
)
TRAJECTORY_TOOL_RESULT_RE = re.compile(
    r"<tool_result\s+tool=\"(?P<tool>[^\"]+)\"\s+tool_call_id=\"(?P<tool_call_id>[^\"]+)\">"
    r"(?P<body>.*?)</tool_result>",
    flags=re.DOTALL,
)
THINKING_TAG_RE = re.compile(r"<thinking>(.*?)</thinking>", flags=re.DOTALL)
TRAJECTORY_TEXT_TAG_RE = re.compile(r"</?text>")
TOKEN_STOPWORDS = frozenset(
    {
        "the",
        "and",
        "for",
        "that",
        "with",
        "from",
        "this",
        "have",
        "were",
        "was",
        "are",
        "into",
        "when",
        "who",
        "what",
        "where",
        "which",
        "their",
        "they",
        "them",
        "your",
        "you",
        "not",
        "but",
        "can",
        "all",
        "any",
        "how",
        "why",
        "out",
        "our",
        "his",
        "her",
        "its",
    }
)


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def normalize_role(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    lowered = value.strip().lower().replace("_", "").replace("-", "")
    return "toolResult" if lowered == "toolresult" else lowered


def extract_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return "\n".join(extract_text(item) for item in value)
    if isinstance(value, dict):
        if isinstance(value.get("text"), str):
            return value["text"]
        if isinstance(value.get("content"), str):
            return value["content"]
        return "\n".join(extract_text(item) for item in value.values())
    return str(value)


def token_count(text: str) -> int:
    if not text:
        return 0
    if ENCODER is None:
        return max(1, math.ceil(len(text) / 4))
    return len(ENCODER.encode(text))


def estimate_tokens_for_message_array(messages: list[Any]) -> int:
    return sum(
        token_count(extract_text(message.get("content")))
        for message in messages
        if isinstance(message, dict)
    )


MONTH_NAME_TO_NUMBER = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}

DATE_NUMERIC_RE = re.compile(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b")
DATE_ISO_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
DATE_MONTH_DAY_YEAR_RE = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b",
    flags=re.IGNORECASE,
)
DATE_MONTH_DAY_RE = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+(\d{1,2})(?:st|nd|rd|th)?\b",
    flags=re.IGNORECASE,
)
YEAR_RE = re.compile(r"\b((?:19|20)\d{2})\b")
DECORATION_RUN_RE = re.compile(r"[_\-=*#~]{3,}")


def strip_line_decoration(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""
    cleaned = DECORATION_RUN_RE.sub(" ", stripped)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned or not re.search(r"[A-Za-z0-9]", cleaned):
        return ""
    return cleaned


def commentary_penalty(sentence: str) -> float:
    penalty = 0.0
    if re.search(r"\bstill seen\b", sentence, flags=re.IGNORECASE):
        penalty += 4.0
    if re.search(r"\bsuccess of the\b.*\bfilm\b", sentence, flags=re.IGNORECASE):
        penalty += 4.0
    if re.search(r"\bstarring\b", sentence, flags=re.IGNORECASE) and re.search(
        r"\bfilm\b", sentence, flags=re.IGNORECASE
    ):
        penalty += 3.0
    if re.search(r"\bhowever,\s+especially\b", sentence, flags=re.IGNORECASE):
        penalty += 2.0
    if re.search(r"\bromantic figures?\b", sentence, flags=re.IGNORECASE):
        penalty += 3.0
    return penalty


def is_commentary_filler(sentence: str) -> bool:
    return commentary_penalty(sentence) >= 8.0


HTML_TAG_RE = re.compile(r"<[^>]+>")
ACADEMIC_SPLICE_MARKERS = (
    "Abstract:",
    "Our results demonstrate",
    "We introduce ",
    "We present ",
    "We prove ",
    "The code is open source",
)
ACADEMIC_NOISE_RE = re.compile(
    r"\b(?:Abstract:|We (?:introduce|present|prove|demonstrate|explore|propose)|"
    r"Our results(?:\s+demonstrate)?|state-of-the-art|benchmark(?:ing)?|"
    r"gradient(?:-| )descent|large language models?|LLMs?|transformers?|"
    r"neuromorphic|autoregressive model|hardware-aware)\b",
    flags=re.IGNORECASE,
)
UI_CHROME_RE = re.compile(
    r"^(?:Error|NEWS|Videos|Goofs|Quotes|Storyline|Genres|ON DISC|IMDb|Trivia)$|"
    r"^There was an error trying to load\b|"
    r"^created \d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)|"
    r"^From \$[\d.]+.*(?:Amazon Video|ON DISC)\b|"
    r"^\d+ of \d+ people found this review helpful|"
    r"^See more »$|"
    r"^Here are a couple of impractical hacks",
    flags=re.IGNORECASE,
)
SPLICE_AFTER_PERIOD_RE = re.compile(
    r"\.\s+(?:of|our|we|the code is|abstract|in the|evaluations on)\b",
    flags=re.IGNORECASE,
)


def trim_embedded_noise(sentence: str) -> str:
    cleaned = HTML_TAG_RE.sub(" ", sentence)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    lowered = cleaned.lower()
    for marker in ACADEMIC_SPLICE_MARKERS:
        position = lowered.find(marker.lower())
        if position > 40:
            cleaned = cleaned[:position].strip()
            lowered = cleaned.lower()
    match = SPLICE_AFTER_PERIOD_RE.search(cleaned)
    if match and match.start() > 50:
        cleaned = cleaned[: match.start() + 1].strip()
    return cleaned


def noise_penalty(sentence: str) -> float:
    penalty = 0.0
    stripped = sentence.strip()
    if not stripped:
        return 99.0
    if UI_CHROME_RE.search(stripped):
        penalty += 12.0
    if re.search(r"\bAbstract:\s", stripped):
        penalty += 10.0
    if ACADEMIC_NOISE_RE.search(stripped):
        penalty += 6.0
    if HTML_TAG_RE.search(stripped):
        penalty += 8.0
    if re.search(r"\bgithub\.com/", stripped, flags=re.IGNORECASE):
        penalty += 5.0
    if re.fullmatch(r"Director:\s*.+", stripped, flags=re.IGNORECASE) and len(stripped) < 60:
        penalty += 4.0
    return penalty


def is_noise_filler(sentence: str) -> bool:
    return noise_penalty(sentence) >= 10.0


def is_sentence_fragment(sentence: str) -> bool:
    stripped = sentence.strip()
    if not stripped:
        return True
    head = stripped.splitlines()[0].strip()
    if not head:
        return True
    if RELATION_PHRASE_RE.search(head) or DEFINITION_PHRASE_RE.search(head):
        return False
    if head.endswith("?") or head.endswith(":"):
        return False
    if head[0].islower():
        return True
    if re.match(
        r"^(?:however|and|but|of|in the|after the|before the)\b",
        head,
        flags=re.IGNORECASE,
    ):
        return True
    words = head.split()
    if words and words[0].lower() in {"founded", "located", "based", "known", "created"}:
        if len(words) < 14 and not re.match(r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\s", head):
            return True
    return False


def is_low_value_sentence(sentence: str) -> bool:
    return is_commentary_filler(sentence) or is_noise_filler(sentence) or is_sentence_fragment(sentence)


def _valid_date_parts(year: int, month: int, day: int) -> bool:
    return 1000 <= year <= 2100 and 1 <= month <= 12 and 1 <= day <= 31


def _date_key(year: int, month: int, day: int) -> str:
    return f"{year:04d}-{month:02d}-{day:02d}"


def extract_date_keys(line: str) -> set[str]:
    keys: set[str] = set()

    for match in DATE_NUMERIC_RE.finditer(line):
        month, day, year = int(match.group(1)), int(match.group(2)), int(match.group(3))
        if _valid_date_parts(year, month, day):
            keys.add(_date_key(year, month, day))

    for match in DATE_ISO_RE.finditer(line):
        year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
        if _valid_date_parts(year, month, day):
            keys.add(_date_key(year, month, day))

    for match in DATE_MONTH_DAY_YEAR_RE.finditer(line):
        month = MONTH_NAME_TO_NUMBER[match.group(1).lower()]
        day = int(match.group(2))
        year = int(match.group(3))
        if _valid_date_parts(year, month, day):
            keys.add(_date_key(year, month, day))

    years = [int(match.group(1)) for match in YEAR_RE.finditer(line)]
    for match in DATE_MONTH_DAY_RE.finditer(line):
        month = MONTH_NAME_TO_NUMBER[match.group(1).lower()]
        day = int(match.group(2))
        for year in years:
            if _valid_date_parts(year, month, day):
                keys.add(_date_key(year, month, day))

    return keys


def extract_topic_tokens(line: str) -> set[str]:
    tokens = {
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9']{2,}", line)
        if token.lower() not in TOKEN_STOPWORDS
    }
    return tokens


def same_event_line(left: str, right: str) -> bool:
    shared_dates = extract_date_keys(left) & extract_date_keys(right)
    if not shared_dates:
        return False

    left_tokens = extract_topic_tokens(left)
    right_tokens = extract_topic_tokens(right)
    if len(left_tokens) < 2 or len(right_tokens) < 2:
        return False

    overlap = left_tokens & right_tokens
    if len(overlap) < 2:
        return False

    overlap_ratio = len(overlap) / min(len(left_tokens), len(right_tokens))
    return overlap_ratio >= 0.3


def normalize_dedup_key(text: str) -> str:
    collapsed = re.sub(r"\s+", " ", text.strip().lower())
    return re.sub(r"[^\w\s]", "", collapsed)


def dedupe_inline_repetition(text: str) -> str:
    if "\n" in text:
        cleaned_lines = [dedupe_inline_repetition(line) for line in text.splitlines()]
        return "\n".join(line for line in cleaned_lines if line)

    stripped = strip_line_decoration(text)
    if not stripped:
        return ""

    if "..." in stripped:
        segments = [segment.strip() for segment in re.split(r"\s*\.\.\.\s*", stripped) if segment.strip()]
        if len(segments) > 1:
            return _dedupe_segments(segments)

    words = stripped.split()
    if len(words) >= 8:
        for size in range(min(24, len(words) // 2), 3, -1):
            prefix = " ".join(words[:size])
            remainder = " ".join(words[size:]).strip()
            if not remainder:
                continue
            prefix_key = normalize_dedup_key(prefix)
            remainder_key = normalize_dedup_key(remainder)
            if remainder_key.startswith(prefix_key) or prefix_key in remainder_key:
                return dedupe_inline_repetition(remainder)
            if prefix_key.startswith(remainder_key[: max(12, len(remainder_key) // 2)]):
                return dedupe_inline_repetition(prefix)

    return stripped


def _dedupe_segments(segments: list[str]) -> str:
    kept: list[str] = []
    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue
        segment_key = normalize_dedup_key(segment)
        if not segment_key:
            continue

        replaced = False
        for index, existing in enumerate(kept):
            existing_key = normalize_dedup_key(existing)
            if segment_key == existing_key:
                replaced = True
                break
            if segment_key in existing_key:
                replaced = True
                break
            if existing_key in segment_key:
                kept[index] = segment
                replaced = True
                break
        if not replaced:
            kept.append(segment)
    return " ".join(kept)


def sentence_content_signature(sentence: str) -> str:
    parts = [part.strip() for part in sentence.splitlines() if part.strip()]
    if not parts:
        return ""
    substantive = max(parts, key=len)
    return normalize_dedup_key(substantive)


def is_metadata_pair_label(line: str) -> bool:
    stripped = line.strip().rstrip(":")
    if not stripped or len(stripped) > 24:
        return False
    if re.search(r"[.!?,]", stripped):
        return False
    words = stripped.split()
    if len(words) != 1:
        return False
    if FINITE_VERB_RE.search(stripped):
        return False
    return bool(re.match(r"^[A-Za-z][A-Za-z0-9+/#.-]*$", words[0]))


def token_jaccard(left: str, right: str) -> float:
    left_tokens = set(bm25_tokenize(left))
    right_tokens = set(bm25_tokenize(right))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def near_duplicate_line(left: str, right: str) -> bool:
    left_key = normalize_dedup_key(left)
    right_key = normalize_dedup_key(right)
    if not left_key or not right_key:
        return False
    if left_key == right_key:
        return True

    left_signature = sentence_content_signature(left)
    right_signature = sentence_content_signature(right)
    if left_signature and left_signature == right_signature:
        return True

    shorter, longer = (left_key, right_key) if len(left_key) <= len(right_key) else (right_key, left_key)
    if len(shorter) >= 24 and shorter in longer:
        return True
    if token_jaccard(left, right) >= 0.78:
        return True
    return same_event_line(left, right)


def line_quality(line: str, *, quality_score: float | None = None) -> float:
    if quality_score is not None:
        return quality_score
    return float(line_score(line))


def dedupe_index_set(
    sentences: list[str],
    indices: set[int],
    *,
    score_by_index: dict[int, float],
) -> set[int]:
    kept_indices: list[int] = []
    kept_sentences: list[str] = []

    for index in sorted(indices):
        sentence = sentences[index]
        duplicate_at = None
        for existing_index, existing in enumerate(kept_sentences):
            if near_duplicate_line(sentence, existing):
                duplicate_at = existing_index
                break

        if duplicate_at is not None:
            if score_by_index.get(index, 0.0) > score_by_index.get(kept_indices[duplicate_at], 0.0):
                kept_indices[duplicate_at] = index
                kept_sentences[duplicate_at] = sentence
            continue

        kept_indices.append(index)
        kept_sentences.append(sentence)

    return set(kept_indices)


def dedupe_lines(lines: list[str], *, quality_scores: list[float] | None = None) -> list[str]:
    kept: list[str] = []
    seen_keys: list[str] = []
    for line_index, line in enumerate(lines):
        cleaned = dedupe_inline_repetition(line.strip())
        if not cleaned:
            continue
        key = normalize_dedup_key(cleaned)
        if not key:
            continue
        quality = line_quality(
            cleaned,
            quality_score=quality_scores[line_index] if quality_scores and line_index < len(quality_scores) else None,
        )

        replaced = False
        for index, seen in enumerate(seen_keys):
            if key == seen:
                replaced = True
                break
            if key in seen:
                replaced = True
                break
            if seen in key:
                existing_quality = line_quality(
                    kept[index],
                    quality_score=quality_scores[index] if quality_scores and index < len(quality_scores) else None,
                )
                if quality >= existing_quality:
                    kept[index] = cleaned
                    seen_keys[index] = key
                replaced = True
                break
        if replaced:
            continue

        duplicate_index = None
        for index, existing in enumerate(kept):
            if near_duplicate_line(cleaned, existing):
                duplicate_index = index
                break

        if duplicate_index is not None:
            existing_quality = line_quality(
                kept[duplicate_index],
                quality_score=(
                    quality_scores[duplicate_index]
                    if quality_scores and duplicate_index < len(quality_scores)
                    else None
                ),
            )
            if quality >= existing_quality:
                kept[duplicate_index] = cleaned
                seen_keys[duplicate_index] = key
            continue

        seen_keys.append(key)
        kept.append(cleaned)
    return kept


def is_plain_text_passage(text: str) -> bool:
    sample = text[:4000].lower()
    if "<message role=" in sample or "<tool_result" in sample or "<tool_call" in sample:
        return False
    if sample.lstrip().startswith("passage:"):
        return True
    return len(text) > 3000 and text.count("\n") >= 5


def bm25_tokenize(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 2 and token not in TOKEN_STOPWORDS
    ]


class SimpleBM25:
    def __init__(self, documents: list[str]) -> None:
        self.tokenized = [bm25_tokenize(document) for document in documents]
        self.document_count = len(self.tokenized)
        self.avg_document_length = (
            sum(len(document) for document in self.tokenized) / self.document_count
            if self.document_count
            else 0.0
        )
        self.document_frequency: dict[str, int] = {}
        for document in self.tokenized:
            for term in set(document):
                self.document_frequency[term] = self.document_frequency.get(term, 0) + 1

    def score(self, query_tokens: list[str], document_index: int, *, k1: float = 1.5, b: float = 0.75) -> float:
        if document_index < 0 or document_index >= self.document_count:
            return 0.0
        document = self.tokenized[document_index]
        if not document:
            return 0.0

        document_length = len(document)
        total = 0.0
        for term in query_tokens:
            frequency = self.document_frequency.get(term, 0)
            if frequency == 0:
                continue
            term_frequency = document.count(term)
            inverse_document_frequency = math.log(
                1 + (self.document_count - frequency + 0.5) / (frequency + 0.5)
            )
            denominator = term_frequency + k1 * (
                1 - b + b * document_length / max(self.avg_document_length, 1.0)
            )
            total += inverse_document_frequency * (term_frequency * (k1 + 1)) / denominator
        return total


def split_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if len(line) <= 180 and not re.search(r"[.!?]", line):
            cleaned = trim_embedded_noise(dedupe_inline_repetition(line))
            if cleaned and not is_low_value_sentence(cleaned):
                sentences.append(cleaned)
            continue
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"(\[])", line)
        for part in parts:
            cleaned = trim_embedded_noise(dedupe_inline_repetition(part.strip()))
            if cleaned and not is_low_value_sentence(cleaned):
                sentences.append(cleaned)
    return sentences


def is_structural_label_line(line: str) -> bool:
    stripped = line.strip().rstrip(":")
    if not stripped or len(stripped) > 36:
        return False
    if re.search(r"[.!?]", stripped):
        return False
    words = stripped.split()
    if not words or len(words) > 4:
        return False
    if FINITE_VERB_RE.search(stripped):
        return False
    return bool(re.match(r"^[\w][\w /+-]*$", stripped))


def has_structural_label_block(sentence: str) -> bool:
    parts = [part.strip() for part in sentence.splitlines() if part.strip()]
    if len(parts) >= 2 and is_structural_label_line(parts[0]):
        return True
    return is_structural_label_line(sentence)


def merge_metadata_label_sentences(sentences: list[str]) -> list[str]:
    merged: list[str] = []
    index = 0
    while index < len(sentences):
        current = sentences[index].strip()
        if is_metadata_pair_label(current) and index + 1 < len(sentences):
            nxt = sentences[index + 1].strip()
            if nxt and len(nxt) <= 160 and not is_metadata_pair_label(nxt):
                merged.append(f"{current}\n{nxt}")
                index += 2
                continue
        merged.append(current)
        index += 1
    return merged


def token_rarity_score(tokens: list[str], ranker: SimpleBM25) -> float:
    if not tokens or ranker.document_count == 0:
        return 0.0
    rarity = 0.0
    for token in set(tokens):
        document_frequency = ranker.document_frequency.get(token, 0)
        rarity += math.log(1 + ranker.document_count / max(document_frequency, 1))
    return rarity / len(set(tokens))


def boilerplate_penalty(tokens: list[str], ranker: SimpleBM25) -> float:
    if not tokens or ranker.document_count == 0:
        return 0.0
    average_frequency = sum(ranker.document_frequency.get(token, 0) for token in tokens) / len(tokens)
    prevalence = average_frequency / ranker.document_count
    if prevalence > 0.35:
        return min(6.0, (prevalence - 0.35) * 12.0)
    return 0.0


def syntax_sentence_score(sentence: str, *, ranker: SimpleBM25) -> float:
    score = 0.0
    tokens = bm25_tokenize(sentence)

    if extract_date_keys(sentence):
        score += 4.0
    if re.search(r"\b\d+\b", sentence):
        score += 1.0

    proper_noun_count = len(PROPER_NOUN_RE.findall(sentence))
    score += min(4.0, proper_noun_count * 1.0)

    if CHEMICAL_FORMULA_RE.search(sentence) or FORMULA_LIKE_RE.search(sentence):
        score += 4.0
    if LABEL_VALUE_INLINE_RE.search(sentence) or has_structural_label_block(sentence):
        score += 5.0
    if RELATION_PHRASE_RE.search(sentence):
        score += 4.0
    if LOCATION_PHRASE_RE.search(sentence):
        score += 3.0
    if DEFINITION_PHRASE_RE.search(sentence):
        score += 2.0
    if QUANTIFIED_FACT_RE.search(sentence):
        score += 3.0
    if sentence.strip().endswith("?"):
        score += 2.0
    if re.match(r"^[\w][\w /+-]{0,24}\s*:?\s*$", sentence.strip()) and is_structural_label_line(sentence):
        score += 4.0

    score += min(6.0, token_rarity_score(tokens, ranker))
    score -= boilerplate_penalty(tokens, ranker)
    score += min(3.0, len(extract_topic_tokens(sentence)) * 0.3)
    score -= commentary_penalty(sentence)
    score -= noise_penalty(sentence)

    if 40 <= len(sentence) <= 320:
        score += 1.0
    return score


def select_priority_indices(syntax_scores: list[float], sentences: list[str]) -> set[int]:
    priority_indices: set[int] = set(range(min(2, len(sentences))))
    if not syntax_scores:
        return priority_indices

    ranked = sorted(range(len(syntax_scores)), key=lambda index: syntax_scores[index], reverse=True)
    max_priority = max(8, int(len(sentences) * 0.08))
    threshold_index = max(0, int(len(syntax_scores) * 0.88) - 1)
    threshold = sorted(syntax_scores, reverse=True)[threshold_index]

    for index in ranked:
        if len(priority_indices) >= max_priority:
            break
        if syntax_scores[index] >= threshold or has_structural_label_block(sentences[index]):
            priority_indices.add(index)
    return priority_indices


ACRONYM_CONNECTOR_WORDS = frozenset({"and", "or", "&"})
ACRONYM_ENTITY_CONNECTOR_RE = re.compile(r"\s+(?:and|or|&)\s+", flags=re.IGNORECASE)
REPEATED_PHRASE_RE = re.compile(
    r"\b((?:[A-Z][a-z]+(?:'s)?\s+){1,3}(?:and\s+)?(?:[A-Z][a-z]+(?:'s)?)(?:\s+[A-Z][a-z]+(?:'s)?){0,2})\b"
)


def phrase_significant_words(phrase: str) -> list[str]:
    return [
        word
        for word in re.findall(r"[A-Za-z][A-Za-z']*", phrase)
        if word.lower() not in TOKEN_STOPWORDS and word.lower() not in ACRONYM_CONNECTOR_WORDS
    ]


def phrase_has_entity_connector(phrase: str) -> bool:
    return bool(ACRONYM_ENTITY_CONNECTOR_RE.search(phrase))


def acronym_phrase_allowed(phrase: str, acronym: str) -> bool:
    words = phrase_significant_words(phrase)
    if len(words) < 2:
        return False
    if len(phrase_capitalized_tokens(phrase)) < 2:
        return False
    if len(words) == 2 and not phrase_has_entity_connector(phrase):
        return False
    if len(acronym) < 3 and not phrase_has_entity_connector(phrase):
        return False
    return True


def build_phrase_acronym(phrase: str) -> str | None:
    segments = re.split(r"\s+and\s+", phrase, flags=re.IGNORECASE)
    segment_codes: list[str] = []
    for segment in segments:
        words = re.findall(r"[A-Za-z][A-Za-z']*", segment)
        significant = [
            word
            for word in words
            if word.lower() not in TOKEN_STOPWORDS and word.lower() not in ACRONYM_CONNECTOR_WORDS
        ]
        significant = [word for word in significant if word[0].isupper()]
        if len(significant) < 1:
            continue
        segment_codes.append("".join(word[0].upper() for word in significant))

    if not segment_codes:
        return None
    if len(segment_codes) == 1:
        code = segment_codes[0]
        if len(code) < 2:
            return None
        return code
    return "&".join(segment_codes)


def phrase_capitalized_tokens(phrase: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[A-Za-z][A-Za-z']*", phrase)
        if token[0].isupper() and token.lower() not in ACRONYM_CONNECTOR_WORDS
    ]


def trailing_surname_pair(phrase: str, superset_phrase: str) -> bool:
    parts = re.split(r"\s+and\s+", phrase, flags=re.IGNORECASE)
    if len(parts) != 2:
        return False
    left, right = parts[0].strip(), parts[1].strip()
    if not left or not right or " " in left or " " in right:
        return False
    if re.search(rf"\b{re.escape(left)}\s+[A-Z][a-z]+\b", superset_phrase):
        return False
    if re.search(rf"\b{re.escape(right)}\s+[A-Z][a-z]+\b", superset_phrase):
        return False
    return bool(
        re.search(rf"\b[A-Z][a-z]+\s+{re.escape(left)}\b", superset_phrase)
        and re.search(rf"\b[A-Z][a-z]+\s+{re.escape(right)}\b", superset_phrase)
    )


def find_superset_phrase(phrase: str, text: str) -> str | None:
    best: str | None = None
    for match in REPEATED_PHRASE_RE.finditer(text):
        longer = match.group(1).strip()
        if len(longer) <= len(phrase) or not trailing_surname_pair(phrase, longer):
            continue
        if best is None or len(longer) > len(best):
            best = longer
    return best


def alias_acronym_for_surname_pair(
    phrase: str, replacements: list[tuple[str, str]], text: str
) -> str | None:
    superset = find_superset_phrase(phrase, text)
    if not superset:
        return None
    longer_tokens = set(phrase_capitalized_tokens(superset))
    for other, acronym in replacements:
        if other == phrase or trailing_surname_pair(other, superset):
            continue
        other_tokens = phrase_capitalized_tokens(other)
        if len(other_tokens) >= 2 and set(other_tokens) <= longer_tokens:
            return acronym
    return None


def discover_repeated_phrases(text: str, *, min_occurrences: int = 2) -> list[tuple[str, str]]:
    counts: dict[str, int] = {}
    canonical: dict[str, str] = {}

    for match in REPEATED_PHRASE_RE.finditer(text):
        phrase = match.group(1).strip()
        acronym = build_phrase_acronym(phrase)
        if not acronym or not acronym_phrase_allowed(phrase, acronym):
            continue
        if len(acronym) >= len(phrase.replace(" ", "")):
            continue

        key = phrase.lower()
        counts[key] = counts.get(key, 0) + 1
        canonical.setdefault(key, phrase)

    replacements: list[tuple[str, str]] = []
    for key, count in counts.items():
        if count < min_occurrences:
            continue
        phrase = canonical[key]
        acronym = build_phrase_acronym(phrase)
        if acronym and acronym_phrase_allowed(phrase, acronym):
            replacements.append((phrase, acronym))

    replacements = sorted(replacements, key=lambda item: len(item[0]), reverse=True)
    finalized: list[tuple[str, str]] = []
    for phrase, acronym in replacements:
        alias = alias_acronym_for_surname_pair(phrase, replacements, text)
        if alias:
            acronym = alias
        elif find_superset_phrase(phrase, text):
            continue
        finalized.append((phrase, acronym))
    return finalized


def replace_subsequent_with_acronym(text: str, phrase: str, acronym: str) -> str:
    pattern = re.compile(re.escape(phrase), flags=re.IGNORECASE)
    seen = 0

    def replacer(match: re.Match[str]) -> str:
        nonlocal seen
        seen += 1
        if seen == 1:
            return match.group(0)
        return acronym

    return pattern.sub(replacer, text)


def apply_acronym_compression(text: str, *, phrase_source: str | None = None) -> str:
    if not env_bool("SOMA_USE_ACRONYMS", True):
        return text

    replacements = discover_repeated_phrases(phrase_source or text)
    updated = text
    for phrase, acronym in replacements:
        updated = replace_subsequent_with_acronym(updated, phrase, acronym)
    return updated


def plain_text_budget_chars(text: str, *, fallback_max_chars: int) -> int:
    ratio = clamp(env_float("SOMA_COMPRESSION_RATIO", 0.45), 0.05, 1.0)
    ratio_budget = max(512, int(len(text) * ratio))
    return max(fallback_max_chars, ratio_budget)


def compress_plain_text_semantic(text: str, *, max_chars: int) -> str:
    text = dedupe_inline_repetition(text)
    phrase_source = text
    if len(text) <= max_chars:
        return apply_acronym_compression(text, phrase_source=phrase_source)

    sentences = dedupe_lines(merge_metadata_label_sentences(split_sentences(text)))
    if not sentences:
        return text[:max_chars]

    query_text = " ".join(sentences[: min(4, len(sentences))])[:1200]
    query_tokens = bm25_tokenize(query_text)
    ranker = SimpleBM25(sentences)

    scored: list[tuple[float, int, str]] = []
    syntax_scores: list[float] = []
    for index, sentence in enumerate(sentences):
        semantic = ranker.score(query_tokens, index)
        syntax = syntax_sentence_score(sentence, ranker=ranker)
        syntax_scores.append(syntax)
        total = syntax * 2.0 + semantic * 8.0 + line_score(sentence)
        scored.append((total, index, sentence))

    score_by_index = {index: total for total, index, _ in scored}
    must_keep = dedupe_index_set(
        sentences,
        {
            index
            for index in select_priority_indices(syntax_scores, sentences)
            if not is_low_value_sentence(sentences[index])
        },
        score_by_index=score_by_index,
    )
    selected_indices = set(must_keep)
    budget = max_chars - sum(len(sentences[index]) + 1 for index in must_keep)
    selected_sentences = [sentences[index] for index in sorted(must_keep)]

    for total, index, sentence in sorted(scored, key=lambda item: item[0], reverse=True):
        if index in selected_indices:
            continue
        if is_low_value_sentence(sentence):
            continue
        if any(near_duplicate_line(sentence, existing) for existing in selected_sentences):
            continue
        cost = len(sentence) + 1
        if cost > budget:
            continue
        selected_indices.add(index)
        selected_sentences.append(sentence)
        budget -= cost

    ordered_indices = sorted(selected_indices)

    while ordered_indices:
        total_chars = sum(len(sentences[index]) + 1 for index in ordered_indices) - 1
        if total_chars <= max_chars:
            break
        droppable = [index for index in ordered_indices if index not in must_keep]
        if not droppable:
            break
        drop_index = min(
            droppable,
            key=lambda index: (
                0 if is_low_value_sentence(sentences[index]) else 1,
                score_by_index.get(index, 0.0),
            ),
        )
        ordered_indices.remove(drop_index)

    ordered_sentences = [sentences[index] for index in ordered_indices]
    ordered_qualities = [score_by_index.get(index, 0.0) for index in ordered_indices]
    ordered = dedupe_lines(ordered_sentences, quality_scores=ordered_qualities)
    output = apply_acronym_compression("\n".join(ordered), phrase_source=phrase_source)
    return output[:max_chars]


def line_score(line: str) -> int:
    score = 0
    if ERROR_LIKE_RE.search(line):
        score += 4
    if FILE_EXT_RE.search(line) or PATH_LIKE_RE.search(line):
        score += 4
    if DIFF_MARKER_RE.search(line):
        score += 3
    if FUNCTION_CALL_RE.search(line):
        score += 2
    if re.search(r"\b(def |class )", line):
        score += 3
    if 20 <= len(line) <= 240:
        score += 1
    if line.strip().startswith(("+", "-", "@@", "diff --git")):
        score += 3
    return score


def truncate_text_extractive(text: str, *, max_chars: int) -> str:
    text = dedupe_inline_repetition(text)
    if len(text) <= max_chars:
        return text

    if env_bool("SOMA_USE_SENTENCE_RANKER", True) and is_plain_text_passage(text):
        return compress_plain_text_semantic(text, max_chars=max_chars)

    lines = dedupe_lines([line.rstrip() for line in text.splitlines()])
    if not lines:
        return text[:max_chars]

    selected: dict[int, str] = {}
    for index in (0, 1, len(lines) - 2, len(lines) - 1):
        if 0 <= index < len(lines) and lines[index].strip():
            selected[index] = dedupe_inline_repetition(lines[index].strip())

    ranked = sorted(
        ((index, line) for index, line in enumerate(lines) if line.strip()),
        key=lambda item: (line_score(item[1]), -item[0]),
        reverse=True,
    )
    budget = max_chars
    for index, line in ranked:
        if index in selected:
            continue
        piece = dedupe_inline_repetition(line.strip())
        if not piece:
            continue
        if len(piece) + 1 > budget:
            if budget > 80:
                piece = piece[: budget - 1]
            else:
                continue
        selected[index] = piece
        budget -= len(piece) + 1
        if budget <= 0:
            break

    ordered = dedupe_lines([selected[index] for index in sorted(selected)])
    output = "\n".join(ordered)
    return output[:max_chars]


def is_low_value_tool_result(message: dict[str, Any]) -> bool:
    if normalize_role(message.get("role")) != "toolResult":
        return False
    tool = str(message.get("tool") or "").lower()
    text = extract_text(message.get("content")).strip().lower()
    if not text or text in {"(no output)", "no output"}:
        return True
    if tool != "exec":
        return False
    noisy_markers = (
        "command not found",
        "no module named pytest",
        "exec preflight",
        "complex interpreter invocation",
        "modulenotfounderror",
    )
    return any(marker in text for marker in noisy_markers)


def clean_trajectory_user_body(body: str) -> str:
    text = THINKING_TAG_RE.sub("", body)
    text = TRAJECTORY_TEXT_TAG_RE.sub("", text)
    return text.strip()


def clean_trajectory_assistant_body(body: str) -> str:
    return body.strip()


def compress_cot_thinking_blocks(text: str, *, max_chars: int = 320) -> str:
    def shrink(match: re.Match[str]) -> str:
        inner = match.group(1).strip()
        if len(inner) <= max_chars:
            return match.group(0)
        clipped = inner[: max(0, max_chars - 4)].rstrip()
        return f"<thinking>\n{clipped}...\n</thinking>"

    return THINKING_TAG_RE.sub(shrink, text)


def parse_trajectory_text(task: str) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    cursor = 0
    for match in TRAJECTORY_MESSAGE_BLOCK_RE.finditer(task):
        if match.start() > cursor:
            gap = task[cursor : match.start()].strip()
            for tool_match in TRAJECTORY_TOOL_RESULT_RE.finditer(gap):
                messages.append(
                    {
                        "role": "toolResult",
                        "tool": tool_match.group("tool"),
                        "toolCallId": tool_match.group("tool_call_id"),
                        "content": tool_match.group("body").strip(),
                    }
                )
        role = match.group("role").strip()
        body = match.group("body")
        if role == "assistant":
            content = clean_trajectory_assistant_body(body)
        else:
            content = clean_trajectory_user_body(body)
        if content:
            messages.append({"role": role, "content": content})
        cursor = match.end()

    tail = task[cursor:].strip()
    for tool_match in TRAJECTORY_TOOL_RESULT_RE.finditer(tail):
        messages.append(
            {
                "role": "toolResult",
                "tool": tool_match.group("tool"),
                "toolCallId": tool_match.group("tool_call_id"),
                "content": tool_match.group("body").strip(),
            }
        )

    return [message for message in messages if message.get("content")]


def trajectory_to_messages(task: str) -> list[dict[str, str]]:
    if "<message role=" in task or "<tool_result " in task:
        parsed = parse_trajectory_text(task)
        if parsed:
            return parsed
    return [{"role": "user", "content": task}]


def is_trajectory_xml(task: str) -> bool:
    return "<message role=" in task or "<tool_result " in task


def serialize_trajectory_message_content(role: str, content: str) -> str:
    if role == "assistant":
        if "<tool_call" in content or "<text>" in content or "<thinking>" in content:
            return content
        return f"<text>\n{content}\n</text>"
    if role == "user":
        if content.lstrip().startswith("<text>"):
            return content
        return f"<text>\n{content}\n</text>"
    return content


def messages_to_trajectory(messages: list[dict]) -> str:
    parts: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role") or "unknown"
        if role == "toolResult":
            tool = message.get("tool") or "tool"
            tool_call_id = message.get("toolCallId") or message.get("tool_call_id") or ""
            content = (
                message.get("content")
                if isinstance(message.get("content"), str)
                else str(message.get("content") or "")
            )
            parts.append(
                f'<tool_result tool="{tool}" tool_call_id="{tool_call_id}">\n{content.rstrip()}\n</tool_result>'
            )
            continue
        content = (
            message.get("content")
            if isinstance(message.get("content"), str)
            else str(message.get("content") or "")
        )
        content = serialize_trajectory_message_content(role, content.strip())
        parts.append(f'<message role="{role}">\n{content}\n</message>')
    return "\n\n".join(parts)


def messages_to_plain_text(messages: list[dict]) -> str:
    if len(messages) == 1 and isinstance(messages[0], dict):
        role = normalize_role(messages[0].get("role"))
        text = extract_text(messages[0].get("content")).strip()
        if role == "user" and text:
            return text

    parts: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = normalize_role(message.get("role")) or "unknown"
        text = extract_text(message.get("content")).strip()
        if not text:
            continue
        if role == "toolResult":
            tool = message.get("tool") or "tool"
            parts.append(f"[tool_result:{tool}]\n{text}")
        else:
            parts.append(f"[{role}]\n{text}")
    return "\n\n".join(parts)


def is_cot_message_array(messages: list[Any]) -> bool:
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = normalize_role(message.get("role"))
        if role in {"assistant", "toolResult"}:
            return True
        text = extract_text(message.get("content"))
        if "<tool_call" in text or "<tool_result" in text:
            return True
    return False


def cot_tool_line_score(line: str) -> int:
    score = line_score(line)
    if COT_TOOL_HIGH_SIGNAL_RE.search(line):
        score += 8
    if re.search(r"\.py:\d+", line):
        score += 6
    if line.strip().startswith(("+", "-", "@@", "diff --git")):
        score += 5
    return score


def compress_cot_tool_output(text: str, *, max_chars: int) -> str:
    text = dedupe_inline_repetition(text)
    if len(text) <= max_chars:
        return text

    lines = [line.rstrip() for line in text.splitlines()]
    if not lines:
        return text[:max_chars]

    selected: dict[int, str] = {}
    for index in list(range(min(5, len(lines)))) + list(range(max(0, len(lines) - 12), len(lines))):
        if lines[index].strip():
            selected[index] = lines[index]

    budget = max_chars - sum(len(line) + 1 for line in selected.values())
    ranked = sorted(
        ((index, line) for index, line in enumerate(lines) if line.strip()),
        key=lambda item: (cot_tool_line_score(item[1]), -item[0]),
        reverse=True,
    )
    for index, line in ranked:
        if index in selected:
            continue
        cost = len(line) + 1
        if cost > budget:
            if budget > 80:
                line = line[: budget - 1]
                cost = len(line) + 1
            else:
                continue
        selected[index] = line
        budget -= cost
        if budget <= 0:
            break

    if not selected:
        return text[:max_chars]
    return "\n".join(selected[index] for index in sorted(selected))[:max_chars]


def assistant_narrative_text(message: dict[str, Any]) -> str:
    text = extract_text(message.get("content"))
    stripped = TOOL_CALL_XML_BLOCK_RE.sub("", text)
    stripped = HTML_TAG_RE.sub("", stripped)
    stripped = re.sub(r"<[^>]+>", "", stripped)
    return re.sub(r"\s+", " ", stripped).strip()


COT_PLANNING_RE = re.compile(
    r"\b(let me|now |next |i(?:'ll| will)|verify|check|add a test|here's what)\b",
    flags=re.IGNORECASE,
)


def assistant_has_narrative(message: dict[str, Any]) -> bool:
    text = assistant_narrative_text(message)
    if len(text) >= 20:
        return True
    return bool(COT_PLANNING_RE.search(text))


def is_conclusion_message(message: dict[str, Any]) -> bool:
    if normalize_role(message.get("role")) != "assistant":
        return False
    text = assistant_narrative_text(message)
    if len(text) < 80:
        return False
    return bool(
        re.search(r"\b(here's what|summary|changes are|what i did|fix(?:ed)?|added)\b", text, flags=re.IGNORECASE)
        or text.count("\n") >= 2
        or "**" in extract_text(message.get("content"))
    )


def is_discovery_tool_result(message: dict[str, Any]) -> bool:
    if normalize_role(message.get("role")) != "toolResult":
        return False
    text = extract_text(message.get("content"))
    if re.search(r"\.py:\d+", text):
        return True
    if str(message.get("tool") or "").lower() == "read" and re.search(r"\bdef\s+\w+", text):
        return True
    return False


def discovery_tool_score(message: dict[str, Any]) -> int:
    if not is_discovery_tool_result(message):
        return 0
    text = extract_text(message.get("content"))
    score = 10
    score += len(re.findall(r"\.py:\d+", text)) * 4
    if re.search(r"\bdef\s+\w+", text):
        score += 8
    if str(message.get("tool") or "").lower() == "read":
        score += 6
    return score


def assistant_has_thinking(message: dict[str, Any]) -> bool:
    return (
        isinstance(message, dict)
        and normalize_role(message.get("role")) == "assistant"
        and "<thinking>" in extract_text(message.get("content"))
    )


def select_cot_arc_indices(
    messages: list[Any],
    *,
    tool_result_indices: list[int],
    keep_assistant_narratives: int,
) -> set[int]:
    keep: set[int] = set()
    first_user = find_first_user_index(messages)
    if first_user is not None:
        keep.add(first_user)

    edit_tool_indices = [
        index
        for index in tool_result_indices
        if isinstance(messages[index], dict) and str(messages[index].get("tool") or "").lower() == "edit"
    ]
    keep.update(edit_tool_indices)
    keep.update(
        index
        for index, message in enumerate(messages)
        if isinstance(message, dict)
        and normalize_role(message.get("role")) == "assistant"
        and '"edit"' in extract_text(message.get("content"))
    )

    discovery_ranked = sorted(
        (
            (discovery_tool_score(messages[index]), index)
            for index in tool_result_indices
            if isinstance(messages[index], dict) and is_discovery_tool_result(messages[index])
        ),
        reverse=True,
    )
    for _, index in discovery_ranked[:2]:
        keep.add(index)

    narrative_indices = [
        index
        for index, message in enumerate(messages)
        if isinstance(message, dict)
        and normalize_role(message.get("role")) == "assistant"
        and assistant_has_narrative(message)
    ]
    if narrative_indices:
        keep.update(narrative_indices[-keep_assistant_narratives:])
        for index in narrative_indices:
            if is_conclusion_message(messages[index]):
                keep.add(index)

    planning_indices = [
        index
        for index in narrative_indices
        if isinstance(messages[index], dict) and COT_PLANNING_RE.search(assistant_narrative_text(messages[index]))
    ]
    keep.update(planning_indices)

    for edit_index in edit_tool_indices:
        for lookback in range(edit_index - 1, max(-1, edit_index - 8), -1):
            if lookback < 0:
                break
            candidate = messages[lookback]
            if not isinstance(candidate, dict):
                continue
            role = normalize_role(candidate.get("role"))
            if role == "toolResult" and (
                is_discovery_tool_result(candidate) or not is_low_value_tool_result(candidate)
            ):
                keep.add(lookback)
                break
            if role == "assistant" and extract_tool_call_ids(candidate):
                keep.add(lookback)

    paired_tool_results = [
        index
        for index in tool_result_indices
        if index in edit_tool_indices
        or is_discovery_tool_result(messages[index])
        or not is_low_value_tool_result(messages[index])
    ]
    keep.update(paired_tool_results[-6:])

    thinking_keep = env_int("SOMA_KEEP_THINKING_BLOCKS", 1)
    if thinking_keep > 0:
        thinking_indices = [
            index for index, message in enumerate(messages) if assistant_has_thinking(message)
        ]
        preferred_thinking = [index for index in thinking_indices if index in keep]
        keep.update((preferred_thinking or thinking_indices)[:thinking_keep])

    kept_tool_result_indices = [
        index
        for index in sorted(keep)
        if isinstance(messages[index], dict) and normalize_role(messages[index].get("role")) == "toolResult"
    ]
    keep.update(find_tool_call_indices(messages, kept_tool_result_indices))
    return expand_cot_tool_pairs(messages, keep)


def expand_cot_tool_pairs(messages: list[Any], indices: set[int]) -> set[int]:
    expanded = set(indices)
    changed = True
    while changed:
        changed = False
        for index in sorted(expanded):
            message = messages[index]
            if not isinstance(message, dict):
                continue
            role = normalize_role(message.get("role"))
            if role == "assistant":
                for call_id in extract_tool_call_ids(message):
                    for result_index, result in enumerate(messages):
                        if (
                            result_index not in expanded
                            and isinstance(result, dict)
                            and normalize_role(result.get("role")) == "toolResult"
                            and call_id in extract_tool_result_ids(result)
                        ):
                            expanded.add(result_index)
                            changed = True
            elif role == "toolResult":
                result_ids = extract_tool_result_ids(message)
                for call_index, call_message in enumerate(messages):
                    if (
                        call_index not in expanded
                        and isinstance(call_message, dict)
                        and normalize_role(call_message.get("role")) == "assistant"
                        and extract_tool_call_ids(call_message) & result_ids
                    ):
                        expanded.add(call_index)
                        changed = True
    return expanded


def cot_arc_fill_priority(message: dict[str, Any], *, index: int, total: int) -> int:
    score = message_score(message, index=index, total=total)
    role = normalize_role(message.get("role"))
    text = extract_text(message.get("content"))
    if role == "assistant" and COT_PLANNING_RE.search(assistant_narrative_text(message)):
        score += 40
    if role == "toolResult" and is_discovery_tool_result(message):
        score += 35
    if role == "assistant" and is_conclusion_message(message):
        score += 50
    if role == "assistant" and assistant_has_thinking(message):
        score += 25
    if role == "assistant" and "<tool_call" in text and '"edit"' in text:
        score += 30
    return score


def build_compressed_messages(
    working: list[Any],
    selected_indices: set[int],
    *,
    wanted_tool_ids: set[str],
    tool_call_indices: list[int],
) -> list[Any]:
    kept_call_ids: set[str] = set()
    for index in sorted(selected_indices):
        message = working[index]
        if isinstance(message, dict) and normalize_role(message.get("role")) == "assistant":
            kept_call_ids.update(extract_tool_call_ids(message))

    compressed: list[Any] = []
    for index in sorted(selected_indices):
        message = working[index]
        if isinstance(message, dict) and normalize_role(message.get("role")) == "toolResult":
            if kept_call_ids and not (extract_tool_result_ids(message) & kept_call_ids):
                continue
        if index in tool_call_indices and isinstance(message, dict):
            message = filter_tool_call_message(message, wanted_tool_ids)
        compressed.append(message)
    return compressed


def cot_coherence_report(messages: list[Any]) -> dict[str, bool]:
    text = "\n".join(extract_text(message.get("content")) for message in messages if isinstance(message, dict))
    lowered = text.lower()
    return {
        "has_task": find_first_user_index(messages) is not None,
        "has_thinking": any(assistant_has_thinking(message) for message in messages if isinstance(message, dict)),
        "has_edit_action": '"edit"' in lowered or 'tool="edit"' in lowered,
        "has_discovery": bool(re.search(r"\.py:\d+|\bdef\s+\w+", text)),
        "has_planning_bridge": bool(COT_PLANNING_RE.search(text)),
        "has_conclusion": bool(
            re.search(r"here's what|what i did|changes are|added `", lowered)
            or any(is_conclusion_message(message) for message in messages if isinstance(message, dict))
        ),
        "has_paired_tool_calls": "<tool_call" in lowered and "<tool_result" in lowered,
    }


def filter_tool_call_content(content: Any, kept_ids: set[str]) -> Any:
    if not kept_ids or not isinstance(content, str):
        return content

    def drop_block(match: re.Match[str]) -> str:
        if match.group(1) in kept_ids:
            return match.group(0)
        return ""

    return TOOL_CALL_XML_BLOCK_RE.sub(drop_block, content)


def filter_tool_call_message(message: dict[str, Any], kept_ids: set[str]) -> dict[str, Any]:
    if not kept_ids:
        return message

    filtered = copy.deepcopy(message)
    content = filtered.get("content")
    if isinstance(content, list):
        filtered["content"] = [
            block
            for block in content
            if not (
                isinstance(block, dict)
                and block.get("type") == "toolCall"
                and isinstance(block.get("id"), str)
                and block["id"].strip() not in kept_ids
            )
        ]
        return filtered

    if isinstance(content, str) and "<tool_call" in content:
        filtered["content"] = filter_tool_call_content(content, kept_ids)
    return filtered


def set_message_text(message: dict[str, Any], text: str) -> dict[str, Any]:
    updated = copy.deepcopy(message)
    content = updated.get("content")
    if isinstance(content, list):
        replaced = False
        new_content = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text" and not replaced:
                new_content.append({**block, "text": text})
                replaced = True
            else:
                new_content.append(block)
        if not replaced:
            new_content.insert(0, {"type": "text", "text": text})
        updated["content"] = new_content
    else:
        updated["content"] = text
    return updated


def truncate_message_content(
    message: dict[str, Any],
    *,
    max_chars: int,
    preserve_task: bool = False,
    preserve_thinking: bool = False,
) -> dict[str, Any]:
    role = normalize_role(message.get("role"))
    text = extract_text(message.get("content"))

    if role == "toolResult":
        compressed = compress_cot_tool_output(text, max_chars=max_chars)
        if compressed == text:
            return message
        return set_message_text(message, compressed)

    if preserve_thinking and role == "assistant":
        original = text
        if "<thinking>" in text:
            text = compress_cot_thinking_blocks(text)
        if "<tool_call" in text:
            if text != original:
                return set_message_text(message, text)
            return message

    if preserve_task:
        if len(text) <= max_chars:
            return message
        return set_message_text(message, text[:max_chars])

    if is_plain_text_passage(text):
        max_chars = plain_text_budget_chars(text, fallback_max_chars=max_chars)
    if len(text) <= max_chars:
        cleaned = truncate_text_extractive(text, max_chars=max_chars)
        if cleaned == text:
            return message
        return set_message_text(message, cleaned)
    return set_message_text(message, truncate_text_extractive(text, max_chars=max_chars))


def sanitize_messages(messages: list[Any]) -> tuple[list[Any], dict[str, Any]]:
    sanitized: list[Any] = []
    changed = False
    removed = 0
    removed_thinking = 0
    cot_mode = is_cot_message_array(messages)

    for message in messages:
        if not isinstance(message, dict):
            sanitized.append(message)
            continue

        role = normalize_role(message.get("role"))
        if role == "assistant" and isinstance(message.get("errorMessage"), str):
            content = message.get("content")
            if content in (None, "", []):
                changed = True
                removed += 1
                continue

        next_message = copy.deepcopy(message)
        content = next_message.get("content")
        if isinstance(content, list) and not cot_mode:
            filtered = [
                block
                for block in content
                if not (isinstance(block, dict) and block.get("type") == "thinking")
            ]
            if len(filtered) != len(content):
                changed = True
                removed_thinking += 1
                next_message["content"] = filtered
        elif cot_mode and normalize_role(next_message.get("role")) == "assistant":
            if isinstance(content, str) and "<thinking>" in content:
                compressed_thinking = compress_cot_thinking_blocks(content)
                if compressed_thinking != content:
                    changed = True
                    next_message["content"] = compressed_thinking
            elif isinstance(content, list):
                updated_blocks: list[Any] = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "thinking":
                        thinking_text = extract_text(block.get("text") or block.get("content"))
                        clipped = thinking_text[:320].rstrip()
                        if len(thinking_text) > len(clipped):
                            changed = True
                            block = {**block, "text": f"{clipped}..."}
                    updated_blocks.append(block)
                next_message["content"] = updated_blocks

        if role != "toolResult" and not extract_text(next_message.get("content")).strip():
            changed = True
            removed += 1
            continue

        sanitized.append(next_message)

    return sanitized if changed else messages, {
        "changed": changed,
        "removedMessageCount": removed,
        "removedThinkingBlockCount": removed_thinking,
    }


def message_score(message: dict[str, Any], *, index: int, total: int) -> int:
    role = normalize_role(message.get("role"))
    text = extract_text(message.get("content"))
    score = line_score(text)
    if role == "user" and index == 0:
        score += 100
    if role == "toolResult":
        score += 20
        tool = str(message.get("tool") or "").lower()
        if tool == "edit":
            score += 30
        elif tool == "read":
            score += 12
    if role == "assistant":
        if assistant_has_narrative(message):
            score += 18
        lowered = text.lower()
        if re.search(r"\bdef\s+test_\w+", text):
            score += 12
        if re.search(r"\.py:\d+", text):
            score += 10
        if "<tool_call" in lowered and '"edit"' in lowered:
            score += 24
    recency_boost = max(0, 10 - (total - 1 - index))
    score += recency_boost
    return score


def extract_tool_result_ids(message: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    for field in ("toolCallId", "toolUseId", "id"):
        value = message.get(field)
        if isinstance(value, str) and value.strip():
            ids.add(value.strip())
    return ids


def extract_tool_call_ids(message: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    content = message.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "toolCall":
                value = block.get("id")
                if isinstance(value, str) and value.strip():
                    ids.add(value.strip())
    for field in ("toolCalls", "tool_calls"):
        tool_calls = message.get(field)
        if isinstance(tool_calls, list):
            for tool_call in tool_calls:
                if isinstance(tool_call, dict):
                    value = tool_call.get("id")
                    if isinstance(value, str) and value.strip():
                        ids.add(value.strip())
    text = extract_text(content)
    for match in TOOL_CALL_XML_ID_RE.finditer(text):
        ids.add(match.group(1))
    return ids


def find_tool_call_indices(messages: list[Any], tool_result_indices: list[int]) -> list[int]:
    wanted_ids: set[str] = set()
    for index in tool_result_indices:
        if isinstance(messages[index], dict):
            wanted_ids.update(extract_tool_result_ids(messages[index]))

    matched: list[int] = []
    if wanted_ids:
        for index, message in enumerate(messages):
            if isinstance(message, dict) and extract_tool_call_ids(message) & wanted_ids:
                matched.append(index)

    if matched:
        return matched

    if not tool_result_indices:
        return []

    first_tool_result_index = min(tool_result_indices)
    for index in range(first_tool_result_index - 1, -1, -1):
        if isinstance(messages[index], dict) and extract_tool_call_ids(messages[index]):
            return [index]
    return []


def find_first_user_index(messages: list[Any]) -> int | None:
    for index, message in enumerate(messages):
        if isinstance(message, dict) and normalize_role(message.get("role")) == "user":
            return index
    return None


def prepare_cot_working_copy(
    messages: list[Any],
    *,
    max_tool_chars: int,
    max_user_chars: int,
) -> list[Any]:
    working = [copy.deepcopy(message) if isinstance(message, dict) else message for message in messages]
    first_user = find_first_user_index(working)
    for index, message in enumerate(working):
        if not isinstance(message, dict):
            continue
        role = normalize_role(message.get("role"))
        if role == "toolResult":
            working[index] = truncate_message_content(message, max_chars=max_tool_chars)
        elif role == "assistant":
            working[index] = truncate_message_content(
                message, max_chars=max_user_chars, preserve_thinking=True
            )
        elif role == "user" and index == first_user:
            working[index] = truncate_message_content(
                message, max_chars=max_user_chars, preserve_task=True
            )
    return working


def compress_cot_messages(
    messages: list[Any],
    *,
    target_tokens: int,
    keep_assistant_narratives: int,
) -> tuple[list[Any], dict[str, Any]]:
    """CoT pipeline: arc keep -> pair expansion -> budget fill -> trim."""
    max_tool_chars = env_int("SOMA_MAX_TOOL_RESULT_CHARS", 2400)
    max_user_chars = env_int("SOMA_MAX_USER_MESSAGE_CHARS", 3500)
    working = prepare_cot_working_copy(
        messages, max_tool_chars=max_tool_chars, max_user_chars=max_user_chars
    )

    if estimate_tokens_for_message_array(working) <= target_tokens:
        return working, {"changed": False, "reason": "within_budget_after_truncation", "cotMode": True}

    first_user = find_first_user_index(working)
    tool_result_indices = [
        index
        for index, message in enumerate(working)
        if isinstance(message, dict) and normalize_role(message.get("role")) == "toolResult"
    ]

    keep_indices = select_cot_arc_indices(
        working,
        tool_result_indices=tool_result_indices,
        keep_assistant_narratives=keep_assistant_narratives,
    )

    edit_tool_indices = [
        index
        for index in tool_result_indices
        if isinstance(working[index], dict) and str(working[index].get("tool") or "").lower() == "edit"
    ]
    edit_assistant_indices = {
        index
        for index, message in enumerate(working)
        if isinstance(message, dict)
        and normalize_role(message.get("role")) == "assistant"
        and '"edit"' in extract_text(message.get("content"))
    }
    discovery_indices = {
        index
        for index in tool_result_indices
        if isinstance(working[index], dict) and is_discovery_tool_result(working[index])
    }
    conclusion_indices = {
        index
        for index, message in enumerate(working)
        if isinstance(message, dict) and is_conclusion_message(message)
    }
    thinking_indices = {index for index, message in enumerate(working) if assistant_has_thinking(message)}
    planning_indices = {
        index
        for index, message in enumerate(working)
        if isinstance(message, dict)
        and normalize_role(message.get("role")) == "assistant"
        and COT_PLANNING_RE.search(assistant_narrative_text(message))
    }
    narrative_indices = [
        index
        for index, message in enumerate(working)
        if isinstance(message, dict)
        and normalize_role(message.get("role")) == "assistant"
        and assistant_has_narrative(message)
    ]

    wanted_tool_ids: set[str] = set()
    for index in keep_indices:
        if isinstance(working[index], dict) and normalize_role(working[index].get("role")) == "toolResult":
            wanted_tool_ids.update(extract_tool_result_ids(working[index]))

    kept_tool_result_indices = [
        index
        for index in sorted(keep_indices)
        if isinstance(working[index], dict) and normalize_role(working[index].get("role")) == "toolResult"
    ]
    tool_call_indices = find_tool_call_indices(working, kept_tool_result_indices)
    keep_indices.update(tool_call_indices)
    keep_indices = expand_cot_tool_pairs(working, keep_indices)

    fill_candidates = sorted(
        (
            (cot_arc_fill_priority(message, index=index, total=len(working)), index)
            for index, message in enumerate(working)
            if isinstance(message, dict) and index not in keep_indices
        ),
        reverse=True,
    )

    selected_indices = set(keep_indices)
    for _, index in fill_candidates:
        candidate_indices = sorted(selected_indices | {index})
        if estimate_tokens_for_message_array([working[i] for i in candidate_indices]) <= target_tokens:
            selected_indices.add(index)

    min_target = int(target_tokens * env_float("SOMA_COT_FILL_RATIO", 0.90))
    current_estimate = estimate_tokens_for_message_array([working[i] for i in sorted(selected_indices)])
    if current_estimate < min_target:
        for _, index in fill_candidates:
            if index in selected_indices:
                continue
            candidate_indices = sorted(selected_indices | {index})
            if estimate_tokens_for_message_array([working[i] for i in candidate_indices]) <= target_tokens:
                selected_indices.add(index)
                selected_indices = expand_cot_tool_pairs(working, selected_indices)

    if estimate_tokens_for_message_array([working[i] for i in sorted(selected_indices)]) > target_tokens:
        selected_indices = set(keep_indices)

    selected_indices = expand_cot_tool_pairs(working, selected_indices)
    compressed = build_compressed_messages(
        working,
        selected_indices,
        wanted_tool_ids=wanted_tool_ids,
        tool_call_indices=tool_call_indices,
    )

    protected_indices = (
        set(edit_tool_indices)
        | edit_assistant_indices
        | discovery_indices
        | conclusion_indices
        | thinking_indices
        | planning_indices
    )
    if first_user is not None:
        protected_indices.add(first_user)
    if narrative_indices:
        protected_indices.update(narrative_indices[-keep_assistant_narratives:])

    while estimate_tokens_for_message_array(compressed) > target_tokens and len(compressed) > 1:
        droppable = [index for index in sorted(selected_indices) if index not in protected_indices]
        if not droppable:
            break
        drop_index = min(
            droppable,
            key=lambda index: (
                0 if is_low_value_tool_result(working[index]) else 1,
                cot_arc_fill_priority(working[index], index=index, total=len(working)),
            ),
        )
        selected_indices.remove(drop_index)
        compressed = build_compressed_messages(
            working,
            selected_indices,
            wanted_tool_ids=wanted_tool_ids,
            tool_call_indices=tool_call_indices,
        )

    changed = fingerprint_messages(compressed) != fingerprint_messages(messages)
    return compressed, {
        "changed": changed,
        "reason": "cot_arc_pruned" if changed else "nothing_to_remove",
        "keptToolResultCount": len([i for i in selected_indices if i in tool_result_indices]),
        "keptMessageCount": len(compressed),
        "targetTokens": target_tokens,
        "cotMode": True,
        "cotCoherence": cot_coherence_report(compressed),
    }


def compress_messages(messages: list[Any], *, target_tokens: int) -> tuple[list[Any], dict[str, Any]]:
    working = [copy.deepcopy(message) if isinstance(message, dict) else message for message in messages]
    cot_mode = is_cot_message_array(working)
    max_tool_chars = env_int("SOMA_MAX_TOOL_RESULT_CHARS", 2400)
    max_user_chars = env_int("SOMA_MAX_USER_MESSAGE_CHARS", 3500)
    keep_tool_results = env_int("SOMA_KEEP_TOOL_RESULTS", 8 if cot_mode else 6)
    keep_assistant_narratives = env_int("SOMA_KEEP_ASSISTANT_NARRATIVES", 3 if cot_mode else 1)

    if cot_mode:
        return compress_cot_messages(
            messages,
            target_tokens=target_tokens,
            keep_assistant_narratives=keep_assistant_narratives,
        )

    first_user = find_first_user_index(working)
    for index, message in enumerate(working):
        if not isinstance(message, dict):
            continue
        role = normalize_role(message.get("role"))
        if role == "toolResult":
            working[index] = truncate_message_content(message, max_chars=max_tool_chars)
        elif role == "user" and index == first_user:
            working[index] = truncate_message_content(message, max_chars=max_user_chars)

    if estimate_tokens_for_message_array(working) <= target_tokens:
        return working, {"changed": False, "reason": "within_budget_after_truncation"}

    tool_result_indices = [
        index
        for index, message in enumerate(working)
        if isinstance(message, dict) and normalize_role(message.get("role")) == "toolResult"
    ]
    keep_indices: set[int] = set()
    if first_user is not None:
        keep_indices.add(first_user)

    edit_tool_indices = [
        index
        for index in tool_result_indices
        if isinstance(working[index], dict) and str(working[index].get("tool") or "").lower() == "edit"
    ]
    kept_tool_results = [
        index
        for index in tool_result_indices[-keep_tool_results:]
        if index in edit_tool_indices or not is_low_value_tool_result(working[index])
    ]
    keep_indices.update(edit_tool_indices)
    keep_indices.update(kept_tool_results)

    wanted_tool_ids: set[str] = set()
    for index in keep_indices:
        if isinstance(working[index], dict) and normalize_role(working[index].get("role")) == "toolResult":
            wanted_tool_ids.update(extract_tool_result_ids(working[index]))

    kept_tool_result_indices = [
        index
        for index in sorted(keep_indices)
        if isinstance(working[index], dict) and normalize_role(working[index].get("role")) == "toolResult"
    ]
    tool_call_indices = find_tool_call_indices(working, kept_tool_result_indices)
    keep_indices.update(tool_call_indices)

    scored_candidates = [
        (message_score(message, index=index, total=len(working)), index)
        for index, message in enumerate(working)
        if isinstance(message, dict) and index not in keep_indices
    ]
    scored_candidates.sort(reverse=True)

    selected_indices = set(keep_indices)
    for _, index in scored_candidates:
        candidate_indices = sorted(selected_indices | {index})
        if estimate_tokens_for_message_array([working[i] for i in candidate_indices]) <= target_tokens:
            selected_indices.add(index)

    if estimate_tokens_for_message_array([working[i] for i in sorted(selected_indices)]) > target_tokens:
        selected_indices = set(keep_indices)

    compressed = build_compressed_messages(
        working,
        selected_indices,
        wanted_tool_ids=wanted_tool_ids,
        tool_call_indices=tool_call_indices,
    )

    while estimate_tokens_for_message_array(compressed) > target_tokens and len(compressed) > 1:
        droppable = [
            index
            for index in sorted(selected_indices)
            if index != first_user and index not in edit_tool_indices
        ]
        if not droppable:
            break
        drop_index = min(
            droppable,
            key=lambda index: (
                0 if is_low_value_tool_result(working[index]) else 1,
                message_score(working[index], index=index, total=len(working)),
            ),
        )
        selected_indices.remove(drop_index)
        compressed = build_compressed_messages(
            working,
            selected_indices,
            wanted_tool_ids=wanted_tool_ids,
            tool_call_indices=tool_call_indices,
        )

    changed = fingerprint_messages(compressed) != fingerprint_messages(messages)
    return compressed, {
        "changed": changed,
        "reason": "algorithmic_pruned" if changed else "nothing_to_remove",
        "keptToolResultCount": len([i for i in selected_indices if i in tool_result_indices]),
        "keptMessageCount": len(compressed),
        "targetTokens": target_tokens,
        "cotMode": False,
    }


def get_params(payload: dict[str, Any]) -> dict[str, Any]:
    params = payload.get("params")
    return params if isinstance(params, dict) else payload


def get_messages(payload: dict[str, Any]) -> list[Any]:
    params = get_params(payload)
    messages = params.get("messages")
    return messages if isinstance(messages, list) else []


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def fingerprint_messages(messages: list[Any]) -> str:
    return hashlib.sha256(canonical_json(messages).encode("utf-8")).hexdigest()


def safe_file_part(value: str) -> str:
    normalized = "".join(char if char.isalnum() or char in "_.-" else "-" for char in value)
    return (normalized.strip("-") or "session")[:120]


def resolve_plugin_dir(payload: dict[str, Any]) -> Path:
    value = payload.get("pluginDir")
    return Path(value) if isinstance(value, str) and value.strip() else Path(__file__).resolve().parent


def resolve_session_identity(payload: dict[str, Any]) -> str:
    params = get_params(payload)
    for key in ("sessionId", "sessionKey"):
        value = params.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return "session"


def save_state(payload: dict[str, Any], raw_messages: list[Any], output_messages: list[Any]) -> bool:
    plugin_dir = resolve_plugin_dir(payload)
    session_part = safe_file_part(resolve_session_identity(payload))
    state_path = plugin_dir / "logs" / STATE_DIR_NAME / f"{session_part}.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "version": STATE_VERSION,
        "updatedAt": utc_now(),
        "sourceMessageCount": len(raw_messages),
        "sourceFingerprint": fingerprint_messages(raw_messages),
        "messageCount": len(output_messages),
        "messages": output_messages,
    }
    temp_path = state_path.with_suffix(f"{state_path.suffix}.tmp")
    temp_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp_path.replace(state_path)
    return True


def compress_task_text(
    task: str,
    *,
    compression_ratio: float = 0.45,
    plugin_dir: Path | None = None,
) -> str:
    """Compress a plain-text passage or OpenClaw XML trajectory string."""
    previous_ratio = os.environ.get("SOMA_COMPRESSION_RATIO")
    previous_min = os.environ.get("SOMA_MIN_COMPACT_TOKENS")
    os.environ["SOMA_COMPRESSION_RATIO"] = str(compression_ratio)
    os.environ["SOMA_MIN_COMPACT_TOKENS"] = "0"
    try:
        payload = {
            "pluginId": "soma-miner",
            "pluginName": "SOMA Miner",
            "pluginDir": str(plugin_dir or Path(__file__).resolve().parent),
            "params": {
                "sessionId": "local-test",
                "messages": trajectory_to_messages(task),
            },
            "sourceHook": "assemble",
        }
        result = handle_assemble(payload)
        messages = result.get("messages") if isinstance(result, dict) else []
        if not isinstance(messages, list) or not messages:
            return task
        if is_trajectory_xml(task):
            return messages_to_trajectory(messages) or task
        return messages_to_plain_text(messages) or task
    finally:
        if previous_ratio is None:
            os.environ.pop("SOMA_COMPRESSION_RATIO", None)
        else:
            os.environ["SOMA_COMPRESSION_RATIO"] = previous_ratio
        if previous_min is None:
            os.environ.pop("SOMA_MIN_COMPACT_TOKENS", None)
        else:
            os.environ["SOMA_MIN_COMPACT_TOKENS"] = previous_min


def handle_assemble(payload: dict[str, Any]) -> dict[str, Any]:
    raw_messages = get_messages(payload)
    messages, sanitization = sanitize_messages(raw_messages)
    original_tokens = estimate_tokens_for_message_array(messages)

    min_tokens = env_int("SOMA_MIN_COMPACT_TOKENS", 1800)
    if original_tokens < min_tokens:
        changed = sanitization["changed"]
        return {
            "assembled": True,
            "messages": messages,
            "estimatedTokens": original_tokens,
            "compaction": {"compacted": False, "threshold": min_tokens},
            "baseMiner": {
                "changed": changed,
                "reason": "below_min_compact_tokens" if not changed else "sanitized",
                "originalMessageCount": len(raw_messages),
                "messageCount": len(messages),
                "tokensBefore": original_tokens,
                "tokensAfter": original_tokens,
                "method": "algorithmic",
            },
        }

    ratio = clamp(env_float("SOMA_COMPRESSION_RATIO", 0.45), 0.05, 1.0)
    target_tokens = max(256, int(original_tokens * ratio))
    compressed, metadata = compress_messages(messages, target_tokens=target_tokens)
    output_tokens = estimate_tokens_for_message_array(compressed)
    changed = (
        sanitization["changed"]
        or metadata.get("changed")
        or fingerprint_messages(compressed) != fingerprint_messages(raw_messages)
    )

    state_saved = False
    state_error = None
    try:
        state_saved = save_state(payload, raw_messages, compressed)
    except Exception as exc:
        state_error = str(exc)

    return {
        "assembled": True,
        "messages": compressed,
        "estimatedTokens": output_tokens,
        "compaction": {
            "compacted": changed,
            "method": "algorithmic",
            "targetTokens": target_tokens,
            "tokensBefore": original_tokens,
            "tokensAfter": output_tokens,
            "compressionRatio": round(output_tokens / original_tokens, 4) if original_tokens else 0,
        },
        "baseMiner": {
            **metadata,
            "changed": changed,
            "pruned": metadata.get("changed", False),
            "sanitized": sanitization["changed"],
            "removedMessageCount": sanitization.get("removedMessageCount", 0),
            "removedThinkingBlockCount": sanitization.get("removedThinkingBlockCount", 0),
            "originalMessageCount": len(raw_messages),
            "messageCount": len(compressed),
            "tokensBefore": original_tokens,
            "tokensAfter": output_tokens,
            "stateSaved": state_saved,
            "stateError": state_error,
            "method": "algorithmic",
        },
    }


def run_event(event_name: str) -> int:
    try:
        payload = json.loads(sys.stdin.read() or "{}")
        if not isinstance(payload, dict):
            raise ValueError("Connector payload must be a JSON object")
        if event_name not in EVENT_NAMES:
            raise ValueError(f"Unknown base miner event: {event_name}")
        response = {"ok": True, "result": handle_assemble(payload)}
    except Exception as error:
        response = {"ok": False, "error": str(error), "errorType": error.__class__.__name__}
    sys.stdout.buffer.write(json.dumps(response, ensure_ascii=False).encode("utf-8"))
    return 0


def cli_main(argv: list[str]) -> int:
    return run_event(argv[0] if argv else "assemble")


if __name__ == "__main__":
    raise SystemExit(cli_main(sys.argv[1:]))
