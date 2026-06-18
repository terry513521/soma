"""Sample LLM-backed miner for text / CoT compression.

This file keeps the same public contract as the sample miner:

    main(task: str, compression_ratio: float | None = None) -> str

Set OPENROUTER_API_KEY before running if you want real LLM compression.
The default model is chosen for long-context, low-cost summarization, but can
be changed with SOMA_LLM_MODEL.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request

try:
    import tiktoken

    ENCODER = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover - fallback for minimal local environments
    ENCODER = None


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "google/gemini-2.5-flash"
DEFAULT_COMPRESSION_RATIO = 0.45
REQUEST_TIMEOUT_SECONDS = 120

SYSTEM_PROMPT = (
    "You are a SOMA trajectory compiler. Your input is an OpenClaw/SWE-bench coding-agent "
    "trajectory: user requests, tool calls, tool outputs, file snippets, terminal results, "
    "errors, hypotheses, and decisions. Compile it into handoff memory for the next coding "
    "agent.\n\n"
    "KEEP (preserve exactly or with minimal trimming):\n"
    "- original bug/request and acceptance criteria\n"
    "- important files, functions, classes, symbols, and config keys\n"
    "- root cause and current diagnosis\n"
    "- critical command outputs that changed understanding\n"
    "- test failures, expected vs actual behavior, and stack traces\n"
    "- constraints, environment facts, and blockers\n"
    "- current fix direction and patch plan\n"
    "- recent tool results and the commands that produced them\n\n"
    "REMOVE or SUMMARIZE:\n"
    "- duplicate logs and repeated context\n"
    "- long irrelevant file dumps (keep only the relevant snippet)\n"
    "- old failed attempts that no longer affect the next step\n"
    "- verbose thinking and filler narration\n"
    "- unimportant command output with no diagnostic value\n"
    "- already-stated facts restated later in the trajectory\n\n"
    "Rules:\n"
    "- Never invent facts, files, tests, or conclusions.\n"
    "- If uncertain, mark as uncertain instead of guessing.\n"
    "- Prefer keeping recent evidence over old exploratory noise.\n"
    "- Do not collapse everything into a tiny abstract summary.\n"
    "- Keep tool/message structure when helpful; summarize only low-value sections.\n\n"
    "Output only the compressed handoff. Useful sections: Bug/Request, Relevant "
    "Files/Functions, Root Cause, Critical Commands/Outputs, Test Failures, Constraints, "
    "Fix Direction, Recent Tool Results, Risks."
)


def build_user_prompt(text: str, target_tokens: int) -> str:
    return (
        f"Compress this OpenClaw agent trajectory to approximately {target_tokens} tokens. "
        "Stay near that budget. Treat it as live repair state for the next SWE-bench agent.\n\n"
        f"{text}"
    )


def main(task: str, compression_ratio: float | None = None) -> str:
    """Compress task text while preserving facts needed by a coding agent."""
    ratio = _clamp_ratio(compression_ratio or DEFAULT_COMPRESSION_RATIO)
    target_tokens = max(128, int(token_count(task) * ratio))

    if token_count(task) <= target_tokens:
        return task

    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        return extractive_fallback(task, target_tokens)

    try:
        return compress_with_openrouter(task, target_tokens=target_tokens, api_key=api_key)
    except Exception:
        # Miner code should fail soft: returning something useful is better than
        # crashing the benchmark run.
        return extractive_fallback(task, target_tokens)


def compress_with_openrouter(text: str, *, target_tokens: int, api_key: str) -> str:
    model = os.getenv("SOMA_LLM_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    temperature = float(os.getenv("SOMA_LLM_TEMPERATURE", "0.1"))

    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max(256, min(8192, int(target_tokens * 1.15))),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(text, target_tokens)},
        ],
    }

    request = urllib.request.Request(
        OPENROUTER_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "http://localhost"),
            "X-Title": os.getenv("OPENROUTER_APP_NAME", "SOMA LLM Compression Miner"),
        },
        method="POST",
    )

    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                raw = response.read().decode("utf-8")
            data = json.loads(raw)
            compressed = data["choices"][0]["message"]["content"].strip()
            return enforce_token_limit(compressed, target_tokens)
        except (urllib.error.URLError, urllib.error.HTTPError, KeyError, json.JSONDecodeError) as exc:
            last_error = exc
            time.sleep(1.5 * (attempt + 1))

    raise RuntimeError(f"OpenRouter compression failed: {last_error}")


def extractive_fallback(text: str, target_tokens: int) -> str:
    """Deterministic fallback that keeps high-signal lines first."""
    if target_tokens <= 0:
        return ""

    lines = [line.rstrip() for line in text.splitlines()]
    scored = sorted(
        enumerate(lines),
        key=lambda item: (_line_score(item[1]), -item[0]),
        reverse=True,
    )

    selected: list[tuple[int, str]] = []
    budget = target_tokens
    for index, line in scored:
        stripped = line.strip()
        if not stripped:
            continue
        line_tokens = token_count(stripped)
        if line_tokens > max(32, target_tokens // 3):
            stripped = enforce_token_limit(stripped, max(32, target_tokens // 3))
            line_tokens = token_count(stripped)
        if line_tokens <= budget:
            selected.append((index, stripped))
            budget -= line_tokens
        if budget <= 0:
            break

    if not selected:
        return enforce_token_limit(text, target_tokens)

    selected.sort(key=lambda item: item[0])
    return enforce_token_limit("\n".join(line for _, line in selected), target_tokens)


def token_count(text: str) -> int:
    if not text:
        return 0
    if ENCODER is None:
        return max(1, len(text) // 4)
    return len(ENCODER.encode(text))


def enforce_token_limit(text: str, token_limit: int) -> str:
    if token_limit <= 0:
        return ""
    if ENCODER is None:
        return text[: token_limit * 4].rstrip()
    ids = ENCODER.encode(text)
    if len(ids) <= token_limit:
        return text
    return ENCODER.decode(ids[:token_limit]).rstrip()


def _clamp_ratio(ratio: float) -> float:
    return max(0.01, min(1.0, float(ratio)))


def _line_score(line: str) -> int:
    score = 0
    lowered = line.lower()
    patterns = [
        r"\b(error|exception|traceback|failed|failure|assert|expected|actual)\b",
        r"\b(test|pytest|unittest|spec|repro|regression)\b",
        r"\b(file|path|class|function|method|import|module)\b",
        r"\b(issue|bug|fix|requirement|constraint|must|should)\b",
        r"[A-Za-z0-9_/.-]+\.(py|js|ts|tsx|java|go|rs|cpp|c|h|md)\b",
        r"\b[A-Za-z_][A-Za-z0-9_]*\(",
    ]
    for pattern in patterns:
        if re.search(pattern, line):
            score += 4
    if "```" in line or lowered.startswith(("diff --git", "@@", "+", "-")):
        score += 3
    if 20 <= len(line) <= 240:
        score += 1
    return score
