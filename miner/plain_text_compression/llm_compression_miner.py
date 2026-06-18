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
DEFAULT_COMPRESSION_RATIO = 0.2
REQUEST_TIMEOUT_SECONDS = 120


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
            {
                "role": "system",
                "content": (
                    "You are a SOMA trajectory compiler. Your input is not ordinary text: "
                    "it is an OpenClaw/SWE-bench coding-agent trajectory containing user "
                    "requests, thoughts, tool calls, tool outputs, file snippets, terminal "
                    "results, errors, hypotheses, and partial decisions. Your job is to "
                    "compile that trajectory into dense handoff memory for the next coding "
                    "agent.\n\n"
                    "Goal: preserve everything needed for the next agent to create the "
                    "correct patch, while deleting tokens that do not change the repair "
                    "state.\n\n"
                    "Never invent facts, files, test results, or conclusions. If evidence is "
                    "uncertain, mark it as uncertain. Keep exact names for files, symbols, "
                    "tests, errors, config keys, commands, and observed outputs.\n\n"
                    "Prioritize in this order: original problem and acceptance criteria; "
                    "failing tests/errors; relevant files/classes/functions; code behavior "
                    "observed from tool outputs; constraints and environment facts; current "
                    "diagnosis; patch plan; failed attempts that prevent repeated mistakes.\n\n"
                    "Remove duplicated reasoning, social text, repeated logs, dead-end "
                    "speculation without evidence, and verbose outputs after extracting "
                    "their useful facts.\n\n"
                    "Output only the compressed handoff. Use compact sections: Problem, "
                    "Evidence, Relevant Code, Tests/Errors, Decisions, Patch Plan, Risks."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Compress this OpenClaw agent trajectory to about {target_tokens} "
                    "tokens or less. Treat it as live repair state, not as prose to "
                    "summarize. If the budget is tight, prefer exact actionable evidence "
                    "over narrative explanation. The next SWE-bench agent should be able "
                    "to continue from your handoff and produce the correct patch.\n\n"
                    f"{text}"
                ),
            },
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
