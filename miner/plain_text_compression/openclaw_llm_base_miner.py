#!/usr/bin/env python3
"""OpenClaw-compatible LLM trajectory compressor for SOMA local testing.

Upload/test this file as the SOMA plugin backend (`base_miner.py`). It implements
the same subprocess contract used by `SOMA-plugin` and `compression_service`:

    python base_miner.py assemble < stdin-json

Environment:
    OPENROUTER_API_KEY             Optional. If missing, deterministic fallback is used.
    SOMA_LLM_MODEL                 Default: google/gemini-2.5-flash
    SOMA_COMPRESSION_RATIO         Default: 0.25
    SOMA_MIN_COMPACT_TOKENS        Default: 1800
    SOMA_FORCE_LLM_COMPACTION      Set to 1/true to compact even short histories.
    SOMA_KEEP_RECENT_MESSAGES      Default: 6
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
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

try:
    import tiktoken

    ENCODER = tiktoken.get_encoding("cl100k_base")
except Exception:
    ENCODER = None


EVENT_NAMES = frozenset({"assemble"})
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "google/gemini-2.5-flash"
STATE_VERSION = 1


def handle_assemble(payload: dict[str, Any]) -> dict[str, Any]:
    params = get_params(payload)
    raw_messages = params.get("messages") if isinstance(params.get("messages"), list) else []
    messages, sanitization = sanitize_messages(raw_messages)
    original_tokens = estimate_tokens_for_message_array(messages)

    force = env_bool("SOMA_FORCE_LLM_COMPACTION", False)
    min_tokens = env_int("SOMA_MIN_COMPACT_TOKENS", 1800)
    if not force and original_tokens < min_tokens:
        return build_result(
            messages=messages,
            original_messages=raw_messages,
            original_tokens=original_tokens,
            changed=sanitization["changed"],
            reason="below_min_compact_tokens" if not sanitization["changed"] else "sanitized",
            compaction={"compacted": False, "threshold": min_tokens},
            payload=payload,
        )

    ratio = env_float("SOMA_COMPRESSION_RATIO", 0.25)
    target_tokens = max(256, int(original_tokens * clamp(ratio, 0.01, 1.0)))
    serialized = serialize_trajectory(messages)

    compressed = compress_text(serialized, target_tokens=target_tokens)
    output_messages = build_handoff_messages(messages, compressed)
    output_tokens = estimate_tokens_for_message_array(output_messages)
    changed = fingerprint_messages(output_messages) != fingerprint_messages(raw_messages)

    return build_result(
        messages=output_messages,
        original_messages=raw_messages,
        original_tokens=original_tokens,
        changed=changed,
        reason="llm_handoff_compacted" if changed else "nothing_to_remove",
        compaction={
            "compacted": changed,
            "method": "llm_handoff",
            "targetTokens": target_tokens,
            "tokensBefore": original_tokens,
            "tokensAfter": output_tokens,
            "compressionRatio": round(output_tokens / original_tokens, 4) if original_tokens else 0,
            "model": os.getenv("SOMA_LLM_MODEL", DEFAULT_MODEL),
            "usedOpenRouter": bool(os.getenv("OPENROUTER_API_KEY", "").strip()),
        },
        payload=payload,
    )


def build_handoff_messages(messages: list[Any], handoff: str) -> list[Any]:
    first_user = find_first_user_message(messages)
    recent = keep_recent_messages(messages, env_int("SOMA_KEEP_RECENT_MESSAGES", 6))
    handoff_message = {
        "role": "user",
        "content": (
            "[SOMA compressed trajectory handoff]\n"
            "The previous OpenClaw trajectory was compacted. Use this as repair state; "
            "continue solving the original task and produce the patch.\n\n"
            f"{handoff}"
        ),
    }

    output: list[Any] = []
    first_user_to_keep = compact_first_user_message(first_user)
    if first_user_to_keep is not None:
        output.append(first_user_to_keep)
    output.append(handoff_message)

    for message in recent:
        if first_user is not None and fingerprint_messages([message]) == fingerprint_messages([first_user]):
            continue
        output.append(message)
    return output


def compact_first_user_message(message: Any | None) -> Any | None:
    if not isinstance(message, dict):
        return None

    max_tokens = env_int("SOMA_PRESERVE_FIRST_USER_MAX_TOKENS", 900)
    content = extract_text(message.get("content"))
    if token_count(content) <= max_tokens:
        return message

    excerpt = extract_original_problem_excerpt(content)
    excerpt = enforce_token_limit(excerpt or content, max_tokens)
    compacted = copy.deepcopy(message)
    compacted["content"] = (
        "[SOMA preserved original task excerpt]\n"
        "The full first user message was too large and was compacted before handoff.\n\n"
        f"{excerpt}"
    )
    return compacted


def extract_original_problem_excerpt(text: str) -> str:
    match = re.search(
        r"<message\s+role=[\"']user[\"']>\s*(.*?)\s*</message>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match:
        return match.group(1).strip()
    return text.strip()


def compress_text(text: str, *, target_tokens: int) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        return extractive_fallback(text, target_tokens)
    try:
        return compress_with_openrouter(text, target_tokens=target_tokens, api_key=api_key)
    except Exception:
        return extractive_fallback(text, target_tokens)


def compress_with_openrouter(text: str, *, target_tokens: int, api_key: str) -> str:
    model = os.getenv("SOMA_LLM_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    payload = {
        "model": model,
        "temperature": env_float("SOMA_LLM_TEMPERATURE", 0.1),
        "max_tokens": max(256, min(8192, int(target_tokens * 1.15))),
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a SOMA trajectory compiler. Your input is an OpenClaw/SWE-bench "
                    "agent trajectory with user requests, tool calls, file snippets, terminal "
                    "results, errors, hypotheses, and partial decisions. Compile it into dense "
                    "handoff memory for the next coding agent.\n\n"
                    "Preserve exact issue requirements, file paths, symbols, tests, commands, "
                    "errors, observed behavior, constraints, diagnosis, patch plan, and failed "
                    "attempts that prevent repeated mistakes. Never invent facts. Mark uncertain "
                    "evidence as uncertain. Remove duplicated reasoning, filler, and verbose logs "
                    "after extracting useful facts.\n\n"
                    "Output only compact sections: Problem, Evidence, Relevant Code, Tests/Errors, "
                    "Decisions, Patch Plan, Risks."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Compress this OpenClaw trajectory to about {target_tokens} tokens or less. "
                    "It will be used as live repair state by the next SWE-bench coding agent.\n\n"
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
            "X-Title": os.getenv("OPENROUTER_APP_NAME", "SOMA OpenClaw LLM Miner"),
        },
        method="POST",
    )

    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=env_int("SOMA_LLM_TIMEOUT_SECONDS", 120)) as response:
                data = json.loads(response.read().decode("utf-8"))
            return enforce_token_limit(data["choices"][0]["message"]["content"].strip(), target_tokens)
        except (urllib.error.URLError, urllib.error.HTTPError, KeyError, json.JSONDecodeError) as exc:
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"OpenRouter compression failed: {last_error}")


def serialize_trajectory(messages: list[Any]) -> str:
    parts: list[str] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            parts.append(f"<message index={index}>\n{stringify(message)}\n</message>")
            continue
        role = normalize_role(message.get("role")) or "unknown"
        content = extract_text(message.get("content"))
        extra = []
        for field in ("toolCallId", "toolUseId", "id", "name"):
            value = message.get(field)
            if isinstance(value, str) and value:
                extra.append(f'{field}="{value}"')
        header = f"<message index={index} role=\"{role}\" {' '.join(extra)}>"
        parts.append(f"{header}\n{content}\n</message>")
    return "\n\n".join(parts)


def extractive_fallback(text: str, target_tokens: int) -> str:
    lines = [line.rstrip() for line in text.splitlines()]
    scored = sorted(enumerate(lines), key=lambda item: (_line_score(item[1]), -item[0]), reverse=True)
    selected: list[tuple[int, str]] = []
    budget = target_tokens
    for index, line in scored:
        stripped = line.strip()
        if not stripped:
            continue
        line_tokens = token_count(stripped)
        if line_tokens > max(48, target_tokens // 4):
            stripped = enforce_token_limit(stripped, max(48, target_tokens // 4))
            line_tokens = token_count(stripped)
        if line_tokens <= budget:
            selected.append((index, stripped))
            budget -= line_tokens
        if budget <= 0:
            break
    selected.sort(key=lambda item: item[0])
    return enforce_token_limit("\n".join(line for _, line in selected) or text, target_tokens)


def sanitize_messages(messages: list[Any]) -> tuple[list[Any], dict[str, Any]]:
    sanitized: list[Any] = []
    changed = False
    removed = 0
    for message in messages:
        if not isinstance(message, dict):
            sanitized.append(message)
            continue
        next_message = copy.deepcopy(message)
        content = next_message.get("content")
        if isinstance(content, list):
            filtered = [block for block in content if not (isinstance(block, dict) and block.get("type") == "thinking")]
            if len(filtered) != len(content):
                changed = True
                next_message["content"] = filtered
        if normalize_role(next_message.get("role")) != "toolResult" and not extract_text(next_message.get("content")).strip():
            changed = True
            removed += 1
            continue
        sanitized.append(next_message)
    return sanitized if changed else messages, {"changed": changed, "removedMessageCount": removed}


def keep_recent_messages(messages: list[Any], count: int) -> list[Any]:
    if count <= 0:
        return []
    return messages[-count:]


def find_first_user_message(messages: list[Any]) -> Any | None:
    for message in messages:
        if isinstance(message, dict) and normalize_role(message.get("role")) == "user":
            return message
    return None


def build_result(
    *,
    messages: list[Any],
    original_messages: list[Any],
    original_tokens: int,
    changed: bool,
    reason: str,
    compaction: dict[str, Any],
    payload: dict[str, Any],
) -> dict[str, Any]:
    state_error = None
    state_saved = False
    try:
        state_saved = save_state(payload, original_messages, messages)
    except Exception as exc:
        state_error = str(exc)

    output_tokens = estimate_tokens_for_message_array(messages)
    return {
        "assembled": True,
        "messages": messages,
        "estimatedTokens": output_tokens,
        "compaction": compaction,
        "baseMiner": {
            "changed": changed,
            "pruned": changed,
            "reason": reason,
            "originalMessageCount": len(original_messages),
            "messageCount": len(messages),
            "tokensBefore": original_tokens,
            "tokensAfter": output_tokens,
            "stateSaved": state_saved,
            "stateError": state_error,
            "updatedAt": utc_now(),
        },
    }


def save_state(payload: dict[str, Any], raw_messages: list[Any], output_messages: list[Any]) -> bool:
    plugin_dir = resolve_plugin_dir(payload)
    session_part = safe_file_part(resolve_session_identity(payload))
    state_path = plugin_dir / "logs" / "state" / f"{session_part}.json"
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


def get_params(payload: dict[str, Any]) -> dict[str, Any]:
    params = payload.get("params")
    return params if isinstance(params, dict) else payload


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


def stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


def normalize_role(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    lowered = value.strip().lower().replace("_", "").replace("-", "")
    return "toolResult" if lowered == "toolresult" else lowered


def estimate_tokens_for_message_array(messages: list[Any]) -> int:
    return sum(token_count(extract_text(message.get("content"))) for message in messages if isinstance(message, dict))


def token_count(text: str) -> int:
    if not text:
        return 0
    if ENCODER is None:
        return max(1, math.ceil(len(text) / 4))
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


def fingerprint_messages(messages: list[Any]) -> str:
    return hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def safe_file_part(value: str) -> str:
    normalized = "".join(char if char.isalnum() or char in "_.-" else "-" for char in value)
    return (normalized.strip("-") or "session")[:120]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


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
