#!/usr/bin/env python3
"""Test compression miner on CoT-Compression-1 CSV challenges."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import sys
from pathlib import Path


def approx_tokens(text: str) -> int:
    try:
        import tiktoken

        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:
        return max(1, len(text) // 4)


def load_main(miner_path: Path):
    spec = importlib.util.spec_from_file_location("candidate_miner", miner_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import miner: {miner_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    main_fn = getattr(module, "main", None)
    if not callable(main_fn):
        raise AttributeError(f"Miner must define main(task, compression_ratio): {miner_path}")
    return main_fn


def load_qa(qa_path: Path) -> dict[str, list[tuple[str, str]]]:
    grouped: dict[str, list[tuple[str, str]]] = {}
    with qa_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            challenge_id = row.get("challenge_id") or ""
            answer = (row.get("answer_text") or "").strip()
            if challenge_id and answer:
                grouped.setdefault(challenge_id, []).append((row.get("question_text") or "", answer))
    return grouped


def answer_present(answer: str, text: str) -> bool:
    if answer in text:
        return True
    normalized = " ".join(answer.split())
    if normalized in " ".join(text.split()):
        return True
    keywords = [
        token.lower()
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_.-]{2,}", answer)
        if token.lower()
        not in {
            "the",
            "and",
            "that",
            "this",
            "with",
            "from",
            "into",
            "will",
            "should",
            "have",
            "been",
            "were",
            "when",
            "what",
            "where",
            "which",
            "their",
            "there",
            "they",
            "then",
            "also",
            "does",
            "using",
            "used",
            "method",
            "class",
            "function",
            "test",
            "file",
            "code",
            "change",
            "added",
            "make",
            "need",
            "must",
        }
    ]
    if not keywords:
        return False
    lowered = text.lower()
    hits = sum(1 for keyword in keywords if keyword in lowered)
    return hits >= max(2, int(len(keywords) * 0.45))


def main() -> None:
    parser = argparse.ArgumentParser(description="Test miner on CoT-Compression-1 challenges.csv")
    parser.add_argument(
        "--data",
        default=str(Path(__file__).with_name("sample_tasks") / "CoT-Compression-1" / "challenges.csv"),
    )
    parser.add_argument(
        "--qa",
        default=str(Path(__file__).with_name("sample_tasks") / "CoT-Compression-1" / "challenge_QA.csv"),
    )
    parser.add_argument(
        "--miner",
        default=str(Path(__file__).with_name("algorithmic_compression_miner.py")),
    )
    parser.add_argument("--field", default="challenge_text")
    parser.add_argument("--ratio", type=float, default=0.45)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).with_name("algorithmic_test_results") / "cot_compression_1"),
    )
    args = parser.parse_args()

    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

    data_path = Path(args.data).resolve()
    qa_path = Path(args.qa).resolve()
    miner_path = Path(args.miner).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    compress = load_main(miner_path)
    qa_by_challenge = load_qa(qa_path)
    summary_path = output_dir / "summary.jsonl"

    with data_path.open(encoding="utf-8", newline="") as handle, summary_path.open(
        "w", encoding="utf-8"
    ) as summary_file:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader, start=1):
            if index > args.limit:
                break

            text = (row.get(args.field) or "").strip()
            if not text:
                continue

            compressed = compress(text, args.ratio)
            output_path = output_dir / f"record_{index:04d}.txt"
            output_path.write_text(compressed, encoding="utf-8")

            original_tokens = approx_tokens(text)
            compressed_tokens = approx_tokens(compressed)
            actual_ratio = compressed_tokens / original_tokens if original_tokens else 0.0
            challenge_id = row.get("challenge_id") or ""
            qa_items = qa_by_challenge.get(challenge_id, [])
            qa_hits = sum(1 for _, answer in qa_items if answer_present(answer, compressed))

            summary = {
                "index": index,
                "challenge_id": challenge_id,
                "challenge_name": row.get("challenge_name"),
                "original_tokens_approx": original_tokens,
                "compressed_tokens_approx": compressed_tokens,
                "actual_ratio_approx": round(actual_ratio, 4),
                "target_ratio": args.ratio,
                "qa_hits": qa_hits,
                "qa_total": len(qa_items),
                "output_path": str(output_path),
            }
            summary_file.write(json.dumps(summary, ensure_ascii=False) + "\n")
            print(
                f"{index:04d} ({row.get('challenge_name', 'unknown')}): "
                f"{original_tokens} -> {compressed_tokens} tokens ({actual_ratio:.2%}), "
                f"QA {qa_hits}/{len(qa_items)}"
            )

    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
