#!/usr/bin/env python3
"""Test compression miner on Context-Compression-1 CSV challenges."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Test miner on Context-Compression-1 challenges.csv")
    parser.add_argument(
        "--data",
        default=str(
            Path(__file__).with_name("sample_tasks")
            / "Context-Compression-1"
            / "challenges.csv"
        ),
    )
    parser.add_argument(
        "--miner",
        default=str(Path(__file__).with_name("algorithmic_compression_miner.py")),
    )
    parser.add_argument("--field", default="challenge_text")
    parser.add_argument("--ratio", type=float, default=0.45)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).with_name("algorithmic_test_results") / "context_compression_1"),
    )
    args = parser.parse_args()

    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

    data_path = Path(args.data).resolve()
    miner_path = Path(args.miner).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    compress = load_main(miner_path)
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
            summary = {
                "index": index,
                "challenge_id": row.get("challenge_id"),
                "challenge_name": row.get("challenge_name"),
                "original_tokens_approx": original_tokens,
                "compressed_tokens_approx": compressed_tokens,
                "actual_ratio_approx": round(actual_ratio, 4),
                "target_ratio": args.ratio,
                "output_path": str(output_path),
            }
            summary_file.write(json.dumps(summary, ensure_ascii=False) + "\n")
            print(
                f"{index:04d} ({row.get('challenge_name', 'unknown')}): "
                f"{original_tokens} -> {compressed_tokens} tokens ({actual_ratio:.2%})"
            )

    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
