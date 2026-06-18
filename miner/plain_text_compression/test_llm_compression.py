"""Test an LLM-backed compression miner on local data.

Examples:
    python miner/plain_text_compression/test_llm_compression.py data.jsonl
    python miner/plain_text_compression/test_llm_compression.py data.csv --field problem_statement
    python miner/plain_text_compression/test_llm_compression.py data.txt --ratio 0.15 --limit 5

For real LLM calls:
    set OPENROUTER_API_KEY=sk-or-...

Optional:
    set SOMA_LLM_MODEL=google/gemini-2.5-flash
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
from typing import Any, Callable


DEFAULT_MINER = Path(__file__).with_name("llm_compression_miner.py")
DEFAULT_OUTPUT_DIR = Path(__file__).with_name("llm_test_results")
TEXT_FIELD_CANDIDATES = (
    "source_text",
    "task",
    "text",
    "prompt",
    "context",
    "messages",
    "trajectory",
    "problem_statement",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an LLM compression miner on local test data")
    parser.add_argument("data_path", help="Path to .jsonl, .json, .csv, or .txt data")
    parser.add_argument("--miner", default=str(DEFAULT_MINER), help="Path to miner python file")
    parser.add_argument("--field", default=None, help="Field/column containing text to compress")
    parser.add_argument("--ratio", type=float, default=0.45, help="Compression ratio, e.g. 0.45")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of records to run")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for outputs")
    args = parser.parse_args()

    data_path = Path(args.data_path).resolve()
    miner_path = Path(args.miner).resolve()
    output_dir = Path(args.output_dir).resolve()

    compress = load_miner_main(miner_path)
    records = list(load_records(data_path, field=args.field))
    if args.limit is not None:
        records = records[: args.limit]

    if not records:
        raise ValueError(f"No usable records found in {data_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.jsonl"

    with summary_path.open("w", encoding="utf-8") as summary_file:
        for index, text in enumerate(records, start=1):
            compressed = compress(text, args.ratio)
            output_path = output_dir / f"record_{index:04d}.txt"
            output_path.write_text(compressed, encoding="utf-8")

            original_tokens = approx_tokens(text)
            compressed_tokens = approx_tokens(compressed)
            ratio = compressed_tokens / original_tokens if original_tokens else 0.0
            row = {
                "index": index,
                "original_tokens_approx": original_tokens,
                "compressed_tokens_approx": compressed_tokens,
                "actual_ratio_approx": round(ratio, 4),
                "output_path": str(output_path),
            }
            summary_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(
                f"{index:04d}: {original_tokens} -> {compressed_tokens} tokens "
                f"({ratio:.2%}) {output_path}"
            )

    print(f"\nSummary written to {summary_path}")


def load_miner_main(path: Path) -> Callable[[str, float], str]:
    if not path.exists():
        raise FileNotFoundError(f"Miner file not found: {path}")

    spec = importlib.util.spec_from_file_location("candidate_miner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import miner file: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    main_fn = getattr(module, "main", None)
    if not callable(main_fn):
        raise AttributeError(f"Miner file must define callable main(task, compression_ratio): {path}")
    return main_fn


def load_records(path: Path, *, field: str | None) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return load_jsonl(path, field=field)
    if suffix == ".json":
        return load_json(path, field=field)
    if suffix == ".csv":
        return load_csv(path, field=field)
    return [path.read_text(encoding="utf-8")]


def load_jsonl(path: Path, *, field: str | None) -> list[str]:
    records: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        value = extract_text(json.loads(line), field=field)
        if value:
            records.append(value)
    return records


def load_json(path: Path, *, field: str | None) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [text for item in payload if (text := extract_text(item, field=field))]
    text = extract_text(payload, field=field)
    return [text] if text else []


def load_csv(path: Path, *, field: str | None) -> list[str]:
    records: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            value = extract_text(row, field=field)
            if value:
                records.append(value)
    return records


def extract_text(value: Any, *, field: str | None) -> str:
    if isinstance(value, str):
        return value

    if field and isinstance(value, dict):
        return stringify(value.get(field))

    if isinstance(value, dict):
        for candidate in TEXT_FIELD_CANDIDATES:
            text = stringify(value.get(candidate))
            if text:
                return text
        return stringify(value)

    return stringify(value)


def stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False, indent=2)


def approx_tokens(text: str) -> int:
    try:
        import tiktoken

        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:
        return max(1, len(text) // 4)


if __name__ == "__main__":
    main()
