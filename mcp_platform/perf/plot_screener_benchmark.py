#!/usr/bin/env python3
"""
Create benchmark comparison plots for screener query experiments.

Input CSV columns:
  scenario,variant,planning_ms,execution_ms

Example:
  count,old,8.455,494.797
  count,new_cte,6.546,222.086
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from typing import Dict, List, Sequence, Tuple


SAMPLE_ROWS = [
    {"scenario": "count", "variant": "old", "planning_ms": 8.455, "execution_ms": 494.797},
    {"scenario": "count", "variant": "new_cte", "planning_ms": 6.546, "execution_ms": 222.086},
    {"scenario": "top_miner_ids", "variant": "old", "planning_ms": 11.442, "execution_ms": 40276.693},
    {"scenario": "top_miner_ids", "variant": "new_cte", "planning_ms": 6.639, "execution_ms": 720.403},
    {
        "scenario": "top_miner_ids",
        "variant": "new_cte_pushdown",
        "planning_ms": 10.596,
        "execution_ms": 299.305,
    },
    {"scenario": "top_ss58", "variant": "old", "planning_ms": 10.863, "execution_ms": 42285.540},
    {"scenario": "top_ss58", "variant": "new_cte", "planning_ms": 6.588, "execution_ms": 798.234},
]


def _unique_in_order(values: Sequence[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def _label(text: str) -> str:
    return text.replace("_", " ")


def _load_csv(path: str) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        expected = {"scenario", "variant", "planning_ms", "execution_ms"}
        missing = expected - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
        for row in reader:
            rows.append(
                {
                    "scenario": row["scenario"],
                    "variant": row["variant"],
                    "planning_ms": float(row["planning_ms"]),
                    "execution_ms": float(row["execution_ms"]),
                }
            )
    return rows


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        exe = sys.executable
        print(
            "matplotlib is required for the currently running interpreter.",
            file=sys.stderr,
        )
        print(f"Python executable: {exe}", file=sys.stderr)
        print(f"Install with: {exe} -m pip install matplotlib", file=sys.stderr)
        raise SystemExit(1)
    return plt


def _build_lookup(rows: Sequence[Dict[str, float]], metric: str) -> Dict[Tuple[str, str], float]:
    lookup: Dict[Tuple[str, str], float] = {}
    for row in rows:
        lookup[(row["scenario"], row["variant"])] = float(row[metric])
    return lookup


def plot_grouped_metric(
    plt,
    rows: Sequence[Dict[str, float]],
    metric: str,
    output_path: str,
    title: str,
    log_scale: bool = False,
) -> None:
    scenarios = _unique_in_order([str(r["scenario"]) for r in rows])
    variants = _unique_in_order([str(r["variant"]) for r in rows])
    lookup = _build_lookup(rows, metric)

    fig, ax = plt.subplots(figsize=(12, 6))
    x_values = list(range(len(scenarios)))
    width = 0.8 / max(len(variants), 1)

    for idx, variant in enumerate(variants):
        offset = (idx - (len(variants) - 1) / 2.0) * width
        bar_positions = [x + offset for x in x_values]
        values = [lookup.get((scenario, variant), math.nan) for scenario in scenarios]
        ax.bar(bar_positions, values, width=width, label=_label(variant))

    ax.set_xticks(x_values)
    ax.set_xticklabels([_label(s) for s in scenarios], rotation=15)
    ax.set_ylabel("milliseconds")
    ax.set_title(title)
    if log_scale:
        ax.set_yscale("log")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_speedup(
    plt,
    rows: Sequence[Dict[str, float]],
    baseline: str,
    output_path: str,
    title: str,
) -> None:
    scenarios = _unique_in_order([str(r["scenario"]) for r in rows])
    variants = [v for v in _unique_in_order([str(r["variant"]) for r in rows]) if v != baseline]
    lookup = _build_lookup(rows, "execution_ms")

    fig, ax = plt.subplots(figsize=(12, 6))
    x_values = list(range(len(scenarios)))
    width = 0.8 / max(len(variants), 1)

    for idx, variant in enumerate(variants):
        offset = (idx - (len(variants) - 1) / 2.0) * width
        bar_positions = [x + offset for x in x_values]
        values: List[float] = []
        for scenario in scenarios:
            old = lookup.get((scenario, baseline))
            new = lookup.get((scenario, variant))
            if old is None or new is None or new <= 0:
                values.append(math.nan)
            else:
                values.append(old / new)
        ax.bar(bar_positions, values, width=width, label=f"{_label(variant)} vs {_label(baseline)}")

    ax.set_xticks(x_values)
    ax.set_xticklabels([_label(s) for s in scenarios], rotation=15)
    ax.set_ylabel("speedup (x)")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def print_summary(rows: Sequence[Dict[str, float]], baseline: str) -> None:
    scenarios = _unique_in_order([str(r["scenario"]) for r in rows])
    variants = _unique_in_order([str(r["variant"]) for r in rows])
    exec_lookup = _build_lookup(rows, "execution_ms")
    plan_lookup = _build_lookup(rows, "planning_ms")

    print("\nExecution/Planning summary (ms):")
    for scenario in scenarios:
        print(f"\n- {scenario}")
        for variant in variants:
            exec_ms = exec_lookup.get((scenario, variant))
            plan_ms = plan_lookup.get((scenario, variant))
            if exec_ms is None or plan_ms is None:
                continue
            print(f"  {variant:18} planning={plan_ms:9.3f}  execution={exec_ms:11.3f}")

    if baseline in variants:
        print(f"\nSpeedups vs '{baseline}' (execution only):")
        for scenario in scenarios:
            old_exec = exec_lookup.get((scenario, baseline))
            if old_exec is None:
                continue
            for variant in variants:
                if variant == baseline:
                    continue
                new_exec = exec_lookup.get((scenario, variant))
                if new_exec is None or new_exec <= 0:
                    continue
                print(f"  {scenario:14} {variant:18} {old_exec / new_exec:8.2f}x")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot screener benchmark results from CSV.")
    parser.add_argument(
        "--input",
        default="",
        help="CSV file with columns: scenario,variant,planning_ms,execution_ms",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(__file__), "plots"),
        help="Directory for output PNG files",
    )
    parser.add_argument(
        "--baseline",
        default="old",
        help="Variant name used as baseline for speedup chart",
    )
    parser.add_argument(
        "--title-prefix",
        default="SOMA screener benchmark",
        help="Title prefix used in generated plots",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.input:
        rows = _load_csv(args.input)
        data_source = args.input
    else:
        rows = SAMPLE_ROWS
        data_source = "embedded sample data"

    if not rows:
        print("No rows to plot.", file=sys.stderr)
        return 1

    os.makedirs(args.output_dir, exist_ok=True)
    plt = _require_matplotlib()

    execution_png = os.path.join(args.output_dir, "execution_ms_linear.png")
    execution_log_png = os.path.join(args.output_dir, "execution_ms_log.png")
    planning_png = os.path.join(args.output_dir, "planning_ms_linear.png")
    speedup_png = os.path.join(args.output_dir, f"speedup_vs_{args.baseline}.png")

    plot_grouped_metric(
        plt=plt,
        rows=rows,
        metric="execution_ms",
        output_path=execution_png,
        title=f"{args.title_prefix}: execution time (linear)",
        log_scale=False,
    )
    plot_grouped_metric(
        plt=plt,
        rows=rows,
        metric="execution_ms",
        output_path=execution_log_png,
        title=f"{args.title_prefix}: execution time (log scale)",
        log_scale=True,
    )
    plot_grouped_metric(
        plt=plt,
        rows=rows,
        metric="planning_ms",
        output_path=planning_png,
        title=f"{args.title_prefix}: planning time (linear)",
        log_scale=False,
    )
    plot_speedup(
        plt=plt,
        rows=rows,
        baseline=args.baseline,
        output_path=speedup_png,
        title=f"{args.title_prefix}: speedup vs {args.baseline}",
    )

    print(f"Data source: {data_source}")
    print(f"Saved: {execution_png}")
    print(f"Saved: {execution_log_png}")
    print(f"Saved: {planning_png}")
    print(f"Saved: {speedup_png}")
    print_summary(rows, baseline=args.baseline)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
