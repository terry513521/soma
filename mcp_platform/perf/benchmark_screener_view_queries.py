#!/usr/bin/env python3
"""
Benchmark screener eligibility queries used in app/db/interfaces.

This benchmark is legacy-view only (v_miner_screener_eligible_ranked).
It measures:
1) Raw SQL query timings for each query shape used by interfaces.
2) End-to-end interface call timings.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Awaitable, Callable


# Make local repos importable when running directly.
THIS_FILE = Path(__file__).resolve()
MCP_ROOT = THIS_FILE.parents[1]  # /root/SOMA/mcp_platform
SOMA_ROOT = MCP_ROOT.parent      # /root/SOMA
SHARED_ROOT = SOMA_ROOT.parent / "SOMA-shared"  # /root/SOMA-shared
if str(MCP_ROOT) not in sys.path:
    sys.path.insert(0, str(MCP_ROOT))
if SHARED_ROOT.exists() and str(SHARED_ROOT) not in sys.path:
    sys.path.insert(0, str(SHARED_ROOT))

DEFAULT_ENV_PATH = MCP_ROOT / ".env"
try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None
if load_dotenv is not None and DEFAULT_ENV_PATH.exists():
    load_dotenv(dotenv_path=DEFAULT_ENV_PATH, override=True)

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import settings
from app.db.interfaces import (
    fetch_top_screener_miner_ids_for_competition,
    fetch_top_screener_ss58_for_competition,
    get_screener_total_eligible_for_competition,
    get_screener_total_eligible_limit1_for_competition,
)


@dataclass(frozen=True)
class CaseSpec:
    name: str
    description: str
    runner: Callable[[AsyncSession, int, float, int], Awaitable[tuple[int, str]]]


@dataclass
class CaseRun:
    case_name: str
    iteration: int
    status: str
    wall_ms: float
    row_count: int
    first_value: str
    error: str


async def _sql_total_eligible_count(
    db: AsyncSession,
    competition_id: int,
    _top_fraction: float,
    _top_limit: int,
) -> tuple[int, str]:
    result = await db.execute(
        text(
            "SELECT COUNT(*) AS total_eligible "
            "FROM v_miner_screener_eligible_ranked "
            "WHERE competition_id = :competition_id"
        ),
        {"competition_id": competition_id},
    )
    rows = result.fetchall()
    first_val = str(rows[0][0]) if rows and rows[0][0] is not None else ""
    return len(rows), first_val


async def _sql_total_eligible_limit1(
    db: AsyncSession,
    competition_id: int,
    _top_fraction: float,
    _top_limit: int,
) -> tuple[int, str]:
    result = await db.execute(
        text(
            "SELECT total_eligible "
            "FROM v_miner_screener_eligible_ranked "
            "WHERE competition_id = :competition_id "
            "LIMIT 1"
        ),
        {"competition_id": competition_id},
    )
    rows = result.fetchall()
    first_val = str(rows[0][0]) if rows and rows[0][0] is not None else ""
    return len(rows), first_val


async def _sql_top_miner_ids(
    db: AsyncSession,
    competition_id: int,
    _top_fraction: float,
    top_limit: int,
) -> tuple[int, str]:
    result = await db.execute(
        text(
            "SELECT miner_id "
            "FROM v_miner_screener_eligible_ranked "
            "WHERE competition_id = :competition_id "
            "  AND rank <= :top_limit "
            "ORDER BY rank ASC"
        ),
        {"competition_id": competition_id, "top_limit": top_limit},
    )
    rows = result.fetchall()
    first_val = str(rows[0][0]) if rows and rows[0][0] is not None else ""
    return len(rows), first_val


async def _sql_top_ss58(
    db: AsyncSession,
    competition_id: int,
    _top_fraction: float,
    top_limit: int,
) -> tuple[int, str]:
    result = await db.execute(
        text(
            "SELECT m.ss58 "
            "FROM v_miner_screener_eligible_ranked r "
            "JOIN miners m ON m.id = r.miner_id "
            "WHERE r.competition_id = :competition_id "
            "  AND r.rank <= :top_limit "
            "  AND m.miner_banned_status IS FALSE "
            "ORDER BY r.rank ASC"
        ),
        {"competition_id": competition_id, "top_limit": top_limit},
    )
    rows = result.fetchall()
    first_val = str(rows[0][0]) if rows and rows[0][0] is not None else ""
    return len(rows), first_val


async def _flow_top_miner_ids(
    db: AsyncSession,
    competition_id: int,
    top_fraction: float,
    _top_limit: int,
) -> tuple[int, str]:
    miner_ids, _total, _limit = await fetch_top_screener_miner_ids_for_competition(
        db,
        competition_id=competition_id,
        top_screener_scripts=top_fraction,
    )
    first_val = str(miner_ids[0]) if miner_ids else ""
    return len(miner_ids), first_val


async def _flow_top_ss58(
    db: AsyncSession,
    competition_id: int,
    top_fraction: float,
    _top_limit: int,
) -> tuple[int, str]:
    ss58_list, _total, _limit = await fetch_top_screener_ss58_for_competition(
        db,
        competition_id=competition_id,
        top_screener_scripts=top_fraction,
    )
    first_val = str(ss58_list[0]) if ss58_list else ""
    return len(ss58_list), first_val


def _case_specs() -> dict[str, CaseSpec]:
    return {
        "sql_total_eligible_count": CaseSpec(
            name="sql_total_eligible_count",
            description="Raw SQL COUNT(*) on legacy screener view",
            runner=_sql_total_eligible_count,
        ),
        "sql_total_eligible_limit1": CaseSpec(
            name="sql_total_eligible_limit1",
            description="Raw SQL total_eligible LIMIT 1 on legacy view",
            runner=_sql_total_eligible_limit1,
        ),
        "sql_top_miner_ids": CaseSpec(
            name="sql_top_miner_ids",
            description="Raw SQL top miner ids by rank <= top_limit",
            runner=_sql_top_miner_ids,
        ),
        "sql_top_ss58": CaseSpec(
            name="sql_top_ss58",
            description="Raw SQL top ss58 by rank <= top_limit",
            runner=_sql_top_ss58,
        ),
        "flow_fetch_top_miner_ids": CaseSpec(
            name="flow_fetch_top_miner_ids",
            description="Interface flow: fetch_top_screener_miner_ids_for_competition",
            runner=_flow_top_miner_ids,
        ),
        "flow_fetch_top_ss58": CaseSpec(
            name="flow_fetch_top_ss58",
            description="Interface flow: fetch_top_screener_ss58_for_competition",
            runner=_flow_top_ss58,
        ),
    }


def _safe_p95(values: list[float]) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    idx = max(0, min(len(sorted_values) - 1, math.ceil(0.95 * len(sorted_values)) - 1))
    return float(sorted_values[idx])


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required. Install with: python -m pip install matplotlib"
        ) from exc
    return plt


def _parse_case_list(raw: str) -> list[str]:
    names = [x.strip() for x in raw.split(",") if x.strip()]
    if not names:
        raise ValueError("No cases provided.")
    available = _case_specs()
    unknown = [name for name in names if name not in available]
    if unknown:
        raise ValueError(
            f"Unknown case names: {', '.join(unknown)}. "
            f"Available: {', '.join(sorted(available))}"
        )
    return names


async def _resolve_top_limit(
    session_maker: async_sessionmaker,
    *,
    competition_id: int,
    top_fraction: float,
) -> tuple[int, int]:
    async with session_maker() as db:
        total = await get_screener_total_eligible_for_competition(
            db,
            competition_id=competition_id,
        )
        total_limit1 = await get_screener_total_eligible_limit1_for_competition(
            db,
            competition_id=competition_id,
        )
    total_eligible = max(total, total_limit1)
    top_limit = (
        int(math.ceil(total_eligible * top_fraction))
        if total_eligible > 0 and top_fraction > 0
        else 0
    )
    return total_eligible, top_limit


async def _run_one_case(
    *,
    session_maker: async_sessionmaker,
    spec: CaseSpec,
    iteration: int,
    competition_id: int,
    top_fraction: float,
    top_limit: int,
) -> CaseRun:
    async with session_maker() as db:
        status = "ok"
        err = ""
        started = time.perf_counter()
        row_count = 0
        first_value = ""
        try:
            row_count, first_value = await spec.runner(
                db,
                competition_id,
                top_fraction,
                top_limit,
            )
        except Exception as exc:
            status = "error"
            err = repr(exc)
        wall_ms = (time.perf_counter() - started) * 1000.0

    return CaseRun(
        case_name=spec.name,
        iteration=iteration,
        status=status,
        wall_ms=wall_ms,
        row_count=row_count,
        first_value=first_value,
        error=err,
    )


def _write_raw_csv(path: Path, rows: list[CaseRun]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "case_name",
                "iteration",
                "status",
                "wall_ms",
                "row_count",
                "first_value",
                "error",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.case_name,
                    row.iteration,
                    row.status,
                    f"{row.wall_ms:.6f}",
                    row.row_count,
                    row.first_value,
                    row.error,
                ]
            )


def _summarize(rows: list[CaseRun]) -> list[dict[str, Any]]:
    grouped: dict[str, list[CaseRun]] = {}
    for row in rows:
        grouped.setdefault(row.case_name, []).append(row)

    summary: list[dict[str, Any]] = []
    for case_name, case_rows in sorted(grouped.items()):
        ms_values = [r.wall_ms for r in case_rows]
        row_counts = [float(r.row_count) for r in case_rows]
        summary.append(
            {
                "case_name": case_name,
                "runs": len(case_rows),
                "ok_runs": sum(1 for r in case_rows if r.status == "ok"),
                "wall_ms_mean": mean(ms_values) if ms_values else 0.0,
                "wall_ms_p95": _safe_p95(ms_values),
                "row_count_mean": mean(row_counts) if row_counts else 0.0,
                "sample_first_value": next(
                    (r.first_value for r in case_rows if r.first_value),
                    "",
                ),
            }
        )
    return summary


def _write_summary_csv(path: Path, summary_rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "case_name",
                "runs",
                "ok_runs",
                "wall_ms_mean",
                "wall_ms_p95",
                "row_count_mean",
                "sample_first_value",
            ]
        )
        for row in summary_rows:
            writer.writerow(
                [
                    row["case_name"],
                    row["runs"],
                    row["ok_runs"],
                    f"{row['wall_ms_mean']:.6f}",
                    f"{row['wall_ms_p95']:.6f}",
                    f"{row['row_count_mean']:.6f}",
                    row["sample_first_value"],
                ]
            )


def _plot(summary_rows: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    plt = _require_matplotlib()
    output_dir.mkdir(parents=True, exist_ok=True)

    names = [row["case_name"] for row in summary_rows]
    mean_values = [row["wall_ms_mean"] for row in summary_rows]
    p95_values = [row["wall_ms_p95"] for row in summary_rows]
    x = list(range(len(names)))
    width = 0.35

    fig1, ax1 = plt.subplots(figsize=(13, 6))
    ax1.bar([i - width / 2 for i in x], mean_values, width=width, label="mean")
    ax1.bar([i + width / 2 for i in x], p95_values, width=width, label="p95")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=20, ha="right")
    ax1.set_ylabel("Time (ms)")
    ax1.set_title("Legacy Screener Query Timings")
    ax1.grid(axis="y", alpha=0.3)
    ax1.legend()
    fig1.tight_layout()
    p1 = output_dir / "screener_queries_time_mean_p95.png"
    fig1.savefig(p1, dpi=150)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(13, 6))
    ax2.bar(x, mean_values, width=0.55)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=20, ha="right")
    ax2.set_ylabel("Mean time (ms)")
    ax2.set_title("Legacy Screener Query Mean Time")
    ax2.grid(axis="y", alpha=0.3)
    fig2.tight_layout()
    p2 = output_dir / "screener_queries_time_mean.png"
    fig2.savefig(p2, dpi=150)
    plt.close(fig2)

    return [p1, p2]


async def _run(args: argparse.Namespace) -> int:
    dsn = settings.get_postgres_dsn()
    if not dsn:
        raise RuntimeError(
            "POSTGRES_DSN could not be resolved from environment/settings "
            "(set POSTGRES_DSN or RDS_SECRET_ID in mcp_platform/.env)."
        )

    engine = create_async_engine(dsn, pool_pre_ping=True)
    session_maker = async_sessionmaker(bind=engine, expire_on_commit=False)

    selected_case_names = _parse_case_list(args.cases)
    all_specs = _case_specs()
    selected_specs = [all_specs[name] for name in selected_case_names]

    total_eligible, top_limit = await _resolve_top_limit(
        session_maker,
        competition_id=args.competition_id,
        top_fraction=args.top_fraction,
    )

    rows: list[CaseRun] = []
    total_loops = args.warmup + args.iterations
    for i in range(total_loops):
        for spec in selected_specs:
            run_row = await _run_one_case(
                session_maker=session_maker,
                spec=spec,
                iteration=i + 1,
                competition_id=args.competition_id,
                top_fraction=args.top_fraction,
                top_limit=top_limit,
            )
            if i >= args.warmup:
                rows.append(run_row)

    await engine.dispose()

    output_dir = Path(args.output_dir).resolve()
    raw_csv = output_dir / "screener_view_queries_benchmark_raw.csv"
    summary_csv = output_dir / "screener_view_queries_benchmark_summary.csv"
    _write_raw_csv(raw_csv, rows)
    summary_rows = _summarize(rows)
    _write_summary_csv(summary_csv, summary_rows)
    plot_files = _plot(summary_rows, output_dir=output_dir)

    print(f"competition_id: {args.competition_id}")
    print(f"top_fraction: {args.top_fraction}")
    print(f"total_eligible: {total_eligible}")
    print(f"top_limit: {top_limit}")
    print(f"iterations (recorded): {args.iterations}")
    print(f"cases: {', '.join(selected_case_names)}")
    print(f"raw results: {raw_csv}")
    print(f"summary: {summary_csv}")
    for path in plot_files:
        print(f"plot: {path}")
    return 0


def _parse_args() -> argparse.Namespace:
    available_case_names = ",".join(_case_specs().keys())
    parser = argparse.ArgumentParser(
        description="Benchmark legacy screener queries used by interface functions.",
    )
    parser.add_argument(
        "--competition-id",
        type=int,
        default=40,
        help="Competition id for benchmarked queries (default: 40).",
    )
    parser.add_argument(
        "--top-fraction",
        type=float,
        default=0.2,
        help="Top fraction used for top_screener queries (default: 0.2).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=20,
        help="Recorded iterations per case (default: 20).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Warmup iterations (not recorded) (default: 5).",
    )
    parser.add_argument(
        "--cases",
        type=str,
        default=available_case_names,
        help=(
            "Comma-separated case names. Available: "
            f"{available_case_names}"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(MCP_ROOT / "perf" / "plots" / "screener_view_queries_benchmark"),
        help="Output directory for CSV and plots.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
