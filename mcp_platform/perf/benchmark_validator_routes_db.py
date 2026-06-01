#!/usr/bin/env python3
"""
Benchmark validator route DB time before vs after screener view changes.

This benchmark compares two variants:
1) before_legacy_view
   Uses legacy screener selection query via v_miner_screener_eligible_ranked.
2) after_small_views
   Uses current app.db.interfaces.screener_eligibility implementation.

Important:
- Tests are pinned to a specific competition_id (default: 40).
- Results include both end-to-end route wall time and SQL-only DB time.
- Plots are generated automatically.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import importlib
import math
import os
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from types import SimpleNamespace
from typing import Any
import types


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
    # Override existing shell vars (e.g. DEBUG=release) with project env values.
    load_dotenv(dotenv_path=DEFAULT_ENV_PATH, override=True)

from fastapi import HTTPException
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core.config import settings
from app.db.interfaces.screener_eligibility import (
    fetch_top_screener_ss58_for_competition as new_fetch_top_screener_ss58_for_competition,
)
from app.db.views import V_MINER_SCREENER_ELIGIBLE_RANKED
from soma_shared.contracts.common.signatures import Signature, SignedEnvelope
from soma_shared.contracts.validator.v1.messages import (
    GetBestMinersUidRequest,
    GetChallengesRequest,
    PostChallengeScores,
    ScoreSubmissionType,
    ValidatorRegisterRequest,
)
from soma_shared.db.metrics import DatabaseMetricsCollector
from soma_shared.db.models.miner import Miner
from soma_shared.db.models.validator import Validator

# Avoid importing app.api.routes.__init__ (which imports frontend deps) by
# creating a lightweight package stub for direct submodule imports.
if "app.api.routes" not in sys.modules:
    routes_pkg = types.ModuleType("app.api.routes")
    routes_pkg.__path__ = [str(MCP_ROOT / "app" / "api" / "routes")]
    sys.modules["app.api.routes"] = routes_pkg

# validator.py imports bittensor but benchmark paths below do not use it.
if "bittensor" not in sys.modules:
    sys.modules["bittensor"] = types.ModuleType("bittensor")

validator_routes = importlib.import_module("app.api.routes.validator")


VARIANT_BEFORE = "before_legacy_view"
VARIANT_AFTER = "after_small_views"
VARIANTS = (VARIANT_BEFORE, VARIANT_AFTER)
ROUTES = (
    "register",
    "request_challenge",
    "score_challenges",
    "get_best_miners",
)


def _fake_sign_payload_model(
    payload: Any,
    *,
    nonce: str,
    use_coldkey: bool = False,
    verbose: bool = False,
    wallet: Any = None,
    keypair: Any = None,
) -> Signature:
    del payload, use_coldkey, verbose, wallet, keypair
    signer = os.getenv("WALLET_HOTKEY") or "benchmark-validator-signer"
    return Signature(
        signer_ss58=signer,
        nonce=nonce,
        signature="benchmark-signature",
    )


@dataclass
class RouteResult:
    route: str
    variant: str
    iteration: int
    status: str
    wall_ms: float
    db_total_ms: float
    db_queries: int
    db_avg_ms: float
    db_errors: int
    error: str


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


def _signature(hotkey: str) -> Signature:
    return Signature(
        signer_ss58=hotkey,
        nonce=uuid.uuid4().hex,
        signature="benchmark-signature",
    )


def _mk_request(
    *,
    path: str,
    method: str,
    app_state: SimpleNamespace,
    request_id: str | None = None,
) -> Any:
    return SimpleNamespace(
        url=SimpleNamespace(path=path),
        method=method,
        state=SimpleNamespace(request_id=request_id or uuid.uuid4().hex),
        app=SimpleNamespace(state=app_state),
        headers={},
        client=None,
    )


async def _legacy_fetch_top_screener_ss58_for_competition(
    db,
    *,
    competition_id: int,
    top_screener_scripts: float,
) -> tuple[list[str], int, int]:
    total_eligible_raw = await db.scalar(
        select(func.count())
        .select_from(V_MINER_SCREENER_ELIGIBLE_RANKED)
        .where(V_MINER_SCREENER_ELIGIBLE_RANKED.c.competition_id == competition_id)
    )
    total_eligible = int(total_eligible_raw or 0)
    top_limit = (
        int(math.ceil(total_eligible * top_screener_scripts))
        if total_eligible > 0 and top_screener_scripts > 0
        else 0
    )
    if top_limit <= 0:
        return [], total_eligible, top_limit

    rows = await db.execute(
        select(Miner.ss58)
        .select_from(V_MINER_SCREENER_ELIGIBLE_RANKED)
        .join(Miner, Miner.id == V_MINER_SCREENER_ELIGIBLE_RANKED.c.miner_id)
        .where(V_MINER_SCREENER_ELIGIBLE_RANKED.c.competition_id == competition_id)
        .where(V_MINER_SCREENER_ELIGIBLE_RANKED.c.rank <= top_limit)
        .where(Miner.miner_banned_status.is_(False))
        .order_by(V_MINER_SCREENER_ELIGIBLE_RANKED.c.rank.asc())
    )
    ss58_list = [str(row.ss58) for row in rows if row.ss58]
    return ss58_list, total_eligible, top_limit


@contextmanager
def _patch_variant(variant: str, competition_id: int):
    original_fetch = validator_routes.fetch_top_screener_ss58_for_competition
    original_get_active_competition_id = validator_routes._get_active_competition_id

    async def _fixed_competition_id(_db) -> int:
        return int(competition_id)

    if variant == VARIANT_BEFORE:
        validator_routes.fetch_top_screener_ss58_for_competition = (
            _legacy_fetch_top_screener_ss58_for_competition
        )
    elif variant == VARIANT_AFTER:
        validator_routes.fetch_top_screener_ss58_for_competition = (
            new_fetch_top_screener_ss58_for_competition
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")

    # Force benchmark to use competition_id=40 (or provided value).
    validator_routes._get_active_competition_id = _fixed_competition_id
    try:
        yield
    finally:
        validator_routes.fetch_top_screener_ss58_for_competition = original_fetch
        validator_routes._get_active_competition_id = original_get_active_competition_id


async def _fetch_metagraph_snapshot(session_maker, limit: int) -> dict[str, list[int | str]]:
    async with session_maker() as db:
        rows = (
            await db.execute(
                select(Miner.ss58)
                .where(Miner.miner_banned_status.is_(False))
                .order_by(Miner.id.asc())
                .limit(limit)
            )
        ).scalars().all()

    hotkeys = [str(ss58) for ss58 in rows if ss58]
    uids = list(range(len(hotkeys)))
    return {"hotkeys": hotkeys, "uids": uids}


async def _ensure_validator_exists(
    session_maker,
    *,
    hotkey: str,
    app_state: SimpleNamespace,
) -> None:
    async with session_maker() as db:
        request = _mk_request(
            path="/validator/register",
            method="POST",
            app_state=app_state,
            request_id=f"seed-register-{uuid.uuid4().hex}",
        )
        envelope = SignedEnvelope(
            payload=ValidatorRegisterRequest(
                validator_hotkey=hotkey,
                serving_ip="1.1.1.1",
                serving_port=8091,
            ),
            sig=_signature(hotkey),
        )
        try:
            await validator_routes.register(
                request=request,
                _req=envelope,
                db=db,
                _stake_check=None,
            )
        except HTTPException:
            # If this fails due existing state constraints, validator may still exist.
            pass
        await db.commit()


async def _invoke_route(
    *,
    route: str,
    db,
    app_state: SimpleNamespace,
    benchmark_hotkey: str,
) -> None:
    if route == "register":
        request = _mk_request(
            path="/validator/register",
            method="POST",
            app_state=app_state,
        )
        envelope = SignedEnvelope(
            payload=ValidatorRegisterRequest(
                validator_hotkey=benchmark_hotkey,
                serving_ip="1.1.1.1",
                serving_port=8091,
            ),
            sig=_signature(benchmark_hotkey),
        )
        await validator_routes.register(
            request=request,
            _req=envelope,
            db=db,
            _stake_check=None,
        )
        return

    if route == "request_challenge":
        # Keep this route DB-only by forcing early validator-status rejection.
        await db.execute(
            update(Validator)
            .where(Validator.ss58 == benchmark_hotkey)
            .values(current_status="paused", is_archive=False)
        )
        await db.commit()
        request = _mk_request(
            path="/validator/request_challenge",
            method="POST",
            app_state=app_state,
        )
        envelope = SignedEnvelope(
            payload=GetChallengesRequest(),
            sig=_signature(benchmark_hotkey),
        )
        await validator_routes.request_challenge(
            request=request,
            _req=envelope,
            db=db,
            _stake_check=None,
        )
        return

    if route == "score_challenges":
        request = _mk_request(
            path="/validator/score_challenges",
            method="POST",
            app_state=app_state,
        )
        envelope = SignedEnvelope(
            payload=PostChallengeScores(
                batch_id="-999999999",
                submission_type=ScoreSubmissionType.ERROR,
                error_code="provider_timeout",
                error_message="benchmark synthetic error",
                retryable=True,
            ),
            sig=_signature(benchmark_hotkey),
        )
        await validator_routes.score_challenges(
            request=request,
            _req=envelope,
            db=db,
            _stake_check=None,
        )
        return

    if route == "get_best_miners":
        request = _mk_request(
            path="/validator/get_best_miners",
            method="POST",
            app_state=app_state,
        )
        envelope = SignedEnvelope(
            payload=GetBestMinersUidRequest(),
            sig=_signature(benchmark_hotkey),
        )
        await validator_routes.get_best_miners(
            request=request,
            _req=envelope,
            db=db,
        )
        return

    raise ValueError(f"Unknown route: {route}")


async def _run_one_case(
    *,
    route: str,
    variant: str,
    iteration: int,
    session_maker,
    collector: DatabaseMetricsCollector,
    app_state: SimpleNamespace,
    benchmark_hotkey: str,
    competition_id: int,
) -> RouteResult:
    async with session_maker() as db:
        token = collector.begin_request_scope()
        status = "ok"
        err = ""
        started = time.perf_counter()
        try:
            with _patch_variant(variant, competition_id):
                await _invoke_route(
                    route=route,
                    db=db,
                    app_state=app_state,
                    benchmark_hotkey=benchmark_hotkey,
                )
        except HTTPException as exc:
            status = f"http_{exc.status_code}"
        except Exception as exc:  # pragma: no cover - benchmark safety
            status = "error"
            err = repr(exc)
        wall_ms = (time.perf_counter() - started) * 1000.0
        snapshot = collector.current_request_snapshot()
        collector.end_request_scope(token)

    if snapshot is None:
        return RouteResult(
            route=route,
            variant=variant,
            iteration=iteration,
            status=status,
            wall_ms=wall_ms,
            db_total_ms=0.0,
            db_queries=0,
            db_avg_ms=0.0,
            db_errors=0,
            error=err,
        )
    return RouteResult(
        route=route,
        variant=variant,
        iteration=iteration,
        status=status,
        wall_ms=wall_ms,
        db_total_ms=float(snapshot.total_duration_ms),
        db_queries=int(snapshot.total_queries),
        db_avg_ms=float(snapshot.avg_duration_ms),
        db_errors=int(snapshot.total_errors),
        error=err,
    )


def _write_raw_csv(path: Path, rows: list[RouteResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "route",
                "variant",
                "iteration",
                "status",
                "wall_ms",
                "db_total_ms",
                "db_queries",
                "db_avg_ms",
                "db_errors",
                "error",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.route,
                    row.variant,
                    row.iteration,
                    row.status,
                    f"{row.wall_ms:.6f}",
                    f"{row.db_total_ms:.6f}",
                    row.db_queries,
                    f"{row.db_avg_ms:.6f}",
                    row.db_errors,
                    row.error,
                ]
            )


def _summarize(rows: list[RouteResult]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[RouteResult]] = {}
    for row in rows:
        grouped.setdefault((row.route, row.variant), []).append(row)

    summary: list[dict[str, Any]] = []
    for (route, variant), group_rows in sorted(grouped.items()):
        wall_values = [r.wall_ms for r in group_rows]
        db_values = [r.db_total_ms for r in group_rows]
        query_values = [float(r.db_queries) for r in group_rows]
        summary.append(
            {
                "route": route,
                "variant": variant,
                "runs": len(group_rows),
                "ok_runs": sum(1 for r in group_rows if r.status == "ok"),
                "wall_ms_mean": mean(wall_values) if wall_values else 0.0,
                "wall_ms_p95": _safe_p95(wall_values),
                "db_total_ms_mean": mean(db_values) if db_values else 0.0,
                "db_total_ms_p95": _safe_p95(db_values),
                "db_queries_mean": mean(query_values) if query_values else 0.0,
            }
        )
    return summary


def _write_summary_csv(path: Path, summary_rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "route",
                "variant",
                "runs",
                "ok_runs",
                "wall_ms_mean",
                "wall_ms_p95",
                "db_total_ms_mean",
                "db_total_ms_p95",
                "db_queries_mean",
            ]
        )
        for row in summary_rows:
            writer.writerow(
                [
                    row["route"],
                    row["variant"],
                    row["runs"],
                    row["ok_runs"],
                    f"{row['wall_ms_mean']:.6f}",
                    f"{row['wall_ms_p95']:.6f}",
                    f"{row['db_total_ms_mean']:.6f}",
                    f"{row['db_total_ms_p95']:.6f}",
                    f"{row['db_queries_mean']:.6f}",
                ]
            )


def _plot(summary_rows: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    plt = _require_matplotlib()
    output_dir.mkdir(parents=True, exist_ok=True)

    routes = sorted({row["route"] for row in summary_rows})
    before = {row["route"]: row for row in summary_rows if row["variant"] == VARIANT_BEFORE}
    after = {row["route"]: row for row in summary_rows if row["variant"] == VARIANT_AFTER}

    x = list(range(len(routes)))
    width = 0.35

    # 1) DB total time bars
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    before_db = [before.get(r, {}).get("db_total_ms_mean", float("nan")) for r in routes]
    after_db = [after.get(r, {}).get("db_total_ms_mean", float("nan")) for r in routes]
    ax1.bar([i - width / 2 for i in x], before_db, width=width, label=VARIANT_BEFORE)
    ax1.bar([i + width / 2 for i in x], after_db, width=width, label=VARIANT_AFTER)
    ax1.set_xticks(x)
    ax1.set_xticklabels(routes, rotation=15)
    ax1.set_ylabel("DB time mean (ms)")
    ax1.set_title("Validator Route DB Time: before vs after")
    ax1.grid(axis="y", alpha=0.3)
    ax1.legend()
    fig1.tight_layout()
    db_plot = output_dir / "validator_routes_db_time_mean.png"
    fig1.savefig(db_plot, dpi=150)
    plt.close(fig1)

    # 2) Wall time bars
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    before_wall = [before.get(r, {}).get("wall_ms_mean", float("nan")) for r in routes]
    after_wall = [after.get(r, {}).get("wall_ms_mean", float("nan")) for r in routes]
    ax2.bar([i - width / 2 for i in x], before_wall, width=width, label=VARIANT_BEFORE)
    ax2.bar([i + width / 2 for i in x], after_wall, width=width, label=VARIANT_AFTER)
    ax2.set_xticks(x)
    ax2.set_xticklabels(routes, rotation=15)
    ax2.set_ylabel("Wall time mean (ms)")
    ax2.set_title("Validator Route Wall Time: before vs after")
    ax2.grid(axis="y", alpha=0.3)
    ax2.legend()
    fig2.tight_layout()
    wall_plot = output_dir / "validator_routes_wall_time_mean.png"
    fig2.savefig(wall_plot, dpi=150)
    plt.close(fig2)

    # 3) Speedup bars (before / after)
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    db_speedups: list[float] = []
    wall_speedups: list[float] = []
    for route in routes:
        b_db = before.get(route, {}).get("db_total_ms_mean", 0.0)
        a_db = after.get(route, {}).get("db_total_ms_mean", 0.0)
        b_wall = before.get(route, {}).get("wall_ms_mean", 0.0)
        a_wall = after.get(route, {}).get("wall_ms_mean", 0.0)
        db_speedups.append((b_db / a_db) if a_db > 0 else float("nan"))
        wall_speedups.append((b_wall / a_wall) if a_wall > 0 else float("nan"))
    ax3.bar([i - width / 2 for i in x], db_speedups, width=width, label="DB speedup (x)")
    ax3.bar([i + width / 2 for i in x], wall_speedups, width=width, label="Wall speedup (x)")
    ax3.set_xticks(x)
    ax3.set_xticklabels(routes, rotation=15)
    ax3.set_ylabel("Speedup (before / after)")
    ax3.set_title("Validator Route Speedup")
    ax3.grid(axis="y", alpha=0.3)
    ax3.legend()
    fig3.tight_layout()
    speedup_plot = output_dir / "validator_routes_speedup.png"
    fig3.savefig(speedup_plot, dpi=150)
    plt.close(fig3)

    return [db_plot, wall_plot, speedup_plot]


async def _run(args: argparse.Namespace) -> int:
    dsn = settings.get_postgres_dsn()
    if not dsn:
        raise RuntimeError(
            "POSTGRES_DSN could not be resolved from environment/settings "
            "(set POSTGRES_DSN or RDS_SECRET_ID in mcp_platform/.env)."
        )

    engine = create_async_engine(
        dsn,
        pool_pre_ping=True,
    )
    session_maker = async_sessionmaker(bind=engine, expire_on_commit=False)
    collector = DatabaseMetricsCollector(
        slow_query_threshold_seconds=9999.0,  # keep benchmark output clean
        log_slow_queries=False,
    )
    collector.install(engine)

    benchmark_hotkey = args.validator_hotkey

    # Build request app-state shared by all calls.
    snapshot = await _fetch_metagraph_snapshot(session_maker, args.metagraph_limit)
    app_state = SimpleNamespace(
        registered_validators={},
        validator_fetch_block_until={},
        metagraph_service=SimpleNamespace(latest_snapshot=snapshot),
    )

    # Prevent wallet/bittensor signing dependencies from affecting DB benchmark.
    # validator.py evaluates `settings.wallet` when building responses; pre-fill
    # a lightweight fake wallet object and fake signer.
    try:
        settings._wallet = SimpleNamespace(
            hotkey=SimpleNamespace(ss58_address=benchmark_hotkey)
        )
    except Exception:
        pass
    validator_routes.sign_payload_model = _fake_sign_payload_model

    # Ensure benchmark validator exists.
    await _ensure_validator_exists(
        session_maker,
        hotkey=benchmark_hotkey,
        app_state=app_state,
    )

    rows: list[RouteResult] = []
    routes = [r.strip() for r in args.routes.split(",") if r.strip()]
    for route in routes:
        if route not in ROUTES:
            raise ValueError(f"Unknown route '{route}'. Valid routes: {', '.join(ROUTES)}")

    for i in range(args.iterations + args.warmup):
        variants = list(VARIANTS if i % 2 == 0 else tuple(reversed(VARIANTS)))
        for route in routes:
            for variant in variants:
                result = await _run_one_case(
                    route=route,
                    variant=variant,
                    iteration=i + 1,
                    session_maker=session_maker,
                    collector=collector,
                    app_state=app_state,
                    benchmark_hotkey=benchmark_hotkey,
                    competition_id=args.competition_id,
                )
                if i >= args.warmup:
                    rows.append(result)

    await engine.dispose()

    output_dir = Path(args.output_dir).resolve()
    raw_csv = output_dir / "validator_route_db_benchmark_raw.csv"
    summary_csv = output_dir / "validator_route_db_benchmark_summary.csv"
    _write_raw_csv(raw_csv, rows)
    summary_rows = _summarize(rows)
    _write_summary_csv(summary_csv, summary_rows)
    plot_files = _plot(summary_rows, output_dir=output_dir)

    print(f"competition_id: {args.competition_id}")
    print(f"iterations (recorded): {args.iterations}")
    print(f"routes: {', '.join(routes)}")
    print(f"raw results: {raw_csv}")
    print(f"summary: {summary_csv}")
    for p in plot_files:
        print(f"plot: {p}")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark validator.py route DB calls before/after screener view changes.",
    )
    parser.add_argument(
        "--competition-id",
        type=int,
        default=40,
        help="Competition id used for benchmarked screener logic (default: 40).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=8,
        help="Recorded iterations per route/variant (default: 8).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Warmup iterations (not recorded) (default: 2).",
    )
    parser.add_argument(
        "--routes",
        type=str,
        default="register,request_challenge,score_challenges,get_best_miners",
        help="Comma-separated routes to benchmark.",
    )
    parser.add_argument(
        "--metagraph-limit",
        type=int,
        default=2048,
        help="Max miners loaded into synthetic metagraph snapshot (default: 2048).",
    )
    parser.add_argument(
        "--validator-hotkey",
        type=str,
        default="5BenchmarkValidatorHotkey111111111111111111111111",
        help="Hotkey used for benchmark route calls.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str((MCP_ROOT / "perf" / "plots" / "validator_route_db_benchmark")),
        help="Output directory for CSV and plots.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
