from __future__ import annotations

import asyncio
import math
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logging import get_logger
from app.services.blob.patch_artifact_storage import PatchArtifactStorage
from app.services.blob.s3 import S3BlobStorage
from app.services.sandbox.remote_compact_bench_manager import RemoteCompactBenchManager
from soma_shared.db.models.miner import Miner
from soma_shared.db.models.miner_upload import MinerUpload
from soma_shared.db.models.script import Script
from soma_shared.db.models.swe_bench_run import SweBenchRun
from soma_shared.db.models.swe_bench_run_validation import SweBenchRunValidation
from soma_shared.db.models.swe_bench_task import SweBenchTask
from soma_shared.db.models.competition import Competition
from soma_shared.db.models.competition_config import CompetitionConfig
from soma_shared.db.models.competition_timeframe import CompetitionTimeframe
from soma_shared.db.session import get_db_session, get_engine


logger = get_logger(__name__)

_ORCHESTRATOR_LOCK_KEY = "swebench-orchestrator-v1"
_SEED_IDLE_LOG_INTERVAL_SECONDS = 60
_LAST_IDLE_SEED_LOG_AT: datetime | None = None


@dataclass(frozen=True)
class _ScriptRef:
    script_id: int
    miner_fk: int


def start_swebench_orchestrator_task(app) -> None:
    interval = max(0.5, float(settings.swebench_orchestrator_interval_seconds))
    task = asyncio.create_task(_run_orchestrator_loop(app, interval))
    app.state.swebench_orchestrator_task = task
    logger.info(
        "swebench_orchestrator_started",
        extra={
            "interval_seconds": interval,
            "dispatch_batch_size": int(settings.swebench_dispatch_batch_size),
        },
    )


async def stop_swebench_orchestrator_task(app) -> None:
    task = getattr(app.state, "swebench_orchestrator_task", None)
    if task is None:
        return
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    logger.info("swebench_orchestrator_stopped")


async def _run_orchestrator_loop(app, interval_seconds: float) -> None:
    try:
        while True:
            try:
                await _run_orchestration_tick(app)
            except Exception:
                logger.exception("swebench_orchestration_tick_failed")
            await asyncio.sleep(interval_seconds)
    except asyncio.CancelledError:
        logger.info("swebench_orchestrator_cancelled")


async def _run_orchestration_tick(app) -> None:
    lock_conn = None
    lock_acquired = False
    try:
        engine = get_engine()
        lock_conn = await engine.connect()
        if lock_conn.dialect.name == "postgresql":
            lock_acquired = bool(
                (
                    await lock_conn.execute(
                        text("SELECT pg_try_advisory_lock(hashtext(:lock_key))"),
                        {"lock_key": _ORCHESTRATOR_LOCK_KEY},
                    )
                ).scalar()
            )
            if not lock_acquired:
                return

        now = datetime.now(timezone.utc)

        async for db in get_db_session():
            active_competition_ids = await _get_active_competition_ids(db, now)
            seeded_runs = 0
            for competition_id in active_competition_ids:
                seeded_runs += await _seed_runs_for_competition(db, competition_id, now)
            await db.commit()
            _maybe_log_seed_pass(
                active_competitions=len(active_competition_ids),
                seeded_runs=seeded_runs,
                now=now,
            )
            break

        dispatched, deferred, failed = await _dispatch_due_runs(app, now)
        if dispatched or deferred or failed:
            logger.info(
                "swebench_orchestrator_dispatch_pass",
                extra={
                    "dispatched": dispatched,
                    "deferred": deferred,
                    "failed": failed,
                },
            )
    finally:
        if lock_conn is not None:
            try:
                if lock_acquired:
                    await lock_conn.execute(
                        text("SELECT pg_advisory_unlock(hashtext(:lock_key))"),
                        {"lock_key": _ORCHESTRATOR_LOCK_KEY},
                    )
            finally:
                await lock_conn.close()


def _maybe_log_seed_pass(*, active_competitions: int, seeded_runs: int, now: datetime) -> None:
    global _LAST_IDLE_SEED_LOG_AT

    if active_competitions > 0 or seeded_runs > 0:
        logger.info(
            "swebench_orchestrator_seed_pass",
            extra={
                "active_competitions": active_competitions,
                "seeded_runs": seeded_runs,
            },
        )
        _LAST_IDLE_SEED_LOG_AT = None
        return

    if _LAST_IDLE_SEED_LOG_AT is None:
        should_log_idle = True
    else:
        elapsed_seconds = (now - _LAST_IDLE_SEED_LOG_AT).total_seconds()
        should_log_idle = elapsed_seconds >= _SEED_IDLE_LOG_INTERVAL_SECONDS

    if should_log_idle:
        logger.info(
            "swebench_orchestrator_seed_pass_idle",
            extra={
                "active_competitions": active_competitions,
                "seeded_runs": seeded_runs,
                "interval_seconds": _SEED_IDLE_LOG_INTERVAL_SECONDS,
            },
        )
        _LAST_IDLE_SEED_LOG_AT = now


async def _get_active_competition_ids(db: AsyncSession, now: datetime) -> list[int]:
    rows = (
        await db.execute(
            select(Competition.id)
            .join(CompetitionConfig, CompetitionConfig.competition_fk == Competition.id)
            .join(
                CompetitionTimeframe,
                CompetitionTimeframe.competition_config_fk == CompetitionConfig.id,
            )
            .where(CompetitionConfig.is_active.is_(True))
            .where(CompetitionTimeframe.upload_starts_at <= now)
            .where(CompetitionTimeframe.eval_ends_at >= now)
        )
    ).all()
    return [int(row[0]) for row in rows]


async def _seed_runs_for_competition(
    db: AsyncSession,
    competition_id: int,
    now: datetime,
) -> int:
    tasks = (
        (
            await db.execute(
                select(SweBenchTask)
                .where(SweBenchTask.competition_fk == competition_id)
                .order_by(SweBenchTask.id.asc())
            )
        )
        .scalars()
        .all()
    )
    if not tasks:
        return 0

    task_repeats: dict[int, int] = {
        int(task.id): max(1, int(task.planned_repeats or 1)) for task in tasks
    }
    screener_task_ids = [int(task.id) for task in tasks if bool(task.is_screener)]

    created = 0
    created += await _seed_baseline_runs(
        db,
        tasks=tasks,
        task_repeats=task_repeats,
        now=now,
    )

    scripts = await _load_latest_scripts_for_competition(db, competition_id)
    for script in scripts:
        created += await _seed_script_runs(
            db,
            script=script,
            tasks=tasks,
            task_repeats=task_repeats,
            screener_task_ids=screener_task_ids,
            now=now,
        )

    return created


async def _load_latest_scripts_for_competition(
    db: AsyncSession,
    competition_id: int,
) -> list[_ScriptRef]:
    rows = (
        await db.execute(
            select(
                Script.id,
                Script.miner_fk,
                MinerUpload.created_at,
            )
            .join(MinerUpload, MinerUpload.script_fk == Script.id)
            .join(Miner, Miner.id == Script.miner_fk)
            .where(MinerUpload.competition_fk == competition_id)
            .where(Miner.miner_banned_status.is_(False))
            .order_by(MinerUpload.created_at.desc())
        )
    ).all()

    by_miner: dict[int, _ScriptRef] = {}
    for row in rows:
        script_id = int(row[0])
        miner_fk = int(row[1])
        if miner_fk in by_miner:
            continue
        by_miner[miner_fk] = _ScriptRef(script_id=script_id, miner_fk=miner_fk)
    return list(by_miner.values())


async def _seed_baseline_runs(
    db: AsyncSession,
    *,
    tasks: list[SweBenchTask],
    task_repeats: dict[int, int],
    now: datetime,
) -> int:
    task_ids = [int(task.id) for task in tasks]
    existing = set(
        (
            int(row[0]),
            int(row[1]),
        )
        for row in (
            await db.execute(
                select(SweBenchRun.task_fk, SweBenchRun.attempt_no)
                .where(SweBenchRun.baseline_run.is_(True))
                .where(SweBenchRun.miner_fk.is_(None))
                .where(SweBenchRun.script_fk.is_(None))
                .where(SweBenchRun.task_fk.in_(task_ids))
            )
        ).all()
    )

    created = 0
    for task in tasks:
        task_id = int(task.id)
        for attempt_no in range(1, task_repeats[task_id] + 1):
            key = (task_id, attempt_no)
            if key in existing:
                continue
            await _create_run_and_validation(
                db,
                task_fk=task_id,
                attempt_no=attempt_no,
                baseline_run=True,
                miner_fk=None,
                script_fk=None,
                now=now,
            )
            existing.add(key)
            created += 1
    return created


async def _seed_script_runs(
    db: AsyncSession,
    *,
    script: _ScriptRef,
    tasks: list[SweBenchTask],
    task_repeats: dict[int, int],
    screener_task_ids: list[int],
    now: datetime,
) -> int:
    created = 0

    if screener_task_ids:
        created += await _seed_script_task_subset(
            db,
            script=script,
            task_ids=screener_task_ids,
            task_repeats=task_repeats,
            now=now,
        )

    screening_complete, screening_passed = await _evaluate_screening_for_script(
        db,
        script=script,
        screener_task_ids=screener_task_ids,
        task_repeats=task_repeats,
    )

    if not screening_complete or not screening_passed:
        return created

    all_task_ids = [int(task.id) for task in tasks]
    created += await _seed_script_task_subset(
        db,
        script=script,
        task_ids=all_task_ids,
        task_repeats=task_repeats,
        now=now,
    )
    return created


async def _seed_script_task_subset(
    db: AsyncSession,
    *,
    script: _ScriptRef,
    task_ids: list[int],
    task_repeats: dict[int, int],
    now: datetime,
) -> int:
    if not task_ids:
        return 0

    existing = set(
        (
            int(row[0]),
            int(row[1]),
        )
        for row in (
            await db.execute(
                select(SweBenchRun.task_fk, SweBenchRun.attempt_no)
                .where(SweBenchRun.baseline_run.is_(False))
                .where(SweBenchRun.script_fk == script.script_id)
                .where(SweBenchRun.miner_fk == script.miner_fk)
                .where(SweBenchRun.task_fk.in_(task_ids))
            )
        ).all()
    )

    created = 0
    for task_id in task_ids:
        repeats = max(1, int(task_repeats.get(int(task_id), 1)))
        for attempt_no in range(1, repeats + 1):
            key = (int(task_id), attempt_no)
            if key in existing:
                continue
            await _create_run_and_validation(
                db,
                task_fk=int(task_id),
                attempt_no=attempt_no,
                baseline_run=False,
                miner_fk=script.miner_fk,
                script_fk=script.script_id,
                now=now,
            )
            existing.add(key)
            created += 1
    return created


async def _evaluate_screening_for_script(
    db: AsyncSession,
    *,
    script: _ScriptRef,
    screener_task_ids: list[int],
    task_repeats: dict[int, int],
) -> tuple[bool, bool]:
    if not screener_task_ids:
        return True, True

    rows = (
        await db.execute(
            select(
                SweBenchRun.task_fk,
                SweBenchRun.attempt_no,
                SweBenchRunValidation.resolved,
                SweBenchRunValidation.scored_at,
            )
            .join(
                SweBenchRunValidation,
                SweBenchRunValidation.run_fk == SweBenchRun.id,
            )
            .where(SweBenchRun.baseline_run.is_(False))
            .where(SweBenchRun.script_fk == script.script_id)
            .where(SweBenchRun.miner_fk == script.miner_fk)
            .where(SweBenchRun.task_fk.in_(screener_task_ids))
        )
    ).all()

    by_task_attempt: dict[tuple[int, int], tuple[bool | None, datetime | None]] = {}
    for row in rows:
        by_task_attempt[(int(row[0]), int(row[1]))] = (row[2], row[3])

    passed_task_count = 0
    for task_id in screener_task_ids:
        repeats = max(1, int(task_repeats.get(int(task_id), 1)))
        attempt_resolved: list[bool] = []
        for attempt_no in range(1, repeats + 1):
            state = by_task_attempt.get((int(task_id), attempt_no))
            if state is None:
                return False, False
            resolved_value, scored_at = state
            if scored_at is None or resolved_value is None:
                return False, False
            attempt_resolved.append(bool(resolved_value))

        if sum(1 for value in attempt_resolved if value) > (len(attempt_resolved) // 2):
            passed_task_count += 1

    required_passes = _required_screening_task_passes(len(screener_task_ids))
    return True, passed_task_count >= required_passes


def _required_screening_task_passes(total_screener_tasks: int) -> int:
    if total_screener_tasks <= 0:
        return 0

    ratio = float(settings.swebench_screening_pass_ratio)
    ratio = min(1.0, max(0.0, ratio))
    ratio_required = int(math.ceil(total_screener_tasks * ratio))
    min_required = max(0, int(settings.swebench_screening_min_passed_tasks))

    required = max(ratio_required, min_required)
    required = max(1, required)
    return min(total_screener_tasks, required)


async def _create_run_and_validation(
    db: AsyncSession,
    *,
    task_fk: int,
    attempt_no: int,
    baseline_run: bool,
    miner_fk: int | None,
    script_fk: int | None,
    now: datetime,
) -> None:
    run = SweBenchRun(
        task_fk=task_fk,
        request_fk=None,
        attempt_no=attempt_no,
        miner_fk=miner_fk,
        script_fk=script_fk,
        diff_storage_uuid=str(uuid.uuid4()),
        tokens_used=None,
        time_taken_seconds=None,
        agent_steps=None,
        status="pending",
        last_error=None,
        baseline_run=baseline_run,
    )
    db.add(run)
    await db.flush()

    validation = SweBenchRunValidation(
        run_fk=run.id,
        request_fk=None,
        validator_fk=None,
        resolved=None,
        logs=None,
        claimed_at=None,
        claim_expires_at=None,
        scored_at=None,
    )
    db.add(validation)


async def _dispatch_due_runs(
    app,
    now: datetime,
) -> tuple[int, int, int]:
    manager = _get_compact_bench_manager(app)
    output_storage = _get_output_storage(app)
    s3_storage = _get_s3_storage(app)

    dispatched = 0
    deferred = 0
    failed = 0

    async for db in get_db_session():
        due_rows = (
            await db.execute(
                select(SweBenchRun, SweBenchTask)
                .join(SweBenchTask, SweBenchTask.id == SweBenchRun.task_fk)
                .where(SweBenchRun.status == "pending")
                .order_by(SweBenchRun.created_at.asc())
                .limit(max(1, int(settings.swebench_dispatch_batch_size)))
                .with_for_update(skip_locked=True)
            )
        ).all()

        if not due_rows:
            await db.rollback()
            break

        expires_in = int(max(60.0, float(settings.sandbox_timeout_per_task_seconds) + 300.0))

        for run, task in due_rows:
            storage_key = output_storage.build_key(run.diff_storage_uuid)
            storage_presigned_url = await s3_storage.generate_presigned_url(
                storage_key,
                "put_object",
                expires_in=expires_in,
            )

            ok, error, retryable = await manager.dispatch_swebench_run(
                run_id=int(run.id),
                benchmark=str(settings.swebench_benchmark_name),
                instance_id=str(task.instance_id),
                storage_uuid=str(run.diff_storage_uuid),
                storage_presigned_url=storage_presigned_url,
                task_context={
                    "competition_fk": int(task.competition_fk),
                    "miner_fk": run.miner_fk,
                    "script_fk": run.script_fk,
                    "attempt_no": int(run.attempt_no),
                    "planned_repeats": int(task.planned_repeats),
                    "baseline_run": bool(run.baseline_run),
                    "is_screener": bool(task.is_screener),
                },
            )

            if ok:
                run.status = "dispatched"
                run.last_error = None
                dispatched += 1
                continue

            run.last_error = error
            if retryable:
                # Keep run pending; orchestrator will retry in next polling tick.
                run.status = "pending"
                deferred += 1
            else:
                run.status = "failed"
                failed += 1

        await db.commit()
        break

    return dispatched, deferred, failed
def _get_s3_storage(app) -> S3BlobStorage:
    s3_storage = getattr(app.state, "swebench_s3_storage", None)
    if s3_storage is None:
        s3_storage = S3BlobStorage()
        app.state.swebench_s3_storage = s3_storage
    return s3_storage


def _get_output_storage(app) -> PatchArtifactStorage:
    output_storage = getattr(app.state, "swebench_output_storage", None)
    if output_storage is None:
        output_storage = PatchArtifactStorage(_get_s3_storage(app))
        app.state.swebench_output_storage = output_storage
    return output_storage


def _get_compact_bench_manager(app) -> RemoteCompactBenchManager:
    manager = getattr(app.state, "swebench_compact_bench_manager", None)
    if manager is None:
        service_url = settings.compact_bench_service_url or settings.sandbox_service_url
        if not service_url:
            raise RuntimeError(
                "COMPACT_BENCH_SERVICE_URL or SANDBOX_SERVICE_URL must be set in configuration"
            )
        manager = RemoteCompactBenchManager(
            sandbox_service_url=service_url,
            execution_timeout_seconds=settings.sandbox_timeout_per_task_seconds,
            submission_timeout_seconds=settings.sandbox_submission_timeout_seconds,
        )
        app.state.swebench_compact_bench_manager = manager
    return manager
