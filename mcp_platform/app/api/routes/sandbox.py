from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from fastapi import APIRouter, Depends, Request

from soma_shared.contracts.sandbox.v1.messages import (
    CompactBenchReportRequest,
    CompactBenchReportResponse,
)
from soma_shared.db.models import SweBenchRun
from soma_shared.db.session import get_db_session

from app.api.routes.utils import _require_private_network
from app.core.logging import get_logger


logger = get_logger(__name__)


router = APIRouter(
    prefix="/api/private/sandbox",
    tags=["sandbox"],
)


async def _persist_compact_bench_report(
    db: AsyncSession,
    *,
    payload: CompactBenchReportRequest,
) -> bool:
    run = (
        await db.execute(
            select(SweBenchRun).where(SweBenchRun.id == payload.run_id)
        )
    ).scalar_one_or_none()
    if run is None:
        logger.warning(
            "compact_bench_report_run_missing",
            extra={
                "run_id": payload.run_id,
            },
        )
        return False

    run.agent_steps = payload.agent_steps
    # TODO: replace this with the S3 path/UUID in the callback contract instead of full patch payload.
    if payload.patch_diff:
        run.diff_storage_uuid = payload.patch_diff
    run.tokens_used = payload.total_tokens
    run.time_taken_seconds = payload.execution_time_seconds

    await db.commit()
    return True


@router.post(
    "/compact-bench/report",
    response_model=CompactBenchReportResponse,
)
async def report_compact_bench_result(
    payload: CompactBenchReportRequest,
    request: Request,
    db: AsyncSession = Depends(get_db_session),
    _: None = Depends(_require_private_network),
) -> CompactBenchReportResponse:
    persisted = False
    try:
        persisted = await _persist_compact_bench_report(db, payload=payload)
    except Exception:
        await db.rollback()
        logger.exception(
            "compact_bench_report_db_write_failed",
            extra={
                "run_id": payload.run_id,
                "path": str(request.url.path),
            },
        )
    return CompactBenchReportResponse(success=persisted)