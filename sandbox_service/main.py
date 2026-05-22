"""Standalone sandbox service for asynchronous compact-bench execution."""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import traceback
from pathlib import Path

# Add parent directory to path to find mcp_platform module
sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
from soma_shared.contracts.sandbox.v1.messages import (
    CompactBenchReportRequest,
    CompactBenchRunTaskRequest,
    CompactBenchRunTaskResponse,
)
from app.compact_bench_executor import CompactBenchExecutor


# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)


# FastAPI app
app = FastAPI(
    title="Sandbox Service",
    description="Remote sandbox execution service for SOMA platform",
    version="1.0.0",
)


@app.on_event("startup")
async def startup():
    """Initialize compact-bench executor and shared capacity controls."""
    get_compact_bench_executor()
    cpu_count = os.cpu_count() or 2
    max_concurrent = int(min(12, max(1, cpu_count - 1)))
    app.state.sandbox_semaphore = asyncio.Semaphore(max_concurrent)
    logger.info("Sandbox semaphore initialized with max_concurrent=%d", max_concurrent)


@app.on_event("shutdown")
async def shutdown() -> None:
    """Release persistent compact-bench helper resources."""
    executor = getattr(app.state, "compact_bench_executor", None)
    if executor is not None:
        executor.shutdown()


def get_compact_bench_executor() -> CompactBenchExecutor:
    """Get or create compact-bench executor instance."""
    if not hasattr(app.state, "compact_bench_executor"):
        app.state.compact_bench_executor = CompactBenchExecutor()
    return app.state.compact_bench_executor


async def _acquire_capacity_slot(*, operation_kind: str, operation_id: str) -> None:
    semaphore: asyncio.Semaphore = app.state.sandbox_semaphore
    try:
        async with asyncio.timeout(0):
            await semaphore.acquire()
    except (asyncio.TimeoutError, TimeoutError):
        logger.warning(
            "%s service at capacity, rejecting request: id=%s",
            operation_kind,
            operation_id,
        )
        raise HTTPException(
            status_code=429,
            detail=f"{operation_kind} service is at capacity. Please try again later.",
        )


def _release_capacity_slot() -> None:
    semaphore: asyncio.Semaphore = app.state.sandbox_semaphore
    semaphore.release()


def _get_compact_bench_report_url() -> str:
    report_url = os.getenv("COMPACT_BENCH_REPORT_URL", "").strip()
    if not report_url:
        raise RuntimeError("COMPACT_BENCH_REPORT_URL must be set for compact-bench callbacks")
    return report_url


def _get_compact_bench_report_timeout() -> float:
    raw_timeout = os.getenv("COMPACT_BENCH_REPORT_TIMEOUT_SECONDS", "15")
    try:
        timeout = float(raw_timeout)
    except ValueError:
        timeout = 15.0
    return max(1.0, timeout)


async def _send_compact_bench_report(report: CompactBenchReportRequest) -> None:
    report_url = _get_compact_bench_report_url()
    logger.info(
        "Sending compact-bench report: run_id=%s ok_status=%s report_url=%s",
        report.run_id,
        report.ok_status,
        report_url,
    )
    async with httpx.AsyncClient() as http_client:
        response = await http_client.post(
            report_url,
            json=report.model_dump(mode="json"),
            timeout=_get_compact_bench_report_timeout(),
        )
        response.raise_for_status()
    logger.info(
        "Compact-bench report delivered: run_id=%s status_code=%s",
        report.run_id,
        response.status_code,
    )


async def _execute_compact_bench_task_in_background(request: CompactBenchRunTaskRequest) -> None:
    try:
        executor = get_compact_bench_executor()
        logger.info(
            "Starting compact-bench background execution: run_id=%s benchmark=%s instance_id=%s",
            request.run_id,
            request.benchmark,
            request.instance_id,
        )
        output = await asyncio.to_thread(
            executor.execute_task,
            batch_id=str(request.run_id),
            task=request,
            timeout_per_task=request.openclaw_timeout,
        )
        logger.info(
            "Compact-bench execution finished: run_id=%s ok_status=%s execution_time_seconds=%s",
            request.run_id,
            output.report.ok_status,
            output.report.execution_time_seconds,
        )

        await _send_compact_bench_report(output.report)
    except Exception as exc:
        tb = traceback.format_exc()
        logger.error(
            "Compact-bench task failed before reporting: run_id=%s, error=%s\n%s",
            request.run_id,
            str(exc),
            tb,
        )
        fallback_report = CompactBenchReportRequest(
            run_id=request.run_id,
            ok_status=False,
            error=f"{type(exc).__name__}: {exc}",
            execution_time_seconds=None,
            total_tokens=None,
            agent_steps=None,
            patch_capture_status=False,
            patch_diff=None,
            metadata={
                "benchmark": request.benchmark,
                "instance_id": request.instance_id,
                "traceback": tb,
            },
        )
        try:
            await _send_compact_bench_report(fallback_report)
        except Exception:
            logger.exception(
                "Compact-bench report callback failed irrecoverably",
                extra={"run_id": request.run_id},
            )
    finally:
        logger.info("Releasing compact-bench capacity slot: run_id=%s", request.run_id)
        _release_capacity_slot()

@app.post("/run_compact_bench_task", response_model=CompactBenchRunTaskResponse)
async def run_compact_bench_task(
    request: CompactBenchRunTaskRequest,
) -> CompactBenchRunTaskResponse:
    """Accept a compact-bench task for asynchronous execution and callback reporting."""

    _get_compact_bench_report_url()
    await _acquire_capacity_slot(
        operation_kind="Compact-bench",
        operation_id=str(request.run_id),
    )
    logger.info(
        "Accepted compact-bench task: run_id=%s benchmark=%s instance_id=%s",
        request.run_id,
        request.benchmark,
        request.instance_id,
    )
    asyncio.create_task(_execute_compact_bench_task_in_background(request))
    return CompactBenchRunTaskResponse(success=True)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "sandbox"}


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("SANDBOX_SERVICE_PORT", "8001"))
    host = os.getenv("SANDBOX_SERVICE_HOST", "0.0.0.0")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
    )
