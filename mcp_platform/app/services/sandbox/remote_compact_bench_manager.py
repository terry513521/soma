from __future__ import annotations

import itertools
from typing import Any

import httpx

from soma_shared.contracts.sandbox.v1.messages import (
    CompactBenchRunTaskRequest,
    CompactBenchRunTaskResponse,
)

from app.core.logging import get_logger
from app.services.sandbox.base import (
    SandboxExecutionError,
    SandboxTaskArtifact,
    SandboxTaskResult,
)


logger = get_logger(__name__)
_RUN_ID_COUNTER = itertools.count(1)


class RemoteCompactBenchManager:
    """Execution backend that delegates benchmark solving to compact-bench."""

    def __init__(
        self,
        *,
        sandbox_service_url: str,
        execution_timeout_seconds: float,
        submission_timeout_seconds: float,
    ):
        self._sandbox_service_url = sandbox_service_url.rstrip("/")
        self._execution_timeout_seconds = execution_timeout_seconds
        self._submission_timeout_seconds = submission_timeout_seconds

    async def execute_task(
        self,
        *,
        batch_id: str,
        storage_uuid: str,
        script_presigned_url: str,
        challenge_text: str,
        task_context: dict[str, Any],
    ) -> SandboxTaskResult:
        run_id = next(_RUN_ID_COUNTER)
        payload = self._build_task_request(
            run_id=run_id,
            task_context=task_context,
            storage_uuid=storage_uuid,
            script_presigned_url=script_presigned_url,
            challenge_text=challenge_text,
        )

        task_error: str | None = None
        dispatch_accepted = False
        dispatch_timeout = max(1.0, float(self._submission_timeout_seconds))

        async with httpx.AsyncClient() as client:
            try:
                dispatch_result = await self._dispatch_task(
                    client=client,
                    payload=payload,
                    timeout=dispatch_timeout,
                )
            except Exception as exc:
                task_error = self._format_dispatch_error(exc)
                dispatch_result = None
        if dispatch_result is not None and dispatch_result.success:
            dispatch_accepted = True
        elif dispatch_result is not None and not dispatch_result.success:
            task_error = "Compact-bench service rejected task dispatch"

        return SandboxTaskResult(
            artifact=SandboxTaskArtifact(
                text="",
                kind="patch",
                metadata={
                    "benchmark": task_context.get("benchmark"),
                    "instance_id": task_context.get("instance_id"),
                    "run_id": run_id,
                    "dispatch_accepted": dispatch_accepted,
                },
            ),
            task_error=task_error,
            execution_time=None,
            metadata={
                "backend": "compact_bench",
                "batch_id": batch_id,
            },
        )

    def _build_task_request(
        self,
        *,
        run_id: int,
        task_context: dict[str, Any],
        storage_uuid: str,
        script_presigned_url: str,
        challenge_text: str,
    ) -> CompactBenchRunTaskRequest:
        benchmark = str(task_context.get("benchmark") or "").strip()
        instance_id = str(task_context.get("instance_id") or "").strip()
        if not benchmark or not instance_id:
            raise SandboxExecutionError(
                "Compact-bench task_context must contain non-empty 'benchmark' and 'instance_id'."
            )

        metadata = dict(task_context.get("metadata") or {})
        metadata.setdefault("storage_uuid", storage_uuid)
        for key in (
            "competition_fk",
            "request_fk",
            "miner_fk",
            "script_fk",
            "validator_fk",
            "attempt_no",
            "planned_repeats",
            "agent_steps",
        ):
            value = task_context.get(key)
            if value is not None:
                metadata.setdefault(key, value)
        for key in ("baseline_run", "is_screener"):
            value = task_context.get(key)
            if value is not None:
                metadata.setdefault(key, bool(value))
        if challenge_text:
            metadata.setdefault("problem_statement", challenge_text)

        return CompactBenchRunTaskRequest(
            benchmark=benchmark,
            instance_id=instance_id,
            run_id=run_id,
            script_presigned_url=script_presigned_url,
            agent_name=str(task_context.get("agent_name") or "openclaw").strip() or "openclaw",
            model=str(task_context.get("model")).strip() if task_context.get("model") else None,
            openclaw_timeout=(
                int(task_context["openclaw_timeout"])
                if task_context.get("openclaw_timeout") is not None
                else int(max(1.0, self._execution_timeout_seconds))
            ),
            openclaw_disable_somarizer=bool(task_context.get("openclaw_disable_somarizer", False)),
            metadata=metadata,
        )

    async def _dispatch_task(
        self,
        *,
        client: httpx.AsyncClient,
        payload: CompactBenchRunTaskRequest,
        timeout: float,
    ) -> CompactBenchRunTaskResponse:
        response = await client.post(
            f"{self._sandbox_service_url}/run_compact_bench_task",
            json=payload.model_dump(mode="json"),
            timeout=timeout,
        )
        response.raise_for_status()
        return CompactBenchRunTaskResponse.model_validate(response.json())

    def _format_dispatch_error(self, exc: Exception) -> str:
        if isinstance(exc, httpx.TimeoutException):
            return "Compact-bench service acknowledgement timed out"
        if isinstance(exc, httpx.HTTPStatusError):
            if exc.response.status_code == 429:
                return (
                    "Platform is at capacity. The compact-bench service is currently handling the maximum "
                    "number of concurrent requests. Please try again later."
                )
            return f"Compact-bench service returned HTTP {exc.response.status_code}"
        return f"Failed to communicate with compact-bench service: {exc}"

    def shutdown(self) -> None:
        logger.info("[RemoteCompactBench] Shutdown complete")