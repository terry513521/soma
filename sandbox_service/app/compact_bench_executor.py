from __future__ import annotations

import importlib.util
import json
import os
import shlex
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from soma_shared.contracts.sandbox.v1.messages import (
    CompactBenchReportRequest,
    CompactBenchRunTaskRequest,
)


@dataclass(slots=True)
class CompactBenchExecutionOutput:
    report: CompactBenchReportRequest
    patch_text: str


def _slug(value: str, *, default: str) -> str:
    sanitized = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "-" for ch in value.lower())
    sanitized = sanitized.strip("-._")
    return sanitized or default


def _token_count(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float) and value >= 0 and value.is_integer():
        return int(value)
    return None


def _step_count(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float) and value >= 0 and value.is_integer():
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return int(stripped)
    return None

class CompactBenchExecutor:
    """Runs compact-bench solve commands and captures produced patches."""

    def __init__(
        self,
        *,
        python_executable: str | None = None,
        output_root: str | Path | None = None,
    ):
        default_output_root = Path("/tmp") / "soma-compact-bench-service"
        self._python_executable = python_executable or os.getenv("COMPACT_BENCH_PYTHON_EXECUTABLE") or sys.executable
        self._output_root = Path(output_root or os.getenv("COMPACT_BENCH_OUTPUT_ROOT") or default_output_root).expanduser().resolve()

        if importlib.util.find_spec("soma_compact_bench") is None:
            raise RuntimeError(
                "The 'soma_compact_bench' package is not installed in the sandbox-service environment. "
                "Install dependencies from requirements.txt before starting the service."
            )

    def execute_task(
        self,
        *,
        batch_id: str,
        task: CompactBenchRunTaskRequest,
        timeout_per_task: float | None,
    ) -> CompactBenchExecutionOutput:
        output_dir = self._output_root / _slug(batch_id, default="batch") / _slug(
            f"{task.instance_id}-{uuid.uuid4().hex[:8]}",
            default="task",
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        command = self._build_command(task=task, output_dir=output_dir)
        env = os.environ.copy()

        effective_timeout = task.openclaw_timeout if task.openclaw_timeout is not None else timeout_per_task
        timeout = max(1.0, float(effective_timeout)) if effective_timeout is not None else None
        started_at = time.monotonic()
        try:
            process = subprocess.run(
                command,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            duration = time.monotonic() - started_at
            metadata = dict(task.metadata)
            metadata.update(
                {
                    "benchmark": task.benchmark,
                    "instance_id": task.instance_id,
                    "status": "timeout",
                    "command": shlex.join(command),
                    "output_dir": str(output_dir),
                }
            )
            report = CompactBenchReportRequest(
                run_id=task.run_id,
                ok_status=False,
                error=str(exc),
                execution_time_seconds=duration,
                total_tokens=None,
                agent_steps=None,
                patch_capture_status=False,
                patch_diff=None,
                metadata=metadata,
            )
            return CompactBenchExecutionOutput(report=report, patch_text="")

        duration = time.monotonic() - started_at
        row = self._read_result_row(output_dir)
        row_metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        patch_capture = row_metadata.get("patch_capture") if isinstance(row_metadata.get("patch_capture"), dict) else {}
        patch_path = patch_capture.get("patch_path") if isinstance(patch_capture, dict) else None
        patch_text = ""
        if isinstance(patch_path, str) and patch_path.strip():
            patch_file = Path(patch_path)
            if patch_file.is_file():
                patch_text = patch_file.read_text(encoding="utf-8")

        status = str(row.get("status") or ("completed" if process.returncode == 0 else "runtime-error"))
        error_text = str(row.get("error") or "").strip() or process.stderr.strip() or None
        success = process.returncode == 0 and status == "completed"
        total_tokens, agent_steps = self._extract_execution_metrics(
            row=row,
            metadata=row_metadata,
        )
        metadata = dict(task.metadata)
        metadata.update(row_metadata)
        metadata.update(
            {
                "benchmark": task.benchmark,
                "instance_id": task.instance_id,
                "status": status,
                "command": shlex.join(command),
                "returncode": process.returncode,
                "output_dir": str(output_dir),
                "total_tokens": total_tokens,
            }
        )

        report = CompactBenchReportRequest(
            run_id=task.run_id,
            ok_status=success,
            error=error_text,
            execution_time_seconds=duration,
            total_tokens=total_tokens,
            agent_steps=agent_steps,
            patch_capture_status=False,
            patch_diff=patch_text or None,
            metadata=metadata,
        )
        return CompactBenchExecutionOutput(report=report, patch_text=patch_text)

    def _build_command(self, *, task: CompactBenchRunTaskRequest, output_dir: Path) -> list[str]:
        # TODO: define the executable command and runtime flags directly in this service
        # instead of passing command-shaping inputs through the payload contract.
        command = [
            self._python_executable,
            "-m",
            "soma_compact_bench.benchmark.solve",
            "--agent-name",
            task.agent_name,
            "--benchmark",
            task.benchmark,
            "--instance-id",
            task.instance_id,
            "--output-dir",
            str(output_dir),
            "--execute",
        ]
        if task.model:
            command.extend(["--model", task.model])
        if task.openclaw_disable_somarizer:
            command.append("--openclaw-disable-somarizer")
        return command

    def _read_result_row(self, output_dir: Path) -> dict[str, Any]:
        output_json_path = output_dir / "output.jsonl"
        if not output_json_path.is_file():
            return {}

        for raw_line in output_json_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                return payload
        return {}

    def _extract_execution_metrics(
        self,
        *,
        row: dict[str, Any],
        metadata: dict[str, Any],
    ) -> tuple[int | None, int | None]:
        total_tokens = None
        for candidate in (
            row.get("total_tokens"),
            metadata.get("total_tokens"),
            (metadata.get("total") or {}).get("total_tokens") if isinstance(metadata.get("total"), dict) else None,
            (metadata.get("token_usage") or {}).get("total_tokens") if isinstance(metadata.get("token_usage"), dict) else None,
        ):
            value = _token_count(candidate)
            if value is not None:
                total_tokens = value
                break

        if total_tokens is None:
            token_usage = metadata.get("token_usage")
            if isinstance(token_usage, dict):
                total_payload = token_usage.get("total")
                if isinstance(total_payload, dict):
                    total_tokens = _token_count(total_payload.get("total_tokens"))

        agent_steps = None
        for candidate in (
            row.get("agent_steps"),
            metadata.get("agent_steps"),
            row.get("steps"),
            metadata.get("steps"),
            (metadata.get("agent") or {}).get("steps") if isinstance(metadata.get("agent"), dict) else None,
        ):
            value = _step_count(candidate)
            if value is not None:
                agent_steps = value
                break

        return total_tokens, agent_steps

    def shutdown(self) -> None:
        """Compact-bench executor does not hold persistent resources."""
