from __future__ import annotations

import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import request as urllib_request
from urllib.parse import urlsplit, urlunsplit

from soma_shared.contracts.sandbox.v1.messages import (
    CompactBenchReportRequest,
    CompactBenchRunTaskRequest,
)


@dataclass(slots=True)
class CompactBenchExecutionOutput:
    report: CompactBenchReportRequest
    patch_text: str


@dataclass(slots=True)
class NginxProxyHandle:
    container_name: str
    proxy_base_url: str
    upstream_base_url: str


PLUGIN_VENV_DIRNAME = ".soma-openclaw-venv"
PLUGIN_BACKEND_FILENAME = "base_miner.py"
PLUGIN_COPY_IGNORE_NAMES = {".git", PLUGIN_VENV_DIRNAME}


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


def _run_command(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, capture_output=True, text=True, check=False)


def _docker_container_running(name: str) -> bool:
    result = _run_command(["docker", "inspect", "-f", "{{.State.Running}}", name])
    return result.returncode == 0 and (result.stdout or "").strip().lower() == "true"


def _build_proxy_container_name() -> str:
    return "soma-compact-bench-nginx"


def _normalize_proxy_upstream_base_url(value: str) -> str:
    parsed = urlsplit(value)
    if not parsed.scheme or not parsed.netloc:
        raise RuntimeError("COMPACT_BENCH_LLM_BASE_URL must be an absolute http(s) URL")
    path = parsed.path or "/"
    if not path.endswith("/"):
        path = f"{path}/"
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def _build_nginx_config(*, upstream_base_url: str, listen_port: int) -> str:
    return "\n".join(
        [
            "events {}",
            "http {",
            "  server {",
            f"    listen {listen_port};",
            "    location / {",
            f"      proxy_pass {upstream_base_url};",
            "      proxy_http_version 1.1;",
            "      proxy_set_header Connection \"\";",
            "      proxy_set_header Host $proxy_host;",
            "      proxy_set_header X-Run-Id $http_x_run_id;",
            "      proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;",
            "      proxy_set_header X-Forwarded-Proto $scheme;",
            "    }",
            "  }",
            "}",
            "",
        ]
    )


def _resolve_plugin_template_path() -> Path:
    configured = os.getenv("COMPACT_BENCH_PLUGIN_TEMPLATE_PATH", "").strip()
    if configured:
        path = Path(configured).expanduser().resolve()
    else:
        path = (Path(__file__).resolve().parents[3] / "SOMA-OpenClaw-plugin").resolve()
    if not path.is_dir():
        raise RuntimeError(f"Compact-bench plugin template path does not exist: {path}")
    return path


def _download_miner_code(script_presigned_url: str) -> str:
    timeout_seconds = _get_miner_download_timeout()
    with urllib_request.urlopen(script_presigned_url, timeout=timeout_seconds) as response:
        payload = response.read()
    return payload.decode("utf-8")


def _get_miner_download_timeout() -> float:
    raw_timeout = os.getenv("COMPACT_BENCH_MINER_DOWNLOAD_TIMEOUT_SECONDS", "30").strip()
    try:
        timeout_seconds = float(raw_timeout)
    except ValueError:
        timeout_seconds = 30.0
    return max(1.0, timeout_seconds)


def _copy_plugin_template_checkout(*, template_path: Path, plugin_path: Path) -> None:
    for child in template_path.iterdir():
        if child.name in PLUGIN_COPY_IGNORE_NAMES:
            continue
        destination = plugin_path / child.name
        if child.is_dir():
            shutil.copytree(child, destination)
        else:
            shutil.copy2(child, destination)

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
        self._llm_proxy_handle: NginxProxyHandle | None = None
        self._llm_proxy_lock = threading.Lock()

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

        plugin_path = self._materialize_plugin_checkout(
            output_dir=output_dir,
            script_presigned_url=task.script_presigned_url,
        )
        command = self._build_command(task=task, output_dir=output_dir)
        env = os.environ.copy()
        llm_base_url = os.getenv("COMPACT_BENCH_LLM_BASE_URL", "").strip()
        if llm_base_url:
            env["LLM_BASE_URL"] = self._ensure_llm_proxy(llm_base_url).proxy_base_url
        env["SOMA_OPENCLAW_SOMARIZER_PLUGIN_PATH"] = str(plugin_path)

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
        patch_capture_status = False
        patch_text = ""
        if isinstance(patch_path, str) and patch_path.strip():
            patch_file = Path(patch_path)
            if patch_file.is_file():
                patch_capture_status = True
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
                "plugin_path": str(plugin_path),
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
            patch_capture_status=patch_capture_status,
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
            "--openclaw-run-id-header-value",
            str(task.run_id),
        ]
        if task.model:
            command.extend(["--model", task.model])
        if task.openclaw_disable_somarizer:
            command.append("--openclaw-disable-somarizer")
        return command

    def _materialize_plugin_checkout(self, *, output_dir: Path, script_presigned_url: str) -> Path:
        template_path = _resolve_plugin_template_path()
        plugin_path = output_dir / "soma-miner-plugin"
        plugin_path.mkdir(parents=True, exist_ok=True)

        _copy_plugin_template_checkout(template_path=template_path, plugin_path=plugin_path)

        miner_code = _download_miner_code(script_presigned_url)
        (plugin_path / PLUGIN_BACKEND_FILENAME).write_text(miner_code, encoding="utf-8")

        template_venv_path = template_path / PLUGIN_VENV_DIRNAME
        plugin_venv_path = plugin_path / PLUGIN_VENV_DIRNAME
        if template_venv_path.exists() and not plugin_venv_path.exists():
            plugin_venv_path.symlink_to(template_venv_path, target_is_directory=True)

        return plugin_path

    def _ensure_llm_proxy(self, upstream_base_url: str) -> NginxProxyHandle:
        normalized_upstream_base_url = _normalize_proxy_upstream_base_url(upstream_base_url)
        with self._llm_proxy_lock:
            if (
                self._llm_proxy_handle is not None
                and self._llm_proxy_handle.upstream_base_url == normalized_upstream_base_url
                and _docker_container_running(self._llm_proxy_handle.container_name)
            ):
                return self._llm_proxy_handle

            handle = self._start_llm_proxy(upstream_base_url=normalized_upstream_base_url)
            self._llm_proxy_handle = handle
            return handle

    def _start_llm_proxy(self, *, upstream_base_url: str) -> NginxProxyHandle:
        proxy_port = self._get_llm_proxy_port()
        container_name = _build_proxy_container_name()
        config_path = self._output_root / "llm-proxy.nginx.conf"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            _build_nginx_config(
                upstream_base_url=upstream_base_url,
                listen_port=proxy_port,
            ),
            encoding="utf-8",
        )

        self._stop_llm_proxy(
            NginxProxyHandle(container_name=container_name, proxy_base_url="", upstream_base_url="")
        )

        result = _run_command(
            [
                "docker",
                "run",
                "-d",
                "--rm",
                "--name",
                container_name,
                "--network",
                "bridge",
                "-v",
                f"{config_path}:/etc/nginx/nginx.conf:ro",
                self._get_llm_proxy_image(),
            ]
        )
        if result.returncode != 0:
            message = (result.stderr or result.stdout or "").strip()
            raise RuntimeError(f"Failed to start compact-bench nginx proxy container: {message}")

        inspect_result = _run_command(
            [
                "docker",
                "inspect",
                "-f",
                "{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}",
                container_name,
            ]
        )
        proxy_ip = (inspect_result.stdout or "").strip()
        if inspect_result.returncode != 0 or not proxy_ip:
            self._stop_llm_proxy(
                NginxProxyHandle(container_name=container_name, proxy_base_url="", upstream_base_url="")
            )
            message = (inspect_result.stderr or inspect_result.stdout or "").strip()
            raise RuntimeError(f"Failed to resolve compact-bench nginx proxy IP: {message}")

        return NginxProxyHandle(
            container_name=container_name,
            proxy_base_url=f"http://{proxy_ip}:{proxy_port}",
            upstream_base_url=upstream_base_url,
        )

    def _stop_llm_proxy(self, handle: NginxProxyHandle) -> None:
        _run_command(["docker", "rm", "-f", handle.container_name])

    def _get_llm_proxy_image(self) -> str:
        return os.getenv("COMPACT_BENCH_LLM_PROXY_IMAGE", "nginx:1.27-alpine").strip() or "nginx:1.27-alpine"

    def _get_llm_proxy_port(self) -> int:
        raw_port = os.getenv("COMPACT_BENCH_LLM_PROXY_PORT", "8080").strip()
        try:
            port = int(raw_port)
        except ValueError:
            port = 8080
        return port if port > 0 else 8080

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
        """Release persistent sandbox-side helper resources."""
        with self._llm_proxy_lock:
            if self._llm_proxy_handle is not None:
                self._stop_llm_proxy(self._llm_proxy_handle)
                self._llm_proxy_handle = None
