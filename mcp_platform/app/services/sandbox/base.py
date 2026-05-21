from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class SandboxExecutionError(Exception):
    """Raised when sandbox execution fails for a batch."""


@dataclass(slots=True)
class SandboxTaskArtifact:
    """Normalized per-task artifact returned by an execution backend."""

    text: str = ""
    kind: str = "compressed_text"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SandboxTaskResult:
    """Normalized single-task result used by the compact-bench backend."""

    artifact: SandboxTaskArtifact
    task_error: str | None = None
    execution_time: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
