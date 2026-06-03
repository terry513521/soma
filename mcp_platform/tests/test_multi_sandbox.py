from __future__ import annotations

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

TESTS_DIR = os.path.dirname(__file__)
MCP_PLATFORM_DIR = os.path.abspath(os.path.join(TESTS_DIR, ".."))
if MCP_PLATFORM_DIR not in sys.path:
    sys.path.insert(0, MCP_PLATFORM_DIR)

os.environ.setdefault("PRIVATE_NETWORK_CIDRS", '["127.0.0.1/32"]')
os.environ.setdefault("TRUSTED_PROXY_CIDRS", '["127.0.0.1/32"]')

import sys
import types

def _install_soma_shared_stub() -> None:
    try:
        import soma_shared.contracts.sandbox.v1.messages  # noqa: F401
        return
    except ImportError:
        pass
    messages = types.ModuleType("soma_shared.contracts.sandbox.v1.messages")
    messages.CompactBenchReportRequest = type("CompactBenchReportRequest", (), {})
    def _make_request_stub(name):
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
        def model_dump(self, **kwargs):
            return self.__dict__
        @classmethod
        def model_validate(cls, data):
            obj = cls.__new__(cls)
            obj.__dict__.update(data if isinstance(data, dict) else {})
            obj.success = data.get("success", True) if isinstance(data, dict) else True
            return obj
        return type(name, (), {"__init__": __init__, "model_dump": model_dump, "model_validate": model_validate})

    messages.CompactBenchRunTaskRequest = _make_request_stub("CompactBenchRunTaskRequest")
    messages.CompactBenchRunTaskResponse = _make_request_stub("CompactBenchRunTaskResponse")
    for module_name in (
        "soma_shared",
        "soma_shared.contracts",
        "soma_shared.contracts.sandbox",
        "soma_shared.contracts.sandbox.v1",
    ):
        sys.modules.setdefault(module_name, types.ModuleType(module_name))
    sys.modules["soma_shared.contracts.sandbox.v1.messages"] = messages

_install_soma_shared_stub()

from app.services.sandbox.remote_compact_bench_manager import RemoteCompactBenchManager


def _make_manager(urls: list[str] | None = None, **kwargs) -> RemoteCompactBenchManager:
    return RemoteCompactBenchManager(
        sandbox_service_urls=urls if urls is not None else [],
        execution_timeout_seconds=10.0,
        submission_timeout_seconds=10.0,
        default_model="qwen/qwen3-coder",
        **kwargs,
    )


def test_round_robin_distributes_across_urls():
    urls = ["http://sandbox-a", "http://sandbox-b", "http://sandbox-c"]
    manager = _make_manager(urls)

    picked = [manager._pick_sandbox_url() for _ in range(6)]

    # Round-robin in order, each URL chosen exactly twice.
    assert picked == [
        "http://sandbox-a",
        "http://sandbox-b",
        "http://sandbox-c",
        "http://sandbox-a",
        "http://sandbox-b",
        "http://sandbox-c",
    ]
    for url in urls:
        assert picked.count(url) == 2


def test_single_url_always_picked():
    manager = _make_manager(["http://only-sandbox"])

    picked = [manager._pick_sandbox_url() for _ in range(3)]

    assert picked == ["http://only-sandbox"] * 3


def test_empty_urls_raises():
    manager = _make_manager([])

    with pytest.raises(RuntimeError):
        manager._pick_sandbox_url()


async def test_dispatch_uses_round_robin():
    urls = ["http://sandbox-a", "http://sandbox-b", "http://sandbox-c"]
    manager = _make_manager(urls)

    requested_urls: list[str] = []

    async def _fake_post(self, url, *args, **kwargs):
        requested_urls.append(url)
        response = MagicMock()
        response.raise_for_status = MagicMock(return_value=None)
        response.json = MagicMock(return_value={"success": True})
        return response

    with patch("httpx.AsyncClient.post", new=_fake_post):
        for idx in range(3):
            ok, error, _retryable = await manager.dispatch_swebench_run(
                run_id=idx + 1,
                benchmark="SWE-bench/SWE-bench_Verified",
                instance_id=f"instance-{idx}",
                storage_uuid=f"uuid-{idx}",
                script_presigned_url="http://example.com/script.py",
            )
            assert ok is True
            assert error is None

    # One request per dispatch, and each sandbox URL received exactly one.
    assert len(requested_urls) == 3
    base_urls = [url.rsplit("/run_compact_bench_task", 1)[0] for url in requested_urls]
    for url in urls:
        assert base_urls.count(url) == 1


def test_legacy_single_url_param():
    manager = RemoteCompactBenchManager(
        sandbox_service_url="http://old",
        execution_timeout_seconds=10.0,
        submission_timeout_seconds=10.0,
        default_model="qwen/qwen3-coder",
    )

    assert manager._sandbox_service_urls == ["http://old"]
    assert manager._pick_sandbox_url() == "http://old"
