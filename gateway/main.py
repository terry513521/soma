from __future__ import annotations

import asyncio
import logging
import os
from typing import AsyncIterator

import boto3
import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine


logger = logging.getLogger("soma.gateway")
logging.basicConfig(
    level=os.getenv("GATEWAY_LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


app = FastAPI(
    title="SOMA Gateway",
    description="Gateway service that proxies OpenAI-compatible requests",
    version="1.0.0",
)


def _resolve_postgres_dsn() -> str:
    dsn = os.getenv("POSTGRES_DSN")
    if not dsn:
        raise RuntimeError("POSTGRES_DSN is required for gateway DB lookups")
    if dsn.startswith("postgresql+asyncpg://"):
        return dsn
    if dsn.startswith("postgresql://"):
        return dsn.replace("postgresql://", "postgresql+asyncpg://", 1)
    if dsn.startswith("postgres://"):
        return dsn.replace("postgres://", "postgresql+asyncpg://", 1)
    return dsn


@app.on_event("startup")
async def startup() -> None:
    app.state.db_engine = create_async_engine(_resolve_postgres_dsn(), future=True)


@app.on_event("shutdown")
async def shutdown() -> None:
    engine: AsyncEngine | None = getattr(app.state, "db_engine", None)
    if engine is not None:
        await engine.dispose()


def _ssm_client():
    region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
    kwargs: dict[str, str] = {}
    if region:
        kwargs["region_name"] = region
    return boto3.client("ssm", **kwargs)


def _resolve_ssm_parameter_name(api_key_path: str) -> str:
    prefix = os.getenv("OPENROUTER_SSM_PREFIX", "/s114/dev")
    clean_prefix = prefix.rstrip("/")
    clean_suffix = api_key_path.strip("/")
    if not clean_suffix:
        raise ValueError("api_key_path must not be empty")
    return f"{clean_prefix}/{clean_suffix}"


def _resolve_api_key_from_ssm(api_key_path: str) -> str:
    parameter_name = _resolve_ssm_parameter_name(api_key_path)
    client = _ssm_client()
    resp = client.get_parameter(Name=parameter_name, WithDecryption=True)
    value = (resp.get("Parameter") or {}).get("Value")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"SSM parameter {parameter_name!r} is empty or missing")
    return value


def _resolve_upstream_url(path: str) -> str:
    base_url = os.getenv("GATEWAY_UPSTREAM_BASE_URL", "https://openrouter.ai/api/v1")
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


async def _resolve_api_key_path_from_db(run_id: int) -> str:
    engine: AsyncEngine = app.state.db_engine
    query = text(
        """
        SELECT mok.secret_ref
        FROM swe_bench_runs sbr
        JOIN miner_openrouter_api_keys mok ON mok.miner_fk = sbr.miner_fk
        WHERE sbr.id = :run_id
          AND mok.revoked_at IS NULL
        LIMIT 1
        """
    )
    async with engine.connect() as conn:
        result = await conn.execute(query, {"run_id": run_id})
        api_key_path = result.scalar_one_or_none()
    if not api_key_path or not str(api_key_path).strip():
        raise ValueError(
            f"No active miner OpenRouter key path found for swe_bench_runs.id={run_id}",
        )
    return str(api_key_path)


def _extract_forward_headers(request: Request) -> dict[str, str]:
    skip = {"host", "content-length", "authorization"}
    headers: dict[str, str] = {}
    for key, value in request.headers.items():
        lower = key.lower()
        if lower in skip or lower.startswith("x-run-id"):
            continue
        headers[key] = value
    return headers


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "healthy", "service": "gateway"}


async def _resolve_authorization_header(run_id: int) -> str:
    try:
        api_key_path = await _resolve_api_key_path_from_db(run_id)
        api_key_value = await asyncio.to_thread(_resolve_api_key_from_ssm, api_key_path)
    except Exception as exc:
        logger.exception("Failed to resolve API key for run_id=%s", run_id)
        raise HTTPException(status_code=400, detail=f"API key resolution failed: {exc}") from exc
    return f"Bearer {api_key_value}"


async def _stream_upstream_response(
    method: str,
    url: str,
    headers: dict[str, str],
    body_bytes: bytes,
    timeout: httpx.Timeout,
) -> tuple[AsyncIterator[bytes], int, dict[str, str]]:
    client = httpx.AsyncClient(timeout=timeout)
    stream_ctx = client.stream(method=method, url=url, headers=headers, content=body_bytes)
    response = await stream_ctx.__aenter__()

    async def _iterator() -> AsyncIterator[bytes]:
        try:
            async for chunk in response.aiter_bytes():
                yield chunk
        finally:
            await stream_ctx.__aexit__(None, None, None)
            await client.aclose()

    response_headers = {
        "content-type": response.headers.get("content-type", "text/event-stream"),
        "x-request-id": response.headers.get("x-request-id", ""),
    }
    return _iterator(), response.status_code, response_headers


@app.api_route("/v1/{path:path}", methods=["POST"])
async def proxy_openai_compatible(
    path: str,
    request: Request,
    x_run_id: str | None = Header(default=None, alias="X-Run-Id"),
) -> Response:
    if not x_run_id:
        raise HTTPException(status_code=400, detail="Missing required header: X-Run-Id")
    try:
        run_id = int(x_run_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="X-Run-Id must be an integer") from exc

    headers = _extract_forward_headers(request)
    headers["Authorization"] = await _resolve_authorization_header(run_id)

    body_bytes = await request.body()
    try:
        parsed = await request.json()
    except Exception:
        parsed = None

    url = _resolve_upstream_url(path)
    timeout = httpx.Timeout(float(os.getenv("GATEWAY_UPSTREAM_TIMEOUT_SECONDS", "60")))

    if isinstance(parsed, dict) and bool(parsed.get("stream")):
        stream, status_code, response_headers = await _stream_upstream_response(
            method="POST",
            url=url,
            headers=headers,
            body_bytes=body_bytes,
            timeout=timeout,
        )
        return StreamingResponse(
            stream,
            status_code=status_code,
            media_type=response_headers.get("content-type", "text/event-stream"),
            headers={"x-request-id": response_headers.get("x-request-id", "")},
        )

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, headers=headers, content=body_bytes)
    except Exception as exc:
        logger.exception("Upstream request failed for run_id=%s", run_id)
        raise HTTPException(status_code=502, detail=f"Upstream request failed: {exc}") from exc

    return Response(
        content=resp.content,
        status_code=resp.status_code,
        media_type=resp.headers.get("content-type"),
        headers={"x-request-id": resp.headers.get("x-request-id", "")},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "gateway.main:app",
        host=os.getenv("GATEWAY_HOST", "0.0.0.0"),
        port=int(os.getenv("GATEWAY_PORT", "8010")),
        log_level=os.getenv("GATEWAY_LOG_LEVEL", "info").lower(),
    )
