from __future__ import annotations

import asyncio
import json
import logging
import os

import boto3
import httpx
from fastapi import FastAPI
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from soma_shared.contracts.sandbox.v1.messages import (
    ApiGatewayProxyPayload,
    ApiGatewayRequest,
    ApiGatewayResponse,
)


logger = logging.getLogger("soma.gateway")
logging.basicConfig(
    level=os.getenv("GATEWAY_LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


app = FastAPI(
    title="SOMA Gateway",
    description="Gateway service that proxies requests using S3 + secret store",
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
    prefix = os.getenv("GATEWAY_SSM_PREFIX", "/s114/dev")
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


def _build_request_content(payload: ApiGatewayProxyPayload) -> str | bytes | None:
    if payload.body is None:
        return None

    if isinstance(payload.body, str):
        return payload.body

    return json.dumps(payload.body)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "healthy", "service": "gateway"}


@app.post("/proxy", response_model=ApiGatewayResponse)
async def proxy(request: ApiGatewayRequest) -> ApiGatewayResponse:
    payload = request.body

    headers = dict(payload.headers)

    try:
        api_key_path = await _resolve_api_key_path_from_db(request.run_id)
        api_key_value = await asyncio.to_thread(_resolve_api_key_from_ssm, api_key_path)
    except Exception as exc:
        logger.exception("Failed to resolve API key for run_id=%s", request.run_id)
        return ApiGatewayResponse(
            success=False,
            error=f"API key resolution failed: {exc}",
        )
    headers["Authorization"] = f"Bearer {api_key_value}"

    try:
        content = await asyncio.to_thread(_build_request_content, payload)
    except Exception as exc:
        logger.exception("Failed to resolve proxy body for run_id=%s", request.run_id)
        return ApiGatewayResponse(
            success=False,
            error=f"Body resolution failed: {exc}",
        )

    timeout = httpx.Timeout(payload.timeout_seconds)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.request(
                method=payload.method.upper(),
                url=payload.url,
                headers=headers,
                content=content,
            )
    except Exception as exc:
        logger.exception("Upstream request failed for run_id=%s", request.run_id)
        return ApiGatewayResponse(
            success=False,
            error=f"Upstream request failed: {exc}",
        )

    response_body = resp.text
    response_headers = {
        "content-type": resp.headers.get("content-type", ""),
        "x-request-id": resp.headers.get("x-request-id", ""),
    }
    if not resp.is_success:
        return ApiGatewayResponse(
            success=False,
            error=f"Upstream status {resp.status_code}",
            status_code=resp.status_code,
            headers=response_headers,
            body=response_body,
        )
    return ApiGatewayResponse(
        success=True,
        status_code=resp.status_code,
        headers=response_headers,
        body=response_body,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "gateway.main:app",
        host=os.getenv("GATEWAY_HOST", "0.0.0.0"),
        port=int(os.getenv("GATEWAY_PORT", "8010")),
        log_level=os.getenv("GATEWAY_LOG_LEVEL", "info").lower(),
    )
