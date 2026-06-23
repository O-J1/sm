from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

import httpx


TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}


class SmithsonianAPIError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class SmithsonianClient:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        timeout: float,
        retry_limit: int,
        max_connections: int,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._retry_limit = retry_limit
        self._semaphore = asyncio.Semaphore(max_connections)
        limits = httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_connections)
        self._client = httpx.AsyncClient(timeout=timeout, limits=limits, follow_redirects=True)

    async def close(self) -> None:
        await self._client.aclose()

    async def search(
        self,
        *,
        q: str,
        start: int = 0,
        rows: int = 10,
        sort: str = "id",
        record_type: str = "edanmdm",
        row_group: str = "objects",
    ) -> dict[str, Any]:
        return await self._get_json(
            "/search",
            {
                "q": q,
                "start": start,
                "rows": rows,
                "sort": sort,
                "type": record_type,
                "row_group": row_group,
            },
        )

    async def content(self, identifier: str) -> dict[str, Any]:
        return await self._get_json(f"/content/{identifier}", {})

    async def terms(self, category: str, *, starts_with: str | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if starts_with:
            params["starts_with"] = starts_with
        return await self._get_json(f"/terms/{category}", params)

    async def stats(self) -> dict[str, Any]:
        return await self._get_json("/stats", {})

    async def _get_json(self, path: str, params: Mapping[str, Any]) -> dict[str, Any]:
        request_params = {key: value for key, value in params.items() if value is not None}
        request_params["api_key"] = self._api_key
        url = f"{self._base_url}{path}"

        async with self._semaphore:
            for attempt in range(self._retry_limit + 1):
                try:
                    response = await self._client.get(url, params=request_params)
                except httpx.HTTPError as exc:
                    if attempt >= self._retry_limit:
                        raise SmithsonianAPIError(str(exc)) from exc
                    await asyncio.sleep(_backoff_seconds(attempt))
                    continue

                if response.status_code in TRANSIENT_STATUS_CODES and attempt < self._retry_limit:
                    await asyncio.sleep(_retry_delay(response, attempt))
                    continue

                if response.status_code >= 400:
                    raise SmithsonianAPIError(
                        f"Smithsonian API returned HTTP {response.status_code}: {response.text[:500]}",
                        status_code=response.status_code,
                    )

                try:
                    payload = response.json()
                except ValueError as exc:
                    raise SmithsonianAPIError("Smithsonian API returned invalid JSON") from exc
                if not isinstance(payload, dict):
                    raise SmithsonianAPIError("Smithsonian API returned a non-object JSON payload")
                return payload

        raise SmithsonianAPIError("Smithsonian API request exhausted retries")


def response_body(payload: dict[str, Any]) -> dict[str, Any]:
    response = payload.get("response")
    if not isinstance(response, dict):
        raise SmithsonianAPIError("Smithsonian API payload did not include a response object")
    return response


def search_rows(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    body = response_body(payload)
    raw_rows = body.get("rows")
    row_count = body.get("rowCount", 0)
    if not isinstance(raw_rows, list):
        raise SmithsonianAPIError("Smithsonian search response did not include rows")
    rows = [row for row in raw_rows if isinstance(row, dict)]
    try:
        total = int(row_count)
    except (TypeError, ValueError):
        total = len(rows)
    return rows, total


def _retry_delay(response: httpx.Response, attempt: int) -> float:
    retry_after = response.headers.get("Retry-After")
    if retry_after:
        try:
            return min(float(retry_after), 120.0)
        except ValueError:
            pass
    return _backoff_seconds(attempt)


def _backoff_seconds(attempt: int) -> float:
    return min(2.0**attempt, 60.0)