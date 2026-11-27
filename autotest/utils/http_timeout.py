"""HTTP / SSE timeouts and partial-stream capture for gateway triage."""

from __future__ import annotations

import json
import os
import time
from typing import Any

_DEFAULT_READ_SEC = 600.0
_DEFAULT_CONNECT_SEC = 30.0


def endpoint_http_read_timeout_sec() -> float:
    """Max wait per read (stream: between SSE chunks; batch: full body)."""
    raw = os.environ.get('ENDPOINT_HTTP_TIMEOUT_SEC', '').strip()
    if not raw:
        return _DEFAULT_READ_SEC
    try:
        value = float(raw)
    except ValueError:
        return _DEFAULT_READ_SEC
    return max(1.0, value)


def endpoint_http_connect_timeout_sec() -> float:
    raw = os.environ.get('ENDPOINT_HTTP_CONNECT_TIMEOUT_SEC', '').strip()
    if not raw:
        return _DEFAULT_CONNECT_SEC
    try:
        value = float(raw)
    except ValueError:
        return _DEFAULT_CONNECT_SEC
    return max(1.0, value)


def openai_client_timeout():
    """``timeout`` argument for ``openai.OpenAI`` (httpx)."""
    import httpx

    read = endpoint_http_read_timeout_sec()
    connect = endpoint_http_connect_timeout_sec()
    return httpx.Timeout(connect=connect, read=read, write=connect, pool=connect)


def requests_timeout_tuple() -> tuple[float, float]:
    """``requests`` ``timeout=(connect, read)`` for streaming POSTs."""
    return (endpoint_http_connect_timeout_sec(), endpoint_http_read_timeout_sec())


def is_timeout_error(exc: BaseException) -> bool:
    if exc.__class__.__name__ in {
        'ReadTimeout',
        'ConnectTimeout',
        'TimeoutException',
        'APITimeoutError',
    }:
        return True
    cause = getattr(exc, '__cause__', None)
    if cause is not None and cause is not exc:
        return is_timeout_error(cause)
    return False


class StreamPartialTimeoutError(TimeoutError):
    """Raised when a stream ends early; carries partial payload for logs."""

    def __init__(
        self,
        message: str,
        *,
        chunk_count: int = 0,
        log_file: str | None = None,
        partial: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.chunk_count = chunk_count
        self.log_file = log_file
        self.partial = partial or {}


def append_timeout_artifact(
    log_file: str | None,
    exc: BaseException,
    *,
    stream: bool,
    chunk_count: int = 0,
    recent_chunk_reprs: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    if not log_file:
        return
    lines = [
        '',
        '=== HTTP TIMEOUT (partial data below) ===',
        f'time_utc: {time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}',
        f'stream: {stream}',
        f'chunk_count: {chunk_count}',
        f'error: {exc!r}',
    ]
    if recent_chunk_reprs:
        lines.append(f'recent_chunks: {len(recent_chunk_reprs)}')
        for i, text in enumerate(recent_chunk_reprs[-20:], 1):
            lines.append(f'  chunk[{i}]: {text[:4000]}')
    if extra:
        try:
            lines.append('extra_json: ' + json.dumps(extra, ensure_ascii=False)[:12000])
        except (TypeError, ValueError):
            lines.append(f'extra: {extra!r}'[:8000])
    try:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')
    except OSError:
        pass


def append_http_sse_partial(
    log_file: str | None,
    exc: BaseException,
    *,
    raw_lines: list[str],
    raw_sse_chunks: list[dict],
    result: dict[str, Any],
) -> None:
    extra = {
        'sse_lines': len(raw_lines),
        'parsed_chunks': len(raw_sse_chunks),
        'stream_complete': result.get('stream_complete'),
        'finish_reason': result.get('finish_reason'),
        'last_raw_lines': raw_lines[-30:],
        'last_sse_samples': raw_sse_chunks[-5:],
    }
    append_timeout_artifact(
        log_file,
        exc,
        stream=True,
        chunk_count=result.get('chunk_count') or len(raw_sse_chunks),
        extra=extra,
    )
