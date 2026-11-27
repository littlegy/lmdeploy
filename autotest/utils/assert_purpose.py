"""Shared assert failure messages for functional tests (pytest-visible
断言目的)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def fail_msg(purpose: str, detail: Any = '') -> str:
    """Build ``assert`` message: ``断言目的：…`` plus optional detail."""
    if detail == '' or detail is None:
        return f'断言目的：{purpose}'
    return f'断言目的：{purpose} | 详情：{detail!r}'


def assert_helper(purpose: str, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
    """Run ``fn(*args, **kwargs)``; on ``AssertionError``, prepend
    *purpose*."""
    try:
        return fn(*args, **kwargs)
    except AssertionError as exc:
        raise AssertionError(f'{fail_msg(purpose)}\n{exc}') from exc
