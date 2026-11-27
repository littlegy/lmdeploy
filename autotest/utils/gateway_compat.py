"""lmdeploy autotest：Chat / Completions / Responses 请求字段与 extra_body 约定。"""

from __future__ import annotations

from typing import Any


def expand_min_output_tokens(fields: dict[str, Any], *, base_url: str | None = None) -> dict[str, Any]:
    del base_url
    out = dict(fields)
    min_val = out.pop('min_new_tokens', None)
    min_tokens = out.pop('min_tokens', None)
    merged = min_val if min_val is not None else min_tokens
    if merged is None:
        return out
    out['min_new_tokens'] = merged
    return out


def chat_sampling_extra_body(base: dict[str, Any] | None = None, /, **fields: Any) -> dict[str, Any]:
    merged: dict[str, Any] = {**(base or {}), **fields}
    return expand_min_output_tokens(merged)


def chat_max_output_kwargs(limit: int, *, base_url: str | None = None) -> dict[str, int]:
    del base_url
    return {'max_tokens': limit}


def chat_invalid_max_output_fields() -> list[tuple[dict[str, int], str]]:
    return [
        ({'max_tokens': 0}, 'max_tokens_zero'),
        ({'max_tokens': -1}, 'max_tokens_negative'),
    ]


def chat_thinking_off_template_kwargs() -> dict[str, Any]:
    return {'enable_thinking': False, 'thinking': False}


def chat_thinking_off_extra_body(*, base_url: str | None = None) -> dict[str, Any]:
    del base_url
    return chat_sampling_extra_body(
        chat_template_kwargs=chat_thinking_off_template_kwargs(),
        include_reasoning=False,
    )


def chat_protocol_probe_create_kwargs() -> dict[str, Any]:
    from utils.constant import thinking_forced_model_enabled

    if not thinking_forced_model_enabled():
        return {'extra_body': chat_thinking_off_extra_body()}
    return {
        'reasoning_effort': 'low',
        'extra_body': chat_sampling_extra_body(
            thinking={'type': 'enabled'},
            chat_template_kwargs={
                'enable_thinking': True,
                'reasoning_effort': 'low',
            },
        ),
    }


def chat_protocol_probe_max_completion_cap() -> tuple[int, int]:
    from utils.constant import thinking_forced_model_enabled

    if thinking_forced_model_enabled():
        return (64, 8)
    return (5, 1)


def responses_protocol_probe_reasoning() -> dict[str, Any]:
    from utils.constant import thinking_forced_model_enabled

    if thinking_forced_model_enabled():
        return {'effort': 'low'}
    return {'effort': 'none'}


def messages_protocol_probe_thinking() -> dict[str, Any]:
    from utils.constant import thinking_forced_model_enabled

    if thinking_forced_model_enabled():
        return {'type': 'enabled'}
    return {'type': 'disabled'}


def chat_tools_create_kwargs(
    tools: list | None,
    tool_choice,
    *,
    base_url: str | None = None,
) -> dict[str, Any]:
    del base_url
    if tools is None:
        return {}
    return {'tools': tools, 'tool_choice': tool_choice}


def responses_tool_choice_kwargs(
    tool_choice,
    tools: list | None = None,
    *,
    base_url: str | None = None,
) -> dict[str, Any]:
    del base_url
    out: dict[str, Any] = {'tool_choice': tool_choice}
    if tools is not None:
        out['tools'] = tools
    return out


def chat_low_repetition_extra_body() -> dict[str, Any]:
    """极低 repetition + min_new_tokens（Chat extra_body / Completions 同源）。"""
    from utils.constant import CHAT_LOW_REPETITION_PENALTY, LOW_REPETITION_MIN_NEW_TOKENS

    return chat_sampling_extra_body(
        repetition_penalty=CHAT_LOW_REPETITION_PENALTY,
        min_new_tokens=LOW_REPETITION_MIN_NEW_TOKENS,
    )


def chat_repetition_above_one_extra_body() -> dict[str, Any]:
    from utils.constant import REPETITION_PENALTY_ABOVE_ONE

    return chat_sampling_extra_body(repetition_penalty=REPETITION_PENALTY_ABOVE_ONE)


def responses_low_repetition_extra_body() -> dict[str, Any]:
    """Responses 无 min_new_tokens：用 ignore_eos 拉长输出便于检出重复。"""
    from utils.constant import CHAT_LOW_REPETITION_PENALTY

    return {
        'repetition_penalty': CHAT_LOW_REPETITION_PENALTY,
        'ignore_eos': True,
    }


def engine_repetition_penalty_field(repetition_penalty: float | None = None) -> dict[str, float]:
    from utils.constant import REPETITION_PENALTY_ABOVE_ONE

    value = REPETITION_PENALTY_ABOVE_ONE if repetition_penalty is None else repetition_penalty
    return {'repetition_penalty': value}


def engine_top_k_extra_body(top_k: int, *, thinking_off: bool = True) -> dict[str, Any]:
    """Chat Responses 共用 top_k extra_body；Messages 用 engine_top_k_fields 写 JSON 顶栏。"""
    sampling = chat_sampling_extra_body(top_k=top_k)
    if not thinking_off:
        return sampling
    return {**chat_thinking_off_extra_body(), **sampling}


def engine_top_k_fields(top_k: int) -> dict[str, int]:
    return {'top_k': top_k}