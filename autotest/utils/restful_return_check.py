import json
import math
import os
import re
from collections.abc import Iterator
from functools import lru_cache
from typing import Any

import requests
from openai import OpenAI
from utils.assert_purpose import fail_msg
from utils.config_utils import (
    _entry_engine_config,
    get_model_path_from_config,
    iter_model_yaml_entries,
)
from utils.constant import BASE_URL, VISION_SMOKE_MIN_COMPLETION_TOKENS
from utils.gateway_compat import chat_protocol_probe_create_kwargs
from utils.toolkit import _load_tokenizer_cached, encode_text

# Preprocess rejects oversize input with this OpenAI error substring.
CONTEXT_LENGTH_ERROR = 'context length'

def assert_chat_stream_completion_token_cap(
    all_chunks: list[dict],
    max_completion_tokens: int,
    *,
    slack: int = 1,
) -> None:
    """Stream ended at length cap: ``usage.completion_tokens`` within ``max ± slack``."""
    usage = chat_stream_final_usage(all_chunks)
    assert usage is not None, fail_msg(
        '流式 length 截断须含 usage（需 stream_options.include_usage=True）',
        all_chunks[-3:],
    )
    completion = usage['completion_tokens']
    assert max_completion_tokens <= completion <= max_completion_tokens + slack, fail_msg(
        '流式在 max_tokens 截断时 completion_tokens 须在 [max, max+slack] 内',
        f'completion_tokens={completion} range=[{max_completion_tokens}, '
        f'{max_completion_tokens + slack}]',
    )
    payload_chunks = chat_stream_payload_chunks(all_chunks)
    assert payload_chunks, fail_msg('流式须至少一条含 choices 的 chunk', all_chunks)
    assert payload_chunks[-1]['choices'][0]['finish_reason'] == 'length', fail_msg(
        '流式在生成长度上限结束须 finish_reason=length',
        payload_chunks[-1],
    )


def chat_stream_final_usage(output_list: list[dict]) -> dict | None:
    """Last ``usage`` object from a chat completion stream (may be usage-only
    chunk)."""
    usage = None
    for item in output_list:
        chunk_usage = item.get('usage')
        if chunk_usage:
            usage = chunk_usage
    return usage


def assert_chat_stream_include_usage(chunks: list[dict]) -> dict:
    """Assert ``stream_options.include_usage`` chunks (OpenAI-compatible
    stream).

    ``completion_tokens`` is non-decreasing; the final usage matches
    ``prompt_tokens + completion_tokens``.
    """
    last_completion = 0
    final_usage = None
    saw_usage = False
    for chunk in chunks:
        usage = chunk.get('usage')
        if not usage:
            continue
        saw_usage = True
        prompt = usage.get('prompt_tokens')
        completion = usage.get('completion_tokens')
        total = usage.get('total_tokens')
        assert prompt is not None and completion is not None and total is not None, fail_msg(
            'include_usage 流式 chunk 的 usage 须含 prompt/completion/total', chunk)
        assert completion >= last_completion, fail_msg(
            '流式 usage.completion_tokens 须单调非递减',
            f'{last_completion} -> {completion}',
        )
        assert total == prompt + completion, fail_msg(
            '流式 usage.total_tokens 须等于 prompt + completion', usage)
        last_completion = completion
        final_usage = usage
    assert saw_usage, fail_msg(
        'include_usage 流式须至少一条带 usage 的 chunk',
        'set stream_options.include_usage=True',
    )
    return final_usage


def _choice_output_ids(container: dict) -> list[int] | None:
    """Lmdeploy ``return_token_ids`` → ``choices[].output_ids``
    (``protocol.ChatCompletionResponseChoice``)."""
    value = container.get('output_ids')
    if isinstance(value, list) and value:
        return value
    return None


def completion_choice_token_ids(choice: dict) -> list[int] | None:
    """Completion choice ``output_ids`` when supported (chat path is canonical
    in lmdeploy)."""
    return _choice_output_ids(choice)


def chat_choice_token_ids(choice: dict) -> list[int] | None:
    """Non-stream chat choice, ``message``, or stream ``delta``
    ``output_ids``."""
    for container in (choice, choice.get('message') or {}, choice.get('delta') or {}):
        if not isinstance(container, dict):
            continue
        ids = _choice_output_ids(container)
        if ids:
            return ids
    return None


def assert_chat_return_token_ids(body: dict) -> None:
    """Non-stream chat with ``return_token_ids=True``."""
    choice = body['choices'][0]
    token_ids = chat_choice_token_ids(choice)
    assert token_ids, fail_msg(
        'return_token_ids 响应须含 choices[].output_ids',
        choice,
    )
    usage = body['usage']
    completion_tokens = usage['completion_tokens']
    assert len(token_ids) == completion_tokens, fail_msg(
        'token_ids 长度须与 usage.completion_tokens 一致',
        f'len(token_ids)={len(token_ids)} completion_tokens={completion_tokens}',
    )
    msg = choice.get('message') or {}
    visible = chat_assistant_visible_text(msg)
    assert visible, fail_msg('return_token_ids 响应须含非空 assistant 文本', msg)


def assert_completions_return_token_ids(body: dict) -> None:
    """Non-stream completion with ``return_token_ids=True`` (``output_ids``
    when present)."""
    choice = body['choices'][0]
    token_ids = completion_choice_token_ids(choice)
    assert token_ids, fail_msg(
        'return_token_ids 响应 choice 须含 output_ids', choice.keys())
    usage = body['usage']
    completion_tokens = usage['completion_tokens']
    assert len(token_ids) == completion_tokens, fail_msg(
        'token_ids 长度须与 usage.completion_tokens 一致',
        f'len(token_ids)={len(token_ids)} completion_tokens={completion_tokens}',
    )
    text = choice.get('text')
    assert isinstance(text, str) and text, fail_msg(
        'return_token_ids 响应须含非空 text', choice)


def assert_chat_sse_stream_ends_with_done(resp: requests.Response) -> None:
    """Raw chat SSE must terminate with ``data: [DONE]`` (OpenAI §2.3)."""
    assert resp.status_code == 200, fail_msg(
        'Chat 流式探针须返回 HTTP 200',
        f'status={resp.status_code} body={resp.text[:300]!r}',
    )
    saw_payload = False
    saw_done = False
    try:
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw:
                continue
            line = raw.strip()
            if line == 'data: [DONE]':
                saw_done = True
                break
            if line.startswith('data: ') and line != 'data: [DONE]':
                saw_payload = True
    finally:
        resp.close()
    assert saw_payload, fail_msg('Chat 流式须至少一条 data JSON chunk', saw_payload)
    assert saw_done, fail_msg('Chat 流式须以 data: [DONE] 结束', saw_done)


def chat_stream_payload_chunks(output_list: list[dict]) -> list[dict]:
    """Drop usage-only SSE chunks with empty ``choices``."""
    chunks = [item for item in output_list if item.get('choices')]
    assert chunks, fail_msg('Chat 流式须至少一条含 choices 的 chunk', output_list[-3:])
    return chunks


def completion_stream_payload_chunks(output_list: list[dict]) -> list[dict]:
    """Drop trailing usage-only completion stream chunks."""
    chunks = [item for item in output_list if item.get('choices')]
    assert chunks, fail_msg('Completions 流式须至少一条含 choices 的 chunk', output_list[-3:])
    return chunks


def aggregate_completion_stream_text(chunks: list[dict]) -> str:
    text = ''
    for item in chunks:
        for choice in item['choices']:
            piece = choice.get('text')
            if isinstance(piece, str):
                text += piece
    return text


def wire_checked_response_json(response: requests.Response) -> Any:
    return json.loads(response.text)


def post_openai_json(
    path: str,
    model_name: str,
    payload: dict,
    *,
    timeout: int | float = 600,
    **request_kwargs: Any,
) -> Any:
    """POST OpenAI-compatible route (non-stream) with wire UTF-8 literal
    checks."""
    url = f'{BASE_URL}{path}'
    body = dict(payload)
    body['model'] = model_name
    resp = requests.post(
        url,
        json=body,
        headers=gateway_auth_headers(),
        timeout=timeout,
        **request_kwargs,
    )
    resp.raise_for_status()
    return wire_checked_response_json(resp)


def post_completions_json(model_name: str, payload: dict, **request_kwargs: Any) -> dict:
    return post_openai_json('/v1/completions', model_name, payload, **request_kwargs)


def post_chat_completions_json(model_name: str, payload: dict, **request_kwargs: Any) -> dict:
    return post_openai_json('/v1/chat/completions', model_name, payload, **request_kwargs)


def post_responses_json(model_name: str, payload: dict, **request_kwargs: Any) -> dict:
    return post_openai_json('/v1/responses', model_name, payload, **request_kwargs)


def iter_openai_sse_data_json(response) -> Iterator[tuple[str, dict]]:
    """Yield ``(raw_data_payload, parsed_json)`` for each ``data:`` SSE
    line."""
    for line in response.iter_lines(decode_unicode=True):
        if not line or not line.startswith('data: '):
            continue
        data = line[6:]
        if data == '[DONE]':
            break
        try:
            item = json.loads(data)
        except json.JSONDecodeError:
            continue
        if not isinstance(item, dict):
            continue
        yield data, item


def collect_openai_sse_data_chunks(response) -> list[dict]:
    return [item for _, item in iter_openai_sse_data_json(response)]


def post_chat_completions_stream_chunks(
    model_name: str,
    payload: dict,
    *,
    choices_only: bool = True,
) -> list[dict]:
    """POST ``/v1/chat/completions`` with ``stream=true`` and wire UTF-8
    literal checks."""
    url = f'{BASE_URL}/v1/chat/completions'
    body = dict(payload)
    body['model'] = model_name
    body['stream'] = True
    resp = requests.post(url, json=body, headers=gateway_auth_headers(), stream=True, timeout=600)
    resp.raise_for_status()
    chunks = collect_openai_sse_data_chunks(resp)
    return chat_stream_payload_chunks(chunks) if choices_only else chunks


def post_completions_stream_chunks(model_name: str, payload: dict) -> list[dict]:
    """POST ``/v1/completions`` with ``stream=true`` and wire UTF-8 literal
    checks."""
    url = f'{BASE_URL}/v1/completions'
    body = dict(payload)
    body['model'] = model_name
    body['stream'] = True
    resp = requests.post(url, json=body, headers=gateway_auth_headers(), stream=True, timeout=600)
    resp.raise_for_status()
    return completion_stream_payload_chunks(collect_openai_sse_data_chunks(resp))


def iter_responses_sse_event_json(response) -> Iterator[dict]:
    """Yield parsed ``data:`` JSON for Responses ``event:`` / ``data:``
    streams."""
    for line in response.iter_lines(decode_unicode=True):
        if not line:
            continue
        if line.startswith('data:'):
            data_str = line[len('data:'):].strip()
            if not data_str:
                continue
            try:
                item = json.loads(data_str)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                yield item


def post_responses_stream_events(model_name: str, payload: dict) -> list[dict]:
    """POST ``/v1/responses`` with ``stream=true`` and wire UTF-8 literal
    checks."""
    url = f'{BASE_URL}/v1/responses'
    body = dict(payload)
    body['model'] = model_name
    body['stream'] = True
    resp = requests.post(url, json=body, headers=gateway_auth_headers(), stream=True, timeout=600)
    resp.raise_for_status()
    return list(iter_responses_sse_event_json(resp))

# Upper bound for "large but valid" payload tests (CI scale, not full context).
CI_LARGE_PAYLOAD_TOKEN_CAP = 32_000

# Avoid rebuilding identical multi‑100k-token strings across parametrized cases.
_SESSION_SIZED_CONTENT_CACHE: dict[tuple[Any, ...], str] = {}

_MAX_GROW_TEXT_CHARS = 50_000_000


def _count_tokens(model_path: str, text: str, *, add_special_tokens: bool) -> int:
    return len(encode_text(model_path, text, add_special_tokens=add_special_tokens))


def grow_text_to_at_least_token_count(
    model_path: str,
    target: int,
    *,
    unit: str = 'x',
    add_special_tokens: bool = False,
) -> str:
    """Build the shortest ``unit`` repetition whose token count is >=
    ``target``."""
    if target <= 0:
        return ''
    probe_len = 4096
    probe_tokens = _count_tokens(
        model_path, unit * probe_len, add_special_tokens=add_special_tokens)
    chars_per_token = probe_len / max(probe_tokens, 1)

    lo = 1
    hi = max(int(target * chars_per_token * 1.5) + probe_len, probe_len * 2)
    while _count_tokens(model_path, unit * hi, add_special_tokens=add_special_tokens) < target:
        hi *= 2
        if hi > _MAX_GROW_TEXT_CHARS:
            raise RuntimeError(
                f'cannot synthesize >={target} tokens (reached {hi} chars)',
            )

    best = unit * hi
    while lo <= hi:
        mid = (lo + hi) // 2
        if mid <= 0:
            break
        candidate = unit * mid
        if _count_tokens(model_path, candidate, add_special_tokens=add_special_tokens) >= target:
            best = candidate
            hi = mid - 1
        else:
            lo = mid + 1
    return best


def _shrink_unit_text_below_token_count(
    model_path: str,
    char_len: int,
    limit: int,
    *,
    unit: str = 'x',
    add_special_tokens: bool = False,
) -> str:
    """Longest ``unit`` repetition with token count strictly below
    ``limit``."""
    if char_len <= 0:
        return ''
    lo, hi = 0, char_len
    best = ''
    while lo <= hi:
        mid = (lo + hi) // 2
        if mid <= 0:
            break
        candidate = unit * mid
        if _count_tokens(model_path, candidate, add_special_tokens=add_special_tokens) >= limit:
            hi = mid - 1
        else:
            best = candidate
            lo = mid + 1
    return best

# Legacy anthropic large-payload test body size (128 KiB repeated filler).
CI_LARGE_PAYLOAD_CHAR_BUDGET = 128 * 1024


def _error_message_from_body(body: dict) -> str:
    if body.get('message'):
        return str(body['message'])
    err = body.get('error')
    if isinstance(err, dict) and err.get('message'):
        return str(err['message'])
    if body.get('detail'):
        return str(body['detail'])
    return ''


def assert_openai_invalid_request_error(
    response: requests.Response | dict,
    *,
    message_substr: str | None = None,
    strict_openai_shape: bool | None = None,
) -> dict:
    """Assert invalid request rejected with HTTP 400.

    lmdeploy: OpenAI ``invalid_request_error`` object shape.
    Other gateways: 400 + non-empty error message (SDK body or JSON).
    """
    if strict_openai_shape is None:
        strict_openai_shape = True

    if isinstance(response, requests.Response):
        assert response.status_code == 400, fail_msg(
            'OpenAI 形非法请求须返回 HTTP 400',
            f'expected 400, got {response.status_code}: {response.text[:500]}',
        )
        body = response.json()
    else:
        body = response

    if strict_openai_shape:
        assert body.get('object') == 'error', fail_msg(
            'lmdeploy 非法请求错误体须含 object=error', body)
        assert body.get('type') == 'invalid_request_error', fail_msg(
            '非法请求错误 type 须为 invalid_request_error', body)
        assert body.get('code') == 400, fail_msg('非法请求错误 code 须为 400', body)
        message = body.get('message')
    else:
        message = _error_message_from_body(body)
    assert message, fail_msg(
        '非法请求错误体须含非空 message',
        repr(body)[:500],
    )
    if message_substr is not None:
        assert message_substr.lower() in message.lower(), fail_msg(
            f'错误 message 须包含预期片段 {message_substr!r}',
            message,
        )
    return body


def _chat_reasoning_and_content(part: dict) -> tuple[str | None, str | None]:
    """Thinking text: lmdeploy ``reasoning_content`` on message/delta."""
    reasoning = part.get('reasoning_content')
    reasoning = reasoning if isinstance(reasoning, str) else None
    content = part.get('content')
    return reasoning, content if isinstance(content, str) else None


def get_chat_message_text(choice):
    msg = choice.get('message') or {}
    reasoning, content = _chat_reasoning_and_content(msg)
    return ''.join(piece for piece in (reasoning, content) if isinstance(piece, str))


def get_chat_delta_text(choice):
    delta = choice.get('delta') or {}
    reasoning, content = _chat_reasoning_and_content(delta)
    return ''.join(piece for piece in (reasoning, content) if isinstance(piece, str))


def assert_chat_message_empty(choice):
    assert not get_chat_message_text(choice), fail_msg(
        'Chat 非流式 choice 的 assistant 可见文本须为空', choice)


def assert_chat_delta_empty(choice):
    assert not get_chat_delta_text(choice), fail_msg(
        'Chat 流式 chunk 的 delta 可见文本须为空', choice)


def assert_chat_completions_batch_return(
        output,
        model_name,
        check_logprobs: bool = False,
        logprobs_num: int = 5,
        *,
        expected_choice_count: int = 1,
):
    assert_usage(output.get('usage'))
    assert output.get('id') is not None, fail_msg(
        'Chat 非流式成功响应须含非空 id', output)
    assert output.get('object') == 'chat.completion', fail_msg(
        'Chat 非流式 object 须为 chat.completion', output)
    assert output.get('model') == model_name, fail_msg(
        'Chat 响应 model 须与请求一致', output)
    output_message = output.get('choices')
    assert len(output_message) == expected_choice_count, fail_msg(
        f'Chat 非流式须恰好 {expected_choice_count} 条 choice',
        output,
    )
    for message in output_message:
        assert message.get('finish_reason') in ['stop', 'length'], fail_msg(
            'Chat choice.finish_reason 须为 stop 或 length', message)
        index = message.get('index')
        assert index in range(expected_choice_count), fail_msg(
            f'Chat choice.index 须在 [0, {expected_choice_count})', message)
        msg = message.get('message') or {}
        reasoning, content = _chat_reasoning_and_content(msg)
        assert (isinstance(content, str) and len(content) > 0) or (
            isinstance(reasoning, str) and len(reasoning) > 0), fail_msg(
            'Chat assistant 须含非空 content 或 reasoning_content', msg)
        assert msg.get('role') == 'assistant', fail_msg(
            'Chat assistant message.role 须为 assistant', msg)
        if check_logprobs:
            assert len(message.get('logprobs').get('content')) == output.get('usage').get(
                'completion_tokens'), fail_msg(
                'Chat logprobs 条目数须等于 completion_tokens',
                message.get('logprobs'),
            )
            for logprob in message.get('logprobs').get('content'):
                assert_logprobs(logprob, logprobs_num)


def assert_completions_batch_return(
        output,
        model_name,
        check_logprobs: bool = False,
        logprobs_num: int = 5,
):
    assert_usage(output.get('usage'))
    assert output.get('id') is not None, fail_msg(
        'Completions 非流式成功响应须含非空 id', output)
    assert output.get('object') == 'text_completion', fail_msg(
        'Completions object 须为 text_completion', output)
    assert output.get('model') == model_name, fail_msg(
        'Completions 响应 model 须与请求一致', output)
    output_message = output.get('choices')
    assert len(output_message) == 1, fail_msg(
        'Completions 非流式须恰好一条 choice', output)
    for message in output_message:
        assert message.get('finish_reason') in ['stop', 'length'], fail_msg(
            'Completions finish_reason 须为 stop 或 length', message)
        assert message.get('index') == 0, fail_msg(
            'Completions 单 choice index 须为 0', message)
        assert len(message.get('text')) > 0, fail_msg(
            'Completions 须含非空补全 text', message)
        if check_logprobs:
            assert len(message.get('logprobs').get('content')) == output.get('usage').get(
                'completion_tokens'), fail_msg(
                'Completions logprobs 条目数须等于 completion_tokens',
                message.get('logprobs'),
            )
            for logprob in message.get('logprobs').get('content'):
                assert_logprobs(logprob, logprobs_num)


def assert_usage(usage):
    assert usage.get('prompt_tokens') > 0, fail_msg(
        '成功补全 usage.prompt_tokens 须为正值', usage)
    assert usage.get('total_tokens') > 0, fail_msg(
        '成功补全 usage.total_tokens 须为正值', usage)
    assert usage.get('completion_tokens') > 0, fail_msg(
        '成功补全 usage.completion_tokens 须为正值', usage)
    assert usage.get('completion_tokens') + usage.get('prompt_tokens') == usage.get(
        'total_tokens'), fail_msg(
        'usage.total_tokens 须等于 prompt_tokens + completion_tokens', usage)


def usage_cached_tokens(usage: dict, *, kind: str = 'chat') -> int | None:
    """Read ``cached_tokens`` from OpenAI usage (chat or responses).

    Returns ``None`` when the server omits prompt/input token details.
    """
    if kind == 'responses':
        details = usage.get('input_tokens_details')
    elif kind == 'chat':
        details = usage.get('prompt_tokens_details')
    else:
        raise ValueError(f'unsupported usage kind: {kind}')
    if not isinstance(details, dict):
        return None
    value = details.get('cached_tokens')
    return value if isinstance(value, int) and not isinstance(value, bool) else None


# OpenAI-compatible model-not-found wording (lmdeploy api_server uses similar phrases).
OPENAI_MODEL_NOT_FOUND_RE = re.compile(
    r'(?i)(model.*does not exist|model.*not found|'
    r"model.*['\"]error['\"].*(not found|does not exist))"
)


def assert_responses_usage(usage):
    assert usage['input_tokens'] > 0, fail_msg(
        'Responses usage.input_tokens 须为正值', usage)
    assert usage['output_tokens'] > 0, fail_msg(
        'Responses usage.output_tokens 须为正值', usage)
    assert usage['total_tokens'] == usage['input_tokens'] + usage['output_tokens'], fail_msg(
        'Responses usage.total 须等于 input + output', usage)
    details = usage.get('output_tokens_details')
    if isinstance(details, dict) and 'reasoning_tokens' in details:
        assert details['reasoning_tokens'] >= 0, fail_msg(
            'Responses reasoning_tokens 须非负', details)


_RESPONSES_REASONING_BLOCK_TYPES = frozenset({'reasoning_text', 'text'})


def _responses_reasoning_text_blocks(item: dict) -> list[str]:
    """Reasoning body from a ``type=reasoning`` output item (may use
    ``summary``)."""
    chunks: list[str] = []
    for block in item.get('content') or []:
        if not isinstance(block, dict):
            continue
        if block.get('type') not in _RESPONSES_REASONING_BLOCK_TYPES:
            continue
        text = block.get('text')
        if isinstance(text, str):
            chunks.append(text)
    for part in item.get('summary') or []:
        if not isinstance(part, dict):
            continue
        text = part.get('text')
        if isinstance(text, str):
            chunks.append(text)
    return chunks


def responses_reasoning_text(output: dict) -> str:
    """Concatenate reasoning output items (empty when none)."""
    parts: list[str] = []
    for item in output.get('output') or []:
        if isinstance(item, dict) and item.get('type') == 'reasoning':
            parts.extend(_responses_reasoning_text_blocks(item))
    return ''.join(parts)


def responses_completed_streamable_text(output: dict) -> str:
    """Message + reasoning text as returned on a completed Responses body."""
    return responses_output_text(output) + responses_reasoning_text(output)


def responses_stream_text(events) -> str:
    """Aggregate streamed assistant-visible text (message + reasoning
    deltas)."""
    delta_types = frozenset({
        'response.output_text.delta',
        'response.reasoning_text.delta',
    })
    parts: list[str] = []
    for event in events:
        etype = getattr(event, 'type', None)
        if etype is None and isinstance(event, dict):
            etype = event.get('type')
        if etype not in delta_types:
            continue
        delta = getattr(event, 'delta', None)
        if delta is None and isinstance(event, dict):
            delta = event.get('delta')
        if isinstance(delta, str):
            parts.append(delta)
    return ''.join(parts)


def _responses_message_text_blocks(item: dict) -> list[str]:
    """Assistant message text from ``output[]`` content blocks."""
    chunks: list[str] = []
    for block in item.get('content') or []:
        if not isinstance(block, dict):
            continue
        block_type = block.get('type')
        if block_type not in ('output_text', 'text'):
            continue
        text = block.get('text')
        if isinstance(text, str):
            chunks.append(text)
    return chunks


def responses_output_text(output: dict, *, require_nonempty: bool = False) -> str:
    """Canonical assistant text for ``POST /v1/responses``.

    lmdeploy Text V1 mirrors the first message in top-level ``output_text``;
    otherwise read ``output[].content[]``.
    """
    derived = ''
    for item in output.get('output') or []:
        if not isinstance(item, dict) or item.get('type') != 'message':
            continue
        derived += ''.join(_responses_message_text_blocks(item))
    top = output.get('output_text')
    if isinstance(top, str) and top:
        if derived:
            assert top == derived, fail_msg(
                'Responses 顶层 output_text 须与 output[] 推导文本一致',
                f'top={top!r} derived={derived!r}',
            )
        return top
    if require_nonempty:
        assert derived, fail_msg('Responses 须含非空 assistant 可见文本', output)
    return derived


def assert_responses_batch_return(
        output,
        model_name,
        *,
        status: str = 'completed',
):
    """Assert ``POST /v1/responses`` JSON body.

    Returns ``function_call`` output items (possibly empty).
    """
    assert output['id'], fail_msg('Responses 成功体须含非空 id', output)
    assert output['object'] == 'response', fail_msg('Responses object 须为 response', output)
    assert output['model'] == model_name, fail_msg('Responses model 须与请求一致', output)
    assert output['status'] == status, fail_msg(
        f'Responses status 须为 {status!r}',
        f'got={output.get("status")!r} incomplete={output.get("incomplete_details")!r} '
        f'error={output.get("error")!r} output_items={len(output.get("output") or [])}',
    )
    assert_responses_usage(output['usage'])
    output_items = output['output']
    assert output_items, fail_msg(
        'Responses 成功体 output[] 须非空',
        f'status={output.get("status")!r} error={output.get("error")!r}',
    )
    messages = []
    function_calls = []
    for item in output_items:
        assert item['type'] in ('message', 'function_call', 'reasoning'), fail_msg(
            'Responses output item type 须为 message/function_call/reasoning', item)
        if item['type'] == 'reasoning':
            reasoning_body = ''.join(_responses_reasoning_text_blocks(item))
            encrypted = item.get('encrypted_content')
            assert reasoning_body or encrypted, fail_msg(
                'Responses reasoning 项须含可见 reasoning 或 encrypted_content', item)
            continue
        if item['type'] == 'message':
            assert item['role'] == 'assistant', fail_msg(
                'Responses message.role 须为 assistant', item)
            content = item['content']
            assert content, fail_msg('Responses message.content 须非空', item)
            first = content[0]
            assert first['type'] in ('output_text', 'text'), fail_msg(
                'Responses 文本块 type 须为 output_text 或 text', first)
            assert isinstance(first['text'], str), fail_msg(
                'Responses 文本块 text 须为 str', first)
            messages.append(item)
        else:
            assert item['name'], fail_msg('Responses function_call 须含 name', item)
            assert item['call_id'], fail_msg('Responses function_call 须含 call_id', item)
            assert isinstance(item['arguments'], str), fail_msg(
                'Responses function_call.arguments 须为 str', item)
            function_calls.append(item)
    if messages:
        message_text = ''.join(_responses_message_text_blocks(messages[0]))
        top = output.get('output_text')
        if isinstance(top, str):
            assert top == message_text, fail_msg(
                'Responses output_text 须与首条 message 文本一致',
                {'top': top, 'message_text': message_text},
            )
    if status == 'completed':
        assert output.get('incomplete_details') is None, fail_msg(
            'completed Responses 不得含 incomplete_details', output.get('incomplete_details'))
        if not function_calls:
            visible = responses_output_text(output, require_nonempty=False)
            reasoning = responses_reasoning_text(output)
            assert (visible.strip() or reasoning.strip()), fail_msg(
                'completed 且无 tool call 时须有可见 message 或 reasoning 文本', output)
    else:
        assert status == 'incomplete', fail_msg(
            '非 completed 时本 helper 仅支持 status=incomplete', status)
        msg_status = messages[0].get('status') if messages else None
        if msg_status is not None:
            assert msg_status == 'incomplete', fail_msg(
                'incomplete Responses 的 message.status 须为 incomplete', msg_status)
        incomplete = output.get('incomplete_details') or {}
        assert incomplete.get('reason') == 'max_output_tokens', fail_msg(
            'incomplete Responses 须因 max_output_tokens 截断', incomplete)
    return function_calls


def assert_responses_error(
    response: requests.Response,
    *,
    status_code: int,
    error_type: str,
    param: str | None = None,
    message_substr: str | None = None,
    strict_text_v1_shape: bool | None = None,
) -> dict:
    """Assert Responses HTTP error (strict lmdeploy Text V1 or generic
    gateway)."""
    if strict_text_v1_shape is None:
        strict_text_v1_shape = True

    assert response.status_code == status_code, fail_msg(
        f'Responses 错误须返回 HTTP {status_code}',
        response.text[:500],
    )
    body = response.json()
    if strict_text_v1_shape:
        err = body['error']
        assert err['message'], fail_msg('Responses 错误须含非空 error.message', err)
        assert err['code'] == status_code, fail_msg(
            'Responses error.code 须与 HTTP 状态码一致', err)
        assert err['type'] == error_type, fail_msg(
            f'Responses error.type 须为 {error_type!r}', err)
        if param is not None:
            assert err['param'] == param, fail_msg(
                f'Responses error.param 须为 {param!r}', err)
        if message_substr is not None:
            assert message_substr in err['message'], fail_msg(
                f'Responses 错误 message 须包含 {message_substr!r}', err['message'])
        return body

    message = _error_message_from_body(body)
    assert message, fail_msg('Responses 错误体须含可解析的错误 message', repr(body)[:500])
    err = body.get('error') if isinstance(body.get('error'), dict) else {}
    if param is not None and err.get('param') is not None:
        actual = str(err['param'])
        # FastAPI may return dotted ``param`` paths; lmdeploy Text V1 may use short names.
        segments = actual.split('.')
        param_ok = (
            actual == param
            or actual.endswith(f'.{param}')
            or actual.endswith(param)
            or param in segments
        )
        assert param_ok, fail_msg(
            f'Responses 错误 param 须匹配 {param!r}', actual)
    if message_substr is not None:
        assert message_substr in message, fail_msg(
            f'Responses 错误 message 须包含 {message_substr!r}', message)
    return body


def assert_logprobs(logprobs, logprobs_num):
    assert_logprob_element(logprobs)
    assert len(logprobs.get('top_logprobs')) >= 0, fail_msg(
        'logprobs.top_logprobs 长度须非负', logprobs)
    assert type(logprobs.get('top_logprobs')) is list, fail_msg(
        'logprobs.top_logprobs 须为 list', logprobs)
    assert len(logprobs.get('top_logprobs')) <= logprobs_num, fail_msg(
        f'logprobs.top_logprobs 长度须 ≤ {logprobs_num}', logprobs)
    for logprob_element in logprobs.get('top_logprobs'):
        assert_logprob_element(logprob_element)


def assert_logprob_element(logprob):
    assert len(logprob.get('token')) > 0 and type(logprob.get('token')) is str, fail_msg(
        'logprob 元素须含非空 str token', logprob)
    assert len(logprob.get('bytes')) > 0 and type(logprob.get('bytes')) is list, fail_msg(
        'logprob 元素须含非空 bytes 列表', logprob)
    assert type(logprob.get('logprob')) is float, fail_msg(
        'logprob 元素 logprob 须为 float', logprob)


def chat_assistant_visible_text(message: dict) -> str:
    """Non-empty assistant text from ``content`` or ``reasoning_content``."""
    content = message.get('content')
    if isinstance(content, str) and content.strip():
        return content.strip()
    reasoning = message.get('reasoning_content')
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning.strip()
    return ''


def assert_chat_vision_smoke_choice(choice: dict, *, raw: str = '') -> str:
    """Vision accept smoke: visible answer + hint when budget exhausted in reasoning."""
    message = choice.get('message') or {}
    finish = choice.get('finish_reason')
    visible = chat_assistant_visible_text(message)
    if visible:
        return visible
    hint = (
        f'finish_reason={finish!r}; thinking models may use the full completion budget '
        f'in reasoning before content — set max_tokens>={VISION_SMOKE_MIN_COMPLETION_TOKENS}'
    )
    detail = raw.strip() or repr(message)[:500]
    raise AssertionError(fail_msg(
        f'视觉 smoke 须含非空 assistant 文本（content 或 reasoning）；{hint}',
        detail,
    ))


_VISION_TIGER_HINTS = ('tiger', 'cat', 'big cat', '虎', '猫', 'feline')


def assert_vision_tiger_text(text: str, *, label: str = 'VLM') -> None:
    """Resource tiger.jpeg smoke: answer must name the animal."""
    lower = (text or '').lower()
    assert any(hint in lower for hint in _VISION_TIGER_HINTS), fail_msg(
        f'{label} 应识别资源图中的动物语义',
        (text or '')[:800],
    )


def assert_chat_vision_tiger_choice(choice: dict, *, raw: str = '') -> str:
    visible = assert_chat_vision_smoke_choice(choice, raw=raw)
    assert_vision_tiger_text(visible, label='Chat')
    return visible


def assert_chat_completions_stream_return(
        output,
        model_name,
        is_last: bool = False,
        check_logprobs: bool = False,
        logprobs_num: int = 5,
):
    assert output.get('id') is not None, fail_msg(
        'Chat 流式 chunk 须含非空 id', output)
    assert output.get('object') == 'chat.completion.chunk', fail_msg(
        'Chat 流式 object 须为 chat.completion.chunk', output)
    assert output.get('model') == model_name, fail_msg(
        'Chat 流式 model 须与请求一致', output)
    output_message = output.get('choices')
    assert len(output_message) == 1, fail_msg('Chat 流式 chunk 须恰好一条 choice', output)
    for message in output_message:
        assert message.get('index') == 0, fail_msg('Chat 流式 choice.index 须为 0', message)
        delta = message.get('delta') or {}
        role = delta.get('role')
        if role is not None:
            assert role == 'assistant', fail_msg(
                'Chat 流式 delta.role 若存在须为 assistant', delta)
        reasoning, content = _chat_reasoning_and_content(delta)
        if (
            not is_last
            and message.get('finish_reason') is None
            and content is None
            and reasoning is None
        ):
            return
        if not is_last:
            assert isinstance(content, str) or isinstance(reasoning, str), fail_msg(
                'Chat 流式中间 chunk 的 delta 须含 str content 或 reasoning_content', delta)
            assert message.get('finish_reason') is None, fail_msg(
                'Chat 流式中间 chunk 不得含 finish_reason', message)
            if check_logprobs:
                assert (len(message.get('logprobs').get('content')) >= 1), fail_msg(
                    'Chat 流式 logprobs 中间 chunk 须至少一条 logprob', message.get('logprobs'))
                for content in message.get('logprobs').get('content'):
                    assert_logprobs(content, logprobs_num)
        else:
            assert message.get('finish_reason') in [
                'stop', 'length', 'error', 'tool_calls',
            ], fail_msg('Chat 流式末 chunk finish_reason 须为已知终态', message)
            if check_logprobs is True:
                assert message.get('logprobs') is None, fail_msg(
                    'Chat 流式末 chunk 在 check_logprobs 时不应再带 logprobs', message)


def assert_completions_stream_return(
        output,
        model_name,
        is_last: bool = False,
        check_logprobs: bool = False,
        logprobs_num: int = 5,
):
    print(output)
    assert output.get('id') is not None, fail_msg(
        'Completions 流式 chunk 须含非空 id', output)
    assert output.get('object') == 'text_completion', fail_msg(
        'Completions 流式 object 须为 text_completion', output)
    assert output.get('model') == model_name, fail_msg(
        'Completions 流式 model 须与请求一致', output)
    output_message = output.get('choices')
    assert len(output_message) == 1, fail_msg(
        'Completions 流式 chunk 须恰好一条 choice', output)
    for message in output_message:
        assert message.get('index') == 0, fail_msg(
            'Completions 流式 choice.index 须为 0', message)
        assert len(message.get('text')) >= 0, fail_msg(
            'Completions 流式 text 字段须存在（可为空串）', message)
        if is_last is False:
            assert message.get('finish_reason') is None, fail_msg(
                'Completions 流式中间 chunk 不得含 finish_reason', message)
            if check_logprobs:
                assert (len(message.get('logprobs').get('content')) >= 1), fail_msg(
                    'Completions 流式 logprobs 中间 chunk 须至少一条 logprob',
                    message.get('logprobs'),
                )
                for content in message.get('logprobs').get('content'):
                    assert_logprobs(content, logprobs_num)

        if is_last is True:
            assert message.get('finish_reason') in ['stop', 'length'], fail_msg(
                'Completions 流式末 chunk finish_reason 须为 stop 或 length', message)
            if check_logprobs is True:
                assert message.get('logprobs') is None, fail_msg(
                    'Completions 流式末 chunk 在 check_logprobs 时不应再带 logprobs', message)


def has_repeated_fragment(text, repeat_count=5):
    pattern = r'(.+?)\1{' + str(repeat_count - 1) + ',}'
    match = re.search(pattern, text.replace('\n', ''))
    if match:
        repeated_fragment = match.group(1)
        start_pos = match.start()
        return True, {'repeated_fragment': repeated_fragment, 'position': start_pos}
    return False, f'{text} does not contain repeated fragments'


def gateway_auth_headers(
    *,
    content_type: str = 'application/json',
    anthropic_version: str | None = None,
) -> dict[str, str]:
    """Auth for raw ``requests`` calls (OpenAI Bearer + Anthropic ``x-api-
    key``)."""
    headers: dict[str, str] = {}
    if content_type:
        headers['Content-Type'] = content_type
    if anthropic_version:
        headers['anthropic-version'] = anthropic_version
    key = os.environ.get('API_KEY', '').strip()
    if key:
        headers['Authorization'] = f'Bearer {key}'
        headers['x-api-key'] = key
    return headers


def get_client_and_model(base_url: str | None = None) -> tuple[OpenAI, str]:
    """Return ``(OpenAI client, deployed model id)`` for a running
    api_server."""
    from utils.http_timeout import openai_client_timeout

    url = base_url or BASE_URL
    client = OpenAI(
        api_key=os.environ.get('API_KEY', 'YOUR_API_KEY'),
        base_url=f'{url.rstrip("/")}/v1',
        timeout=openai_client_timeout(),
    )
    preferred = os.environ.get('MODEL', '').strip()
    if preferred:
        return client, preferred
    models = client.models.list().data
    if not models:
        raise RuntimeError(f'No model returned from GET {url}/v1/models')
    return client, models[0].id


# Prefix-cache probes (multi-model)
# ---------------------------------
# Env:
#   ``PREFIX_CACHE_MIN_PAD_TOKENS`` — shared prefix length target (default 2048).
#   ``TOKENIZER`` — HF path; pad/probe length is measured with this tokenizer.
#   ``PREFIX_CACHE_FALLBACK_CHARS_PER_TOKEN`` — when TOKENIZER unset (default 6).
#   ``PREFIX_CACHE_REQUIRE_COLD=1`` — first call must report cached_tokens==0.
#
# Assert contract (``assert_prefix_cache_hit`` / ``assert_prefix_cache_usage_pair`` /
# ``assert_usage_prefix_cache``):
#   identical request ×2; prompt/input token counts match;
#   output/completion token counts match when the second call reports no cache read;
#   second ``cached_tokens`` (or Anthropic ``cache_read_input_tokens``) is > 0;
#   second cached ≥ first cached; cached ≤ prompt.
#   Missing cache counter fields in usage → fail (no skip).

_PREFIX_CACHE_PAD_UNIT = 'Prefix-cache probe sentence. '
_PREFIX_CACHE_USER_SUFFIX = 'Reply with one word: ok'


def prefix_cache_min_pad_tokens() -> int:
    return int(os.environ.get('PREFIX_CACHE_MIN_PAD_TOKENS', '2048'))


def _prefix_cache_tokenizer_path() -> str | None:
    path = os.environ.get('TOKENIZER', '').strip()
    return path or None


def _text_token_count(text: str, *, tokenizer_path: str | None) -> int | None:
    if not tokenizer_path:
        return None
    try:
        return len(encode_text(tokenizer_path, text, add_special_tokens=False))
    except Exception:
        return None


def _prefix_cache_unit_repeats_for_token_count(
    token_count: int,
    *,
    suffix: str = '',
    min_repeats: int | None = None,
) -> int:
    min_repeats = 1 if min_repeats is None else min_repeats
    tok = _prefix_cache_tokenizer_path()
    unit = _PREFIX_CACHE_PAD_UNIT
    if tok:
        n = min_repeats
        while n < 20_000:
            text = unit * n + suffix
            count = _text_token_count(text, tokenizer_path=tok)
            if count is not None and count >= token_count:
                return n
            n += 32
        return n
    chars_per = float(os.environ.get('PREFIX_CACHE_FALLBACK_CHARS_PER_TOKEN', '6'))
    target_chars = token_count * chars_per
    return max(min_repeats, math.ceil(target_chars / len(unit)))


def prefix_cache_pad_text(min_pad_tokens: int | None = None) -> str:
    """Repeat pad unit until ≥ ``min_pad_tokens`` (TOKENIZER or char
    fallback)."""
    target = prefix_cache_min_pad_tokens() if min_pad_tokens is None else min_pad_tokens
    n = _prefix_cache_unit_repeats_for_token_count(target, suffix='')
    return _PREFIX_CACHE_PAD_UNIT * n


def prefix_cache_probe_text(min_pad_tokens: int | None = None) -> str:
    """Chat/completions/responses: long shared user prefix + short suffix."""
    target = prefix_cache_min_pad_tokens() if min_pad_tokens is None else min_pad_tokens
    suffix = _PREFIX_CACHE_USER_SUFFIX
    n = _prefix_cache_unit_repeats_for_token_count(target, suffix=suffix)
    return _PREFIX_CACHE_PAD_UNIT * n + suffix


def prefix_cache_pad_repeats(min_pad_tokens: int | None = None) -> int:
    """Repeat count for the pad-only segment (docs / shell parity)."""
    target = prefix_cache_min_pad_tokens() if min_pad_tokens is None else min_pad_tokens
    return _prefix_cache_unit_repeats_for_token_count(target, suffix='')


PREFIX_CACHE_PAD = prefix_cache_pad_text()
PREFIX_CACHE_PROBE = prefix_cache_probe_text()

# VL paths where VLAsyncEngine silently clears enable_prefix_caching.
_VL_PREFIX_CACHE_UNSUPPORTED = (
    'intern-s1-mini',
    'glm-4v',
    'gemma-3-27b',
    'internvl3',
)

def assert_prefix_cache_usage_pair(
    first_cached: int,
    second_cached: int,
    prompt_tokens: int,
    msg: str,
) -> None:
    """Two back-to-back identical prompts: usage must show a prefix-cache hit."""
    assert second_cached > 0, fail_msg('第二次相同前缀请求须报告缓存命中（cached>0）', msg)
    assert second_cached <= prompt_tokens, fail_msg(
        '第二次 cached 不得超过 prompt token 数', msg)
    assert second_cached >= first_cached, fail_msg(
        '第二次 cached 须 ≥ 第一次（前缀缓存只增不减）', msg)
    if os.environ.get('PREFIX_CACHE_REQUIRE_COLD', '').strip() == '1':
        assert first_cached == 0, fail_msg(
            'PREFIX_CACHE_REQUIRE_COLD 下第一次请求 cached 须为 0', msg)


def assert_prefix_cache_hit(http_url: str, image_url: str | None = None):
    """Same prompt twice: second call must report cache hit; generation stable.

    ``completion_tokens`` must match only when the second usage reports no
    cache read; with a cache hit, text and ``finish_reason`` must still match.

    Unsupported VL paths return without asserting so callers can still run
    functional chat cases on the same server. ``image_url`` adds a vision
    part (OpenAI ``image_url``) and requires the reply to name the animal.
    """
    client, model_name = get_client_and_model(http_url)
    model_l = model_name.lower()
    if any(tag in model_l for tag in _VL_PREFIX_CACHE_UNSUPPORTED):
        print(f'prefix-cache assert skipped for VL path: {model_name}')
        return
    # Pad length targets PREFIX_CACHE_MIN_PAD_TOKENS (default 2048) for cache hits.
    if image_url:
        probe = prefix_cache_pad_text() + 'What animal is in the image? Reply with one word.'
        expect = 'tiger'
        content: str | list[dict[str, Any]] = [
            {'type': 'image_url', 'image_url': {'url': image_url}},
            {'type': 'text', 'text': probe},
        ]
    else:
        probe = prefix_cache_probe_text()
        content = probe
    messages = [{'role': 'user', 'content': content}]
    create = dict(
        model=model_name,
        messages=messages,
        temperature=0,
        max_completion_tokens=64,
        **chat_protocol_probe_create_kwargs(),
    )
    first = client.chat.completions.create(**create)
    second = client.chat.completions.create(**create)
    first_choice = first.choices[0]
    second_choice = second.choices[0]
    first_text = first_choice.message.content
    second_text = second_choice.message.content
    first_usage = first.usage.model_dump() if hasattr(first.usage, 'model_dump') else dict(first.usage)
    second_usage = second.usage.model_dump() if hasattr(second.usage, 'model_dump') else dict(second.usage)
    first_cached = usage_cached_tokens(first_usage, kind='chat')
    second_cached = usage_cached_tokens(second_usage, kind='chat')
    assert first_cached is not None and second_cached is not None, fail_msg(
        '前缀缓存探测须能读取 prompt_tokens_details.cached_tokens',
        f'first={first_usage} second={second_usage}',
    )
    prompt_tokens = second.usage.prompt_tokens
    first_completion = first.usage.completion_tokens
    second_completion = second.usage.completion_tokens
    msg = (f'first_cached={first_cached} second_cached={second_cached} '
           f'prompt_tokens={prompt_tokens} '
           f'first_completion={first_completion} second_completion={second_completion} '
           f'first_text={first_text!r} second_text={second_text!r}')
    assert first.usage.prompt_tokens == prompt_tokens, fail_msg(
        '两次相同请求的 prompt_tokens 须一致', msg)
    if second_cached <= 0:
        assert first_completion == second_completion, fail_msg(
            '无缓存命中时两次 completion_tokens 须一致', msg)
    assert_prefix_cache_usage_pair(first_cached, second_cached, prompt_tokens, msg)
    if image_url:
        assert expect in (first_text or '').lower(), fail_msg(
            f'视觉前缀缓存探测回复须含 {expect!r}', msg)
    else:
        assert isinstance(first_text, str) and first_text.strip(), fail_msg(
            '前缀缓存探测第一次须有非空补全文本', msg)
    assert first_text == second_text, fail_msg(
        '相同前缀两次请求的补全文本须一致（温度 0）', msg)
    assert first_choice.finish_reason == second_choice.finish_reason, fail_msg(
        '相同前缀两次请求的 finish_reason 须一致', msg)


def reasoning_evidence(kind: str, body: dict) -> list[str]:
    """Where a response shows a reasoning chain.

    Usage counters are not evidence.
    """
    paths: list[str] = []
    if kind == 'completions':
        for index, choice in enumerate(body.get('choices') or []):
            if not isinstance(choice, dict):
                continue
            value = choice.get('reasoning_content')
            if isinstance(value, str) and value.strip():
                paths.append(f'choices[{index}].reasoning_content')
    elif kind == 'responses':
        for index, item in enumerate(body.get('output') or []):
            if not isinstance(item, dict) or item.get('type') != 'reasoning':
                continue
            if responses_reasoning_text({'output': [item]}) or item.get('encrypted_content'):
                paths.append(f'output[{index}](type=reasoning)')
    elif kind == 'messages':
        for index, block in enumerate(body.get('content') or []):
            if not isinstance(block, dict):
                continue
            block_type = block.get('type')
            key = 'thinking' if block_type == 'thinking' else 'data' if block_type == 'redacted_thinking' else None
            value = block.get(key) if key else None
            if isinstance(value, str) and value.strip():
                paths.append(f'content[{index}].{key}(type={block_type})')
    else:
        raise ValueError(f'unsupported reasoning kind: {kind}')
    return paths


def _usage_at(usage: dict, path: tuple[str, ...]):
    node = usage
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


_REASONING_TOKEN_PATH = {
    'completions': ('completion_tokens_details', 'reasoning_tokens'),
    'responses': ('output_tokens_details', 'reasoning_tokens'),
    'messages': ('output_tokens_details', 'thinking_tokens'),
}


def assert_reasoning_usage(kind: str, body: dict, *, thinking: bool) -> None:
    """If a chain is present, the protocol token field must be an int greater
    than 0.

    No chain means this response is not a reasoning sample: the token field is
    not required. ``thinking=False`` still requires the chain to be absent.
    """
    evidence = reasoning_evidence(kind, body)
    usage = body['usage']
    if not thinking:
        assert not evidence, fail_msg(
            'thinking 关闭时响应体不得含 reasoning/thinking 链路透出', evidence)
        return
    if not evidence:
        return
    path = _REASONING_TOKEN_PATH[kind]
    value = _usage_at(usage, path)
    field = 'usage.' + '.'.join(path)
    assert isinstance(value, int) and not isinstance(value, bool) and value > 0, fail_msg(
        f'存在 reasoning 链时 {field} 须为大于 0 的整数',
        f'{field}={value!r} evidence={evidence} usage={usage}',
    )


def usage_cache_counter(usage: dict, cached_keys: tuple[str, ...]) -> int | None:
    node: Any = usage
    for key in cached_keys:
        if not isinstance(node, dict) or key not in node:
            return None
        child = node[key]
        if child is None:
            return None
        node = child
    return node if isinstance(node, int) and not isinstance(node, bool) else None


def assert_usage_prefix_cache(
    first_usage: dict,
    second_usage: dict,
    *,
    prompt_key: str,
    cached_keys: tuple[str, ...],
    output_key: str,
) -> None:
    """Same contract as ``assert_prefix_cache_hit``, on raw usage dicts."""
    first_prompt = first_usage[prompt_key]
    second_prompt = second_usage[prompt_key]
    cold = usage_cache_counter(first_usage, cached_keys)
    hot = usage_cache_counter(second_usage, cached_keys)
    path = '.'.join(cached_keys)
    assert cold is not None and hot is not None, fail_msg(
        f'usage 须含前缀缓存计数字段 {path}',
        f'first={first_usage} second={second_usage}',
    )
    msg = (f'first_cached={cold} second_cached={hot} '
           f'prompt={first_prompt}/{second_prompt} '
           f'output={first_usage[output_key]}/{second_usage[output_key]}')
    anthropic_cache_read = cached_keys == ('cache_read_input_tokens',)
    if anthropic_cache_read and hot > 0:
        assert first_prompt >= second_prompt, fail_msg(
            'Anthropic 缓存命中时首次 input_tokens 须 ≥ 第二次', msg)
        assert first_prompt == second_prompt + hot, fail_msg(
            'Anthropic 计费：第二次 input + cache_read 须等于首次 input_tokens',
            f'{second_prompt}+{hot} vs {first_prompt}; {msg}',
        )
        prompt_for_block = first_prompt
    else:
        assert first_prompt == second_prompt, fail_msg(
            '两次相同请求的 prompt/input token 数须一致', msg)
        prompt_for_block = second_prompt
    if hot <= 0:
        assert first_usage[output_key] == second_usage[output_key], fail_msg(
            '无缓存命中时两次 output/completion token 数须一致', msg)
    assert_prefix_cache_usage_pair(cold, hot, prompt_for_block, msg)


@lru_cache(maxsize=1)
def deployed_model_name() -> str:
    """Single model id exposed by the RESTFUL api_server."""
    return get_client_and_model(BASE_URL)[1]


def resolve_effective_session_len(config: dict[str, Any], model_id: str) -> int:
    """Context limit aligned with ``async_engine.session_len``.

    Uses yaml ``session-len`` when set, otherwise HF ``_get_and_verify_max_len``.
    Does not apply ``tokenizer.model_max_length`` (server preprocess does not either).
    """
    model_path = get_model_path_from_config(config, model_id)
    session_len = None
    for entry in iter_model_yaml_entries(model_id):
        extra = _entry_engine_config(entry).get('extra') or {}
        if extra.get('session-len') is not None:
            session_len = int(extra['session-len'])
            break
    if session_len is None:
        from transformers import AutoConfig

        from lmdeploy.utils import _get_and_verify_max_len

        hf_cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        session_len = _get_and_verify_max_len(hf_cfg, None)
    return session_len


def build_session_sized_user_content(
    *,
    config: dict[str, Any],
    model_id: str,
    oversize: bool = False,
    max_completion_tokens: int = 0,
    reserve: int = 256,
    slack: int = 32,
    unit: str = 'x',
    token_cap: int | None = CI_LARGE_PAYLOAD_TOKEN_CAP,
) -> str:
    """Size user text relative to server ``session_len`` (local tokenizer).

    ``oversize=True``: raw user token count exceeds server ``session_len`` (400 tests).
    ``oversize=False``: as large as possible while fitting; also caps by finite
    ``tokenizer.model_max_length`` when tighter than HF context (e.g. InternVL3-38B).
    """
    session_len = resolve_effective_session_len(config, model_id)
    model_path = get_model_path_from_config(config, model_id)
    effective_session = session_len
    if not oversize:
        tok_mml = getattr(_load_tokenizer_cached(model_path), 'model_max_length', None)
        if tok_mml is not None and tok_mml < 1_000_000:
            effective_session = min(session_len, int(tok_mml))
    cache_key = (
        model_path,
        effective_session,
        oversize,
        max_completion_tokens,
        reserve,
        slack,
        unit,
        token_cap,
    )
    cached = _SESSION_SIZED_CONTENT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    add_special = False
    if oversize:
        target = effective_session + slack + 1
        text = grow_text_to_at_least_token_count(
            model_path, target, unit=unit, add_special_tokens=add_special)
        _SESSION_SIZED_CONTENT_CACHE[cache_key] = text
        return text

    input_limit = effective_session - max_completion_tokens - reserve
    if token_cap is not None:
        input_limit = min(input_limit, token_cap)
    if input_limit <= 0:
        text = ''
    else:
        text = grow_text_to_at_least_token_count(
            model_path, input_limit, unit=unit, add_special_tokens=add_special)
        session_input_limit = effective_session - max_completion_tokens
        if _count_tokens(model_path, text, add_special_tokens=add_special) >= session_input_limit:
            text = _shrink_unit_text_below_token_count(
                model_path,
                len(text),
                session_input_limit,
                unit=unit,
                add_special_tokens=add_special,
            )
    _SESSION_SIZED_CONTENT_CACHE[cache_key] = text
    return text
