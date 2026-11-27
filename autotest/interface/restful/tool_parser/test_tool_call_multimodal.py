"""Multimodal image + tool-call tests (Chat completions + Responses +
Messages)."""

import pytest
import requests
from openai import BadRequestError
from utils.anthropic_messages import (
    SEARCH_TOOL_ANTHROPIC,
    USER_ASK_WEATHER_DALLAS_VLM,
    WEATHER_TOOL_ANTHROPIC,
    assert_success_message_json,
    assert_weather_tool_city_state,
)
from utils.constant import BASE_URL, CAPPED_MAX_COMPLETION_TOKENS, THINKING_MAX_COMPLETION_TOKENS
from utils.gateway_compat import responses_tool_choice_kwargs
from utils.restful_return_check import (
    assert_responses_batch_return,
    deployed_model_name,
    responses_output_text,
)
from utils.tool_reasoning_definitions import (
    SEARCH_TOOL,
    WEATHER_TOOL,
    assert_arguments_parseable,
    assert_no_parser_drop,
    assert_parallel_weather_cities_isolated,
    assert_tool_call_fields,
    assert_tool_name_single_delta,
    build_messages_with_parallel_tool_responses,
    build_messages_with_tool_response,
    collect_stream_parallel_tool_calls,
    collect_stream_tool_call,
    validate_stream_tool_call_chunks,
    validate_stream_tool_call_result,
)

from .conftest import (
    MESSAGES_ASKING_FOR_WEATHER,
    MM_IMAGE_MEDIA_TYPES,
    MM_TEST_IMAGE_BEIJING,
    MM_TEST_IMAGE_POSE,
    MM_TEST_IMAGE_TIGER,
    MM_TEST_VIDEO,
    MM_VIDEO_EXTRA_BODY,
    MM_VIDEO_MEDIA_TYPES,
    _apply_marks_mm,
    _ToolCallTestBase,
    build_mm_dallas_weather_user_message,
    build_mm_dual_image_dallas_messages,
    build_mm_image_and_video_dallas_messages,
    build_mm_miami_weather_user_message,
    build_mm_parallel_weather_messages,
    build_mm_parallel_weather_user_message,
    build_mm_tiger_search_messages,
    build_multimodal_user_message,
    mm_beijing_weather_messages,
    mm_create_extra_body_for_media_type,
    mm_dallas_weather_messages,
    mm_file_to_data_url,
    mm_miami_weather_messages,
    mm_weather_messages_for_media_type,
)
from .test_tool_call_anthropic_sdk import (
    _http_headers,
    _http_tool_use_blocks,
    _sse_aggregate_tool_use_inputs,
    _sse_tool_use_blocks,
    _trace_anthropic_http,
)
from .test_tool_call_responses import (
    _TOOLS as _RESPONSES_FLAT_TOOLS,
)
from .test_tool_call_responses import (
    _WEATHER_NAME,
    _assert_weather_function_call,
    responses_tool_create_kwargs,
    stream_responses_function_call,
)

_MESSAGES_URL = f'{BASE_URL}/v1/messages'
_VLM_TOOL_MAX_TOKENS = THINKING_MAX_COMPLETION_TOKENS

# Chat 深度多模在下方各类；此处仅 Responses + Messages，避免与 Basic/Choice 重复跑 chat。
_MM_CROSS_APIS = (
    pytest.param('responses', id='responses', marks=pytest.mark.responses),
    pytest.param('anthropic', id='anthropic', marks=pytest.mark.anthropic),
)


def _anthropic_vlm_user_turn(image_url: str) -> dict:
    return {
        'role': 'user',
        'content': [
            {'type': 'text', 'text': USER_ASK_WEATHER_DALLAS_VLM},
            {'type': 'image', 'source': {'type': 'url', 'url': image_url}},
        ],
    }


def _anthropic_vlm_weather_request(image_url: str, **extra) -> dict:
    return {
        'model': deployed_model_name(),
        'max_tokens': _VLM_TOOL_MAX_TOKENS,
        'temperature': 0,
        'system': MESSAGES_ASKING_FOR_WEATHER[0]['content'],
        'messages': [_anthropic_vlm_user_turn(image_url)],
        'tools': [WEATHER_TOOL_ANTHROPIC, SEARCH_TOOL_ANTHROPIC],
        **extra,
    }


def _anthropic_post_messages(*, req_json: dict, stream: bool) -> requests.Response:
    return requests.post(
        _MESSAGES_URL,
        headers=_http_headers(),
        json={**req_json, 'stream': stream},
        stream=stream,
        timeout=180,
    )


def _dallas_weather_batch(base: _ToolCallTestBase, mm_api: str) -> tuple[str, dict]:
    image_url = base._require_mm_image(MM_TEST_IMAGE_TIGER)
    messages = mm_dallas_weather_messages(image_url)
    if mm_api == 'responses':
        client, model_name = base._get_client()
        fcs = assert_responses_batch_return(
            client.responses.create(
                model=model_name,
                **responses_tool_create_kwargs(tool_choice='required', input=messages),
            ).model_dump(),
            model_name,
        )
        assert fcs
        _assert_weather_function_call(fcs[0])
        return fcs[0]['name'], assert_arguments_parseable(fcs[0]['arguments'])
    req = _anthropic_vlm_weather_request(image_url)
    resp = _anthropic_post_messages(req_json=req, stream=False)
    assert resp.status_code == 200, resp.text
    blocks = _http_tool_use_blocks(assert_success_message_json(resp.json()))
    assert blocks, resp.text[:400]
    assert_weather_tool_city_state(blocks[0]['input'], ctx=f'mm_cross/{mm_api}/batch')
    return blocks[0]['name'], blocks[0]['input']


def _dallas_weather_stream(base: _ToolCallTestBase, mm_api: str) -> tuple[str, dict]:
    image_url = base._require_mm_image(MM_TEST_IMAGE_TIGER)
    messages = mm_dallas_weather_messages(image_url)
    if mm_api == 'responses':
        client, model_name = base._get_client()
        fc = stream_responses_function_call(
            client,
            model_name,
            **responses_tool_create_kwargs(tool_choice='required', input=messages),
        )
        _assert_weather_function_call(fc)
        return fc['name'], assert_arguments_parseable(fc['arguments'])
    req = _anthropic_vlm_weather_request(image_url)
    resp = _anthropic_post_messages(req_json=req, stream=True)
    assert resp.status_code == 200, resp.text
    raw = ''.join(chunk.decode('utf-8') for chunk in resp.iter_content(chunk_size=None) if chunk)
    _trace_anthropic_http(
        base._log_file,
        f'dallas_weather_stream/{mm_api}',
        url=_MESSAGES_URL,
        http_status=resp.status_code,
        request_json=req,
        response_text=raw,
    )
    names = [b['name'] for b in _sse_tool_use_blocks(raw)]
    assert WEATHER_TOOL_ANTHROPIC['name'] in names, names
    inputs = _sse_aggregate_tool_use_inputs(raw)
    assert inputs, raw[:800]
    assert_weather_tool_city_state(inputs[0], ctx=f'mm_cross/{mm_api}/stream')
    return names[0], inputs[0]


# ===========================================================================
# Basic multimodal tool call
# ===========================================================================


@_apply_marks_mm
class TestToolCallMultimodalBasic(_ToolCallTestBase):
    """Basic multimodal tool call: structure, finish_reason, field validation."""

    def test_non_streaming(self, backend, model_case):
        image_url = self._require_mm_image(MM_TEST_IMAGE_TIGER)
        messages = mm_dallas_weather_messages(image_url)
        client, model_name = self._get_client()

        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0,
            max_completion_tokens=CAPPED_MAX_COMPLETION_TOKENS,
            tools=[WEATHER_TOOL, SEARCH_TOOL],
            logprobs=False,
        )

        assert response.object == 'chat.completion'
        assert response.id is not None and len(response.id) > 0
        assert response.model is not None and len(response.model) > 0
        assert len(response.choices) == 1

        choice = response.choices[0]
        tool_calls = choice.message.tool_calls

        assert choice.message.role == 'assistant'
        assert choice.finish_reason == 'tool_calls', (
            f'Expected tool_calls with image+text; got finish_reason={choice.finish_reason!r}, '
            f'content={choice.message.content!r}')
        assert tool_calls is not None and len(tool_calls) >= 1

        tc = tool_calls[0]
        assert_tool_call_fields(tc)
        assert tc.type == 'function'
        assert tc.function.name == WEATHER_TOOL['function']['name']

        parsed_args = assert_arguments_parseable(tc.function.arguments)
        assert 'city' in parsed_args and 'state' in parsed_args
        assert 'dallas' in parsed_args['city'].lower()
        assert 'tx' in parsed_args['state'].lower()

        assert response.usage is not None
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0

    def test_streaming(self, backend, model_case):
        image_url = self._require_mm_image(MM_TEST_IMAGE_TIGER)
        messages = mm_dallas_weather_messages(image_url)
        r = self._stream_tool_call(messages, tools=[WEATHER_TOOL, SEARCH_TOOL])

        assert r['role'] == 'assistant'
        assert r['chunk_count'] > 0, 'Expected at least one SSE chunk'
        validate_stream_tool_call_result(
            r,
            expected_function_name=WEATHER_TOOL['function']['name'],
            **self._parser_validation_kwargs([WEATHER_TOOL, SEARCH_TOOL]),
        )

        streamed_args = assert_arguments_parseable(r['args_str'])
        assert 'city' in streamed_args and 'state' in streamed_args
        assert 'dallas' in streamed_args['city'].lower()
        assert 'tx' in streamed_args['state'].lower()

    def test_streaming_function_name_not_fragmented(self, backend, model_case):
        image_url = self._require_mm_image(MM_TEST_IMAGE_TIGER)
        messages = mm_dallas_weather_messages(image_url)
        client, model_name = self._get_client()
        stream = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0,
            max_completion_tokens=CAPPED_MAX_COMPLETION_TOKENS,
            tools=[WEATHER_TOOL],
            tool_choice={
                'type': 'function',
                'function': {'name': 'get_current_weather'},
            },
            logprobs=False,
            stream=True,
        )
        chunks = list(stream)
        validate_stream_tool_call_chunks(chunks, check_incremental_arguments=False)
        assert_tool_name_single_delta(iter(chunks), 'get_current_weather')

    @pytest.mark.parametrize('media_type', MM_VIDEO_MEDIA_TYPES)
    def test_streaming_function_name_not_fragmented_video(
            self, backend, model_case, media_type):
        source = self._require_mm_media_source(media_type)
        messages = mm_weather_messages_for_media_type(media_type, source)
        client, model_name = self._get_client()
        create_kwargs = {}
        extra_body = mm_create_extra_body_for_media_type(media_type)
        if extra_body is not None:
            create_kwargs['extra_body'] = extra_body
        stream = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0,
            max_completion_tokens=CAPPED_MAX_COMPLETION_TOKENS,
            tools=[WEATHER_TOOL],
            tool_choice={
                'type': 'function',
                'function': {'name': 'get_current_weather'},
            },
            logprobs=False,
            stream=True,
            **create_kwargs,
        )
        chunks = list(stream)
        validate_stream_tool_call_chunks(chunks, check_incremental_arguments=False)
        assert_tool_name_single_delta(iter(chunks), 'get_current_weather')

    def test_search_tool_with_image_context(self, backend, model_case):
        """web_search from an image-grounded user query."""
        image_url = self._require_mm_image(MM_TEST_IMAGE_TIGER)
        messages = build_mm_tiger_search_messages(image_url)
        client, model_name = self._get_client()

        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0,
            max_completion_tokens=CAPPED_MAX_COMPLETION_TOKENS,
            tools=[SEARCH_TOOL],
            logprobs=False,
        )

        choice = response.choices[0]
        tool_calls = choice.message.tool_calls
        assert choice.finish_reason == 'tool_calls', (
            f'Expected tool_calls for search+image; got finish_reason={choice.finish_reason!r}, '
            f'content={choice.message.content!r}')
        assert tool_calls is not None and len(tool_calls) >= 1
        tc = tool_calls[0]
        assert_tool_call_fields(tc)
        assert tc.function.name == 'web_search'
        parsed = assert_arguments_parseable(tc.function.arguments)
        assert 'query' in parsed
        assert isinstance(parsed['query'], str) and len(parsed['query']) > 0

    def test_search_tool_with_image_context_streaming(self, backend, model_case):
        """Streaming web_search from an image-grounded user query."""
        image_url = self._require_mm_image(MM_TEST_IMAGE_TIGER)
        messages = build_mm_tiger_search_messages(image_url)
        r = self._stream_tool_call(messages, tools=[SEARCH_TOOL])

        assert r['role'] == 'assistant'
        assert r['chunk_count'] > 0, 'Expected at least one SSE chunk'
        validate_stream_tool_call_result(
            r,
            expected_function_name=SEARCH_TOOL['function']['name'],
            **self._parser_validation_kwargs([SEARCH_TOOL]),
        )
        parsed = assert_arguments_parseable(r['args_str'])
        assert 'query' in parsed
        assert isinstance(parsed['query'], str) and len(parsed['query']) > 0


# ===========================================================================
# Stream / non-stream consistency
# ===========================================================================


@_apply_marks_mm
class TestToolCallMultimodalStreamConsistency(_ToolCallTestBase):
    """Streaming and non-streaming multimodal tool call results must match."""

    def test_stream_nonstream_consistency(self, backend, model_case):
        image_url = self._require_mm_image(MM_TEST_IMAGE_TIGER)
        messages = mm_dallas_weather_messages(image_url)
        client, model_name = self._get_client()
        common_kwargs = dict(
            model=model_name,
            messages=messages,
            temperature=0,
            max_completion_tokens=CAPPED_MAX_COMPLETION_TOKENS,
            tools=[WEATHER_TOOL, SEARCH_TOOL],
            logprobs=False,
        )

        ns_resp = client.chat.completions.create(**common_kwargs)
        ns_choice = ns_resp.choices[0]
        assert ns_choice.finish_reason == 'tool_calls'
        assert ns_choice.message.tool_calls is not None and len(ns_choice.message.tool_calls) >= 1
        ns_tc = ns_choice.message.tool_calls[0]
        assert_tool_call_fields(ns_tc)
        ns_name = ns_tc.function.name
        ns_args = assert_arguments_parseable(ns_tc.function.arguments)

        stream = client.chat.completions.create(**common_kwargs, stream=True)
        r = collect_stream_tool_call(stream)
        validate_stream_tool_call_result(
            r,
            **self._parser_validation_kwargs([WEATHER_TOOL, SEARCH_TOOL]),
        )
        s_args = assert_arguments_parseable(r['args_str'])

        assert ns_name == r['function_name']
        assert ns_args == s_args
        assert ns_name in ('get_current_weather', 'web_search')

    @pytest.mark.parametrize('media_type', MM_VIDEO_MEDIA_TYPES)
    def test_stream_nonstream_consistency_video(
            self, backend, model_case, media_type):
        source = self._require_mm_media_source(media_type)
        messages = mm_weather_messages_for_media_type(media_type, source)
        client, model_name = self._get_client()
        extra_body = mm_create_extra_body_for_media_type(media_type)
        common_kwargs = dict(
            model=model_name,
            messages=messages,
            temperature=0,
            max_completion_tokens=CAPPED_MAX_COMPLETION_TOKENS,
            tools=[WEATHER_TOOL],
            logprobs=False,
        )
        if extra_body is not None:
            common_kwargs['extra_body'] = extra_body

        ns_resp = client.chat.completions.create(**common_kwargs)
        ns_choice = ns_resp.choices[0]
        assert ns_choice.finish_reason == 'tool_calls', (
            f'Expected tool_calls for media_type={media_type!r}; '
            f'got finish_reason={ns_choice.finish_reason!r}')
        assert ns_choice.message.tool_calls is not None and len(
            ns_choice.message.tool_calls) >= 1
        ns_tc = ns_choice.message.tool_calls[0]
        assert_tool_call_fields(ns_tc)
        ns_name = ns_tc.function.name
        ns_args = assert_arguments_parseable(ns_tc.function.arguments)

        stream = client.chat.completions.create(**common_kwargs, stream=True)
        r = collect_stream_tool_call(stream)
        validate_stream_tool_call_result(
            r,
            expected_function_name=WEATHER_TOOL['function']['name'],
            **self._parser_validation_kwargs([WEATHER_TOOL]),
        )
        s_args = assert_arguments_parseable(r['args_str'])

        assert ns_name == r['function_name'] == 'get_current_weather'
        assert ns_args == s_args


# ===========================================================================
# tool_choice variants
# ===========================================================================


@_apply_marks_mm
class TestToolCallMultimodalChoice(_ToolCallTestBase):
    """tool_choice with image in user message."""

    def test_tool_choice_none(self, backend, model_case):
        image_url = self._require_mm_image(MM_TEST_IMAGE_TIGER)
        messages = [build_multimodal_user_message('Describe this image.', image_url)]
        client, model_name = self._get_client()

        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0,
            max_completion_tokens=CAPPED_MAX_COMPLETION_TOKENS,
            tools=[WEATHER_TOOL, SEARCH_TOOL],
            tool_choice='none',
            logprobs=False,
        )

        choice = response.choices[0]
        assert choice.message.role == 'assistant'
        assert choice.message.tool_calls is None or len(choice.message.tool_calls) == 0
        assert choice.message.content is not None
        assert len(choice.message.content.strip()) > 0
        assert choice.finish_reason in ('stop', 'length')

    def test_tool_choice_auto(self, backend, model_case):
        """tool_choice='auto' with image: model may call tool or reply in
        text."""
        image_url = self._require_mm_image(MM_TEST_IMAGE_TIGER)
        messages = mm_dallas_weather_messages(image_url)
        client, model_name = self._get_client()

        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0,
            max_completion_tokens=CAPPED_MAX_COMPLETION_TOKENS,
            tools=[WEATHER_TOOL, SEARCH_TOOL],
            tool_choice='auto',
            logprobs=False,
        )

        choice = response.choices[0]
        assert choice.message.role == 'assistant'
        assert choice.finish_reason in ('stop', 'length', 'tool_calls')
        if choice.message.tool_calls and len(choice.message.tool_calls) > 0:
            for tc in choice.message.tool_calls:
                assert_tool_call_fields(tc)
                assert tc.function.name in ('get_current_weather', 'web_search')
                assert_arguments_parseable(tc.function.arguments)
            assert choice.finish_reason == 'tool_calls'
        else:
            assert choice.message.content and len(choice.message.content.strip()) > 0
            assert choice.finish_reason in ('stop', 'length')

    def test_tool_choice_specific_function(self, backend, model_case):
        image_url = self._require_mm_image(MM_TEST_IMAGE_TIGER)
        messages = mm_dallas_weather_messages(image_url)
        client, model_name = self._get_client()

        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0,
            max_completion_tokens=CAPPED_MAX_COMPLETION_TOKENS,
            tools=[WEATHER_TOOL, SEARCH_TOOL],
            tool_choice={
                'type': 'function',
                'function': {'name': 'get_current_weather'},
            },
            logprobs=False,
        )

        choice = response.choices[0]
        tool_calls = choice.message.tool_calls
        assert tool_calls is not None and len(tool_calls) >= 1
        tc = tool_calls[0]
        assert_tool_call_fields(tc)
        assert tc.function.name == 'get_current_weather'
        assert_arguments_parseable(tc.function.arguments)

    def test_tool_choice_required(self, backend, model_case):
        pose_url = self._require_mm_image(MM_TEST_IMAGE_POSE)
        messages = mm_miami_weather_messages(pose_url)
        client, model_name = self._get_client()

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0,
                max_completion_tokens=CAPPED_MAX_COMPLETION_TOKENS,
                tools=[WEATHER_TOOL, SEARCH_TOOL],
                tool_choice='required',
                logprobs=False,
            )
        except BadRequestError as e:
            pytest.skip(f'tool_choice="required" rejected by server (HTTP 400): {e}')

        choice = response.choices[0]
        assert choice.message.role == 'assistant'
        assert choice.message.tool_calls is not None
        assert len(choice.message.tool_calls) >= 1
        for tc in choice.message.tool_calls:
            assert_tool_call_fields(tc)
            assert_arguments_parseable(tc.function.arguments)

    def test_tool_choice_required_streaming(self, backend, model_case):
        pose_url = self._require_mm_image(MM_TEST_IMAGE_POSE)
        messages = mm_miami_weather_messages(pose_url)
        client, model_name = self._get_client()

        try:
            stream = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0,
                max_completion_tokens=CAPPED_MAX_COMPLETION_TOKENS,
                tools=[WEATHER_TOOL, SEARCH_TOOL],
                tool_choice='required',
                logprobs=False,
                stream=True,
            )
        except BadRequestError as e:
            pytest.skip(f'tool_choice="required" streaming rejected by server (HTTP 400): {e}')
        r = collect_stream_tool_call(stream)
        validate_stream_tool_call_result(
            r,
            expected_function_name=None,
            **self._parser_validation_kwargs([WEATHER_TOOL, SEARCH_TOOL]),
        )


# ===========================================================================
# Arguments and multi-turn
# ===========================================================================


@_apply_marks_mm
class TestToolCallMultimodalArgumentsAndMultiTurn(_ToolCallTestBase):
    """Weather args with image context; tool result follow-up."""

    def test_beijing_image_weather_args(self, backend, model_case):
        image_url = self._require_mm_image(MM_TEST_IMAGE_BEIJING)
        messages = mm_beijing_weather_messages(image_url)
        client, model_name = self._get_client()

        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0,
            max_completion_tokens=CAPPED_MAX_COMPLETION_TOKENS,
            tools=[WEATHER_TOOL],
            tool_choice={
                'type': 'function',
                'function': {'name': 'get_current_weather'},
            },
            logprobs=False,
        )

        choice = response.choices[0]
        tool_calls = choice.message.tool_calls
        assert choice.finish_reason == 'tool_calls'
        assert tool_calls is not None and len(tool_calls) >= 1
        tc = tool_calls[0]
        assert_tool_call_fields(tc)
        assert tc.function.name == 'get_current_weather'
        parsed = assert_arguments_parseable(tc.function.arguments)
        city = parsed['city'].lower() if 'city' in parsed else ''
        assert 'beijing' in city or '北京' in city

    def test_image_tool_call_then_text_followup(self, backend, model_case):
        image_url = self._require_mm_image(MM_TEST_IMAGE_TIGER)
        messages = mm_dallas_weather_messages(image_url)
        r = self._stream_tool_call(messages, tools=[WEATHER_TOOL])
        validate_stream_tool_call_result(
            r,
            expected_function_name=WEATHER_TOOL['function']['name'],
            **self._parser_validation_kwargs([WEATHER_TOOL]),
        )
        self._append_assistant_and_tool_messages(messages, r)
        messages.append({
            'role': 'user',
            'content': 'Thanks. In one short sentence, was it hot or mild?',
        })

        client, model_name = self._get_client()
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0,
            max_completion_tokens=CAPPED_MAX_COMPLETION_TOKENS,
            tools=[WEATHER_TOOL],
            logprobs=False,
        )
        choice = response.choices[0]
        assert choice.finish_reason in ('stop', 'length')
        assert choice.message.tool_calls is None or len(choice.message.tool_calls) == 0
        assert choice.message.content and len(choice.message.content.strip()) > 0

    def test_image_tool_call_then_streaming_text_followup(self, backend, model_case):
        """After image+tool round, streaming follow-up should be text-only."""
        image_url = self._require_mm_image(MM_TEST_IMAGE_TIGER)
        messages = mm_dallas_weather_messages(image_url)
        r = self._stream_tool_call(messages, tools=[WEATHER_TOOL])
        validate_stream_tool_call_result(
            r,
            expected_function_name=WEATHER_TOOL['function']['name'],
            **self._parser_validation_kwargs([WEATHER_TOOL]),
        )
        self._append_assistant_and_tool_messages(messages, r)
        messages.append({
            'role': 'user',
            'content': 'Thanks. In one short sentence, was it hot or mild?',
        })

        client, model_name = self._get_client()
        stream = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0,
            max_completion_tokens=CAPPED_MAX_COMPLETION_TOKENS,
            tools=[WEATHER_TOOL],
            logprobs=False,
            stream=True,
        )

        chunks = []
        finish_reason = None
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                chunks.append(delta.content)
            if chunk.choices[0].finish_reason is not None:
                finish_reason = chunk.choices[0].finish_reason
            tool_calls = delta.tool_calls
            assert tool_calls is None or len(tool_calls) == 0

        assert finish_reason in ('stop', 'length')
        assert len(''.join(chunks).strip()) > 0

    def test_parallel_results_text_followup_with_image_history(self, backend, model_case):
        """Parallel tool results in history; first user turn had an image."""
        image_url = self._require_mm_image(MM_TEST_IMAGE_TIGER)
        messages = build_messages_with_parallel_tool_responses()
        messages[1] = build_mm_parallel_weather_user_message(image_url)

        client, model_name = self._get_client()
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0,
            max_completion_tokens=CAPPED_MAX_COMPLETION_TOKENS,
            tools=[WEATHER_TOOL],
            logprobs=False,
        )

        choice = response.choices[0]
        assert choice.message.role == 'assistant'
        assert choice.finish_reason in ('stop', 'length')
        assert choice.message.tool_calls is None or len(choice.message.tool_calls) == 0
        assert choice.message.content and len(choice.message.content) > 0

        content = choice.message.content
        has_dallas = 'Dallas' in content or '98' in content
        has_sf = 'San Francisco' in content or '65' in content
        assert has_dallas or has_sf

    def test_multi_turn_search_after_weather_with_image(self, backend, model_case):
        """Image in first turn; after weather tool result, user asks for web
        search."""
        image_url = self._require_mm_image(MM_TEST_IMAGE_TIGER)
        messages = build_messages_with_tool_response()
        messages[1] = build_mm_dallas_weather_user_message(image_url)
        messages.append({
            'role': 'user',
            'content': 'Now search the web for how to stay cool in hot weather.',
        })

        client, model_name = self._get_client()
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0,
            max_completion_tokens=CAPPED_MAX_COMPLETION_TOKENS,
            tools=[WEATHER_TOOL, SEARCH_TOOL],
            logprobs=False,
        )

        choice = response.choices[0]
        assert choice.message.role == 'assistant'
        if choice.message.tool_calls and len(choice.message.tool_calls) > 0:
            tc = choice.message.tool_calls[0]
            assert_tool_call_fields(tc)
            if tc.function.name == 'web_search':
                parsed = assert_arguments_parseable(tc.function.arguments)
                assert 'query' in parsed
        else:
            assert choice.message.content and len(choice.message.content.strip()) > 0

    def test_multi_turn_search_after_weather_with_image_streaming(self, backend, model_case):
        """Streaming follow-up after image+weather history; parser must not
        drop output."""
        image_url = self._require_mm_image(MM_TEST_IMAGE_TIGER)
        messages = build_messages_with_tool_response()
        messages[1] = build_mm_dallas_weather_user_message(image_url)
        messages.append({
            'role': 'user',
            'content': 'Now search the web for how to stay cool in hot weather.',
        })

        r = self._stream_tool_call(
            messages,
            tools=[WEATHER_TOOL, SEARCH_TOOL],
            max_completion_tokens=CAPPED_MAX_COMPLETION_TOKENS,
        )
        assert r['finish_reason'] in ('stop', 'length', 'tool_calls')
        if r['finish_reason'] == 'tool_calls':
            validate_stream_tool_call_result(
                r,
                expected_function_name=None,
                **self._parser_validation_kwargs([WEATHER_TOOL, SEARCH_TOOL]),
            )
            if r['function_name'] == 'web_search':
                parsed = assert_arguments_parseable(r['args_str'])
                assert 'query' in parsed
        else:
            assert len(r['content'].strip()) > 0
            assert_no_parser_drop(
                r['raw_text'],
                r['tool_calls'],
                r['decoded_str'],
                **self._parser_validation_kwargs([SEARCH_TOOL]),
            )

    def test_second_turn_user_image_weather_tool(self, backend, model_case):
        """Second user turn sends a new image and should still trigger weather
        tool."""
        first_image = self._require_mm_image(MM_TEST_IMAGE_TIGER)
        second_image = self._require_mm_image(MM_TEST_IMAGE_POSE)
        messages = mm_dallas_weather_messages(first_image)
        r = self._stream_tool_call(messages, tools=[WEATHER_TOOL])
        validate_stream_tool_call_result(
            r,
            expected_function_name=WEATHER_TOOL['function']['name'],
            **self._parser_validation_kwargs([WEATHER_TOOL]),
        )
        self._append_assistant_and_tool_messages(messages, r)
        messages.append(build_mm_miami_weather_user_message(second_image))

        client, model_name = self._get_client()
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0,
            max_completion_tokens=CAPPED_MAX_COMPLETION_TOKENS,
            tools=[WEATHER_TOOL],
            logprobs=False,
        )

        choice = response.choices[0]
        tool_calls = choice.message.tool_calls
        assert choice.finish_reason == 'tool_calls', (
            f'Expected tool_calls on second-turn image+weather; got finish_reason={choice.finish_reason!r}, '
            f'content={choice.message.content!r}')
        assert tool_calls is not None and len(tool_calls) >= 1
        tc = tool_calls[0]
        assert_tool_call_fields(tc)
        assert tc.function.name == 'get_current_weather'
        parsed = assert_arguments_parseable(tc.function.arguments)
        assert 'city' in parsed
        assert 'miami' in parsed['city'].lower()

    def test_multimodal_history_text_followup(self, backend, model_case):
        """Image in first user turn; follow-up after tool result (text
        reply)."""
        image_url = self._require_mm_image(MM_TEST_IMAGE_TIGER)
        messages = build_messages_with_tool_response()
        messages[1] = build_mm_dallas_weather_user_message(image_url)

        client, model_name = self._get_client()
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0,
            max_completion_tokens=CAPPED_MAX_COMPLETION_TOKENS,
            tools=[WEATHER_TOOL, SEARCH_TOOL],
            logprobs=False,
        )
        choice = response.choices[0]
        assert choice.finish_reason in ('stop', 'length')
        assert choice.message.role == 'assistant'
        assert choice.message.tool_calls is None or len(choice.message.tool_calls) == 0
        assert choice.message.content and len(choice.message.content) > 0


# ===========================================================================
# Content order, dual images, tool result with image
# ===========================================================================


@_apply_marks_mm
class TestToolCallMultimodalContentOrder(_ToolCallTestBase):
    """Text/image part order in user content should not break tool calling."""

    def test_weather_tool_text_before_image(self, backend, model_case):
        image_url = self._require_mm_image(MM_TEST_IMAGE_TIGER)
        messages = mm_dallas_weather_messages(image_url)
        client, model_name = self._get_client()

        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0,
            max_completion_tokens=CAPPED_MAX_COMPLETION_TOKENS,
            tools=[WEATHER_TOOL],
            logprobs=False,
        )

        choice = response.choices[0]
        assert choice.finish_reason == 'tool_calls'
        tc = choice.message.tool_calls[0]
        assert_tool_call_fields(tc)
        assert tc.function.name == 'get_current_weather'

    def test_weather_tool_image_before_text(self, backend, model_case):
        image_url = self._require_mm_image(MM_TEST_IMAGE_TIGER)
        messages = mm_dallas_weather_messages(image_url, image_first=True)
        client, model_name = self._get_client()

        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0,
            max_completion_tokens=CAPPED_MAX_COMPLETION_TOKENS,
            tools=[WEATHER_TOOL],
            logprobs=False,
        )

        choice = response.choices[0]
        assert choice.finish_reason == 'tool_calls'
        tc = choice.message.tool_calls[0]
        assert_tool_call_fields(tc)
        assert tc.function.name == 'get_current_weather'
        parsed = assert_arguments_parseable(tc.function.arguments)
        assert 'city' in parsed
        assert 'dallas' in parsed['city'].lower()


@_apply_marks_mm
class TestToolCallMultimodalDualImage(_ToolCallTestBase):
    """Multiple images in one user turn with tools."""

    def test_dual_images_weather_tool(self, backend, model_case):
        tiger_url = self._require_mm_image(MM_TEST_IMAGE_TIGER)
        pose_url = self._require_mm_image(MM_TEST_IMAGE_POSE)
        messages = build_mm_dual_image_dallas_messages(tiger_url, pose_url)
        client, model_name = self._get_client()

        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0,
            max_completion_tokens=CAPPED_MAX_COMPLETION_TOKENS,
            tools=[WEATHER_TOOL],
            logprobs=False,
        )

        choice = response.choices[0]
        assert choice.finish_reason == 'tool_calls'
        tool_calls = choice.message.tool_calls
        assert tool_calls is not None and len(tool_calls) >= 1
        tc = tool_calls[0]
        assert_tool_call_fields(tc)
        assert tc.function.name == 'get_current_weather'
        parsed = assert_arguments_parseable(tc.function.arguments)
        assert 'city' in parsed and 'state' in parsed
        assert 'dallas' in parsed['city'].lower()
        assert 'tx' in parsed['state'].lower()


@_apply_marks_mm
class TestToolCallMultimodalMixedMedia(_ToolCallTestBase):
    """Same-turn mixed modalities (image + video) with tools."""

    def _image_and_video_weather_messages(self):
        image_url = self._require_mm_image(MM_TEST_IMAGE_TIGER)
        video_url = self._require_mm_resource(MM_TEST_VIDEO)
        return build_mm_image_and_video_dallas_messages(image_url, video_url)

    def test_image_and_video_weather_tool(self, backend, model_case):
        messages = self._image_and_video_weather_messages()
        client, model_name = self._get_client()
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0,
            max_completion_tokens=CAPPED_MAX_COMPLETION_TOKENS,
            tools=[WEATHER_TOOL],
            logprobs=False,
            extra_body=dict(MM_VIDEO_EXTRA_BODY),
        )
        choice = response.choices[0]
        assert choice.finish_reason == 'tool_calls', (
            f'Expected tool_calls for image+video; got '
            f'finish_reason={choice.finish_reason!r}, '
            f'content={choice.message.content!r}')
        tool_calls = choice.message.tool_calls
        assert tool_calls is not None and len(tool_calls) >= 1
        tc = tool_calls[0]
        assert_tool_call_fields(tc)
        assert tc.function.name == 'get_current_weather'
        parsed = assert_arguments_parseable(tc.function.arguments)
        assert 'city' in parsed and 'state' in parsed
        assert 'dallas' in parsed['city'].lower()
        assert 'tx' in parsed['state'].lower()

    def test_image_and_video_weather_tool_streaming(self, backend, model_case):
        messages = self._image_and_video_weather_messages()
        r = self._stream_tool_call(
            messages,
            tools=[WEATHER_TOOL],
            extra_body=dict(MM_VIDEO_EXTRA_BODY),
        )
        assert r['role'] == 'assistant'
        assert r['chunk_count'] > 0
        validate_stream_tool_call_result(
            r,
            expected_function_name=WEATHER_TOOL['function']['name'],
            **self._parser_validation_kwargs([WEATHER_TOOL]),
        )
        parsed = assert_arguments_parseable(r['args_str'])
        assert 'city' in parsed and 'state' in parsed
        assert 'dallas' in parsed['city'].lower()
        assert 'tx' in parsed['state'].lower()


@_apply_marks_mm
class TestToolCallMultimodalToolResultImage(_ToolCallTestBase):
    """After a tool round-trip, follow-up user turn may include image_url.

    OpenAI Chat Completions only allows ``text`` in ``role=tool`` content;
    images belong on a subsequent ``user`` message (not inside tool).
    """

    def test_tool_result_with_image_then_text_reply(self, backend, model_case):
        user_image = self._require_mm_image(MM_TEST_IMAGE_TIGER)
        tool_image = self._require_mm_image(MM_TEST_IMAGE_POSE)
        messages = mm_dallas_weather_messages(user_image)
        r = self._stream_tool_call(messages, tools=[WEATHER_TOOL])
        validate_stream_tool_call_result(
            r,
            expected_function_name=WEATHER_TOOL['function']['name'],
            **self._parser_validation_kwargs([WEATHER_TOOL]),
        )
        self._append_assistant_and_tool_messages(messages, r)
        # Tool result: text only (OpenAI-compatible).
        messages[-1]['content'] = 'Weather icon attached.'
        # Image on the next user turn, not inside the tool message.
        messages.append({
            'role': 'user',
            'content': [
                {
                    'type': 'text',
                    'text': 'Describe the pose in this image from the weather tool.',
                },
                {'type': 'image_url', 'image_url': {'url': tool_image}},
            ],
        })

        client, model_name = self._get_client()
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0,
            max_completion_tokens=CAPPED_MAX_COMPLETION_TOKENS,
            tools=[WEATHER_TOOL],
            logprobs=False,
        )
        choice = response.choices[0]
        assert choice.finish_reason in ('stop', 'length')
        assert choice.message.tool_calls is None or len(choice.message.tool_calls) == 0
        assert choice.message.content and len(choice.message.content.strip()) > 0


# ===========================================================================
# Parallel tool calls with image
# ===========================================================================


@_apply_marks_mm
class TestToolCallMultimodalParallel(_ToolCallTestBase):
    """Parallel weather tool calls when user turn also contains an image."""

    def test_parallel_weather_with_image(self, backend, model_case):
        image_url = self._require_mm_image(MM_TEST_IMAGE_TIGER)
        messages = build_mm_parallel_weather_messages(image_url)
        client, model_name = self._get_client()

        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0,
            max_completion_tokens=CAPPED_MAX_COMPLETION_TOKENS,
            tools=[WEATHER_TOOL],
            logprobs=False,
        )

        tool_calls = response.choices[0].message.tool_calls
        assert tool_calls is not None and len(tool_calls) >= 2, (
            f'Expected ≥2 parallel tool calls for two cities, got {len(tool_calls) if tool_calls else 0}')

        parsed_list = []
        for tc in tool_calls:
            assert_tool_call_fields(tc)
            assert tc.function.name == 'get_current_weather'
            parsed = assert_arguments_parseable(tc.function.arguments)
            assert 'city' in parsed and 'state' in parsed
            parsed_list.append(parsed)
        assert_parallel_weather_cities_isolated(parsed_list)
        assert response.choices[0].finish_reason == 'tool_calls'

    def test_parallel_weather_with_image_streaming(self, backend, model_case):
        image_url = self._require_mm_image(MM_TEST_IMAGE_TIGER)
        messages = build_mm_parallel_weather_messages(image_url)
        client, model_name = self._get_client()

        stream = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0,
            max_completion_tokens=CAPPED_MAX_COMPLETION_TOKENS,
            tools=[WEATHER_TOOL],
            logprobs=False,
            stream=True,
        )

        chunks = list(stream)
        validate_stream_tool_call_chunks(chunks, check_incremental_arguments=False)
        tc_data, fr_count = collect_stream_parallel_tool_calls(iter(chunks))
        assert fr_count == 1
        assert len(tc_data) >= 2, (
            f'Expected ≥2 parallel streaming tool calls, got {len(tc_data)} indices: {list(tc_data.keys())}')

        parsed_list = []
        for idx, data in tc_data.items():
            assert data['name'] == 'get_current_weather', (
                f'Index {idx}: expected get_current_weather, got {data["name"]!r}')
            assert len(data['args_str']) > 0
            parsed = assert_arguments_parseable(data['args_str'])
            assert 'city' in parsed and 'state' in parsed
            parsed_list.append(parsed)
        assert_parallel_weather_cities_isolated(parsed_list)

    def test_parallel_tool_calls_false_with_image(self, backend, model_case):
        image_url = self._require_mm_image(MM_TEST_IMAGE_TIGER)
        messages = build_mm_parallel_weather_messages(image_url)
        client, model_name = self._get_client()

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0,
                max_completion_tokens=CAPPED_MAX_COMPLETION_TOKENS,
                tools=[WEATHER_TOOL],
                parallel_tool_calls=False,
                logprobs=False,
            )
        except BadRequestError as e:
            pytest.skip(f'parallel_tool_calls=False rejected by server (HTTP 400): {e}')

        tool_calls = response.choices[0].message.tool_calls
        assert tool_calls is not None and len(tool_calls) == 1
        assert_tool_call_fields(tool_calls[0])
        assert tool_calls[0].function.name == 'get_current_weather'
        assert response.choices[0].finish_reason == 'tool_calls'


# ===========================================================================
# MULTIMODAL_TYPES coverage (image + video variants with local fixtures)
# ===========================================================================


@_apply_marks_mm
class TestToolCallMultimodalMediaTypes(_ToolCallTestBase):
    """Smoke tool_call for each MULTIMODAL_TYPES variant with local fixtures
    (non-stream + stream for image/video)."""

    def _weather_tool_call_for_media_type(self, media_type: str) -> dict:
        source = self._require_mm_media_source(media_type)
        messages = mm_weather_messages_for_media_type(media_type, source)
        client, model_name = self._get_client()
        create_kwargs = {}
        extra_body = mm_create_extra_body_for_media_type(media_type)
        if extra_body is not None:
            create_kwargs['extra_body'] = extra_body

        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0,
            max_completion_tokens=CAPPED_MAX_COMPLETION_TOKENS,
            tools=[WEATHER_TOOL],
            logprobs=False,
            **create_kwargs,
        )

        choice = response.choices[0]
        tool_calls = choice.message.tool_calls
        assert choice.finish_reason == 'tool_calls', (
            f'Expected tool_calls for media_type={media_type!r}; '
            f'got finish_reason={choice.finish_reason!r}, '
            f'content={choice.message.content!r}')
        assert tool_calls is not None and len(tool_calls) >= 1
        tc = tool_calls[0]
        assert_tool_call_fields(tc)
        assert tc.function.name == 'get_current_weather'
        return assert_arguments_parseable(tc.function.arguments)

    def _weather_tool_call_stream_for_media_type(self, media_type: str) -> dict:
        """Streaming weather tool_call for one MULTIMODAL_TYPES entry."""
        source = self._require_mm_media_source(media_type)
        messages = mm_weather_messages_for_media_type(media_type, source)
        create_kwargs = {}
        extra_body = mm_create_extra_body_for_media_type(media_type)
        if extra_body is not None:
            create_kwargs['extra_body'] = extra_body

        r = self._stream_tool_call(messages, tools=[WEATHER_TOOL], **create_kwargs)
        assert r['role'] == 'assistant'
        assert r['chunk_count'] > 0, (
            f'Expected at least one SSE chunk for media_type={media_type!r}')
        validate_stream_tool_call_result(
            r,
            expected_function_name=WEATHER_TOOL['function']['name'],
            **self._parser_validation_kwargs([WEATHER_TOOL]),
        )
        return assert_arguments_parseable(r['args_str'])

    @staticmethod
    def _assert_dallas_weather_args(parsed: dict) -> None:
        assert 'city' in parsed and 'state' in parsed
        assert 'dallas' in parsed['city'].lower()
        assert 'tx' in parsed['state'].lower()

    @staticmethod
    def _assert_sichuan_weather_args(parsed: dict) -> None:
        assert 'city' in parsed and 'state' in parsed
        city = parsed['city'].lower()
        state = parsed['state'].lower()
        assert (
            'chengdu' in city or '成都' in city
            or 'sichuan' in city or 'sichuan' in state or '四川' in state
        )

    @pytest.mark.parametrize('media_type', MM_IMAGE_MEDIA_TYPES)
    def test_weather_tool_image_media_types(self, backend, model_case, media_type):
        parsed = self._weather_tool_call_for_media_type(media_type)
        self._assert_dallas_weather_args(parsed)

    @pytest.mark.parametrize('media_type', MM_IMAGE_MEDIA_TYPES)
    def test_weather_tool_image_media_types_streaming(
            self, backend, model_case, media_type):
        parsed = self._weather_tool_call_stream_for_media_type(media_type)
        self._assert_dallas_weather_args(parsed)

    def test_weather_tool_image_url_data_url(self, backend, model_case):
        """image_url with data:image/...;base64 payload (not a file path)."""
        path = self._require_mm_resource(MM_TEST_IMAGE_TIGER)
        data_url = mm_file_to_data_url(path, mime='image/jpeg')
        messages = mm_weather_messages_for_media_type('image_url', data_url)
        client, model_name = self._get_client()
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0,
            max_completion_tokens=CAPPED_MAX_COMPLETION_TOKENS,
            tools=[WEATHER_TOOL],
            logprobs=False,
        )
        choice = response.choices[0]
        assert choice.finish_reason == 'tool_calls', (
            f'Expected tool_calls for image_url data-URL; '
            f'got finish_reason={choice.finish_reason!r}, '
            f'content={choice.message.content!r}')
        assert choice.message.tool_calls is not None and len(choice.message.tool_calls) >= 1
        tc = choice.message.tool_calls[0]
        assert_tool_call_fields(tc)
        assert tc.function.name == 'get_current_weather'
        self._assert_dallas_weather_args(assert_arguments_parseable(tc.function.arguments))

    def test_weather_tool_image_url_data_url_streaming(self, backend, model_case):
        """Streaming image_url with data:image/...;base64 payload."""
        path = self._require_mm_resource(MM_TEST_IMAGE_TIGER)
        data_url = mm_file_to_data_url(path, mime='image/jpeg')
        messages = mm_weather_messages_for_media_type('image_url', data_url)
        r = self._stream_tool_call(messages, tools=[WEATHER_TOOL])
        assert r['role'] == 'assistant'
        assert r['chunk_count'] > 0
        validate_stream_tool_call_result(
            r,
            expected_function_name=WEATHER_TOOL['function']['name'],
            **self._parser_validation_kwargs([WEATHER_TOOL]),
        )
        self._assert_dallas_weather_args(assert_arguments_parseable(r['args_str']))

    @pytest.mark.parametrize('media_type', MM_VIDEO_MEDIA_TYPES)
    def test_weather_tool_video_media_types(self, backend, model_case, media_type):
        parsed = self._weather_tool_call_for_media_type(media_type)
        self._assert_sichuan_weather_args(parsed)

    @pytest.mark.parametrize('media_type', MM_VIDEO_MEDIA_TYPES)
    def test_weather_tool_video_media_types_streaming(
            self, backend, model_case, media_type):
        parsed = self._weather_tool_call_stream_for_media_type(media_type)
        self._assert_sichuan_weather_args(parsed)


# ===========================================================================
# Cross-protocol: Responses + Messages (Chat → ``TestToolCallMultimodal*``)
# ===========================================================================


@_apply_marks_mm
class TestToolCallMultimodalCrossProtocol(_ToolCallTestBase):

    @pytest.mark.parametrize('mm_api', _MM_CROSS_APIS)
    def test_dallas_weather_non_stream(self, backend, model_case, mm_api):
        del backend, model_case
        _dallas_weather_batch(self, mm_api)

    @pytest.mark.parametrize('mm_api', _MM_CROSS_APIS)
    def test_dallas_weather_stream(self, backend, model_case, mm_api):
        del backend, model_case
        _dallas_weather_stream(self, mm_api)

    @pytest.mark.parametrize('mm_api', _MM_CROSS_APIS)
    def test_stream_nonstream_consistency(self, backend, model_case, mm_api):
        del backend, model_case
        ns_name, ns_args = _dallas_weather_batch(self, mm_api)
        s_name, s_args = _dallas_weather_stream(self, mm_api)
        assert ns_name == s_name
        assert ns_args == s_args

    @pytest.mark.responses
    def test_tool_choice_none_describe_image(self, backend, model_case):
        del backend, model_case
        image_url = self._require_mm_image(MM_TEST_IMAGE_TIGER)
        messages = [build_multimodal_user_message('Describe this image.', image_url)]
        client, model_name = self._get_client()
        output = client.responses.create(
            model=model_name,
            input=messages,
            temperature=0,
            max_output_tokens=CAPPED_MAX_COMPLETION_TOKENS,
            **responses_tool_choice_kwargs('none', _RESPONSES_FLAT_TOOLS),
        ).model_dump()
        fcs = assert_responses_batch_return(output, model_name)
        assert not fcs
        assert responses_output_text(output, require_nonempty=True).strip()

    @pytest.mark.responses
    def test_tool_choice_force_weather_tool(self, backend, model_case):
        del backend, model_case
        image_url = self._require_mm_image(MM_TEST_IMAGE_TIGER)
        messages = mm_dallas_weather_messages(image_url)
        client, model_name = self._get_client()
        fcs = assert_responses_batch_return(
            client.responses.create(
                model=model_name,
                **responses_tool_create_kwargs(
                    tool_choice={'type': 'function', 'name': _WEATHER_NAME},
                    input=messages,
                ),
            ).model_dump(),
            model_name,
        )
        assert fcs
        _assert_weather_function_call(fcs[0])

    def test_anthropic_tool_choice_force_named_with_image(self, backend, model_case):
        del backend, model_case
        image_url = self._require_mm_image(MM_TEST_IMAGE_TIGER)
        req = _anthropic_vlm_weather_request(
            image_url,
            tool_choice={'type': 'tool', 'name': WEATHER_TOOL_ANTHROPIC['name']},
        )
        resp = _anthropic_post_messages(req_json=req, stream=False)
        assert resp.status_code == 200, resp.text
        blocks = _http_tool_use_blocks(assert_success_message_json(resp.json()))
        assert blocks[0]['name'] == WEATHER_TOOL_ANTHROPIC['name']
        assert_weather_tool_city_state(blocks[0]['input'], ctx='cross_protocol/force_named')

    def test_anthropic_tool_choice_any_with_image(self, backend, model_case):
        del backend, model_case
        image_url = self._require_mm_image(MM_TEST_IMAGE_TIGER)
        req = _anthropic_vlm_weather_request(image_url, tool_choice={'type': 'any'})
        resp = _anthropic_post_messages(req_json=req, stream=False)
        assert resp.status_code == 200, resp.text
        blocks = _http_tool_use_blocks(assert_success_message_json(resp.json()))
        allowed = {WEATHER_TOOL_ANTHROPIC['name'], SEARCH_TOOL_ANTHROPIC['name']}
        assert blocks[0]['name'] in allowed, blocks[0]['name']

    @pytest.mark.parametrize('mm_api', _MM_CROSS_APIS)
    def test_tool_result_followup_after_image_tool_use(self, backend, model_case, mm_api):
        del backend, model_case
        image_url = self._require_mm_image(MM_TEST_IMAGE_TIGER)
        messages = mm_dallas_weather_messages(image_url)
        followup_text = 'Sunny, 98F in Dallas, TX.'
        if mm_api == 'responses':
            client, model_name = self._get_client()
            fc = assert_responses_batch_return(
                client.responses.create(
                    model=model_name,
                    **responses_tool_create_kwargs(
                        tool_choice={'type': 'function', 'name': _WEATHER_NAME},
                        input=messages,
                    ),
                ).model_dump(),
                model_name,
            )[0]
            _assert_weather_function_call(fc)
            follow = client.responses.create(
                model=model_name,
                input=[
                    *messages,
                    {
                        'type': 'function_call',
                        'call_id': fc['call_id'],
                        'name': fc['name'],
                        'arguments': fc['arguments'],
                    },
                    {
                        'type': 'function_call_output',
                        'call_id': fc['call_id'],
                        'output': followup_text,
                    },
                ],
                temperature=0,
                max_output_tokens=CAPPED_MAX_COMPLETION_TOKENS,
                **responses_tool_choice_kwargs('none', _RESPONSES_FLAT_TOOLS),
            ).model_dump()
            assert not assert_responses_batch_return(follow, model_name)
            text = responses_output_text(follow, require_nonempty=True)
            assert '98' in text or 'Dallas' in text
            return
        req = _anthropic_vlm_weather_request(image_url)
        resp = _anthropic_post_messages(req_json=req, stream=False)
        assert resp.status_code == 200, resp.text
        tu = _http_tool_use_blocks(assert_success_message_json(resp.json()))[0]
        follow_req = _anthropic_vlm_weather_request(image_url)
        follow_req['messages'] = [
            _anthropic_vlm_user_turn(image_url),
            {
                'role': 'assistant',
                'content': [{
                    'type': 'tool_use',
                    'id': tu['id'],
                    'name': tu['name'],
                    'input': tu['input'],
                }],
            },
            {
                'role': 'user',
                'content': [{
                    'type': 'tool_result',
                    'tool_use_id': tu['id'],
                    'content': followup_text,
                }],
            },
        ]
        resp2 = _anthropic_post_messages(req_json=follow_req, stream=False)
        assert resp2.status_code == 200, resp2.text
        data2 = assert_success_message_json(resp2.json())
        text = ''.join(b['text'] for b in data2['content'] if b['type'] == 'text')
        assert '98' in text or 'Dallas' in text, text[:500]
