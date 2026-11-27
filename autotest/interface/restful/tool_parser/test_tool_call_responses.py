"""Responses Text V1 tool calls (``POST /v1/responses``)."""

import pytest
import requests
from openai import BadRequestError
from utils.constant import BASE_URL, CAPPED_MAX_COMPLETION_TOKENS
from utils.gateway_compat import responses_tool_choice_kwargs
from utils.restful_return_check import (
    assert_responses_batch_return,
    assert_responses_error,
    responses_output_text,
)
from utils.tool_reasoning_definitions import MESSAGES_HELLO, SEARCH_TOOL, WEATHER_TOOL, assert_arguments_parseable

from interface.restful.tool_parser.conftest import MESSAGES_ASKING_FOR_WEATHER, _apply_marks, _ToolCallTestBase

_RESPONSES_URL = f'{BASE_URL}/v1/responses'
_WEATHER_NAME = WEATHER_TOOL['function']['name']
_TOOLS = [
    {'type': 'function', **WEATHER_TOOL['function']},
    {'type': 'function', **SEARCH_TOOL['function']},
]
_SEARCH_NAME = SEARCH_TOOL['function']['name']
_SEARCH_TOOL = {'type': 'function', **SEARCH_TOOL['function']}
_CREATE = dict(
    input=MESSAGES_ASKING_FOR_WEATHER,
    tools=_TOOLS,
    temperature=0,
    max_output_tokens=CAPPED_MAX_COMPLETION_TOKENS,
)


def responses_tool_create_kwargs(*, tool_choice, input=None, **extra) -> dict:
    body = dict(
        input=input if input is not None else MESSAGES_ASKING_FOR_WEATHER,
        tools=_TOOLS,
        temperature=0,
        max_output_tokens=CAPPED_MAX_COMPLETION_TOKENS,
        tool_choice=tool_choice,
    )
    body.update(extra)
    return body


def stream_responses_function_call(client, model_name: str, **create_kwargs) -> dict:
    """Shared streaming tool-call collector for Responses (text + multimodal)."""
    events = list(client.responses.create(model=model_name, stream=True, **create_kwargs))
    assert events[0].type == 'response.created'
    assert events[-1].type == 'response.completed'
    args_delta = ''.join(
        event.delta for event in events if event.type == 'response.function_call_arguments.delta')
    fcs = assert_responses_batch_return(events[-1].response.model_dump(), model_name)
    assert fcs
    assert args_delta == fcs[0]['arguments']
    return fcs[0]


def _assert_weather_function_call(fc):
    assert fc['name'] == _WEATHER_NAME
    parsed = assert_arguments_parseable(fc['arguments'])
    assert 'dallas' in parsed['city'].lower()
    assert 'tx' in parsed['state'].lower()
    return parsed


def _assert_search_function_call(fc):
    assert fc['name'] == _SEARCH_NAME
    parsed = assert_arguments_parseable(fc['arguments'])
    assert parsed.get('query'), parsed
    return parsed


@pytest.mark.responses
@_apply_marks
class TestToolCallResponses(_ToolCallTestBase):

    def test_non_streaming(self, backend, model_case):
        client, model_name = self._get_client()
        fcs = assert_responses_batch_return(
            client.responses.create(
                model=model_name,
                tool_choice='required',
                input=MESSAGES_HELLO,
                tools=[_SEARCH_TOOL],
                temperature=0,
                max_output_tokens=CAPPED_MAX_COMPLETION_TOKENS,
            ).model_dump(),
            model_name,
        )
        assert fcs
        _assert_search_function_call(fcs[0])

    def test_streaming(self, backend, model_case):
        client, model_name = self._get_client()
        fc = stream_responses_function_call(
            client,
            model_name,
            tool_choice='required',
            input=MESSAGES_HELLO,
            tools=[_SEARCH_TOOL],
            temperature=0,
            max_output_tokens=CAPPED_MAX_COMPLETION_TOKENS,
        )
        _assert_search_function_call(fc)

    def test_stream_nonstream_consistency(self, backend, model_case):
        client, model_name = self._get_client()
        kwargs = responses_tool_create_kwargs(tool_choice={'type': 'function', 'name': _WEATHER_NAME})
        ns_fcs = assert_responses_batch_return(
            client.responses.create(model=model_name, **kwargs).model_dump(),
            model_name,
        )
        assert ns_fcs
        ns_fc = ns_fcs[0]
        _assert_weather_function_call(ns_fc)
        stream_fc = stream_responses_function_call(client, model_name, **kwargs)
        _assert_weather_function_call(stream_fc)
        assert ns_fc['name'] == stream_fc['name']
        assert assert_arguments_parseable(ns_fc['arguments']) == assert_arguments_parseable(
            stream_fc['arguments'])

    def test_tool_choice_auto(self, backend, model_case):
        client, model_name = self._get_client()
        output = client.responses.create(
            model=model_name,
            **responses_tool_create_kwargs(tool_choice='auto'),
        ).model_dump()
        fcs = assert_responses_batch_return(output, model_name)
        if fcs:
            assert fcs[0]['name'] in (_WEATHER_NAME, _SEARCH_NAME)
            assert_arguments_parseable(fcs[0]['arguments'])
        else:
            assert responses_output_text(output, require_nonempty=True).strip()

    def test_tool_choice_required(self, backend, model_case):
        client, model_name = self._get_client()
        try:
            output = client.responses.create(
                model=model_name,
                **responses_tool_create_kwargs(tool_choice='required'),
            ).model_dump()
        except BadRequestError as e:
            pytest.skip(f'tool_choice="required" rejected by server (HTTP 400): {e}')
        fcs = assert_responses_batch_return(output, model_name)
        assert fcs
        for fc in fcs:
            assert fc['name'] in (_WEATHER_NAME, _SEARCH_NAME)
            assert_arguments_parseable(fc['arguments'])

    def test_tool_choice_required_streaming(self, backend, model_case):
        client, model_name = self._get_client()
        try:
            fc = stream_responses_function_call(
                client,
                model_name,
                **responses_tool_create_kwargs(tool_choice='required'),
            )
        except BadRequestError as e:
            pytest.skip(f'tool_choice="required" streaming rejected by server (HTTP 400): {e}')
        assert fc['name'] in (_WEATHER_NAME, _SEARCH_NAME)
        assert_arguments_parseable(fc['arguments'])

    def test_named_tool_choice(self, backend, model_case):
        client, model_name = self._get_client()
        fcs = assert_responses_batch_return(
            client.responses.create(
                model=model_name,
                tool_choice={'type': 'function', 'name': _WEATHER_NAME},
                **_CREATE,
            ).model_dump(),
            model_name,
        )
        assert fcs
        _assert_weather_function_call(fcs[0])

    def test_tool_choice_none(self, backend, model_case):
        client, model_name = self._get_client()
        output = client.responses.create(
            model=model_name,
            **_CREATE,
            **responses_tool_choice_kwargs('none', _TOOLS),
        ).model_dump()
        fcs = assert_responses_batch_return(output, model_name)
        assert not fcs
        assert responses_output_text(output, require_nonempty=True).strip()

    def test_function_call_output_followup(self, backend, model_case):
        client, model_name = self._get_client()
        fc = assert_responses_batch_return(
            client.responses.create(
                model=model_name,
                tool_choice={'type': 'function', 'name': _WEATHER_NAME},
                **_CREATE,
            ).model_dump(),
            model_name,
        )[0]
        _assert_weather_function_call(fc)
        output = client.responses.create(
            model=model_name,
            input=[
                *MESSAGES_ASKING_FOR_WEATHER,
                {
                    'type': 'function_call',
                    'call_id': fc['call_id'],
                    'name': fc['name'],
                    'arguments': fc['arguments'],
                },
                {
                    'type': 'function_call_output',
                    'call_id': fc['call_id'],
                    'output': 'Sunny, 98F in Dallas, TX.',
                },
            ],
            temperature=0,
            max_output_tokens=CAPPED_MAX_COMPLETION_TOKENS,
            **responses_tool_choice_kwargs('none', _TOOLS),
        ).model_dump()
        fcs = assert_responses_batch_return(output, model_name)
        assert not fcs
        text = output['output_text']
        assert '98' in text or 'Dallas' in text

    def test_unknown_tool_choice_name(self, backend, model_case):
        _, model_name = self._get_client()
        resp = requests.post(
            _RESPONSES_URL,
            json={
                'model': model_name,
                'input': 'Hi',
                'tools': [_TOOLS[0]],
                'tool_choice': {'type': 'function', 'name': 'missing'},
            },
            timeout=30,
        )
        assert_responses_error(resp, status_code=400, error_type='invalid_request_error', param='tool_choice')

    def test_tools_missing_name(self, backend, model_case):
        _, model_name = self._get_client()
        resp = requests.post(
            _RESPONSES_URL,
            json={
                'model': model_name,
                'input': 'Hi',
                'tools': [{'type': 'function'}],
            },
            timeout=30,
        )
        assert_responses_error(resp, status_code=400, error_type='invalid_request_error', param='tools')