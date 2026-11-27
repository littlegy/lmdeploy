"""POST /v1/responses — lmdeploy Text V1 REST 协议覆盖。"""

import re

import pytest
import requests
from utils.anthropic_messages import RESPONSES_MATH_PROMPT_CN, TINY_1X1_PNG_BASE64
from utils.assert_purpose import fail_msg
from utils.config_utils import get_restful_chat_model_list
from utils.constant import (
    BACKEND_LIST,
    BASE_URL,
    CAPPED_MAX_COMPLETION_TOKENS,
    REPETITION_PROBE_PROMPT,
    THINKING_MAX_COMPLETION_TOKENS,
    VISION_SMOKE_MIN_COMPLETION_TOKENS,
)
from utils.gateway_compat import (
    chat_repetition_above_one_extra_body,
    engine_repetition_penalty_field,
    engine_top_k_extra_body,
    responses_low_repetition_extra_body,
    responses_protocol_probe_reasoning,
)
from utils.mm_restful import (
    is_mm_tool_call_capable,
    require_mm_image_url,
    responses_user_image_input,
    skip_unless_vlm_model,
)
from utils.restful_return_check import (
    assert_responses_batch_return,
    assert_responses_error,
    assert_usage_prefix_cache,
    assert_vision_tiger_text,
    gateway_auth_headers,
    get_client_and_model,
    has_repeated_fragment,
    post_responses_json,
    post_responses_stream_events,
    prefix_cache_probe_text,
    responses_output_text,
    responses_stream_text,
    wire_checked_response_json,
)

_RESPONSES_URL = f'{BASE_URL}/v1/responses'
_MATH_PROMPT = 'What is 13 * 24?'
_QED_INSTRUCTION = 'After the numeric answer, end with exactly: QED'
_MAX_OUTPUT_TOKENS = CAPPED_MAX_COMPLETION_TOKENS * 4

_RESPONSES_INVALID_SAMPLING_CASES = [
    pytest.param({'top_p': 0}, id='top_p_zero'),
    pytest.param({'top_p': 1.01}, id='top_p_above_one'),
    pytest.param({'temperature': -0.01}, id='temperature_negative'),
    pytest.param({'temperature': 2.01}, id='temperature_above_two'),
]


def _responses_json(model_name: str, **extra) -> dict:
    body = {
        'model': model_name,
        'input': _MATH_PROMPT,
        'max_output_tokens': CAPPED_MAX_COMPLETION_TOKENS,
        'temperature': 0.01,
    }
    body.update(extra)
    return body


def _post_responses_json(body: dict, *, stream: bool = False, timeout: int = 30) -> requests.Response:
    payload = dict(body)
    if stream:
        payload['stream'] = True
    return requests.post(
        _RESPONSES_URL,
        headers=gateway_auth_headers(),
        json=payload,
        stream=stream,
        timeout=timeout,
    )


@pytest.fixture(scope='class')
def openai_client_and_model():
    return get_client_and_model(BASE_URL)


@pytest.mark.parametrize('backend', BACKEND_LIST)
@pytest.mark.parametrize('model_case', get_restful_chat_model_list())
class TestRestfulOpenAIResponses:

    def test_return_info(self, backend, model_case, openai_client_and_model):
        del backend, model_case
        _, model_name = openai_client_and_model
        output = post_responses_json(
            model_name,
            {
                'input': _MATH_PROMPT,
                'max_output_tokens': _MAX_OUTPUT_TOKENS,
                'temperature': 0.01,
            },
        )
        assert_responses_batch_return(output, model_name)
        assert '312' in responses_output_text(output), fail_msg(
            '回复须含 13*24 结果', responses_output_text(output)[:300])

    def test_return_info_streaming(self, backend, model_case, openai_client_and_model):
        del backend, model_case
        _, model_name = openai_client_and_model
        events = post_responses_stream_events(
            model_name,
            {
                'input': _MATH_PROMPT,
                'max_output_tokens': _MAX_OUTPUT_TOKENS,
                'temperature': 0.01,
            },
        )
        assert events[0]['type'] == 'response.created', fail_msg('首事件', events[0])
        assert events[-1]['type'] == 'response.completed', fail_msg('末事件', events[-1])
        assert responses_stream_text(events).strip(), fail_msg(
            '流式应有非空输出', events[-1])

    def test_chinese_input_non_stream(self, backend, model_case, openai_client_and_model):
        del backend, model_case
        _, model_name = openai_client_and_model
        output = post_responses_json(
            model_name,
            {
                'input': RESPONSES_MATH_PROMPT_CN,
                'max_output_tokens': _MAX_OUTPUT_TOKENS,
                'temperature': 0.01,
            },
        )
        assert_responses_batch_return(output, model_name)
        text = responses_output_text(output, require_nonempty=True)
        assert '312' in text, fail_msg('中文数学题应含 312', text[:300])
        assert len(re.findall(r'[\u4e00-\u9fff]', text)) >= 2, fail_msg(
            '中文回复应含汉字', text[:300])

    def test_chinese_input_stream(self, backend, model_case, openai_client_and_model):
        del backend, model_case
        _, model_name = openai_client_and_model
        events = post_responses_stream_events(
            model_name,
            {
                'input': RESPONSES_MATH_PROMPT_CN,
                'max_output_tokens': _MAX_OUTPUT_TOKENS,
                'temperature': 0.01,
            },
        )
        text = responses_stream_text(events)
        assert '312' in text, fail_msg('中文流式应含 312', text[:300])
        assert len(re.findall(r'[\u4e00-\u9fff]', text)) >= 2, fail_msg(
            '中文流式应含汉字', text[:300])

    def test_instructions(self, backend, model_case, openai_client_and_model):
        del backend, model_case
        client, model_name = openai_client_and_model
        output = client.responses.create(
            model=model_name,
            instructions=_QED_INSTRUCTION,
            input=_MATH_PROMPT,
            max_output_tokens=_MAX_OUTPUT_TOKENS,
            temperature=0.01,
        ).model_dump()
        assert_responses_batch_return(output, model_name)
        text = responses_output_text(output, require_nonempty=True)
        assert '312' in text and 'QED' in text, fail_msg('instructions 应影响输出', text[:300])

    def test_chat_history_input(self, backend, model_case, openai_client_and_model):
        del backend, model_case
        client, model_name = openai_client_and_model
        output = client.responses.create(
            model=model_name,
            input=[
                {'role': 'system', 'content': _QED_INSTRUCTION},
                {'role': 'user', 'content': 'What is 5 * 3?'},
                {'role': 'assistant', 'content': '15. QED.'},
                {'role': 'user', 'content': f'Multiply the result by 2. {_QED_INSTRUCTION}'},
            ],
            max_output_tokens=_MAX_OUTPUT_TOKENS,
            temperature=0.01,
        ).model_dump()
        assert_responses_batch_return(output, model_name)
        text = responses_output_text(output, require_nonempty=True)
        assert '30' in text and 'QED' in text, fail_msg('多轮 input 应延续上下文', text[:300])

    def test_input_text_content_parts(self, backend, model_case, openai_client_and_model):
        del backend, model_case
        client, model_name = openai_client_and_model
        output = client.responses.create(
            model=model_name,
            input=[{
                'type': 'message',
                'role': 'user',
                'content': [{'type': 'input_text', 'text': _MATH_PROMPT}],
            }],
            max_output_tokens=_MAX_OUTPUT_TOKENS,
            temperature=0.01,
        ).model_dump()
        assert_responses_batch_return(output, model_name)
        assert '312' in responses_output_text(output, require_nonempty=True), fail_msg(
            'input_text 块路径', output.get('output_text', '')[:200])

    def test_unknown_model(self, backend, model_case, openai_client_and_model):
        del backend, model_case, openai_client_and_model
        resp = _post_responses_json(_responses_json('definitely-not-a-deployed-model-name'))
        assert_responses_error(
            resp,
            status_code=404,
            error_type='not_found_error',
            param='model',
        )

    @pytest.mark.parametrize('field_name,value', [
        pytest.param('background', True, id='background'),
        pytest.param('previous_response_id', 'resp_not_supported', id='previous_response_id'),
        pytest.param('conversation', 'conv_not_supported', id='conversation'),
    ])
    def test_text_v1_rejects_unsupported_stateful_fields(
            self, backend, model_case, openai_client_and_model, field_name, value):
        del backend, model_case
        _, model_name = openai_client_and_model
        resp = _post_responses_json(_responses_json(model_name, **{field_name: value}))
        assert_responses_error(
            resp,
            status_code=400,
            error_type='invalid_request_error',
            param=field_name,
            message_substr='not supported by Responses Text V1',
        )

    @pytest.mark.parametrize('stream', [False, True], ids=['batch', 'stream'])
    @pytest.mark.parametrize('extra_fields', _RESPONSES_INVALID_SAMPLING_CASES)
    def test_rejects_invalid_sampling_parameters(
            self, backend, model_case, openai_client_and_model, stream, extra_fields):
        del backend, model_case
        _, model_name = openai_client_and_model
        resp = _post_responses_json(_responses_json(model_name, **extra_fields), stream=stream)
        assert_responses_error(
            resp,
            status_code=400,
            error_type='invalid_request_error',
            strict_text_v1_shape=True,
        )

    def test_minimum_repetition_penalty(self, backend, model_case, openai_client_and_model):
        del backend, model_case
        client, model_name = openai_client_and_model
        response = client.responses.create(
            model=model_name,
            input=REPETITION_PROBE_PROMPT,
            temperature=0.01,
            max_output_tokens=THINKING_MAX_COMPLETION_TOKENS,
            reasoning=responses_protocol_probe_reasoning(),
            extra_body=responses_low_repetition_extra_body(),
        )
        output = response.model_dump()
        assert output['status'] in ('completed', 'incomplete'), fail_msg(
            'Responses status', output.get('status'))
        assert_responses_batch_return(output, model_name, status=output['status'])
        text = responses_output_text(output, require_nonempty=True)
        result, msg = has_repeated_fragment(text)
        assert result, fail_msg('极低 repetition_penalty 下应出现重复片段', msg)

    def test_minimum_repetition_penalty_streaming(self, backend, model_case, openai_client_and_model):
        del backend, model_case
        _, model_name = openai_client_and_model
        events = post_responses_stream_events(
            model_name,
            {
                'input': REPETITION_PROBE_PROMPT,
                'temperature': 0.01,
                'max_output_tokens': THINKING_MAX_COMPLETION_TOKENS,
                'reasoning': responses_protocol_probe_reasoning(),
                **responses_low_repetition_extra_body(),
            },
        )
        assert events[0]['type'] == 'response.created', fail_msg('首事件', events[0])
        assert events[-1]['type'] in ('response.completed', 'response.incomplete'), fail_msg(
            '末事件', events[-1])
        final_body = events[-1]['response']
        assert_responses_batch_return(final_body, model_name, status=final_body['status'])
        streaming_text = responses_stream_text(events)
        result, msg = has_repeated_fragment(streaming_text)
        assert result, fail_msg('极低 repetition_penalty 流式应出现重复片段', msg)

    def test_repetition_penalty_bigger_than_1(self, backend, model_case, openai_client_and_model):
        del backend, model_case
        client, model_name = openai_client_and_model
        output = client.responses.create(
            model=model_name,
            input=REPETITION_PROBE_PROMPT,
            max_output_tokens=THINKING_MAX_COMPLETION_TOKENS,
            temperature=0.01,
            extra_body=chat_repetition_above_one_extra_body(),
        ).model_dump()
        assert_responses_batch_return(output, model_name)

    def test_repetition_penalty_bigger_than_1_streaming(self, backend, model_case, openai_client_and_model):
        del backend, model_case
        _, model_name = openai_client_and_model
        events = post_responses_stream_events(
            model_name,
            {
                'input': REPETITION_PROBE_PROMPT,
                'max_output_tokens': THINKING_MAX_COMPLETION_TOKENS,
                'temperature': 0.01,
                **engine_repetition_penalty_field(),
            },
        )
        assert events[-1]['type'] in ('response.completed', 'response.incomplete'), fail_msg(
            '末事件', events[-1])
        final_body = events[-1]['response']
        assert_responses_batch_return(final_body, model_name, status=final_body['status'])

    def test_top_k_minus_1_disables_top_k_filtering(self, backend, model_case, openai_client_and_model):
        del backend, model_case
        client, model_name = openai_client_and_model
        output = client.responses.create(
            model=model_name,
            input='Hi, pls intro yourself',
            max_output_tokens=THINKING_MAX_COMPLETION_TOKENS,
            temperature=0.01,
            extra_body=engine_top_k_extra_body(-1),
        ).model_dump()
        assert output['status'] in ('completed', 'incomplete'), fail_msg(
            'Responses status', output.get('status'))
        assert_responses_batch_return(output, model_name, status=output['status'])
        text = responses_output_text(output, require_nonempty=True)
        assert text.strip(), fail_msg('top_k=-1 应产出非空文本', text[:200])

    def test_top_k_1_greedy(self, backend, model_case, openai_client_and_model):
        del backend, model_case
        client, model_name = openai_client_and_model

        def _greedy_text() -> str:
            output = client.responses.create(
                model=model_name,
                input=REPETITION_PROBE_PROMPT,
                max_output_tokens=200,
                temperature=0,
                reasoning=responses_protocol_probe_reasoning(),
                extra_body=engine_top_k_extra_body(1),
            ).model_dump()
            assert output['status'] in ('completed', 'incomplete'), fail_msg(
                'Responses status', output.get('status'))
            assert_responses_batch_return(output, model_name, status=output['status'])
            return responses_output_text(output, require_nonempty=True)

        text_a = _greedy_text()
        text_b = _greedy_text()
        assert text_a.strip(), fail_msg('top_k=1 贪心应非空', text_a[:200])
        assert text_a == text_b, fail_msg(
            'top_k=1 同参两次输出应一致', (text_a[:120], text_b[:120]))

    @pytest.mark.prefix_cache
    def test_prefix_cache_hit(self, backend, model_case, openai_client_and_model):
        del backend, model_case
        _, model_name = openai_client_and_model
        body = {
            'model': model_name,
            'input': prefix_cache_probe_text(),
            'temperature': 0,
            'max_output_tokens': 64,
            'reasoning': responses_protocol_probe_reasoning(),
        }
        first = _post_responses_json(body, timeout=120)
        second = _post_responses_json(body, timeout=120)
        assert first.status_code == 200, fail_msg('第一次应 200', first.text[:300])
        assert second.status_code == 200, fail_msg('第二次应 200', second.text[:300])
        first_body = wire_checked_response_json(first)
        second_body = wire_checked_response_json(second)
        assert_usage_prefix_cache(
            first_body['usage'],
            second_body['usage'],
            prompt_key='input_tokens',
            cached_keys=('input_tokens_details', 'cached_tokens'),
            output_key='output_tokens',
        )
        t1 = responses_output_text(first_body, require_nonempty=True)
        t2 = responses_output_text(second_body, require_nonempty=True)
        assert 'ok' in t1.lower() and t1 == t2, fail_msg('前缀缓存输出一致', (t1[:120], t2[:120]))


@pytest.mark.parametrize('backend', BACKEND_LIST)
@pytest.mark.parametrize('model_case', get_restful_chat_model_list())
class TestRestfulResponsesMultimodal:
    """Responses multimodal protocol only (no tools).

    Tool+image → ``tool_parser``.
    """

    def test_text_model_rejects_image(self, backend, model_case, openai_client_and_model):
        del backend
        if is_mm_tool_call_capable(model_case):
            pytest.skip('VL models covered by accept-image cases')
        _, model_name = openai_client_and_model
        resp = _post_responses_json({
            'model': model_name,
            'input': responses_user_image_input(
                image_url=f'data:image/png;base64,{TINY_1X1_PNG_BASE64}',
                text='What is in this image?',
            ),
            'max_output_tokens': 16,
            'temperature': 0,
        }, timeout=60)
        assert_responses_error(
            resp,
            status_code=400,
            error_type='invalid_request_error',
        )

    def test_vision_resource_image_non_stream(self, backend, model_case, config, openai_client_and_model):
        del backend
        skip_unless_vlm_model(model_case)
        _, model_name = openai_client_and_model
        image_url = require_mm_image_url(config)
        resp = _post_responses_json({
            'model': model_name,
            'input': responses_user_image_input(
                image_url=image_url,
                text='In one word, name the animal in the image.',
            ),
            'max_output_tokens': VISION_SMOKE_MIN_COMPLETION_TOKENS,
            'temperature': 0.01,
        }, timeout=180)
        if resp.status_code == 400:
            pytest.skip(f'Responses input_image with file url not supported: {resp.text[:400]}')
        assert resp.status_code == 200, fail_msg('Responses 多模应成功', resp.text[:800])
        body = wire_checked_response_json(resp)
        text = responses_output_text(body, require_nonempty=True)
        assert_vision_tiger_text(text, label='Responses')

    def test_vision_resource_image_stream(self, backend, model_case, config, openai_client_and_model):
        del backend
        skip_unless_vlm_model(model_case)
        _, model_name = openai_client_and_model
        image_url = require_mm_image_url(config)
        events = post_responses_stream_events(
            model_name,
            {
                'input': responses_user_image_input(
                    image_url=image_url,
                    text='In one word, name the animal in the image.',
                ),
                'max_output_tokens': VISION_SMOKE_MIN_COMPLETION_TOKENS,
                'temperature': 0.01,
            },
        )
        assert events[0]['type'] == 'response.created', fail_msg('首事件', events[0])
        assert events[-1]['type'] == 'response.completed', fail_msg('末事件', events[-1])
        text = responses_stream_text(events)
        assert text.strip(), fail_msg('Responses 流式多模应有非空输出', events[-1])
        assert_vision_tiger_text(text, label='Responses stream')
