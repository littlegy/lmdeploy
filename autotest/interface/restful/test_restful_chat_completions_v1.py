import json
import re

import pytest
import requests
from openai import APIStatusError, BadRequestError
from utils.anthropic_messages import TINY_1X1_PNG_BASE64, USER_GREETING_CN
from utils.assert_purpose import fail_msg
from utils.config_utils import get_model_path_from_config, get_restful_chat_model_list
from utils.constant import (
    BACKEND_LIST,
    BASE_URL,
    CAPPED_MAX_COMPLETION_TOKENS,
    REPETITION_PROBE_PROMPT,
    THINKING_MAX_COMPLETION_TOKENS,
    VISION_SMOKE_MIN_COMPLETION_TOKENS,
)
from utils.gateway_compat import (
    chat_invalid_max_output_fields,
    chat_low_repetition_extra_body,
    chat_max_output_kwargs,
    chat_protocol_probe_create_kwargs,
    chat_repetition_above_one_extra_body,
    chat_thinking_off_extra_body,
    engine_top_k_extra_body,
)
from utils.restful_return_check import (
    CONTEXT_LENGTH_ERROR,
    assert_chat_completions_batch_return,
    assert_chat_completions_stream_return,
    assert_chat_return_token_ids,
    assert_chat_sse_stream_ends_with_done,
    assert_chat_stream_include_usage,
    assert_chat_vision_tiger_choice,
    assert_openai_invalid_request_error,
    assert_prefix_cache_hit,
    assert_vision_tiger_text,
    build_session_sized_user_content,
    chat_stream_final_usage,
    chat_stream_payload_chunks,
    gateway_auth_headers,
    get_chat_delta_text,
    get_chat_message_text,
    get_client_and_model,
    has_repeated_fragment,
    post_chat_completions_json,
    post_chat_completions_stream_chunks,
)
from utils.toolkit import encode_text

from utils.mm_restful import (
    build_multimodal_user_message,
    is_mm_tool_call_capable,
    require_mm_image_url,
    skip_unless_vlm_model,
)

_CHAT_COMPLETIONS_URL = f'{BASE_URL}/v1/chat/completions'
_CHAT_MESSAGES = [{'role': 'user', 'content': 'Hi, pls intro yourself'}]

_CHAT_INVALID_SAMPLING_CASES = [
    pytest.param({'top_p': 0}, id='top_p_zero'),
    pytest.param({'top_p': 1.01}, id='top_p_above_one'),
    pytest.param({'top_p': 'test'}, id='top_p_string'),
    pytest.param({'n': 0}, id='n_zero'),
    pytest.param({'n': 'test'}, id='n_string'),
    pytest.param({'temperature': -0.01}, id='temperature_negative'),
    pytest.param({'temperature': 2.01}, id='temperature_above_two'),
    pytest.param({'temperature': 'test'}, id='temperature_string'),
]

_CHAT_INVALID_MESSAGE_CASES = [
    pytest.param({'messages': []}, id='messages_empty'),
    pytest.param({'messages': [{'content': 'hello'}]}, id='missing_role'),
    pytest.param({'messages': [{'role': 'bogus', 'content': 'hi'}]}, id='invalid_role'),
    pytest.param({'messages': [{'role': 'tool', 'content': 'result'}]}, id='tool_missing_call_id'),
]

_LMDEPLOY_CHAT_N_MULTI_SMOKE = 3
_CHAT_N_OVER_FANOUT_MAX = 129
_CHAT_LOGPROBS_NUM = 3

_RESPONSE_FORMAT_WEATHER_SCHEMA = {
    'type': 'object',
    'properties': {
        'city': {'type': 'string'},
        'temperature': {'type': 'number'},
    },
    'required': ['city', 'temperature'],
    'additionalProperties': False,
}


@pytest.fixture(scope='class')
def openai_client_and_model():
    return get_client_and_model(BASE_URL)


@pytest.mark.order(8)
@pytest.mark.flaky(reruns=2)
@pytest.mark.parametrize('backend', BACKEND_LIST)
@pytest.mark.parametrize('model_case', get_restful_chat_model_list())
class TestRestfulOpenAI:

    @pytest.mark.pr_test
    def test_return_info(self, backend, model_case, openai_client_and_model):
        del backend, model_case
        _, model_name = openai_client_and_model
        output = post_chat_completions_json(
            model_name,
            {'messages': _CHAT_MESSAGES, 'temperature': 0.01},
        )
        assert_chat_completions_batch_return(output, model_name)

    @pytest.mark.pr_test
    def test_return_info_streaming(self, backend, model_case, openai_client_and_model):
        client, model_name = openai_client_and_model
        outputs = client.chat.completions.create(model=model_name,
                                                 messages=[
                                                     {
                                                         'role': 'user',
                                                         'content': 'Hi, pls intro yourself'
                                                     },
                                                 ],
                                                 temperature=0.01,
                                                 stream=True)

        outputList = []
        for output in outputs:
            outputList.append(output.model_dump())

        assert_chat_completions_stream_return(outputList[-1], model_name, True)
        for index in range(0, len(outputList) - 1):
            assert_chat_completions_stream_return(outputList[index], model_name)

    def test_single_stopword(self, backend, model_case, openai_client_and_model):
        client, model_name = openai_client_and_model
        outputs = client.chat.completions.create(model=model_name,
                                                 messages=[
                                                     {
                                                         'role': 'user',
                                                         'content': 'Shanghai is'
                                                     },
                                                 ],
                                                 temperature=0.01,
                                                 stop=' is')

        output = outputs.model_dump()
        assert_chat_completions_batch_return(output, model_name)
        assert ' is' not in get_chat_message_text(output.get('choices')[0])
        assert output.get('choices')[0].get('finish_reason') == 'stop'

    @pytest.mark.pr_test
    def test_single_stopword_streaming(self, backend, model_case, openai_client_and_model):
        client, model_name = openai_client_and_model
        outputs = client.chat.completions.create(model=model_name,
                                                 messages=[
                                                     {
                                                         'role': 'user',
                                                         'content': 'Shanghai is'
                                                     },
                                                 ],
                                                 stop=' is',
                                                 temperature=0.01,
                                                 stream=True)

        outputList = []
        for output in outputs:
            outputList.append(output.model_dump())

        assert_chat_completions_stream_return(outputList[-1], model_name, True)
        for index in range(0, len(outputList) - 1):
            assert_chat_completions_stream_return(outputList[index], model_name)
            assert ' is ' not in get_chat_delta_text(outputList[index].get('choices')[0])
        assert outputList[-1].get('choices')[0].get('finish_reason') == 'stop'

    def test_array_stopwords(self, backend, model_case, openai_client_and_model):
        client, model_name = openai_client_and_model
        outputs = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    'role': 'user',
                    'content': 'Shanghai is'
                },
            ],
            temperature=0.01,
            stop=[' is', '上海', ' to'],
        )

        output = outputs.model_dump()
        assert_chat_completions_batch_return(output, model_name)
        assert ' is' not in get_chat_message_text(output.get('choices')[0])
        assert ' 上海' not in get_chat_message_text(output.get('choices')[0])
        assert ' to' not in get_chat_message_text(output.get('choices')[0])
        assert output.get('choices')[0].get('finish_reason') == 'stop'

    def test_array_stopwords_streaming(self, backend, model_case, openai_client_and_model):
        client, model_name = openai_client_and_model
        outputs = client.chat.completions.create(model=model_name,
                                                 messages=[
                                                     {
                                                         'role': 'user',
                                                         'content': 'Shanghai is'
                                                     },
                                                 ],
                                                 stop=[' is', '上海', ' to'],
                                                 temperature=0.01,
                                                 stream=True)

        outputList = []
        for output in outputs:
            outputList.append(output.model_dump())

        assert_chat_completions_stream_return(outputList[-1], model_name, True)
        for index in range(0, len(outputList) - 1):
            assert_chat_completions_stream_return(outputList[index], model_name)
            assert ' is' not in get_chat_delta_text(outputList[index].get('choices')[0])
            assert '上海' not in get_chat_delta_text(outputList[index].get('choices')[0])
            assert ' to ' not in get_chat_delta_text(outputList[index].get('choices')[0])
        assert outputList[-1].get('choices')[0].get('finish_reason') == 'stop'

    @pytest.mark.pr_test
    def test_minimum_topp(self, backend, model_case, openai_client_and_model):
        client, model_name = openai_client_and_model
        outputList = []
        for i in range(3):
            outputs = client.chat.completions.create(model=model_name,
                                                     messages=[
                                                         {
                                                             'role': 'user',
                                                             'content': 'Shanghai is'
                                                         },
                                                     ],
                                                     temperature=0.01,
                                                     top_p=0.0000000001,
                                                     max_tokens=10)
            output = outputs.model_dump()
            outputList.append(output)
            assert_chat_completions_batch_return(output, model_name)
        texts = [get_chat_message_text(output.get('choices')[0]) for output in outputList]
        assert texts[0] == texts[1]
        assert texts[1] == texts[2]

    def test_minimum_topp_streaming(self, backend, model_case, openai_client_and_model):
        client, model_name = openai_client_and_model
        responseList = []
        for i in range(3):
            outputs = client.chat.completions.create(model=model_name,
                                                     messages=[
                                                         {
                                                             'role': 'user',
                                                             'content': 'Hi, pls intro yourself'
                                                         },
                                                     ],
                                                     top_p=0.0000000001,
                                                     max_tokens=10,
                                                     stream=True)

            outputList = []
            for output in outputs:
                outputList.append(output.model_dump())
            assert_chat_completions_stream_return(outputList[-1], model_name, True)
            response = ''
            for index in range(0, len(outputList) - 1):
                assert_chat_completions_stream_return(outputList[index], model_name)
                response += get_chat_delta_text(outputList[index].get('choices')[0])
            responseList.append(response)
        assert responseList[0] == responseList[1] or responseList[1] == responseList[2]

    @pytest.mark.pr_test
    def test_mistake_modelname_return(self, backend, model_case, openai_client_and_model):
        client, _ = openai_client_and_model
        with pytest.raises(Exception, match='The model \'error\' does not exist.'):
            client.chat.completions.create(
                model='error',
                messages=[
                    {
                        'role': 'user',
                        'content': 'Shanghai is'
                    },
                ],
                temperature=0.01,
                stop=[' is', '上海', ' to'],
            )

    def test_mistake_modelname_return_streaming(self, backend, model_case, openai_client_and_model):
        client, _ = openai_client_and_model

        with pytest.raises(Exception, match='The model \'error\' does not exist.'):
            client.chat.completions.create(model='error',
                                           messages=[
                                               {
                                                   'role': 'user',
                                                   'content': 'Hi, pls intro yourself'
                                               },
                                           ],
                                           max_tokens=5,
                                           temperature=0.01,
                                           stream=True)

    @pytest.mark.pr_test
    def test_mutilple_times_response_should_not_same(self, backend, model_case, openai_client_and_model):
        client, model_name = openai_client_and_model
        outputList = []
        for i in range(3):
            outputs = client.chat.completions.create(model=model_name,
                                                     messages=[
                                                         {
                                                             'role': 'user',
                                                             'content': 'Shanghai is'
                                                         },
                                                     ],
                                                     max_tokens=100)
            output = outputs.model_dump()
            outputList.append(output)
            assert_chat_completions_batch_return(output, model_name)
        texts = [get_chat_message_text(output.get('choices')[0]) for output in outputList]
        assert texts[0] != texts[1] or texts[1] != texts[2]

    def test_mutilple_times_response_should_not_same_streaming(self, backend, model_case, openai_client_and_model):
        client, model_name = openai_client_and_model
        responseList = []
        for i in range(3):
            outputs = client.chat.completions.create(model=model_name,
                                                     messages=[
                                                         {
                                                             'role': 'user',
                                                             'content': 'Hi, pls intro yourself'
                                                         },
                                                     ],
                                                     max_tokens=100,
                                                     stream=True)

            outputList = []
            for output in outputs:
                outputList.append(output.model_dump())
            assert_chat_completions_stream_return(outputList[-1], model_name, True)
            response = ''
            for index in range(0, len(outputList) - 1):
                assert_chat_completions_stream_return(outputList[index], model_name)
                response += get_chat_delta_text(outputList[index].get('choices')[0])
            responseList.append(response)
        assert responseList[0] != responseList[1] or responseList[1] == responseList[2]

    def test_longtext_input(self, backend, model_case, openai_client_and_model, config):
        client, model_name = openai_client_and_model
        oversize_content = build_session_sized_user_content(
            config=config, model_id=model_case, oversize=True)
        with pytest.raises(BadRequestError) as ei:
            client.chat.completions.create(model=model_name,
                                           messages=[
                                               {
                                                   'role': 'user',
                                                   'content': oversize_content,
                                               },
                                           ],
                                           max_tokens=100)
        assert ei.value.status_code == 400
        assert_openai_invalid_request_error(ei.value.body, message_substr=CONTEXT_LENGTH_ERROR)

    @pytest.mark.pr_test
    def test_longtext_input_streaming(self, backend, model_case, openai_client_and_model, config):
        client, model_name = openai_client_and_model
        oversize_content = build_session_sized_user_content(
            config=config, model_id=model_case, oversize=True)
        with pytest.raises(BadRequestError) as ei:
            client.chat.completions.create(model=model_name,
                                           messages=[
                                               {
                                                   'role': 'user',
                                                   'content': oversize_content,
                                               },
                                           ],
                                           max_tokens=100,
                                           stream=True)
        assert ei.value.status_code == 400
        assert_openai_invalid_request_error(ei.value.body, message_substr=CONTEXT_LENGTH_ERROR)

    @pytest.mark.pr_test
    def test_max_tokens(self, backend, model_case, openai_client_and_model):
        client, model_name = openai_client_and_model
        outputs = client.chat.completions.create(model=model_name,
                                                 messages=[
                                                     {
                                                         'role': 'user',
                                                         'content': 'Hi, pls intro yourself'
                                                     },
                                                 ],
                                                 max_tokens=5,
                                                 temperature=0.01)
        output = outputs.model_dump()
        assert_chat_completions_batch_return(output, model_name)
        assert output.get('choices')[0].get('finish_reason') == 'length'
        assert output.get('usage').get('completion_tokens') == 6 or output.get('usage').get('completion_tokens') == 5

    def test_max_tokens_streaming(self, backend, model_case, openai_client_and_model, config):
        client, model_name = openai_client_and_model
        model_path = get_model_path_from_config(config, model_case)

        outputs = client.chat.completions.create(model=model_name,
                                                 messages=[
                                                     {
                                                         'role': 'user',
                                                         'content': 'Hi, pls intro yourself'
                                                     },
                                                 ],
                                                 max_tokens=5,
                                                 temperature=0.01,
                                                 stream=True)

        outputList = []
        for output in outputs:
            outputList.append(output.model_dump())

        assert_chat_completions_stream_return(outputList[-1], model_name, True)
        response = ''
        for index in range(0, len(outputList) - 1):
            assert_chat_completions_stream_return(outputList[index], model_name)
            response += get_chat_delta_text(outputList[index].get('choices')[0])
        length = len(encode_text(model_path, response, add_special_tokens=False))
        assert outputList[-1].get('choices')[0].get('finish_reason') == 'length'
        assert length == 5 or length == 6

    @pytest.mark.logprob
    @pytest.mark.not_pytorch
    @pytest.mark.pr_test
    def test_logprobs_legacy_turbomind_pr(self, backend, model_case, openai_client_and_model):
        client, model_name = openai_client_and_model
        outputs = client.chat.completions.create(model=model_name,
                                                 messages=[
                                                     {
                                                         'role': 'user',
                                                         'content': 'Hi, pls intro yourself'
                                                     },
                                                 ],
                                                 max_tokens=5,
                                                 temperature=0.01,
                                                 logprobs=True,
                                                 top_logprobs=10)
        output = outputs.model_dump()
        assert_chat_completions_batch_return(output, model_name, check_logprobs=True, logprobs_num=10)
        assert output.get('choices')[0].get('finish_reason') == 'length'
        assert output.get('usage').get('completion_tokens') == 6 or output.get('usage').get('completion_tokens') == 5

    @pytest.mark.logprob
    @pytest.mark.not_pytorch
    @pytest.mark.pr_test
    def test_logprobs_streaming_legacy_turbomind_pr(self, backend, model_case, openai_client_and_model, config):
        client, model_name = openai_client_and_model
        model_path = get_model_path_from_config(config, model_case)

        outputs = client.chat.completions.create(model=model_name,
                                                 messages=[
                                                     {
                                                         'role': 'user',
                                                         'content': 'Hi, pls intro yourself'
                                                     },
                                                 ],
                                                 max_tokens=5,
                                                 temperature=0.01,
                                                 logprobs=True,
                                                 top_logprobs=10,
                                                 stream=True)

        outputList = []
        for output in outputs:
            outputList.append(output.model_dump())

        assert_chat_completions_stream_return(outputList[-1], model_name, True, check_logprobs=True, logprobs_num=10)
        response = ''
        for index in range(0, len(outputList) - 1):
            assert_chat_completions_stream_return(outputList[index], model_name, check_logprobs=True, logprobs_num=10)
            response += get_chat_delta_text(outputList[index].get('choices')[0])
        length = len(encode_text(model_path, response, add_special_tokens=False))
        assert outputList[-1].get('choices')[0].get('finish_reason') == 'length'
        assert length == 5 or length == 6

    def test_minimum_repetition_penalty(self, backend, model_case, openai_client_and_model):
        client, model_name = openai_client_and_model
        outputs = client.chat.completions.create(
            model=model_name,
            messages=[{'role': 'user', 'content': REPETITION_PROBE_PROMPT}],
            extra_body=chat_low_repetition_extra_body(),
            temperature=0.01,
            max_tokens=200,
        )
        output = outputs.model_dump()
        assert_chat_completions_batch_return(output, model_name)
        result, msg = has_repeated_fragment(get_chat_message_text(output.get('choices')[0]))
        assert result, msg

    def test_minimum_repetition_penalty_streaming(self, backend, model_case, openai_client_and_model):
        client, model_name = openai_client_and_model
        outputs = client.chat.completions.create(
            model=model_name,
            messages=[{'role': 'user', 'content': REPETITION_PROBE_PROMPT}],
            extra_body=chat_low_repetition_extra_body(),
            temperature=0.01,
            max_tokens=200,
            stream=True,
        )
        outputList = [chunk.model_dump() for chunk in outputs]
        assert_chat_completions_stream_return(outputList[-1], model_name, True)
        response = ''
        for index in range(0, len(outputList) - 1):
            assert_chat_completions_stream_return(outputList[index], model_name)
            response += get_chat_delta_text(outputList[index].get('choices')[0])
        result, msg = has_repeated_fragment(response)
        assert result, msg

    def test_repetition_penalty_bigger_than_1(self, backend, model_case, openai_client_and_model):
        client, model_name = openai_client_and_model
        outputs = client.chat.completions.create(
            model=model_name,
            messages=[{'role': 'user', 'content': REPETITION_PROBE_PROMPT}],
            extra_body=chat_repetition_above_one_extra_body(),
            temperature=0.01,
            max_tokens=200,
        )
        output = outputs.model_dump()
        assert_chat_completions_batch_return(output, model_name)

    def test_repetition_penalty_bigger_than_1_streaming(self, backend, model_case, openai_client_and_model):
        client, model_name = openai_client_and_model
        outputs = client.chat.completions.create(
            model=model_name,
            messages=[{'role': 'user', 'content': REPETITION_PROBE_PROMPT}],
            extra_body=chat_repetition_above_one_extra_body(),
            temperature=0.01,
            max_tokens=200,
            stream=True,
        )
        outputList = [chunk.model_dump() for chunk in outputs]
        assert_chat_completions_stream_return(outputList[-1], model_name, True)
        for index in range(0, len(outputList) - 1):
            assert_chat_completions_stream_return(outputList[index], model_name)

    def test_ignore_eos(self, backend, model_case, openai_client_and_model):
        client, model_name = openai_client_and_model
        outputs = client.chat.completions.create(
            model=model_name,
            messages=[{'role': 'user', 'content': 'Hi, what is your name?'}],
            extra_body={'ignore_eos': True},
            max_tokens=100,
            temperature=0.01,
        )
        output = outputs.model_dump()
        assert_chat_completions_batch_return(output, model_name)
        completion_tokens = output.get('usage', {}).get('completion_tokens')
        assert completion_tokens == 101 or completion_tokens == 100
        assert output.get('choices')[0].get('finish_reason') == 'length'

    def test_ignore_eos_streaming(self, backend, model_case, openai_client_and_model, config):
        client, model_name = openai_client_and_model
        model_path = get_model_path_from_config(config, model_case)
        outputs = client.chat.completions.create(
            model=model_name,
            messages=[{'role': 'user', 'content': 'Hi, what is your name?'}],
            extra_body={'ignore_eos': True},
            max_tokens=100,
            temperature=0.01,
            stream=True,
        )
        outputList = [chunk.model_dump() for chunk in outputs]
        assert_chat_completions_stream_return(outputList[-1], model_name, True)
        response = ''
        for index in range(0, len(outputList) - 1):
            assert_chat_completions_stream_return(outputList[index], model_name)
            response += get_chat_delta_text(outputList[index].get('choices')[0])
        length = len(encode_text(model_path, response, add_special_tokens=False))
        assert outputList[-1].get('choices')[0].get('finish_reason') == 'length'
        assert length >= 99 and length <= 101

    def test_max_tokens_default_cap_no_overshoot_followup(self, backend, model_case, openai_client_and_model):
        client, model_name = openai_client_and_model
        prompt = 'Continue writing forever without stopping.'
        max_tokens = CAPPED_MAX_COMPLETION_TOKENS
        overshoot_slack = 1
        outputs = client.chat.completions.create(
            model=model_name,
            messages=[{'role': 'user', 'content': prompt}],
            extra_body={'ignore_eos': True},
            max_tokens=max_tokens,
            temperature=0.01,
        )
        output = outputs.model_dump()
        assert_chat_completions_batch_return(output, model_name)
        assert output.get('choices')[0].get('finish_reason') == 'length'
        completion_tokens = output.get('usage', {}).get('completion_tokens')
        assert completion_tokens is not None, 'Missing usage.completion_tokens'
        assert completion_tokens <= max_tokens + overshoot_slack, (
            f'Length cap overshoot: completion_tokens={completion_tokens} > '
            f'max_tokens={max_tokens}+{overshoot_slack}')
        followup = client.chat.completions.create(
            model=model_name,
            messages=[{'role': 'user', 'content': 'Say hi in one word.'}],
            max_tokens=8,
            temperature=0.01,
        )
        assert_chat_completions_batch_return(followup.model_dump(), model_name)

    def test_max_completion_tokens(self, backend, model_case, openai_client_and_model):
        client, model_name = openai_client_and_model
        outputs = client.chat.completions.create(
            model=model_name,
            messages=[{'role': 'user', 'content': 'Hi, pls intro yourself'}],
            max_completion_tokens=5,
            temperature=0.01,
        )
        output = outputs.model_dump()
        assert_chat_completions_batch_return(output, model_name)
        assert output.get('choices')[0].get('finish_reason') == 'length'
        assert output.get('usage').get('completion_tokens') in (5, 6)

    def test_max_completion_tokens_streaming(self, backend, model_case, openai_client_and_model, config):
        client, model_name = openai_client_and_model
        model_path = get_model_path_from_config(config, model_case)
        outputs = client.chat.completions.create(
            model=model_name,
            messages=[{'role': 'user', 'content': 'Hi, pls intro yourself'}],
            max_completion_tokens=5,
            temperature=0.01,
            stream=True,
        )
        outputList = [chunk.model_dump() for chunk in outputs]
        assert_chat_completions_stream_return(outputList[-1], model_name, True)
        response = ''
        for index in range(0, len(outputList) - 1):
            assert_chat_completions_stream_return(outputList[index], model_name)
            response += get_chat_delta_text(outputList[index].get('choices')[0])
        length = len(encode_text(model_path, response, add_special_tokens=False))
        assert outputList[-1].get('choices')[0].get('finish_reason') == 'length'
        assert length in (5, 6)

    @pytest.mark.parametrize(
        'invalid_payload',
        [
            pytest.param(fields, id=label)
            for fields, label in chat_invalid_max_output_fields()
        ],
    )
    def test_rejects_invalid_request_parameters(
            self, backend, model_case, openai_client_and_model, invalid_payload):
        _, model_name = openai_client_and_model
        resp = requests.post(
            _CHAT_COMPLETIONS_URL,
            headers=gateway_auth_headers(),
            json={
                'model': model_name,
                'messages': _CHAT_MESSAGES,
                **invalid_payload,
            },
            timeout=30,
        )
        assert_openai_invalid_request_error(resp, strict_openai_shape=True)

    @pytest.mark.parametrize('stream', [False, True], ids=['batch', 'stream'])
    @pytest.mark.parametrize('extra_fields', _CHAT_INVALID_SAMPLING_CASES)
    def test_rejects_invalid_sampling_parameters(
            self, backend, model_case, openai_client_and_model, stream, extra_fields):
        _, model_name = openai_client_and_model
        body = {'model': model_name, 'messages': _CHAT_MESSAGES, **extra_fields}
        if stream:
            body['stream'] = True
        resp = requests.post(
            _CHAT_COMPLETIONS_URL,
            headers=gateway_auth_headers(),
            json=body,
            stream=stream,
            timeout=60,
        )
        assert_openai_invalid_request_error(resp, strict_openai_shape=True)

    def test_chinese_user_prompt_non_stream(self, backend, model_case, openai_client_and_model):
        client, model_name = openai_client_and_model
        outputs = client.chat.completions.create(
            model=model_name,
            messages=[{'role': 'user', 'content': USER_GREETING_CN}],
            temperature=0.01,
            extra_body=chat_thinking_off_extra_body(),
        )
        output = outputs.model_dump()
        assert_chat_completions_batch_return(output, model_name)
        text = get_chat_message_text(output['choices'][0])
        assert text.strip(), fail_msg('中文 prompt 场景下 assistant content 应非空', output)
        assert len(re.findall(r'[\u4e00-\u9fff]', text)) >= 2, fail_msg(
            '中文 prompt 非流式回复应含至少 2 个汉字',
            text[:300],
        )

    def test_stream_sse_terminates_with_done(self, backend, model_case, openai_client_and_model):
        _, model_name = openai_client_and_model
        resp = requests.post(
            _CHAT_COMPLETIONS_URL,
            headers=gateway_auth_headers(),
            json={
                'model': model_name,
                'messages': _CHAT_MESSAGES,
                'temperature': 0.01,
                'stream': True,
                **chat_max_output_kwargs(16),
                **chat_protocol_probe_create_kwargs(),
            },
            stream=True,
            timeout=60,
        )
        assert_chat_sse_stream_ends_with_done(resp)

    def test_stream_without_include_usage_omits_usage(self, backend, model_case, openai_client_and_model):
        client, model_name = openai_client_and_model
        outputs = client.chat.completions.create(
            model=model_name,
            messages=_CHAT_MESSAGES,
            temperature=0.01,
            **chat_max_output_kwargs(16),
            stream=True,
            extra_body=chat_thinking_off_extra_body(),
        )
        all_chunks = [chunk.model_dump() for chunk in outputs]
        usage = chat_stream_final_usage(all_chunks)
        assert usage is None, fail_msg(
            '默认流式不应在末 chunk 带 usage',
            all_chunks[-3:],
        )

    @pytest.mark.prefix_cache
    def test_prefix_cache_hit(self, backend, model_case, openai_client_and_model):
        del backend, model_case, openai_client_and_model
        assert_prefix_cache_hit(BASE_URL)

    def test_messages_with_system_role(self, backend, model_case, openai_client_and_model):
        del backend, model_case
        client, model_name = openai_client_and_model
        outputs = client.chat.completions.create(
            model=model_name,
            messages=[
                {'role': 'system', 'content': 'Reply briefly.'},
                {'role': 'user', 'content': 'Say hello in one word.'},
            ],
            temperature=0.01,
            extra_body=chat_thinking_off_extra_body(),
            **chat_max_output_kwargs(32),
        )
        assert_chat_completions_batch_return(outputs.model_dump(), model_name)

    def test_max_tokens_streaming_include_usage(self, backend, model_case, openai_client_and_model):
        del backend, model_case
        _, model_name = openai_client_and_model
        all_chunks = post_chat_completions_stream_chunks(
            model_name,
            {
                'messages': _CHAT_MESSAGES,
                'temperature': 0.01,
                **chat_max_output_kwargs(5),
                'stream_options': {'include_usage': True},
            },
            choices_only=False,
        )
        assert_chat_stream_include_usage(all_chunks)
        usage = chat_stream_final_usage(all_chunks)
        assert usage is not None, fail_msg('stream include_usage 时末 chunk 须带 usage', all_chunks[-3:])
        assert usage['completion_tokens'] in (5, 6), fail_msg(
            'usage.completion_tokens 须落在 max_tokens 允许区间',
            usage.get('completion_tokens'),
        )
        stream_chunks = chat_stream_payload_chunks(all_chunks)
        assert stream_chunks[-1]['choices'][0]['finish_reason'] == 'length', fail_msg(
            '达到 max_tokens 上限时 finish_reason 应为 length',
            stream_chunks[-1].get('choices'),
        )

    @pytest.mark.parametrize('message_patch', _CHAT_INVALID_MESSAGE_CASES)
    def test_rejects_invalid_messages(
            self, backend, model_case, openai_client_and_model, message_patch):
        del backend, model_case
        _, model_name = openai_client_and_model
        resp = requests.post(
            _CHAT_COMPLETIONS_URL,
            headers=gateway_auth_headers(),
            json={'model': model_name, **message_patch},
            timeout=60,
        )
        assert_openai_invalid_request_error(resp, strict_openai_shape=True)

    def test_n_multiple_completions(self, backend, model_case, openai_client_and_model):
        del backend, model_case
        client, model_name = openai_client_and_model
        n = _LMDEPLOY_CHAT_N_MULTI_SMOKE
        outputs = client.chat.completions.create(
            model=model_name,
            messages=_CHAT_MESSAGES,
            n=n,
            temperature=0.01,
            **chat_max_output_kwargs(32),
            **chat_protocol_probe_create_kwargs(),
        )
        assert_chat_completions_batch_return(
            outputs.model_dump(),
            model_name,
            expected_choice_count=n,
        )

    @pytest.mark.parametrize('stream', [False, True], ids=['batch', 'stream'])
    def test_rejects_n_over_fanout_max(
            self, backend, model_case, openai_client_and_model, stream):
        del backend, model_case
        _, model_name = openai_client_and_model
        body = {
            'model': model_name,
            'messages': _CHAT_MESSAGES,
            'n': _CHAT_N_OVER_FANOUT_MAX,
            'stream': stream,
        }
        resp = requests.post(
            _CHAT_COMPLETIONS_URL,
            headers=gateway_auth_headers(),
            json=body,
            stream=stream,
            timeout=60,
        )
        assert_openai_invalid_request_error(resp, strict_openai_shape=True)

    def test_whitelisted_frequency_and_presence_penalty(
            self, backend, model_case, openai_client_and_model):
        del backend, model_case
        client, model_name = openai_client_and_model
        outputs = client.chat.completions.create(
            model=model_name,
            messages=_CHAT_MESSAGES,
            temperature=0.01,
            frequency_penalty=0,
            presence_penalty=0,
        )
        assert_chat_completions_batch_return(outputs.model_dump(), model_name)

    def test_response_format_json_object(self, backend, model_case, openai_client_and_model):
        del backend, model_case
        client, model_name = openai_client_and_model
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[{
                    'role': 'user',
                    'content': (
                        'Return the weather of Beijing as JSON. '
                        "The response must contain a key named 'city'."
                    ),
                }],
                response_format={'type': 'json_object'},
                temperature=0,
                extra_body=chat_thinking_off_extra_body(),
                **chat_max_output_kwargs(512),
            )
        except (BadRequestError, APIStatusError) as exc:
            pytest.skip(f'server rejected response_format: {exc}')
        choice = completion.choices[0]
        assert choice.finish_reason == 'stop', fail_msg(
            'finish_reason 应为 stop', choice.finish_reason)
        parsed = json.loads(choice.message.content or '')
        assert isinstance(parsed, dict) and 'city' in parsed, fail_msg(
            'json_object 须含 city', choice.message.content[:200])

    def test_response_format_json_schema_strict(self, backend, model_case, openai_client_and_model):
        del backend, model_case
        client, model_name = openai_client_and_model
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[{
                    'role': 'user',
                    'content': 'Return the weather of Beijing as JSON following the schema.',
                }],
                response_format={
                    'type': 'json_schema',
                    'json_schema': {
                        'name': 'weather',
                        'strict': True,
                        'schema': _RESPONSE_FORMAT_WEATHER_SCHEMA,
                    },
                },
                temperature=0,
                extra_body=chat_thinking_off_extra_body(),
                **chat_max_output_kwargs(512),
            )
        except (BadRequestError, APIStatusError) as exc:
            pytest.skip(f'server rejected response_format: {exc}')
        choice = completion.choices[0]
        assert choice.finish_reason == 'stop', fail_msg(
            'finish_reason 应为 stop', choice.finish_reason)
        parsed = json.loads(choice.message.content or '')
        assert isinstance(parsed.get('city'), str), fail_msg('city 须为字符串', parsed)
        assert isinstance(parsed.get('temperature'), (int, float)), fail_msg(
            'temperature 须为数值', parsed)

    def test_top_k_minus_1_disables_top_k_filtering(
            self, backend, model_case, openai_client_and_model):
        del backend, model_case
        client, model_name = openai_client_and_model
        outputs = client.chat.completions.create(
            model=model_name,
            messages=[{'role': 'user', 'content': 'Hi, pls intro yourself'}],
            temperature=0.01,
            extra_body=engine_top_k_extra_body(-1),
            **chat_max_output_kwargs(THINKING_MAX_COMPLETION_TOKENS),
        )
        output = outputs.model_dump()
        assert_chat_completions_batch_return(output, model_name)
        assert get_chat_message_text(output['choices'][0]).strip(), fail_msg(
            'top_k=-1 应产出非空文本', output)

    def test_top_k_1_greedy(self, backend, model_case, openai_client_and_model):
        del backend, model_case
        client, model_name = openai_client_and_model

        def _greedy_text(temperature: float) -> str:
            outputs = client.chat.completions.create(
                model=model_name,
                messages=[{'role': 'user', 'content': REPETITION_PROBE_PROMPT}],
                temperature=temperature,
                extra_body=engine_top_k_extra_body(1),
                **chat_max_output_kwargs(64),
            )
            output = outputs.model_dump()
            assert_chat_completions_batch_return(output, model_name)
            return get_chat_message_text(output['choices'][0])

        text_a = _greedy_text(0)
        text_b = _greedy_text(0)
        text_t1 = _greedy_text(1)
        assert text_a.strip() and text_a == text_b == text_t1, fail_msg(
            'top_k=1 贪心输出应确定', (text_a[:80], text_b[:80], text_t1[:80]))

    def test_top_k_negative_invalid_rejected(self, backend, model_case, openai_client_and_model):
        del backend, model_case
        client, model_name = openai_client_and_model
        with pytest.raises(BadRequestError) as ei:
            client.chat.completions.create(
                model=model_name,
                messages=[{'role': 'user', 'content': 'Hi, pls intro yourself'}],
                temperature=0.01,
                extra_body=engine_top_k_extra_body(-5),
                **chat_max_output_kwargs(16),
            )
        assert ei.value.status_code == 400, fail_msg(
            '非法 top_k=-5 应 HTTP 400', ei.value.status_code)

    def test_return_token_ids_non_streaming(self, backend, model_case, openai_client_and_model):
        del backend, model_case
        client, model_name = openai_client_and_model
        completion = client.chat.completions.create(
            model=model_name,
            messages=[{'role': 'user', 'content': 'Complete with one word: ok'}],
            temperature=0,
            **chat_max_output_kwargs(8),
            extra_body={**chat_thinking_off_extra_body(), 'return_token_ids': True},
        )
        assert_chat_return_token_ids(completion.model_dump())

    @pytest.mark.logprob
    def test_logprobs_non_streaming_lmdeploy(self, backend, model_case, openai_client_and_model):
        del backend, model_case
        client, model_name = openai_client_and_model
        outputs = client.chat.completions.create(
            model=model_name,
            messages=_CHAT_MESSAGES,
            temperature=0.01,
            logprobs=True,
            top_logprobs=_CHAT_LOGPROBS_NUM,
            **chat_max_output_kwargs(16),
            extra_body=chat_thinking_off_extra_body(),
        )
        assert_chat_completions_batch_return(
            outputs.model_dump(),
            model_name,
            check_logprobs=True,
            logprobs_num=_CHAT_LOGPROBS_NUM,
        )

    @pytest.mark.logprob
    def test_logprobs_streaming_lmdeploy(self, backend, model_case, openai_client_and_model):
        del backend, model_case
        client, model_name = openai_client_and_model
        outputs = client.chat.completions.create(
            model=model_name,
            messages=_CHAT_MESSAGES,
            temperature=0.01,
            logprobs=True,
            top_logprobs=_CHAT_LOGPROBS_NUM,
            **chat_max_output_kwargs(16),
            stream=True,
            extra_body=chat_thinking_off_extra_body(),
        )
        stream_chunks = chat_stream_payload_chunks([o.model_dump() for o in outputs])
        assert_chat_completions_stream_return(
            stream_chunks[-1], model_name, True,
            check_logprobs=True, logprobs_num=_CHAT_LOGPROBS_NUM,
        )
        for index in range(0, len(stream_chunks) - 1):
            assert_chat_completions_stream_return(
                stream_chunks[index], model_name,
                check_logprobs=True, logprobs_num=_CHAT_LOGPROBS_NUM,
            )


@pytest.mark.order(8)
@pytest.mark.flaky(reruns=2)
@pytest.mark.parametrize('backend', BACKEND_LIST)
@pytest.mark.parametrize('model_case', get_restful_chat_model_list())
class TestRestfulChatMultimodal:
    """OpenAI Chat multimodal protocol only (no ``tools``).

    Tool+image → ``tool_parser``.
    """

    def test_text_model_rejects_image(self, backend, model_case, openai_client_and_model):
        del backend
        if is_mm_tool_call_capable(model_case):
            pytest.skip('VL models covered by accept-image cases')
        _, model_name = openai_client_and_model
        resp = requests.post(
            _CHAT_COMPLETIONS_URL,
            headers=gateway_auth_headers(),
            json={
                'model': model_name,
                'messages': [
                    build_multimodal_user_message(
                        'What is in this image?',
                        f'data:image/png;base64,{TINY_1X1_PNG_BASE64}',
                    ),
                ],
                'temperature': 0,
                **chat_max_output_kwargs(16),
                **chat_thinking_off_extra_body(),
            },
            timeout=60,
        )
        assert_openai_invalid_request_error(resp)

    def test_vision_resource_image_non_stream(self, backend, model_case, config, openai_client_and_model):
        del backend
        skip_unless_vlm_model(model_case)
        _, model_name = openai_client_and_model
        image_url = require_mm_image_url(config)
        resp = requests.post(
            _CHAT_COMPLETIONS_URL,
            headers=gateway_auth_headers(),
            json={
                'model': model_name,
                'messages': [
                    build_multimodal_user_message(
                        'In one word, name the animal in the image.',
                        image_url,
                    ),
                ],
                'temperature': 0.01,
                **chat_max_output_kwargs(VISION_SMOKE_MIN_COMPLETION_TOKENS),
                **chat_protocol_probe_create_kwargs(),
            },
            timeout=180,
        )
        assert resp.status_code == 200, fail_msg('Chat 多模批式应成功', resp.text[:800])
        assert_chat_vision_tiger_choice(resp.json()['choices'][0], raw=resp.text)

    def test_vision_resource_image_stream(self, backend, model_case, config, openai_client_and_model):
        del backend
        skip_unless_vlm_model(model_case)
        client, model_name = openai_client_and_model
        image_url = require_mm_image_url(config)
        outputs = client.chat.completions.create(
            model=model_name,
            messages=[
                build_multimodal_user_message(
                    'In one word, name the animal in the image.',
                    image_url,
                ),
            ],
            temperature=0.01,
            stream=True,
            **chat_max_output_kwargs(VISION_SMOKE_MIN_COMPLETION_TOKENS),
            **chat_protocol_probe_create_kwargs(),
        )
        stream_chunks = chat_stream_payload_chunks([c.model_dump() for c in outputs])
        assert stream_chunks, fail_msg('Chat 多模流式应有 payload chunk', outputs)
        assert_chat_completions_stream_return(stream_chunks[-1], model_name, True)
        text = ''.join(
            get_chat_delta_text(chunk.get('choices', [{}])[0])
            for chunk in stream_chunks
            if chunk.get('choices')
        ).strip()
        assert_vision_tiger_text(text, label='Chat 多模流式')
