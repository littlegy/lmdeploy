import re

import pytest
import requests
from utils.anthropic_messages import (
    COMPLETIONS_PROMPT_CN,
    USER_CHINESE_STRICT_GREETING_CN,
)
from utils.assert_purpose import fail_msg
from utils.config_utils import get_restful_base_model_list
from utils.constant import BACKEND_LIST, BASE_URL
from utils.restful_return_check import (
    OPENAI_MODEL_NOT_FOUND_RE,
    aggregate_completion_stream_text,
    assert_chat_sse_stream_ends_with_done,
    assert_completions_batch_return,
    assert_completions_return_token_ids,
    assert_completions_stream_return,
    assert_openai_invalid_request_error,
    assert_usage_prefix_cache,
    completion_stream_payload_chunks,
    gateway_auth_headers,
    get_client_and_model,
    post_completions_stream_chunks,
    prefix_cache_probe_text,
    wire_checked_response_json,
)

_COMPLETIONS_URL = f'{BASE_URL}/v1/completions'
_COMPLETIONS_PROBE_PROMPT = 'Hi, pls intro yourself'

_COMPLETIONS_INVALID_PROMPT_CASES = [
    pytest.param({}, id='prompt_missing'),
    pytest.param({'prompt': ''}, id='prompt_empty'),
]

_COMPLETIONS_INVALID_SAMPLING_CASES = [
    pytest.param({'top_p': 0}, id='top_p_zero'),
    pytest.param({'top_p': 1.01}, id='top_p_above_one'),
    pytest.param({'top_p': 'test'}, id='top_p_string'),
    pytest.param({'n': 0}, id='n_zero'),
    pytest.param({'n': 'test'}, id='n_string'),
    pytest.param({'temperature': -0.01}, id='temperature_negative'),
    pytest.param({'temperature': 2.01}, id='temperature_above_two'),
    pytest.param({'temperature': 'test'}, id='temperature_string'),
]


@pytest.fixture(scope='class')
def openai_client_and_model():
    return get_client_and_model(BASE_URL)


@pytest.mark.parametrize('backend', BACKEND_LIST)
@pytest.mark.parametrize('model_case', get_restful_base_model_list())
class TestRestfulOpenAICompletions:

    def test_return(self, backend, model_case, openai_client_and_model):
        print(f'[test_return] backend={backend!r} model_case={model_case!r}')
        client, model_name = openai_client_and_model
        response = client.completions.create(
            model=model_name,
            prompt='Hi, pls intro yourself',
            max_tokens=16,
            temperature=0.01,
        )
        item = response.model_dump()
        completion_tokens = item['usage']['completion_tokens']
        assert completion_tokens > 0
        assert completion_tokens <= 17
        assert completion_tokens >= 16
        assert item.get('choices')[0].get('finish_reason') in ['length']
        print(f'[test_return] model_name={model_name!r} last_usage={item.get("usage")!r} '
              f'finish_reason={item.get("choices")[0].get("finish_reason")!r}')
        assert_completions_batch_return(item, model_name)

    def test_return_streaming(self, backend, model_case, openai_client_and_model):
        print(f'[test_return_streaming] backend={backend!r} model_case={model_case!r}')
        client, model_name = openai_client_and_model
        outputs = client.completions.create(
            model=model_name,
            prompt='Hi, pls intro yourself',
            max_tokens=16,
            stream=True,
            temperature=0.01,
        )
        outputList = [chunk.model_dump() for chunk in outputs]
        print(f'[test_return_streaming] model_name={model_name!r} stream_chunks={len(outputList)}')
        assert_completions_stream_return(outputList[-1], model_name, True)
        for index in range(0, len(outputList) - 1):
            assert_completions_stream_return(outputList[index], model_name)

    def test_max_tokens(self, backend, model_case, openai_client_and_model):
        print(f'[test_max_tokens] backend={backend!r} model_case={model_case!r}')
        client, model_name = openai_client_and_model
        response = client.completions.create(
            model=model_name,
            prompt='Hi, pls intro yourself',
            max_tokens=16,
            temperature=0.01,
        )
        item = response.model_dump()
        completion_tokens = item['usage']['completion_tokens']
        assert completion_tokens > 0
        assert completion_tokens <= 17
        assert completion_tokens >= 16
        assert item.get('choices')[0].get('finish_reason') in ['length']
        print(f'[test_max_tokens] completion_tokens={completion_tokens} '
              f'finish_reason={item.get("choices")[0].get("finish_reason")!r}')

    def test_single_stopword(self, backend, model_case, openai_client_and_model):
        print(f'[test_single_stopword] backend={backend!r} model_case={model_case!r}')
        client, model_name = openai_client_and_model
        response = client.completions.create(
            model=model_name,
            prompt='Shanghai is',
            max_tokens=200,
            stop=' Shanghai',
            temperature=0.01,
        )
        item = response.model_dump()
        assert ' Shanghai' not in item.get('choices')[0].get('text')
        assert item.get('choices')[0].get('finish_reason') in ['stop', 'length']
        print(f'[test_single_stopword] finish_reason={item.get("choices")[0].get("finish_reason")!r} '
              f'text_preview={((item.get("choices")[0].get("text")) or "")[:120]!r}')

    def test_array_stopwords(self, backend, model_case, openai_client_and_model):
        print(f'[test_array_stopwords] backend={backend!r} model_case={model_case!r}')
        client, model_name = openai_client_and_model
        response = client.completions.create(
            model=model_name,
            prompt='Shanghai is',
            max_tokens=200,
            stop=[' Shanghai', ' city', ' China'],
            temperature=0.01,
        )
        item = response.model_dump()
        assert ' Shanghai' not in item.get('choices')[0].get('text')
        assert ' city' not in item.get('choices')[0].get('text')
        assert ' China' not in item.get('choices')[0].get('text')
        assert item.get('choices')[0].get('finish_reason') in ['stop', 'length']
        print(f'[test_array_stopwords] finish_reason={item.get("choices")[0].get("finish_reason")!r} '
              f'text_preview={((item.get("choices")[0].get("text")) or "")[:120]!r}')

    def test_completions_stream(self, backend, model_case, openai_client_and_model):
        print(f'[test_completions_stream] backend={backend!r} model_case={model_case!r}')
        client, model_name = openai_client_and_model
        outputs = client.completions.create(
            model=model_name,
            prompt='Shanghai is',
            stream=True,
            temperature=0.01,
        )
        outputList = [chunk.model_dump() for chunk in outputs]
        print(f'[test_completions_stream] model_name={model_name!r} stream_chunks={len(outputList)}')
        for index in range(1, len(outputList) - 1):
            output = outputList[index]
            assert output.get('model') == model_name
            for message in output.get('choices'):
                assert message.get('index') == 0
                assert len(message.get('text')) > 0

        output_last = outputList[len(outputList) - 1]
        assert output_last.get('choices')[0].get('finish_reason') in ['stop', 'length']
        print(f'[test_completions_stream] last_finish_reason={output_last.get("choices")[0].get("finish_reason")!r}')

    def test_completions_stream_stopword(self, backend, model_case, openai_client_and_model):
        print(f'[test_completions_stream_stopword] backend={backend!r} model_case={model_case!r}')
        client, model_name = openai_client_and_model
        outputs = client.completions.create(
            model=model_name,
            prompt='Beijing is',
            stream=True,
            stop=' is',
            temperature=0.01,
        )
        outputList = [chunk.model_dump() for chunk in outputs]
        print(f'[test_completions_stream_stopword] model_name={model_name!r} stream_chunks={len(outputList)}')
        for index in range(1, len(outputList) - 2):
            output = outputList[index]
            assert output.get('model') == model_name
            assert output.get('object') == 'text_completion'
            for message in output.get('choices'):
                assert ' is' not in message.get('text')
                assert message.get('index') == 0
                assert len(message.get('text')) > 0

        output_last = outputList[len(outputList) - 1]
        assert output_last.get('choices')[0].get('text') == ''
        assert output_last.get('choices')[0].get('finish_reason') in ['stop', 'length']
        print(f'[test_completions_stream_stopword] last_finish_reason='
              f'{output_last.get("choices")[0].get("finish_reason")!r}')

    def test_completions_stream_stopwords(self, backend, model_case, openai_client_and_model):
        print(f'[test_completions_stream_stopwords] backend={backend!r} model_case={model_case!r}')
        client, model_name = openai_client_and_model
        outputs = client.completions.create(
            model=model_name,
            prompt='Beijing is',
            stream=True,
            stop=[' Beijing', ' city', ' China'],
            temperature=0.01,
        )
        outputList = [chunk.model_dump() for chunk in outputs]
        print(f'[test_completions_stream_stopwords] model_name={model_name!r} stream_chunks={len(outputList)}')
        for index in range(1, len(outputList) - 2):
            output = outputList[index]
            assert output.get('model') == model_name
            assert output.get('object') == 'text_completion'
            for message in output.get('choices'):
                assert ' Beijing' not in message.get('text')
                assert ' city' not in message.get('text')
                assert ' China' not in message.get('text')
                assert message.get('index') == 0
                assert len(message.get('text')) > 0

        output_last = outputList[len(outputList) - 1]
        assert output_last.get('choices')[0].get('text') == ''
        assert output_last.get('choices')[0].get('finish_reason') in ['stop', 'length']
        print(f'[test_completions_stream_stopwords] last_finish_reason='
              f'{output_last.get("choices")[0].get("finish_reason")!r}')

    def test_batch_prompt_order(self, backend, model_case, openai_client_and_model):
        print(f'[test_batch_prompt_order] backend={backend!r} model_case={model_case!r}')
        client, model_name = openai_client_and_model
        response = client.completions.create(
            model=model_name,
            prompt=['你好', '今天天气怎么样', '你是谁', '帮我写一首以梅花为主题的五言律诗', '5+2等于多少'],
            max_tokens=400,
            extra_body={'min_new_tokens': 50},
        )
        item = response.model_dump()
        print(f'[test_batch_prompt_order] batch_response={item!r}')
        assert '天' in item.get('choices')[1].get('text') or '雨' in item.get('choices')[1].get(
            'text') or '伞' in item.get('choices')[1].get('text'), item.get('choices')[1].get('text')
        assert '梅' in item.get('choices')[3].get('text') or '对仗' in item.get('choices')[3].get(
            'text') or '仄' in item.get('choices')[3].get('text') or '诗' in item.get('choices')[3].get(
                'text'), item.get('choices')[3].get('text')
        assert '7' in item.get('choices')[4].get('text') or '5+2' in item.get('choices')[4].get('text'), item.get(
            'choices')[4].get('text')

    @pytest.mark.parametrize(
        'invalid_payload',
        [
            pytest.param({'max_tokens': 0}, id='max_tokens_zero'),
            pytest.param({'max_tokens': -1}, id='max_tokens_negative'),
        ],
    )
    def test_rejects_invalid_request_parameters(
            self, backend, model_case, openai_client_and_model, invalid_payload):
        del backend, model_case
        _, model_name = openai_client_and_model
        resp = requests.post(
            _COMPLETIONS_URL,
            headers=gateway_auth_headers(),
            json={
                'model': model_name,
                'prompt': _COMPLETIONS_PROBE_PROMPT,
                **invalid_payload,
            },
            timeout=30,
        )
        assert_openai_invalid_request_error(resp, strict_openai_shape=True)

    @pytest.mark.parametrize('stream', [False, True], ids=['batch', 'stream'])
    @pytest.mark.parametrize('extra_fields', _COMPLETIONS_INVALID_SAMPLING_CASES)
    def test_rejects_invalid_sampling_parameters(
            self, backend, model_case, openai_client_and_model, stream, extra_fields):
        del backend, model_case
        _, model_name = openai_client_and_model
        body = {
            'model': model_name,
            'prompt': _COMPLETIONS_PROBE_PROMPT,
            **extra_fields,
            'stream': stream,
        }
        resp = requests.post(
            _COMPLETIONS_URL,
            headers=gateway_auth_headers(),
            json=body,
            stream=stream,
            timeout=60,
        )
        assert_openai_invalid_request_error(resp, strict_openai_shape=True)

    @pytest.mark.parametrize('prompt_patch', _COMPLETIONS_INVALID_PROMPT_CASES)
    def test_rejects_invalid_prompt(
            self, backend, model_case, openai_client_and_model, prompt_patch):
        del backend, model_case
        _, model_name = openai_client_and_model
        resp = requests.post(
            _COMPLETIONS_URL,
            headers=gateway_auth_headers(),
            json={'model': model_name, **prompt_patch},
            timeout=60,
        )
        assert_openai_invalid_request_error(resp, strict_openai_shape=True)

    def test_mistake_modelname_return(self, backend, model_case, openai_client_and_model):
        del backend, model_case, openai_client_and_model
        resp = requests.post(
            _COMPLETIONS_URL,
            headers=gateway_auth_headers(),
            json={
                'model': 'error',
                'prompt': _COMPLETIONS_PROBE_PROMPT,
                'temperature': 0.01,
                'max_tokens': 16,
            },
            timeout=60,
        )
        assert resp.status_code in (400, 404), fail_msg(
            '未知 model 须返回 HTTP 400 或 404',
            f'status={resp.status_code} body={resp.text[:500]!r}',
        )
        if resp.status_code == 400:
            assert_openai_invalid_request_error(resp, strict_openai_shape=True)
        assert OPENAI_MODEL_NOT_FOUND_RE.search((resp.text or '').lower()), fail_msg(
            '未知 model 错误信息须指出模型不可用',
            resp.text[:500],
        )

    def test_mistake_modelname_return_streaming(self, backend, model_case, openai_client_and_model):
        del backend, model_case, openai_client_and_model
        resp = requests.post(
            _COMPLETIONS_URL,
            headers=gateway_auth_headers(),
            json={
                'model': 'error',
                'prompt': _COMPLETIONS_PROBE_PROMPT,
                'temperature': 0.01,
                'max_tokens': 16,
                'stream': True,
            },
            stream=True,
            timeout=60,
        )
        assert resp.status_code in (400, 404), fail_msg(
            '未知 model 流式须返回 HTTP 400 或 404',
            f'status={resp.status_code} body={resp.text[:500]!r}',
        )

    def test_stream_sse_terminates_with_done(self, backend, model_case, openai_client_and_model):
        del backend, model_case
        _, model_name = openai_client_and_model
        resp = requests.post(
            _COMPLETIONS_URL,
            headers=gateway_auth_headers(),
            json={
                'model': model_name,
                'prompt': _COMPLETIONS_PROBE_PROMPT,
                'max_tokens': 8,
                'temperature': 0.01,
                'stream': True,
            },
            stream=True,
            timeout=60,
        )
        assert resp.status_code == 200, fail_msg(
            '流式请求应成功返回 HTTP 200',
            resp.text[:300],
        )
        assert_chat_sse_stream_ends_with_done(resp)

    def test_prefix_cache_hit(self, backend, model_case, openai_client_and_model):
        del backend, model_case
        _, model_name = openai_client_and_model
        body = {
            'model': model_name,
            'prompt': prefix_cache_probe_text(),
            'temperature': 0,
            'max_tokens': 64,
        }
        first = requests.post(
            _COMPLETIONS_URL, headers=gateway_auth_headers(), json=body, timeout=120,
        )
        second = requests.post(
            _COMPLETIONS_URL, headers=gateway_auth_headers(), json=body, timeout=120,
        )
        assert first.status_code == 200, fail_msg('请求应成功返回 HTTP 200', first.text[:300])
        assert second.status_code == 200, fail_msg('请求应成功返回 HTTP 200', second.text[:300])
        first_body = wire_checked_response_json(first)
        second_body = wire_checked_response_json(second)
        assert_usage_prefix_cache(
            first_body['usage'],
            second_body['usage'],
            prompt_key='prompt_tokens',
            cached_keys=('prompt_tokens_details', 'cached_tokens'),
            output_key='completion_tokens',
        )
        first_text = first_body['choices'][0]['text']
        second_text = second_body['choices'][0]['text']
        assert first_text.strip() and first_text == second_text, fail_msg(
            '前缀缓存重复请求的输出应一致',
            (first_text[:200], second_text[:200]),
        )
        assert first_body['choices'][0]['finish_reason'] == second_body['choices'][0]['finish_reason']

    def test_whitelisted_top_p_and_penalties(self, backend, model_case, openai_client_and_model):
        del backend, model_case
        client, model_name = openai_client_and_model
        response = client.completions.create(
            model=model_name,
            prompt='Reply with one word: ok',
            max_tokens=16,
            temperature=0.01,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
        )
        assert_completions_batch_return(response.model_dump(), model_name)

    def test_chinese_prompt_non_stream(self, backend, model_case, openai_client_and_model):
        del backend, model_case
        client, model_name = openai_client_and_model
        response = client.completions.create(
            model=model_name,
            prompt=COMPLETIONS_PROMPT_CN,
            max_tokens=128,
            temperature=0.01,
        )
        item = response.model_dump()
        assert_completions_batch_return(item, model_name)
        text = (item.get('choices') or [{}])[0].get('text') or ''
        assert text.strip(), fail_msg('中文 prompt 补全应非空', text[:200])
        assert len(re.findall(r'[\u4e00-\u9fff]', text)) >= 4, fail_msg(
            '中文 completions 应含汉字', text[:200])

    def test_chinese_prompt_stream(self, backend, model_case, openai_client_and_model):
        del backend, model_case
        client, model_name = openai_client_and_model
        outputs = client.completions.create(
            model=model_name,
            prompt=COMPLETIONS_PROMPT_CN,
            max_tokens=128,
            stream=True,
            temperature=0.01,
        )
        stream_chunks = completion_stream_payload_chunks([c.model_dump() for c in outputs])
        assert_completions_stream_return(stream_chunks[-1], model_name, True)
        text = aggregate_completion_stream_text(stream_chunks)
        assert text.strip() and len(re.findall(r'[\u4e00-\u9fff]', text)) >= 4, fail_msg(
            '中文流式 completions', text[:200])

    def test_chinese_strict_prompt_non_stream(self, backend, model_case, openai_client_and_model):
        del backend, model_case
        client, model_name = openai_client_and_model
        response = client.completions.create(
            model=model_name,
            prompt=USER_CHINESE_STRICT_GREETING_CN,
            max_tokens=128,
            temperature=0.01,
        )
        item = response.model_dump()
        assert_completions_batch_return(item, model_name)
        text = (item.get('choices') or [{}])[0].get('text') or ''
        assert len(re.findall(r'[\u4e00-\u9fff]', text)) >= 4, fail_msg(
            '纯中文约束', text[:200])
        assert not re.search(r'[A-Za-z]', text), fail_msg(
            '纯中文约束不应含英文字母', text[:200])

    def test_return_token_ids_non_streaming(self, backend, model_case, openai_client_and_model):
        del backend, model_case
        client, model_name = openai_client_and_model
        response = client.completions.create(
            model=model_name,
            prompt='Complete with one word: ok',
            max_tokens=8,
            temperature=0,
            extra_body={'return_token_ids': True},
        )
        assert_completions_return_token_ids(response.model_dump())

    @pytest.mark.logprob
    def test_logprobs_non_streaming(self, backend, model_case, openai_client_and_model):
        del backend, model_case
        _, model_name = openai_client_and_model
        resp = requests.post(
            _COMPLETIONS_URL,
            headers=gateway_auth_headers(),
            json={
                'model': model_name,
                'prompt': _COMPLETIONS_PROBE_PROMPT,
                'max_tokens': 16,
                'temperature': 0.01,
                'logprobs': 3,
            },
            timeout=60,
        )
        assert resp.status_code == 200, fail_msg(
            'logprobs 须 HTTP 200（需 --logprobs-mode raw_logprobs）', resp.text[:500])
        assert_completions_batch_return(
            resp.json(), model_name, check_logprobs=True, logprobs_num=3)

    @pytest.mark.logprob
    def test_logprobs_streaming(self, backend, model_case, openai_client_and_model):
        del backend, model_case
        client, model_name = openai_client_and_model
        outputs = client.completions.create(
            model=model_name,
            prompt=_COMPLETIONS_PROBE_PROMPT,
            max_tokens=16,
            temperature=0.01,
            logprobs=3,
            stream=True,
        )
        stream_chunks = completion_stream_payload_chunks([o.model_dump() for o in outputs])
        assert_completions_stream_return(
            stream_chunks[-1], model_name, True, check_logprobs=True, logprobs_num=3)
        for index in range(0, len(stream_chunks) - 1):
            assert_completions_stream_return(
                stream_chunks[index], model_name, check_logprobs=True, logprobs_num=3)

    def test_return_streaming_post_chunks(self, backend, model_case, openai_client_and_model):
        del backend, model_case
        _, model_name = openai_client_and_model
        stream_chunks = post_completions_stream_chunks(
            model_name,
            {
                'prompt': _COMPLETIONS_PROBE_PROMPT,
                'max_tokens': 16,
                'temperature': 0.01,
            },
        )
        assert_completions_stream_return(stream_chunks[-1], model_name, True)
        for index in range(0, len(stream_chunks) - 1):
            assert_completions_stream_return(stream_chunks[index], model_name)
