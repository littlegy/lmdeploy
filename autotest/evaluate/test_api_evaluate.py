import pytest
from utils.config_utils import get_evaluate_pytorch_model_list, get_evaluate_turbomind_model_list, get_workerid
from utils.evaluate_utils import restful_test
from utils.run_restful_chat import start_proxy_server, start_restful_api, stop_proxy_server, stop_restful_api

DEFAULT_PORT = 23333
PROXY_PORT = 8000

EVAL_CONFIGS = {
    'default': {
        'query_per_second': 1,
        'max_out_len': 32768,
        'batch_size': 500,
        'temperature': 0.1,
    }
}


@pytest.fixture(scope='function', autouse=True)
def prepare_environment(request, config, worker_id):
    param = request.param
    model = param['model']
    backend = param['backend']
    model_path = config.get('model_path') + '/' + model
    pid, startRes = start_restful_api(config, param, model, model_path, backend, worker_id)
    yield param
    stop_restful_api(pid, startRes, param)


@pytest.fixture(scope='function', autouse=True)
def prepare_environment_judge_evaluate(request, config, worker_id):
    param = request.param
    model = param['model']
    backend = param['backend']
    model_path = config.get('model_path') + '/' + model
    proxy_config = {
        'model': 'Qwen/Qwen2.5-32B-Instruct',
        'backend': 'turbomind',
        'tp_num': 4,
        'extra': f'--proxy-url http://0.0.0.0:{PROXY_PORT}',
        'cuda_prefix': None
    }
    proxy_pid, proxy_process = start_proxy_server(proxy_config, worker_id)
    pid, startRes = start_restful_api(proxy_config, param, model, model_path, backend, worker_id)
    yield param
    stop_restful_api(pid, startRes, param)
    stop_proxy_server(proxy_pid, proxy_process)


def get_turbomind_model_list(tp_num):
    model_list = get_evaluate_turbomind_model_list(tp_num, kvint_list=[4, 8])
    new_model_list = []
    for model in model_list:
        model['cuda_prefix'] = None
        new_model_list.append(model)
    return new_model_list


def get_pytorch_model_list(tp_num):
    model_list = get_evaluate_pytorch_model_list(tp_num, kvint_list=[4, 8])
    new_model_list = []
    for model in model_list:
        model['cuda_prefix'] = None
        new_model_list.append(model)
    return new_model_list


def run_test(config, run_id, prepare_environment, worker_id, test_type='infer', eval_config_name='default'):
    """Run test with specified evaluation configuration."""
    preset_config = EVAL_CONFIGS.get(eval_config_name, {})

    if get_workerid(worker_id) is None:
        result, msg = restful_test(config,
                                   run_id,
                                   prepare_environment,
                                   worker_id=worker_id,
                                   test_type=test_type,
                                   **preset_config)
    else:
        result, msg = restful_test(config,
                                   run_id,
                                   prepare_environment,
                                   worker_id=worker_id,
                                   port=DEFAULT_PORT + get_workerid(worker_id),
                                   test_type=test_type**preset_config)
    return result, msg


@pytest.mark.turbomind
@pytest.mark.gpu_num_1
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('prepare_environment', get_turbomind_model_list(tp_num=1), indirect=True)
@pytest.mark.parametrize('eval_config', list(EVAL_CONFIGS.keys()))
def test_turbomind_restful_tp1(config, run_id, prepare_environment, worker_id, eval_config):
    result, msg = run_test(config, run_id, prepare_environment, worker_id, eval_config)
    assert result, msg


@pytest.mark.turbomind
@pytest.mark.gpu_num_2
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('prepare_environment', get_turbomind_model_list(tp_num=2), indirect=True)
@pytest.mark.parametrize('eval_config', list(EVAL_CONFIGS.keys()))
def test_turbomind_restful_tp2(config, run_id, prepare_environment, worker_id, eval_config):
    result, msg = run_test(config, run_id, prepare_environment, worker_id, eval_config)
    assert result, msg


@pytest.mark.turbomind
@pytest.mark.gpu_num_4
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('prepare_environment', get_turbomind_model_list(tp_num=4), indirect=True)
@pytest.mark.parametrize('eval_config', list(EVAL_CONFIGS.keys()))
def test_turbomind_restful_tp4(config, run_id, prepare_environment, worker_id, eval_config):
    result, msg = run_test(config, run_id, prepare_environment, worker_id, eval_config)
    assert result, msg


@pytest.mark.turbomind
@pytest.mark.gpu_num_8
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('prepare_environment', get_turbomind_model_list(tp_num=8), indirect=True)
@pytest.mark.parametrize('eval_config', list(EVAL_CONFIGS.keys()))
def test_turbomind_restful_tp8(config, run_id, prepare_environment, worker_id, eval_config):
    result, msg = run_test(config, run_id, prepare_environment, worker_id, eval_config)
    assert result, msg


@pytest.mark.pytorch
@pytest.mark.gpu_num_1
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('prepare_environment', get_pytorch_model_list(tp_num=1), indirect=True)
@pytest.mark.parametrize('eval_config', list(EVAL_CONFIGS.keys()))
def test_pytorch_restful_tp1(config, run_id, prepare_environment, worker_id, eval_config):
    result, msg = run_test(config, run_id, prepare_environment, worker_id, eval_config)
    assert result, msg


@pytest.mark.pytorch
@pytest.mark.gpu_num_2
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('prepare_environment', get_pytorch_model_list(tp_num=2), indirect=True)
@pytest.mark.parametrize('eval_config', list(EVAL_CONFIGS.keys()))
def test_pytorch_restful_tp2(config, run_id, prepare_environment, worker_id, eval_config):
    result, msg = run_test(config, run_id, prepare_environment, worker_id, eval_config)
    assert result, msg


@pytest.mark.pytorch
@pytest.mark.gpu_num_4
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('prepare_environment', get_pytorch_model_list(tp_num=4), indirect=True)
@pytest.mark.parametrize('eval_config', list(EVAL_CONFIGS.keys()))
def test_pytorch_restful_tp4(config, run_id, prepare_environment, worker_id, eval_config):
    result, msg = run_test(config, run_id, prepare_environment, worker_id, eval_config)
    assert result, msg


@pytest.mark.pytorch
@pytest.mark.gpu_num_8
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('prepare_environment', get_pytorch_model_list(tp_num=8), indirect=True)
@pytest.mark.parametrize('eval_config', list(EVAL_CONFIGS.keys()))
def test_pytorch_restful_tp8(config, run_id, prepare_environment, worker_id, eval_config):
    result, msg = run_test(config, run_id, prepare_environment, worker_id, 'infer', eval_config)
    assert result, msg


@pytest.mark.eval
@pytest.mark.pytorch
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('prepare_environment_judge_evaluate', get_pytorch_model_list(tp_num=1), indirect=True)
@pytest.mark.parametrize('eval_config', list(EVAL_CONFIGS.keys()))
def test_pytorch_judgeeval_tp1(config, run_id, prepare_environment_judge_evaluate, worker_id, eval_config):
    result, msg = run_test(config, run_id, prepare_environment_judge_evaluate, worker_id, 'eval', eval_config)
    assert result, msg


@pytest.mark.eval
@pytest.mark.pytorch
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('prepare_environment_judge_evaluate', get_pytorch_model_list(tp_num=2), indirect=True)
@pytest.mark.parametrize('eval_config', list(EVAL_CONFIGS.keys()))
def test_pytorch_judgeeval_tp2(config, run_id, prepare_environment_judge_evaluate, worker_id, eval_config):
    result, msg = run_test(config, run_id, prepare_environment_judge_evaluate, worker_id, 'eval', eval_config)
    assert result, msg


@pytest.mark.eval
@pytest.mark.pytorch
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('prepare_environment_judge_evaluate', get_pytorch_model_list(tp_num=4), indirect=True)
@pytest.mark.parametrize('eval_config', list(EVAL_CONFIGS.keys()))
def test_pytorch_judgeeval_tp4(config, run_id, prepare_environment_judge_evaluate, worker_id, eval_config):
    result, msg = run_test(config, run_id, prepare_environment_judge_evaluate, worker_id, 'eval', eval_config)
    assert result, msg


@pytest.mark.eval
@pytest.mark.turbomind
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('prepare_environment_judge_evaluate', get_turbomind_model_list(tp_num=1), indirect=True)
@pytest.mark.parametrize('eval_config', list(EVAL_CONFIGS.keys()))
def test_turbomind_judgeeval_tp1(config, run_id, prepare_environment_judge_evaluate, worker_id, eval_config):
    result, msg = run_test(config, run_id, prepare_environment_judge_evaluate, worker_id, 'eval', eval_config)
    assert result, msg


@pytest.mark.eval
@pytest.mark.turbomind
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('prepare_environment_judge_evaluate', get_turbomind_model_list(tp_num=2), indirect=True)
@pytest.mark.parametrize('eval_config', list(EVAL_CONFIGS.keys()))
def test_turbomind_judgeeval_tp2(config, run_id, prepare_environment_judge_evaluate, worker_id, eval_config):
    result, msg = run_test(config, run_id, prepare_environment_judge_evaluate, worker_id, 'eval', eval_config)
    assert result, msg


@pytest.mark.eval
@pytest.mark.turbomind
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('prepare_environment_judge_evaluate', get_turbomind_model_list(tp_num=4), indirect=True)
@pytest.mark.parametrize('eval_config', list(EVAL_CONFIGS.keys()))
def test_turbomind_judgeeval_tp4(config, run_id, prepare_environment_judge_evaluate, worker_id, eval_config):
    result, msg = run_test(config, run_id, prepare_environment_judge_evaluate, worker_id, 'eval', eval_config)
    assert result, msg


@pytest.mark.eval
@pytest.mark.turbomind
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('prepare_environment_judge_evaluate', get_turbomind_model_list(tp_num=8), indirect=True)
@pytest.mark.parametrize('eval_config', list(EVAL_CONFIGS.keys()))
def test_turbomind_judgeeval_tp8(config, run_id, prepare_environment_judge_evaluate, worker_id, eval_config):
    result, msg = run_test(config, run_id, prepare_environment_judge_evaluate, worker_id, 'eval', eval_config)
    assert result, msg
