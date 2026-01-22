import time

import pytest
from tools.common_case_config import SPECULATIVE_DECODING_RESTFUL_TEST_LLM
from utils.benchmark_utils import restful_test
from utils.config_utils import get_func_config_list
from utils.ray_distributed_utils import ray_worker_node_wait


def get_models(backend, parallel_config):
    return get_func_config_list(backend, parallel_config, func_type='benchmark')


def _run_ray_distributed_benchmark_test(config, run_config, manager=None):
    """Distributed benchmark test executor (using shared Ray cluster)"""
    assert manager is not None, 'Manager instance must be provided'

    if manager.is_master:
        try:
            result, msg = restful_test(config, run_config, worker_id='gw0')
            assert result, msg
        finally:
            manager.cleanup(force=False)
    else:
        time.sleep(10)
        ray_worker_node_wait(manager, timeout_minutes=4880)


TURBOMIND = 'turbomind'
PYTORCH = 'pytorch'


@pytest.mark.turbomind
@pytest.mark.gpu_num_1
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_models(backend=TURBOMIND, parallel_config={'tp': 1}))
def test_turbomind_throughput_tp1(config, run_config, worker_id):
    result, msg = restful_test(config, run_config, worker_id=worker_id)
    assert result, msg


@pytest.mark.turbomind
@pytest.mark.gpu_num_2
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_models(backend=TURBOMIND, parallel_config={'tp': 2}))
def test_turbomind_throughput_tp2(config, run_config, worker_id):
    result, msg = restful_test(config, run_config, worker_id=worker_id)
    assert result, msg


@pytest.mark.turbomind
@pytest.mark.gpu_num_4
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_models(backend=TURBOMIND, parallel_config={'tp': 4}))
def test_turbomind_throughput_tp4(config, run_config, worker_id):
    result, msg = restful_test(config, run_config, worker_id=worker_id)
    assert result, msg


@pytest.mark.turbomind
@pytest.mark.gpu_num_8
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_models(backend=TURBOMIND, parallel_config={'tp': 8}))
def test_turbomind_throughput_tp8(config, run_config, worker_id):
    result, msg = restful_test(config, run_config, worker_id=worker_id)
    assert result, msg


@pytest.mark.pytorch
@pytest.mark.gpu_num_1
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_models(backend=PYTORCH, parallel_config={'tp': 1}))
def test_pytorch_throughput_tp1(config, run_config, worker_id):
    result, msg = restful_test(config, run_config, worker_id=worker_id)
    assert result, msg


@pytest.mark.pytorch
@pytest.mark.gpu_num_2
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_models(backend=PYTORCH, parallel_config={'tp': 2}))
def test_pytorch_throughput_tp2(config, run_config, worker_id):
    result, msg = restful_test(config, run_config, worker_id=worker_id)
    assert result, msg


@pytest.mark.pytorch
@pytest.mark.gpu_num_4
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_models(backend=PYTORCH, parallel_config={'tp': 4}))
def test_pytorch_throughput_tp4(config, run_config, worker_id):
    result, msg = restful_test(config, run_config, worker_id=worker_id)
    assert result, msg


@pytest.mark.pytorch
@pytest.mark.gpu_num_8
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_models(backend=PYTORCH, parallel_config={'tp': 8}))
def test_pytorch_throughput_tp8(config, run_config, worker_id):
    result, msg = restful_test(config, run_config, worker_id=worker_id)
    assert result, msg


@pytest.mark.pytorch
@pytest.mark.gpu_num_16
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_models(backend=PYTORCH, parallel_config={'tp': 16}))
def test_pytorch_throughput_tp16(config, run_config, worker_id):
    result, msg = restful_test(config, run_config, worker_id=worker_id)
    assert result, msg


@pytest.mark.function
@pytest.mark.flaky(reruns=0)
@pytest.mark.gpu_num_2
@pytest.mark.parametrize('run_config', [{
    'model': 'internlm/internlm2_5-20b-chat',
    'backend': PYTORCH,
    'communicator': 'nccl',
    'quant_policy': 0,
    'parallel_config': {
        'tp': 2
    },
    'extra_params': {}
}, {
    'model': 'internlm/internlm2_5-20b-chat',
    'backend': TURBOMIND,
    'communicator': 'nccl',
    'quant_policy': 4,
    'parallel_config': {
        'tp': 2
    },
    'extra_params': {}
}, {
    'model': 'internlm/internlm2_5-20b-chat',
    'backend': TURBOMIND,
    'communicator': 'cuda_ipc',
    'quant_policy': 8,
    'parallel_config': {
        'tp': 2
    },
    'extra_params': {}
}, {
    'model': 'Qwen/Qwen3-VL-32B-Instruct',
    'backend': PYTORCH,
    'communicator': 'nccl',
    'quant_policy': 8,
    'parallel_config': {
        'tp': 2
    },
    'extra_params': {}
}])
def test_restful_func_tp2(config, run_config, worker_id):
    result, msg = restful_test(config, run_config, worker_id=worker_id, is_smoke=True)

    assert result, msg


# Speculative Decoding benchmark tests
@pytest.mark.pytorch
@pytest.mark.gpu_num_1
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config',
                         [item for item in SPECULATIVE_DECODING_RESTFUL_TEST_LLM if item['parallel_config']['tp'] == 1])
def test_speculative_decoding_throughput_tp1(config, run_config, worker_id):
    result, msg = restful_test(config, run_config, worker_id=worker_id)
    assert result, msg


@pytest.mark.pytorch
@pytest.mark.gpu_num_distributed_tp16
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize(
    'run_config', [item for item in SPECULATIVE_DECODING_RESTFUL_TEST_LLM if item['parallel_config']['tp'] == 16])
def test_speculative_decoding_throughput_tp16(shared_ray_manager, config, run_config, worker_id):
    _run_ray_distributed_benchmark_test(config=config, run_config=run_config, manager=shared_ray_manager)
