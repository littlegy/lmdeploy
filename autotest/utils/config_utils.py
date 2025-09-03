import copy
import os
from collections import OrderedDict

import yaml
from utils.get_run_config import get_tp_num

from lmdeploy.utils import is_bf16_supported


def get_turbomind_model_list(tp_num: int = None, model_type: str = 'chat_model', quant_policy: int = None):
    config = get_config()

    if quant_policy is None:
        case_list = copy.deepcopy(config.get('turbomind_' + model_type))
    else:
        case_list = [
            x for x in config.get('turbomind_' + model_type)
            if x not in config.get('turbomind_quatization').get('no_kvint' + str(quant_policy))
        ]

    quatization_case_config = config.get('turbomind_quatization')
    for key in config.get('turbomind_' + model_type):
        if key in case_list and key not in quatization_case_config.get('no_awq') and not is_quantization_model(key):
            case_list.append(key + '-inner-4bits')
    for key in quatization_case_config.get('gptq'):
        if key in case_list:
            case_list.append(key + '-inner-gptq')

    if tp_num is not None:
        return [item for item in case_list if get_tp_num(config, item) == tp_num]
    else:
        return case_list


def get_torch_model_list(tp_num: int = None,
                         model_type: str = 'chat_model',
                         exclude_dup: bool = False,
                         quant_policy: int = None):
    config = get_config()
    exclude_dup = False

    if exclude_dup:
        if quant_policy is None:
            case_list = [x for x in config.get('pytorch_' + model_type) if x in config.get('turbomind_' + model_type)]
        else:
            case_list = [
                x for x in config.get('pytorch_' + model_type)
                if x in config.get('turbomind_' +
                                   model_type) and x not in config.get('pytorch_quatization').get('no_kvint' +
                                                                                                  str(quant_policy))
            ]
    else:
        if quant_policy is None:
            case_list = config.get('pytorch_' + model_type)
        else:
            case_list = [
                x for x in config.get('pytorch_' + model_type)
                if x not in config.get('pytorch_quatization').get('no_kvint' + str(quant_policy))
            ]

    quatization_case_config = config.get('pytorch_quatization')
    for key in quatization_case_config.get('w8a8'):
        if key in case_list:
            case_list.append(key + '-inner-w8a8')
    for key in quatization_case_config.get('awq'):
        if key in case_list:
            case_list.append(key + '-inner-4bits')

    if tp_num is not None:
        return [item for item in case_list if get_tp_num(config, item) == tp_num]
    else:
        return case_list


def get_all_model_list(tp_num: int = None, quant_policy: int = None, model_type: str = 'chat_model'):

    case_list = get_turbomind_model_list(tp_num=tp_num, model_type=model_type, quant_policy=quant_policy)
    if _is_bf16_supported_by_device():
        for case in get_torch_model_list(tp_num=tp_num, quant_policy=quant_policy, model_type=model_type):
            if case not in case_list:
                case_list.append(case)
    return case_list


def get_communicator_list():
    if _is_bf16_supported_by_device():
        return ['native', 'nccl']
    return ['nccl']


def get_quantization_model_list(type):
    config = get_config()
    if type == 'awq':
        case_list = [
            x
            for x in list(OrderedDict.fromkeys(config.get('turbomind_chat_model') + config.get('turbomind_base_model')))
            if x not in config.get('turbomind_quatization').get('no_awq') and not is_quantization_model(x)
        ]
        for key in config.get('pytorch_quatization').get('awq'):
            if key not in case_list:
                case_list.append(key)
        return case_list
    if type == 'gptq':
        return config.get('turbomind_quatization').get(type)
    if type == 'w8a8':
        return config.get('pytorch_quatization').get(type)
    return []


def get_vl_model_list(tp_num: int = None, quant_policy: int = None):
    config = get_config()
    if quant_policy is None:
        case_list = copy.deepcopy(config.get('vl_model'))
    else:
        case_list = [
            x for x in config.get('vl_model')
            if (x in config.get('turbomind_chat_model') and x not in config.get('turbomind_quatization').get(
                'no_kvint' + str(quant_policy))) or (x in config.get('pytorch_chat_model') and x not in config.get(
                    'pytorch_quatization').get('no_kvint' + str(quant_policy)))
        ]

    for key in config.get('vl_model'):
        if key in config.get('turbomind_chat_model') and key not in config.get('turbomind_quatization').get(
                'no_awq') and not is_quantization_model(key) and key + '-inner-4bits' not in case_list and (
                    quant_policy is not None
                    and key not in config.get('turbomind_quatization').get('no_kvint' + str(quant_policy))):
            case_list.append(key + '-inner-4bits')
        if key in config.get('pytorch_chat_model') and key in config.get('pytorch_quatization').get(
                'awq') and not is_quantization_model(key) and key + '-inner-4bits' not in case_list and (
                    quant_policy is not None
                    and key not in config.get('pytorch_quatization').get('no_kvint' + str(quant_policy))):
            case_list.append(key + '-inner-4bits')
    if tp_num is not None:
        return [item for item in case_list if get_tp_num(config, item) == tp_num]
    else:
        return case_list


def get_cuda_prefix_by_workerid(worker_id, tp_num: int = 1):
    cuda_id = get_cuda_id_by_workerid(worker_id, tp_num)
    if cuda_id is None or 'gw' not in worker_id:
        return None
    else:
        device_type = os.environ.get('DEVICE', 'cuda')
        if device_type == 'ascend':
            return 'ASCEND_RT_VISIBLE_DEVICES=' + cuda_id
        else:
            return 'CUDA_VISIBLE_DEVICES=' + cuda_id


def get_cuda_id_by_workerid(worker_id, tp_num: int = 1):
    if worker_id is None or 'gw' not in worker_id:
        return None
    else:
        if tp_num == 1:
            return worker_id.replace('gw', '')
        elif tp_num == 2:
            cuda_num = int(worker_id.replace('gw', '')) * 2
            return ','.join([str(cuda_num), str(cuda_num + 1)])
        elif tp_num == 4:
            cuda_num = int(worker_id.replace('gw', '')) * 4
            return ','.join([str(cuda_num), str(cuda_num + 1), str(cuda_num + 2), str(cuda_num + 3)])


def get_config():
    # Determine config file based on DEVICE environment variable
    device = os.environ.get('DEVICE', '')
    if device:
        config_path = f'autotest/config-{device}.yaml'
        # Fallback to default config if device-specific config doesn't exist
        if not os.path.exists(config_path):
            config_path = 'autotest/config.yaml'
    else:
        config_path = 'autotest/config.yaml'

    with open(config_path) as f:
        config = yaml.load(f.read(), Loader=yaml.SafeLoader)
    return config


def get_benchmark_model_list(tp_num, is_longtext: bool = False, kvint_list: list = []):
    config = get_config()
    if is_longtext:
        case_list_base = [item for item in config.get('longtext_model')]
    else:
        case_list_base = config.get('benchmark_model')
    quatization_case_config = config.get('turbomind_quatization')
    pytorch_quatization_case_config = config.get('pytorch_quatization')

    case_list = copy.deepcopy(case_list_base)
    for key in case_list_base:
        if key in config.get('turbomind_chat_model') and key not in quatization_case_config.get(
                'no_awq') and not is_quantization_model(key):
            case_list.append(key + '-inner-4bits')

    for key in case_list_base:
        if key in config.get('pytorch_chat_model') and key in pytorch_quatization_case_config.get(
                'w8a8') and not is_quantization_model(key):
            case_list.append(key + '-inner-w8a8')

    model_list = [item for item in case_list if get_tp_num(config, item) == tp_num]

    result = []
    if len(model_list) > 0:
        result += [{
            'model': item,
            'backend': 'turbomind',
            'quant_policy': 0,
            'tp_num': tp_num
        } for item in model_list
                   if item.replace('-inner-4bits', '') in config.get('turbomind_chat_model') or tp_num == 4]
        result += [{
            'model': item,
            'backend': 'pytorch',
            'tp_num': tp_num
        } for item in model_list if '4bits' not in item and (
            item.replace('-inner-w8a8', '') in config.get('pytorch_chat_model') or tp_num == 4)]
        for kvint in kvint_list:
            result += [{
                'model': item,
                'backend': 'turbomind',
                'quant_policy': kvint,
                'tp_num': tp_num
            } for item in model_list if item.replace('-inner-4bits', '') in config.get('turbomind_chat_model')
                       and item.replace('-inner-4bits', '') not in quatization_case_config.get('no_kvint' + str(kvint))]
    return result


def get_chat_template(model_name):
    """根据模型名称返回对应的chat template."""
    # 模型名称到chat template的精确映射
    template_mapping = {
        # Llama系列
        'meta-llama/lama-3.2-1b-instruct': 'llama3_2',
        'meta-llama/lama-3.2-3b-instruct': 'llama3_2',
        'meta-llama/meta-llama-3-1-8b-instruct': 'llama3_1',
        'meta-llama/meta-llama-3-1-8b-instruct-awq': 'llama3_1',
        'meta-llama/meta-llama-3-1-70b-instruct': 'llama3_1',
        'meta-llama/meta-llama-3-8b-instruct': 'llama3',
        'meta-llama/lama-4-scout-17b-16e-instruct': 'llama4',
        'meta-llama/lama-3.2-11b-vision-instruct': 'llama3_2',

        # InternLM系列
        'internlm/intern-s1': 'internlm',
        'internlm/intern-s1-mini': 'internlm',
        'internlm/internlm3-8b-instruct': 'internlm3',
        'internlm/internlm3-8b-instruct-awq': 'internlm3',
        'internlm/internlm2_5-7b': 'internlm2',
        'internlm/internlm2_5-1_8b': 'internlm2',
        'internlm/internlm2_5-20b': 'internlm2',

        # Qwen系列
        'qwen/qwen3-0.6b': 'qwen3',
        'qwen/qwen3-4b': 'qwen3',
        'qwen/qwen3-8b-base': 'qwen3',
        'qwen/qwen3-32b': 'qwen3',
        'qwen/qwen3-30b-a3b': 'qwen3',
        'qwen/qwen3-30b-a3b-base': 'qwen3',
        'qwen/qwen3-235b-a22b': 'qwen3',
        'qwen/qwen2.5-0.5b-instruct': 'qwen2d5',
        'qwen/qwen2.5-7b-instruct': 'qwen2d5',
        'qwen/qwen2.5-14b': 'qwen2d5',
        'qwen/qwen2.5-32b-instruct': 'qwen2d5',
        'qwen/qwen2.5-72b-instruct': 'qwen2d5',
        'qwen/qwen2-57b-a14b-instruct-gptq-int4': 'qwen2d5',

        # Mistral系列
        'mistralai/mistral-7b-instruct-v0.3': 'mistral',
        'mistralai/mistral-nemo-instruct-2407': 'mistral',
        'mistralai/mixtral-8x7b-instruct-v0.1': 'mixtral',

        # DeepSeek系列
        'deepseek-ai/deepseek-r1-distill-llama-8b': 'deepseek-r1',
        'deepseek-ai/deepseek-r1-distill-qwen-32b': 'deepseek-r1',
        'deepseek-ai/deepseek-coder-1.3b-instruct': 'deepseek-coder',

        # 其他模型
        '01-ai/yi-vl-6b': 'yi-vl',
        'liuhaotian/llava-v1.5-13b': 'llava-v1',
        'liuhaotian/llava-v1.6-vicuna-7b': 'llava-v1',
        'codellama/codellama-7b-instruct-hf': 'codellama',
        'codellama/codellama-7b-hf': 'codellama',
        'thudm/chatglm2-6b': 'chatglm',
        'thudm/glm-4v-9b': 'glm4',
        'thudm/codegeex4-all-9b': 'codegeex4',
        'openbmb/minicpm-llama3-v-2_5': 'minicpmv-2d6',
        'openbmb/minicpm-v-2_6': 'minicpmv-2d6',
        'allenai/molmo-7b-d-0924': 'molmo',
        'openai/gpt-oss-20b': 'base',
        'openai/gpt-oss-120b': 'base',
        'google/gemma-3-12b-it': 'gemma',
        'google/gemma-2-9b-it': 'gemma',
        'google/gemma-2-27b-it': 'gemma',
        'google/gemma-7b-it': 'gemma',
        'microsoft/phi-4-mini-instruct': 'phi-4',
        'microsoft/phi-3.5-mini-instruct': 'phi-3',
        'microsoft/phi-3.5-vision-instruct': 'phi-3',
        'microsoft/phi-3-mini-4k-instruct': 'phi-3',
        'microsoft/phi-3-vision-128k-instruct': 'phi-3',
        'bigcode/starcoder2-7b': 'base'
    }

    return template_mapping.get(model_name, 'base')


def get_evaluate_model_list(tp_num, is_longtext: bool = False):
    """Get model list for evaluation tests without quantized models.

    Args:
        tp_num: Number of tensor parallelism
        is_longtext: Whether to use longtext models

    Returns:
        list: List of model configurations without quantized models
    """
    config = get_config()

    # Get base model list
    if is_longtext:
        case_list_base = [item for item in config.get('longtext_model', [])]
    else:
        case_list_base = config.get('evaluate_model', config.get('benchmark_model', []))

    # Filter models by TP number first
    model_list = [item for item in case_list_base if get_tp_num(config, item) == tp_num]

    result = []
    if len(model_list) > 0:
        # Get TurboMind model sets
        turbomind_chat_models = set(config.get('turbomind_chat_model', []))
        turbomind_base_models = set(config.get('turbomind_base_model', []))

        # Add TurboMind models with different communicators
        for item in model_list:
            if item in turbomind_chat_models or item in turbomind_base_models:
                # Determine communicators based on TP number
                communicators = ['native'] if tp_num == 1 else ['native', 'nccl']

                for communicator in communicators:
                    model_config = {
                        'model': item,
                        'backend': 'turbomind',
                        'tp_num': tp_num,
                        'communicator': communicator
                    }

                    # Add chat template for base models
                    if item in turbomind_base_models:
                        chat_template = get_chat_template(item)
                        if chat_template and chat_template != 'base':
                            model_config['extra'] = f'--chat-template {chat_template} '

                    result.append(model_config)

        # Add PyTorch models (excluding quantized models)
        pytorch_chat_models = set(config.get('pytorch_chat_model', []))
        pytorch_base_models = set(config.get('pytorch_base_model', []))
        all_pytorch_models = pytorch_chat_models.union(pytorch_base_models)

        for item in model_list:
            if item in all_pytorch_models:
                model_config = {'model': item, 'backend': 'pytorch', 'tp_num': tp_num}

                # Add chat template for PyTorch base models
                if item in pytorch_base_models:
                    chat_template = get_chat_template(item)
                    if chat_template and chat_template != 'base':
                        model_config['extra'] = f'--chat-template {chat_template} '

                result.append(model_config)

    return result


def get_workerid(worker_id):
    if worker_id is None or 'gw' not in worker_id:
        return None
    else:
        return int(worker_id.replace('gw', ''))


def is_quantization_model(name):
    return 'awq' in name.lower() or '4bits' in name.lower() or 'w4' in name.lower() or 'int4' in name.lower()


def _is_bf16_supported_by_device():
    """Check if bf16 is supported based on the current device."""
    device = os.environ.get('DEVICE', 'cuda')
    if device == 'ascend':
        # For Ascend, bf16 support check would be different
        # Placeholder implementation
        return True
    else:
        # For CUDA and default, use the existing check
        return is_bf16_supported()


def set_device_env_variable(worker_id, tp_num: int = 1):
    """Set device environment variable based on the device type."""
    device = os.environ.get('DEVICE', 'cuda')  # Default to cuda

    if device == 'ascend':
        device_id = get_cuda_id_by_workerid(worker_id, tp_num)
        if device_id is not None:
            os.environ['ASCEND_RT_VISIBLE_DEVICES'] = device_id
    else:  # Default to cuda
        cuda_id = get_cuda_id_by_workerid(worker_id, tp_num)
        if cuda_id is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = cuda_id


def unset_device_env_variable():
    device_type = os.environ.get('DEVICE', 'cuda')
    if device_type == 'ascend':
        if 'ASCEND_RT_VISIBLE_DEVICES' in os.environ:
            del os.environ['ASCEND_RT_VISIBLE_DEVICES']
    else:
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            del os.environ['CUDA_VISIBLE_DEVICES']
