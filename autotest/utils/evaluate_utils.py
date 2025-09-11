# utils/evaluate_utils.py
import os
import subprocess

from mmengine.config import Config

DEFAULT_PORT = 23333


def get_model_type(model_name):
    """根据模型名称确定模型类型（chat/base/vl）"""
    model_name_lower = model_name.lower()

    chat_patterns = [
        'chat',
        'instruct',
        'gemma',
        'llama3',
        'llama2',
        'llama',
    ]
    if any(pattern in model_name_lower for pattern in chat_patterns):
        return 'chat'
    else:
        return 'base'


def restful_test(config, run_id, prepare_environment, worker_id='gw0', port=DEFAULT_PORT):
    try:
        # 从prepare_environment中获取模型信息
        model_name = prepare_environment['model']
        backend_type = prepare_environment['backend']
        tp_num = prepare_environment.get('tp_num', 1)
        communicator = prepare_environment.get('communicator', 'native')

        # 根据模型类型确定评估配置
        model_type = get_model_type(model_name)
        print(f'Model {model_name} identified as {model_type} model')

        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)

        if model_type == 'base':
            config_file = os.path.join(parent_dir, 'evaluate/eval_config_base.py')
        else:
            config_file = os.path.join(parent_dir, 'evaluate/eval_config_chat.py')

        # 获取模型路径
        model_base_path = config.get('model_path', '/nvme/qa_test_models')
        model_path = os.path.join(model_base_path, model_name)

        print(f'Starting OpenCompass evaluation for model: {model_name}')
        print(f'Model path: {model_path}')
        print(f'Backend: {backend_type}')
        print(f'Model type: {model_type}')
        print(f'Config file: {config_file}')

        # 获取日志路径
        log_path = config.get('log_path', '/nvme/qa_test_models/autotest_model/log')
        os.makedirs(log_path, exist_ok=True)

        # 保存当前目录
        original_cwd = os.getcwd()
        work_dir = os.path.join(
            log_path, f"wk_{backend_type}_{model_name.replace('/', '_')}_{model_type}_{communicator}_{worker_id}")
        os.makedirs(work_dir, exist_ok=True)

        try:

            if not os.path.exists(config_file):
                return False, f'Config file {config_file} not found in any expected location'

            # 使用mmengine加载配置
            cfg = Config.fromfile(config_file)

            # 动态修改配置参数
            cfg.MODEL_NAME = model_name
            cfg.MODEL_PATH = model_path
            cfg.API_BASE = f'http://127.0.0.1:{port}/v1'

            # 修改模型配置
            if cfg.models and len(cfg.models) > 0:
                model_cfg = cfg.models[0]
                model_cfg['abbr'] = f'{model_name}-lmdeploy-api'
                model_cfg['openai_api_base'] = f'http://127.0.0.1:{port}/v1'
                model_cfg['path'] = model_path
                # model_cfg['run_cfg']['num_gpus'] = tp_num
                if 'backend' in model_cfg:
                    model_cfg['backend'] = backend_type

                if 'engine_config' in model_cfg and 'communicator' in model_cfg['engine_config']:
                    model_cfg['engine_config']['communicator'] = communicator

            # 生成新的临时配置文件名
            temp_config_file = f'temp_{model_name.replace("/", "_")}_{os.getpid()}.py'
            temp_config_path = os.path.join(log_path, temp_config_file)

            # 保存修改后的配置
            cfg.dump(temp_config_path)
            print(f'Modified config saved to: {temp_config_path}')

            cmd = ['opencompass', temp_config_path, '--reuse', '--max-num-workers', '16', '-w', work_dir]
            print(f"Running command: {' '.join(cmd)}")
            print(f'Work directory: {work_dir}')

            # 执行评估
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=72000)

            # 输出结果
            stdout_output = result.stdout
            stderr_output = result.stderr

            # 保存输出到日志文件
            log_file = os.path.join(
                log_path,
                f"eval_{backend_type}_{model_name.replace('/', '_')}_{model_type}_{communicator}_{worker_id}.log")
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write(f'Model: {model_name}\n')
                f.write(f'Model type: {model_type}\n')
                f.write(f'Config file: {temp_config_file}\n')
                f.write(f'Backend: {backend_type}\n')
                f.write(f'TP Num: {tp_num}\n')
                f.write(f'Command: {" ".join(cmd)}\n')
                f.write(f'Work directory: {work_dir}\n')
                f.write(f'STDOUT:\n{stdout_output}\n')
                if stderr_output:
                    f.write(f'STDERR:\n{stderr_output}\n')
                f.write(f'Return code: {result.returncode}\n')

            print(f'STDOUT:\n{stdout_output}')
            if stderr_output:
                print(f'STDERR:\n{stderr_output}')
            print(f'Return code: {result.returncode}')

            # 检查stdout中是否包含错误信息
            evaluation_failed = False
            error_keywords = ['ERROR -', 'fail, see', 'task .* fail']
            for line in stdout_output.split('\n'):
                if any(keyword in line for keyword in error_keywords):
                    evaluation_failed = True
                    break

            # 判断是否成功 (优化后的判断逻辑)
            if result.returncode == 0 and not evaluation_failed:
                return True, f'Evaluation completed successfully for {model_name} ({model_type})'
            else:
                error_msg = f'Evaluation failed for {model_name} ({model_type}) '
                if result.returncode != 0:
                    error_msg += f'with return code {result.returncode}'
                elif evaluation_failed:
                    error_msg += 'with internal errors detected in logs'

                if stderr_output:
                    error_msg += f'\nSTDERR: {stderr_output}'
                else:
                    # 提取stdout中的错误信息
                    error_lines = []
                    for line in stdout_output.split('\n'):
                        if any(keyword in line for keyword in error_keywords):
                            error_lines.append(line)
                    if error_lines:
                        error_msg += f'\nLog errors: {" | ".join(error_lines[:3])}'  # 只显示前3个错误行

                return False, error_msg

        finally:
            # 恢复原始目录
            os.chdir(original_cwd)
            print(f'Returned to directory: {original_cwd}')

    except subprocess.TimeoutExpired:
        timeout_msg = (f'Evaluation timed out for {model_name} '
                       f'after 7200 seconds')
        return False, timeout_msg
    except Exception as e:
        return False, f'Error during evaluation for {model_name}: {str(e)}'
