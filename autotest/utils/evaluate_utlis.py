# utils/evaluate_utils.py
import os
import subprocess
import tempfile

# import time
# from pathlib import Path


def get_model_type(model_name):
    """根据模型名称确定模型类型（chat/base/vl）"""
    # 检查是否为视觉语言模型
    if any(pattern in model_name.lower() for pattern in ['vl', 'internvl', 'llava', 'vision']):
        return 'vl'

    # 检查是否为聊天模型
    if any(pattern in model_name.lower() for pattern in ['chat', 'instruct', '对话', '对话模型']):
        return 'chat'

    # 检查是否为基础模型
    if any(pattern in model_name.lower() for pattern in ['base']):
        return 'base'

    # 默认认为是聊天模型（大多数模型都是聊天模型）
    return 'chat'


# In utils/evaluate_utils.py, fix the config file selection:


def restful_test(config, run_id, prepare_environment, worker_id='gw0'):
    """RESTful API测试函数，启动LMDeploy服务并运行OpenCompass评估.

    Args:
        config: 测试配置字典，来自config.yaml
        run_id: 运行ID
        prepare_environment: 已经通过fixture准备好的环境参数
        worker_id: 工作进程ID，默认为"gw0"

    Returns:
        tuple: (success: bool, message: str)
    """

    try:
        # 从prepare_environment中获取模型信息
        model_name = prepare_environment['model']
        backend_type = prepare_environment['backend']

        # 根据模型类型确定评估配置
        model_type = get_model_type(model_name)
        print(f'Model {model_name} identified as {model_type} model')

        # 获取对应类型的评估参数
        eval_config_key = f'{model_type}_evaluation'
        eval_config = config.get(eval_config_key, config.get('evaluation', {}))
        datasets = eval_config.get('datasets', ['mmlu'])
        datasets_str = ','.join(datasets) if isinstance(datasets, list) else datasets

        # 修正配置文件选择逻辑
        if model_type == 'base':
            config_file = 'eval_config_base.py'
        else:
            config_file = 'eval_config_chat.py'

        # 获取模型路径
        model_base_path = config.get('model_path', '/nvme/qa_test_models')
        model_path = os.path.join(model_base_path, model_name)

        print(f'Starting OpenCompass evaluation for model: {model_name}')
        print(f'Model path: {model_path}')
        print(f'Backend: {backend_type}')
        print(f'Model type: {model_type}')
        print(f'Datasets: {datasets_str}')
        print(f'Config file: {config_file}')

        # 获取OpenCompass工作目录
        opencompass_dir = config.get('opencompass_dir', './evaluate')
        if not os.path.isabs(opencompass_dir):
            opencompass_dir = os.path.abspath(opencompass_dir)

        if not os.path.exists(opencompass_dir):
            return False, f'OpenCompass directory not found: {opencompass_dir}'

        print(f'OpenCompass directory: {opencompass_dir}')

        # 获取日志路径
        log_path = config.get('log_path', '/nvme/qa_test_models/autotest_model/log')
        os.makedirs(log_path, exist_ok=True)

        # 保存当前目录
        original_cwd = os.getcwd()

        # 创建临时工作目录
        with tempfile.TemporaryDirectory() as work_dir:
            try:
                # 切换到OpenCompass目录
                os.chdir(opencompass_dir)
                print(f'Changed to directory: {opencompass_dir}')

                # 设置环境变量
                env = os.environ.copy()
                env['EVAL_MODEL_NAME'] = model_name
                env['EVAL_MODEL_PATH'] = model_path
                env['EVAL_API_BASE'] = 'http://127.0.0.1:23333/v1'
                env['EVAL_BACKEND'] = backend_type
                env['EVAL_DATASETS'] = datasets_str
                env['EVAL_MODEL_TYPE'] = model_type
                env['LOG_PATH'] = log_path

                # 构建OpenCompass评估命令
                cmd = ['python', 'run.py', config_file, '-w', work_dir]

                print(f"Running command: {' '.join(cmd)}")
                print(f'Work directory: {work_dir}')

                # 执行评估
                result = subprocess.run(cmd,
                                        capture_output=True,
                                        text=True,
                                        timeout=eval_config.get('timeout', 7200),
                                        env=env)

                # 输出结果
                stdout_output = result.stdout
                stderr_output = result.stderr

                # 保存输出到日志文件
                log_file = os.path.join(log_path, f"eval_{model_name.replace('/', '_')}_{model_type}_{worker_id}.log")
                with open(log_file, 'w', encoding='utf-8') as f:
                    f.write(f'Model: {model_name}\n')
                    f.write(f'Model type: {model_type}\n')
                    f.write(f'Config file: {config_file}\n')
                    f.write(f'Datasets: {datasets_str}\n')
                    f.write(f'Backend: {backend_type}\n')
                    f.write(f'STDOUT:\n{stdout_output}\n')
                    if stderr_output:
                        f.write(f'STDERR:\n{stderr_output}\n')
                    f.write(f'Return code: {result.returncode}\n')

                print(f'STDOUT:\n{stdout_output}')
                if stderr_output:
                    print(f'STDERR:\n{stderr_output}')
                print(f'Return code: {result.returncode}')

                # 判断是否成功
                if result.returncode == 0:
                    return True, f'Evaluation completed successfully for {model_name} ({model_type})'
                else:
                    error_msg = (f'Evaluation failed for {model_name} ({model_type}) '
                                 f'with return code {result.returncode}')
                    if stderr_output:
                        error_msg += f'\nSTDERR: {stderr_output}'
                    return False, error_msg

            finally:
                # 恢复原始目录
                os.chdir(original_cwd)
                print(f'Returned to directory: {original_cwd}')

    except subprocess.TimeoutExpired:
        return False, f"Evaluation timed out for {model_name} after {eval_config.get('timeout', 7200)} seconds"
    except Exception as e:
        return False, f'Error during evaluation for {model_name}: {str(e)}'
