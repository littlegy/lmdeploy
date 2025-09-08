from mmengine.config import read_base
from opencompass.models import OpenAISDK

with read_base():
    # 导入数据集配置
    from opencompass.configs.datasets.gpqa.gpqa_openai_simple_evals_gen_5aeece import gpqa_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.IFEval.IFEval_gen_353ae7 import ifeval_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.math.math_0shot_gen_11c4b5 import math_datasets  # noqa: F401, E501

datasets = sum([v for k, v in locals().items() if k.endswith('_datasets')], [])

# 定义模型配置
MODEL_NAME = 'internlm2_5-1_8b'
MODEL_PATH = '/nvme/qa_test_models/internlm/internlm2_5-1_8b'
API_BASE = 'http://127.0.0.1:23333/v1'

# API 元模板配置
api_meta_template = dict(round=[
    dict(role='HUMAN', api_role='HUMAN'),
    dict(role='BOT', api_role='BOT', generate=True),
])

# 创建模型配置
models = [
    dict(
        type=OpenAISDK,
        abbr=f'{MODEL_NAME}-lmdeploy-api',
        openai_api_base=API_BASE,
        key='EMPTY',  # 如果不需要 API key 可以设为 EMPTY
        path=MODEL_PATH,
        meta_template=api_meta_template,
        max_out_len=2048,
        batch_size=1000,
        run_cfg=dict(num_gpus=1, communicator='native'),
        temperature=0.1,
    )
]

summarizer = dict(
    dataset_abbrs=[
        ['GPQA_diamond', 'accuracy'],
        ['math', 'accuracy'],
        ['IFEval', 'Prompt-level-strict-accuracy'],
    ],
    summary_groups=sum([v for k, v in locals().items() if k.endswith('_summary_groups')], []),
)
