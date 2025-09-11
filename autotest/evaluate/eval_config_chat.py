from mmengine.config import read_base
from opencompass.models import OpenAISDK

with read_base():
    # 导入数据集配置
    from opencompass.configs.datasets.gsm8k.gsm8k_gen_1d7fe4 import gsm8k_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.IFEval.IFEval_gen_3321a3 import ifeval_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.mmlu.mmlu_gen_4d595a import mmlu_datasets  # noqa: F401, E501
    from opencompass.configs.summarizers.groups.mmlu import mmlu_summary_groups  # noqa: F401, E501

datasets = sum([v for k, v in locals().items() if k.endswith('_datasets')], [])

# 定义模型配置
MODEL_NAME = 'Qwen2-7B-Instruct'
MODEL_PATH = '/nvme/qa_test_models/Qwen/Qwen2-7B-Instruct'
API_BASE = 'http://127.0.0.1:65525/v1'

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
        batch_size=500,
        temperature=0.1,
    )
]

summarizer = dict(
    dataset_abbrs=[
        ['mmlu', 'naive_average'],
        ['gsm8k', 'accuracy'],
        ['IFEval', 'Prompt-level-strict-accuracy'],
    ],
    summary_groups=sum([v for k, v in locals().items() if k.endswith('_summary_groups')], []),
)
