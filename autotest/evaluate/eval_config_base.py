# eval_config.py
import os
import sys

from mmengine.config import read_base
from opencompass.models import OpenAISDK

# from copy import deepcopy

# Read from environment variables with defaults
MODEL_NAME = os.getenv('EVAL_MODEL_NAME', 'internlm2-chat-7b')
MODEL_PATH = os.getenv('EVAL_MODEL_PATH', 'internlm2-chat-7b')
API_BASE = os.getenv('EVAL_API_BASE', 'http://127.0.0.1:23333/v1')
BACKEND_TYPE = os.getenv('EVAL_BACKEND', 'turbomind')
DATASET_NAMES = os.getenv('EVAL_DATASETS', 'mmlu').split(',')

# Process command line arguments if provided (they take precedence over env vars)
for i, arg in enumerate(sys.argv):
    if arg == '--model-name' and i + 1 < len(sys.argv):
        MODEL_NAME = sys.argv[i + 1]
    elif arg == '--model-path' and i + 1 < len(sys.argv):
        MODEL_PATH = sys.argv[i + 1]
    elif arg == '--api-base' and i + 1 < len(sys.argv):
        API_BASE = sys.argv[i + 1]
    elif arg == '--backend' and i + 1 < len(sys.argv):
        BACKEND_TYPE = sys.argv[i + 1]
    elif arg == '--datasets' and i + 1 < len(sys.argv):
        DATASET_NAMES = sys.argv[i + 1].split(',')

# Dynamically import datasets based on provided names
datasets = []

with read_base():
    # Always import the summarizer groups
    # from opencompass.configs.summarizers.groups.cmmlu import cmmlu_summary_groups
    # from opencompass.configs.summarizers.groups.GaokaoBench import GaokaoBench_summary_groups
    # from opencompass.configs.summarizers.groups.mathbench_v1_2024 import mathbench_2024_summary_groups
    # from opencompass.configs.summarizers.groups.mmlu import mmlu_summary_groups
    # from opencompass.configs.summarizers.groups.mmlu_pro import mmlu_pro_summary_groups

    # Import datasets based on DATASET_NAMES
    for dataset_name in DATASET_NAMES:
        dataset_name = dataset_name.strip().lower()
        try:
            if dataset_name == 'mmlu':
                from opencompass.configs.datasets.mmlu.mmlu_gen_a484b3 import mmlu_datasets
                datasets.extend(mmlu_datasets)
            elif dataset_name == 'ceval':
                from opencompass.configs.datasets.ceval.ceval_gen_5f30c7 import ceval_datasets
                datasets.extend(ceval_datasets)
            elif dataset_name == 'gsm8k':
                from opencompass.configs.datasets.gsm8k.gsm8k_gen_1d7fe4 import gsm8k_datasets
                datasets.extend(gsm8k_datasets)
            elif dataset_name == 'humaneval':
                from opencompass.configs.datasets.humaneval.humaneval_gen_6f1494 import humaneval_datasets
                datasets.extend(humaneval_datasets)
            elif dataset_name == 'cmmlu':
                from opencompass.configs.datasets.cmmlu.cmmlu_gen_041cbf import cmmlu_datasets
                datasets.extend(cmmlu_datasets)
            elif dataset_name == 'mathbench':
                from opencompass.configs.datasets.MathBench.mathbench_2024_few_shot_mixed_4a3fd4 import \
                    mathbench_datasets
                datasets.extend(mathbench_datasets)
            elif dataset_name == 'mmlu_pro':
                from opencompass.configs.datasets.mmlu_pro.mmlu_pro_few_shot_gen_bfaf90 import mmlu_pro_datasets
                datasets.extend(mmlu_pro_datasets)
            elif dataset_name == 'arc_c':
                from opencompass.configs.datasets.ARC_c.ARC_c_few_shot_ppl import ARC_c_datasets
                datasets.extend(ARC_c_datasets)
            elif dataset_name == 'bbh':
                from opencompass.configs.datasets.bbh.bbh_gen_98fba6 import bbh_datasets
                datasets.extend(bbh_datasets)
            elif dataset_name == 'race':
                from opencompass.configs.datasets.race.race_few_shot_ppl import race_datasets
                datasets.extend([race_datasets[1]])  # Only high school level
            elif dataset_name == 'hellaswag':
                from opencompass.configs.datasets.hellaswag.hellaswag_10shot_ppl_59c85e import hellaswag_datasets
                datasets.extend(hellaswag_datasets)
            elif dataset_name == 'winogrande':
                from opencompass.configs.datasets.winogrande.winogrande_5shot_ll_252f01 import winogrande_datasets
                datasets.extend(winogrande_datasets)
            elif dataset_name == 'boolq':
                from opencompass.configs.datasets.SuperGLUE_BoolQ.SuperGLUE_BoolQ_few_shot_ppl import BoolQ_datasets
                datasets.extend(BoolQ_datasets)
            elif dataset_name == 'gpqa':
                from opencompass.configs.datasets.gpqa.gpqa_few_shot_ppl_4b5a83 import gpqa_datasets
                datasets.extend(gpqa_datasets)
            elif dataset_name == 'drop':
                from opencompass.configs.datasets.drop.drop_gen_a2697c import drop_datasets
                datasets.extend(drop_datasets)
            elif dataset_name == 'triviaqa':
                from opencompass.configs.datasets.triviaqa.triviaqa_wiki_1shot_gen_20a989 import triviaqa_datasets
                datasets.extend(triviaqa_datasets)
            elif dataset_name == 'nq':
                from opencompass.configs.datasets.nq.nq_open_1shot_gen_20a989 import nq_datasets
                datasets.extend(nq_datasets)
            elif dataset_name == 'wikibench':
                from opencompass.configs.datasets.wikibench.wikibench_few_shot_ppl_c23d79 import wikibench_datasets
                datasets.extend(wikibench_datasets)
            elif dataset_name == 'gaokaobench':
                from opencompass.configs.datasets.GaokaoBench.GaokaoBench_no_subjective_gen_d21e37 import \
                    GaokaoBench_datasets
                datasets.extend(GaokaoBench_datasets)
            elif dataset_name == 'theoremqa':
                from opencompass.configs.datasets.TheoremQA.TheoremQA_5shot_gen_6f0af8 import TheoremQA_datasets
                datasets.extend(TheoremQA_datasets)
            elif dataset_name == 'mbpp':
                from opencompass.configs.datasets.mbpp.sanitized_mbpp_gen_742f0c import sanitized_mbpp_datasets
                datasets.extend(sanitized_mbpp_datasets)
            elif dataset_name == 'math':
                from opencompass.configs.datasets.math.math_4shot_base_gen_43d5b6 import math_datasets
                datasets.extend(math_datasets)
            elif dataset_name == 'crowspairs':
                from opencompass.configs.datasets.crowspairs.crowspairs_ppl import crowspairs_datasets
                datasets.extend(crowspairs_datasets)
            else:
                print(f"Warning: Unknown dataset '{dataset_name}', skipping...")
        except ImportError as e:
            print(f"Warning: Failed to import dataset '{dataset_name}': {e}")
        except Exception as e:
            print(f"Warning: Error processing dataset '{dataset_name}': {e}")

# Fallback to MMLU if no datasets were successfully loaded
if not datasets:
    print('No valid datasets specified, falling back to MMLU...')
    from opencompass.configs.datasets.mmlu.mmlu_gen_a484b3 import mmlu_datasets
    datasets = mmlu_datasets

api_meta_template = dict(round=[
    dict(role='HUMAN', api_role='HUMAN'),
    dict(role='BOT', api_role='BOT', generate=True),
])

# Create model configuration
models = [
    dict(
        type=OpenAISDK,
        abbr=f'{MODEL_NAME}-lmdeploy-api',
        openai_api_base=API_BASE,
        key='EMPTY',
        path=MODEL_PATH,
        meta_template=api_meta_template,
        max_out_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1),
        temperature=0.1,
    )
]

# Configure summarizer with support for multiple dataset groups
summarizer = dict(
    dataset_abbrs=[
        # Core performance metrics
        ['mmlu', 'naive_average'],
        ['ceval', 'naive_average'],
        ['gsm8k', 'accuracy'],
        ['openai_humaneval', 'humaneval_pass@1'],
        ['cmmlu', 'naive_average'],
        ['mmlu_pro', 'naive_average'],
        ['ARC-c', 'accuracy'],
        ['BoolQ', 'accuracy'],
        ['race-high', 'accuracy'],
        ['hellaswag', 'accuracy'],
        ['winogrande', 'accuracy'],
        ['drop', 'accuracy'],
        ['bbh', 'naive_average'],
        ['math', 'accuracy'],
        ['GaokaoBench', 'weighted_average'],
        ['triviaqa_wiki_1shot', 'score'],
        ['nq_open_1shot', 'score'],
        ['wikibench-wiki-single_choice_cncircular', 'perf_4'],
        ['TheoremQA', 'score'],
        ['sanitized_mbpp', 'score'],
        ['GPQA_diamond', 'accuracy'],
        # MathBench metrics
        '###### MathBench-A: Application Part ######',
        'college',
        'high',
        'middle',
        'primary',
        'arithmetic',
        'mathbench-a (average)',
        '###### MathBench-T: Theory Part ######',
        'college_knowledge',
        'high_knowledge',
        'middle_knowledge',
        'primary_knowledge',
        'mathbench-t (average)',
        '###### Overall: Average between MathBench-A and MathBench-T ######',
        'Overall',
        '',
        # Detailed MMLU breakdowns
        'mmlu-stem',
        'mmlu-social-science',
        'mmlu-humanities',
        'mmlu-other',
        # Detailed CMMLU breakdowns
        'cmmlu-stem',
        'cmmlu-social-science',
        'cmmlu-humanities',
        'cmmlu-other',
        'cmmlu-china-specific',
        # Detailed MMLU-Pro breakdowns
        'mmlu_pro_biology',
        'mmlu_pro_business',
        'mmlu_pro_chemistry',
        'mmlu_pro_computer_science',
        'mmlu_pro_economics',
        'mmlu_pro_engineering',
        'mmlu_pro_health',
        'mmlu_pro_history',
        'mmlu_pro_law',
        'mmlu_pro_math',
        'mmlu_pro_philosophy',
        'mmlu_pro_physics',
        'mmlu_pro_psychology',
        'mmlu_pro_other',
    ],
    summary_groups=sum([v for k, v in locals().items() if k.endswith('_summary_groups')], []),
)
