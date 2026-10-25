import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from benchmark import TOOLS, official_module, score, scorers


def test_official_ifeval_strict_and_loose_are_used():
    ifeval, mmmu = scorers()
    example = dict(key=1, prompt='End the reply with FINISH.',
                   instruction_id_list=['startend:end_checker'], kwargs=[dict(end_phrase='FINISH')])
    row = dict(id='fixture/1', task='ifeval', example=example)
    assert score(row, 'Done. FINISH', ifeval, mmmu)['correct']
    assert not score(row, 'Done.', ifeval, mmmu)['correct']
    assert score(row, 'Done. FINISH', ifeval, mmmu)['strict_instructions'] == [True]


def test_official_mmmu_extraction_and_fallback_are_reproducible():
    ifeval, mmmu = scorers()
    row = dict(id='fixture/2', task='mmmu_standard', options=str(list('abcdefghij')), gold='J')
    assert score(row, 'Answer: J', ifeval, mmmu) == dict(correct=True, answer='J', extraction_fallback=False)
    a = score(row, '', ifeval, mmmu)
    random.seed(88)
    b = score(row, '', ifeval, mmmu)
    assert a == b and a['extraction_fallback']
    expected = mmmu.mmmu_process_results(dict(id='fixture/2', options=row['options'], answer='J', response='Answer: J'))
    assert expected['pred_indexs'] == 'J' and expected['if_right']


def test_official_mmmu_preserves_shuffled_image_mapping():
    helper = official_module('official_mmmu_prompts_test', TOOLS/'MMMU/mmmu-pro/infer/infer_transformers.py')
    helper.prompt_config = dict(standard='Choose a letter.')
    row = dict(question='Which picture?', options=str(['<image 2>', '<image 1>']),
               image_1='first image', image_2='second image')
    prompt, images = helper.process_prompt(row, 'standard (10 options)')
    assert images == ['second image', 'first image']
    assert 'A. [image]\nB. [image]' in prompt


def test_prepared_ifeval_matches_google_repository_input():
    import json
    from benchmark import ROOT
    examples = [json.loads(line) for line in
                (ROOT/'assets/benchmark/ifeval/input_data_google_repository.jsonl').read_text().splitlines()]
    prepared = json.loads((ROOT/'assets/benchmark/prepared_v2/questions.json').read_text())
    actual = [r['example'] for r in prepared if r['task'] == 'ifeval']
    assert actual == examples
    # The Hub copy has a different placeholder requirement on this prompt.
    corrected = next(r for r in actual if r['key'] == 2785)
    assert 'at least 3 placeholders' in corrected['prompt']
