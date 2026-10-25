"""Targeted CPU checks for the dual-mode IFEval extension."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from ifeval_modes import (CONFIG, extract_answer, pending_tasks, read_records,
                          merge_model_shards, read_model_records,
                          record_sampling_matches, render_prompt, sampling_values,
                          shard_question_ids)
from analyze_ifeval_modes import (cluster_interval, comparison_contrasts,
                                  matched_question_ids, paired_question,
                                  selected_family_size)


@pytest.fixture(scope='module')
def processor():
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained('/root/model/Qwen-Qwen3.8-27B', local_files_only=True)


def test_template_applies_xhigh_and_off_is_original_template(processor):
    config = json.loads(CONFIG.read_text())
    row = dict(prompt='A fixed question')
    on = render_prompt(processor, row, config['modes']['on'])
    off = render_prompt(processor, row, config['modes']['off'])
    default = processor.apply_chat_template([dict(role='user', content=[dict(type='text', text=row['prompt'])])],
        tokenize=False, add_generation_prompt=True, enable_thinking=True)
    old_off = processor.apply_chat_template([dict(role='user', content=[dict(type='text', text=row['prompt'])])],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    assert on == default and off == old_off
    medium = processor.apply_chat_template([dict(role='user', content=[dict(type='text', text=row['prompt'])])],
        tokenize=False, add_generation_prompt=True, enable_thinking=True, reasoning_effort='medium')
    assert processor.tokenizer.encode(on) != processor.tokenizer.encode(medium)


@pytest.mark.parametrize('mode,text,final', [
    ('on', 'private plan\n</think>\n\nFINAL', '\n\nFINAL'),
    ('on', '<think>private plan</think>FINAL', 'FINAL'),
    ('on', 'reasoning cut before final', ''),
    ('on', 'private plan</think>', ''),
    ('off', 'FINAL', 'FINAL'),
])
def test_official_parser_keeps_final_and_never_scores_truncated_reasoning(processor, mode, text, final):
    settings = json.loads(CONFIG.read_text())['modes'][mode]
    prompt = render_prompt(processor, dict(prompt='question'), settings)
    generated = SimpleNamespace(text=text, token_ids=processor.tokenizer.encode(text, add_special_tokens=False))
    result = extract_answer(processor.tokenizer, generated, processor.tokenizer.encode(prompt), settings)
    assert result['final_answer'] == final
    assert result['missing_final'] == (not bool(final))
    if mode == 'on':
        assert result['reasoning_tokens'] > 0


def test_sampling_is_explicit_and_seed_is_per_request():
    from vllm import SamplingParams
    config = json.loads(CONFIG.read_text())
    for mode in ['off', 'on']:
        for seed in config['seeds']:
            requested = sampling_values(config['modes'][mode], seed)
            params = SamplingParams(**requested)
            assert requested == {k: getattr(params, k) for k in requested}
            assert params.ignore_eos is False and not params.stop and not params.stop_token_ids


def test_record_sampling_accepts_only_recorded_cap_history():
    config = json.loads(CONFIG.read_text())
    settings = config['modes']['off']
    current = sampling_values(settings, 20261016)
    historical = dict(current, max_tokens=4096)
    assert record_sampling_matches(dict(sampling=historical, engine_sampling=historical), settings, 20261016)
    assert record_sampling_matches(dict(sampling=current, engine_sampling=current), settings, 20261016)
    unsupported = dict(current, max_tokens=32768)
    assert not record_sampling_matches(dict(sampling=unsupported, engine_sampling=unsupported), settings, 20261016)


def test_question_cluster_keeps_three_seeds_and_instruction_ratio():
    # Repeated seeds are perfectly correlated; they must not multiply sample size.
    numerator = [[0., 1.], [1., 0.], [-1., 1.], [0., 2.]]
    denominator = [[1., 2.], [1., 1.], [1., 2.], [1., 3.]]
    point, intervals = cluster_interval(numerator, denominator, 999, 7, [.95, .99])
    np.testing.assert_allclose(point, np.sum(numerator, axis=0)/np.sum(denominator, axis=0))
    # Compare to an independent enumeration of SciPy's paired resampling formula.
    from scipy.stats import bootstrap
    a, b = np.asarray(numerator).T, np.asarray(denominator).T
    direct = bootstrap((a, b), lambda x, y, axis: x.sum(axis)/y.sum(axis),
        paired=True, vectorized=True, axis=-1, n_resamples=999, method='percentile',
        confidence_level=.95, batch=128, rng=np.random.default_rng(7))
    np.testing.assert_array_equal(intervals[.95].low, direct.confidence_interval.low)
    np.testing.assert_array_equal(intervals[.95].high, direct.confidence_interval.high)
    a = dict(id='q', seed=1, correct=True, strict_instructions=[True, True], final_answer='A')
    b = dict(a, correct=False, strict_instructions=[False, True], final_answer='B')
    numer, denom = paired_question([dict(a, seed=s) for s in [1, 2, 3]],
                                  [dict(b, seed=s) for s in [1, 2, 3]])
    np.testing.assert_array_equal(numer, [-1, 1, 0, 1, -1, 1, 0])
    assert denom == [1, 1, 1, 1, 2, 2, 2]


def test_resume_preserves_complete_records_and_repairs_only_torn_tail(tmp_path):
    path = tmp_path/'questions.jsonl'
    row = dict(id='q', mode='off', seed=20261016)
    line = json.dumps(row).encode() + b'\n'
    path.write_bytes(line + b'{"id":')
    with pytest.raises(AssertionError, match='Incomplete'):
        read_records(path)
    assert read_records(path, repair=True) == [row]
    assert path.read_bytes() == line
    assert list(tmp_path.glob('*.partial-*'))[0].read_bytes() == b'{"id":'
    path.write_bytes(line + b'{"id":')
    assert read_records(path, allow_partial_tail=True) == [row]
    assert path.read_bytes() == line + b'{"id":'
    path.write_bytes(line + line)
    with pytest.raises(AssertionError, match='Duplicate'):
        read_records(path)


def test_seed_repeats_do_not_shrink_the_question_confidence_interval():
    # Two independent questions, with three identical seed results per question.
    values = [[-1.], [1.]]
    point, clustered = cluster_interval(values, [[1.], [1.]], 9999, 42)
    _, wrong = cluster_interval([[-1.]]*3 + [[1.]]*3, [[1.]]*6, 9999, 42)
    assert point[0] == 0
    np.testing.assert_array_equal([clustered[.95].low[0], clustered[.95].high[0]], [-1, 1])
    assert wrong[.95].low[0] > -1 and wrong[.95].high[0] < 1


def test_partial_analysis_uses_one_complete_question_set_for_every_cell():
    ids, models, modes, seeds = ['q1', 'q2'], ['a', 'b'], ['off', 'on'], [1, 2, 3]
    tables = {model: {(mode, seed, q): {} for mode in modes for seed in seeds for q in ids}
              for model in models}
    del tables['b']['on', 3, 'q2']
    assert matched_question_ids(ids, models, modes, seeds, tables) == ['q1']


def test_formal_comparison_set_covers_both_modes_and_fast_peers():
    config = json.loads(CONFIG.read_text())
    contrasts = comparison_contrasts(config['models'], list(config['modes']), config['baseline'])
    labels = {label for label, _, _ in contrasts}
    assert len(contrasts) == 19
    assert len(contrasts) * 7 == 133
    assert 'eNeural-Qwen3.8-27B-fast:on minus off' in labels
    assert 'Fast:on minus rdtand-Qwen3.8-27B-PrismaAQUA-5.5bit-vllm:on' in labels
    assert 'Fast:off minus ModelOpt-Qwen3.8-27B-AutoQuantize-5.5bit:off' in labels


def test_predeclared_family_covers_all_planned_comparisons():
    assert selected_family_size(None, 91, 133) == 133
    assert selected_family_size(300, 91, 133) == 300
    with pytest.raises(AssertionError):
        selected_family_size(91, 91, 133)


def test_task_seed_filter_preserves_question_order_and_completed_rows():
    experiment = dict(questions=[dict(id='q1'), dict(id='q2')], seeds=[16, 17, 18])
    completed = {('on', 17, 'q1')}
    pending = pending_tasks(experiment, completed, ['on'], {'q1', 'q2'}, {17})
    assert [(row['id'], mode, seed) for row, mode, seed in pending] == [('q2', 'on', 17)]


def test_question_shards_are_disjoint_and_cover_all_ids():
    experiment = dict(questions=[dict(id=f'q{i}') for i in range(7)])
    left, right = shard_question_ids(experiment, 0, 2), shard_question_ids(experiment, 1, 2)
    assert left.isdisjoint(right)
    assert left | right == {row['id'] for row in experiment['questions']}


def test_shard_merge_keeps_unique_rows_and_archives_source_files(tmp_path):
    import ifeval_modes
    output = tmp_path
    model = 'candidate'
    folder = output/model
    folder.mkdir()
    (output/'experiment.json').write_text(json.dumps(dict(
        question_ids=['q1', 'q2'], modes=['off', 'on'], seeds=[1])))
    rows = [dict(id='q1', mode='off', seed=1),
            dict(id='q2', mode='on', seed=1),
            dict(id='q2', mode='off', seed=1)]
    (folder/'questions.jsonl').write_text(json.dumps(rows[0])+'\n')
    (folder/'questions.shard-0-of-2.jsonl').write_text(json.dumps(rows[1])+'\n')
    (folder/'questions.shard-1-of-2.jsonl').write_text(json.dumps(rows[2])+'\n')
    result = merge_model_shards(output, model)
    merged = read_model_records(folder)
    assert result['merged'] == 2
    assert { (r['id'], r['mode'], r['seed']) for r in merged } == {
        ('q1', 'off', 1), ('q2', 'on', 1), ('q2', 'off', 1)}
    assert not list(folder.glob('questions.shard-*-of-*.jsonl'))
    assert len(list((folder/'merged-shards').glob('*.jsonl'))) == 2


def test_timing_snapshot_survives_interrupt_during_update(tmp_path, monkeypatch):
    import ifeval_modes
    path = tmp_path/'attempt.json'
    ifeval_modes.write_attempt(path, dict(status='running', new_records=1))
    original = path.read_bytes()
    writer = ifeval_modes.write_json
    def interrupt(temporary, value):
        temporary.write_text('{"status":')
        raise KeyboardInterrupt
    monkeypatch.setattr(ifeval_modes, 'write_json', interrupt)
    with pytest.raises(KeyboardInterrupt):
        ifeval_modes.write_attempt(path, dict(status='running', new_records=2))
    assert path.read_bytes() == original
    monkeypatch.setattr(ifeval_modes, 'write_json', writer)
    ifeval_modes.write_attempt(path, dict(status='ok', new_records=2))
    assert json.loads(path.read_text()) == dict(status='ok', new_records=2)


def test_shared_engine_retains_validated_settings(monkeypatch):
    import vllm
    from benchmark import create_llm
    captured = {}
    monkeypatch.setattr(vllm, 'LLM', lambda **kw: captured.update(kw))
    config = json.loads(CONFIG.read_text())
    create_llm(config['models'][0], config, 20261016)
    assert captured['speculative_config'] is None
    assert captured['kv_cache_dtype'] == 'bfloat16'
    assert captured['generation_config'] == 'vllm'
    assert captured['tensor_parallel_size'] == captured['max_num_seqs'] == 1
    assert captured['enforce_eager'] and not captured['enable_prefix_caching']
    assert captured['kernel_config'] == {'linear_backend_per_quant': {'fp8_block_w8a8': 'triton'}}
