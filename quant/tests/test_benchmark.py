import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from benchmark_analysis import paired_rows, intervals, population_intervals, paired_accuracy_test


def test_flip_accounting_and_paired_intervals():
    reference, candidate = [], []
    for i in range(24):
        base = dict(id=str(i), task='mmmu_standard', stratum=str(i // 12),
                    prompt_token_ids=[i], image_sha256=['same'])
        a = dict(base, correct=i % 4 in [0, 1], answer='A')
        b = dict(base, correct=i % 4 in [0, 2], answer='A' if i % 4 == 0 else 'B')
        reference.append(a)
        candidate.append(b)
    rows = paired_rows(reference, list(reversed(candidate)))
    point, limits = intervals(rows, resamples=999)
    ci = limits[.95]
    np.testing.assert_allclose(point, [0., .25, .25, .75])
    assert point[0] == point[2] - point[1]
    assert ci.low[0] < 0 < ci.high[0]
    bad = [dict(r) for r in candidate]
    bad[0]['prompt_token_ids'] = [99]
    with pytest.raises(AssertionError):
        paired_rows(reference, bad)


def test_correlated_questions_keep_the_same_paired_effect():
    rows = [dict(stratum='a' if i < 8 else 'b', accuracy_delta=.125,
                 harmful_flip=0, recovery_flip=.125, answer_flip=.5) for i in range(16)]
    point, limits = intervals(rows, resamples=499)
    ci = limits[.95]
    np.testing.assert_allclose(ci.low, point)
    np.testing.assert_allclose(ci.high, point)


def test_population_quality_uses_conditional_census_and_remaining_weights():
    rows = [dict(id=f'ifeval/{i}', accuracy_delta=d, harmful_flip=int(d<0),
                 recovery_flip=int(d>0), answer_flip=int(d!=0))
            for i, d in enumerate([1, 0, 1, -1])]
    manifest = dict(strata=[dict(task='ifeval', stratum='known', sample=2, population=2),
                           dict(task='ifeval', stratum='remaining', sample=2, population=539)],
        selection={r['id']: dict(kind='probability', sampling_stratum='known' if i<2 else 'remaining')
                   for i, r in enumerate(rows)})
    point, limits = population_intervals(rows, manifest, 'ifeval', resamples=999)
    np.testing.assert_allclose(point, np.array([1., 269.5, 270.5, 540.])/541)
    assert np.isclose(point[0], point[2]-point[1])
    assert limits[.95][0, 0] < 0 < limits[.95][1, 0]
    # The known/census contribution is fixed; only the remaining stratum varies.
    np.testing.assert_allclose(limits[.95][:, 3], 540/541)


def test_exact_paired_check_does_not_overstate_four_one_way_flips():
    rows = [dict(harmful_flip=1, recovery_flip=0) for _ in range(4)]
    result = paired_accuracy_test(rows, 160)
    assert result['discordant'] == 4
    assert result['exact_mcnemar_p'] == .125
    assert result['bonferroni_p'] == 1.
    assert paired_accuracy_test([], 160)['exact_mcnemar_p'] == 1.


def test_confidence_levels_reuse_official_bootstrap_draws(monkeypatch):
    import benchmark_analysis
    original = benchmark_analysis.bootstrap
    counts = []
    def record(*args, **kwargs):
        counts.append(kwargs['n_resamples'])
        return original(*args, **kwargs)
    monkeypatch.setattr(benchmark_analysis, 'bootstrap', record)
    rows = [dict(stratum='x', accuracy_delta=i % 2, harmful_flip=0.,
                 recovery_flip=i % 2, answer_flip=i % 3 == 0) for i in range(20)]
    point, reused = intervals(rows, confidences=[.95, .999], resamples=999)
    assert counts == [999, 0]
    for confidence in [.95, .999]:
        direct_point, direct = intervals(rows, confidences=[confidence], resamples=999)
        np.testing.assert_array_equal(point, direct_point)
        np.testing.assert_array_equal(reused[confidence].low, direct[confidence].low)
        np.testing.assert_array_equal(reused[confidence].high, direct[confidence].high)


def test_decoding_modes_are_separate_and_can_be_paired(tmp_path, monkeypatch):
    import argparse
    import json
    from benchmark_analysis import compare
    monkeypatch.chdir(tmp_path)
    (tmp_path/'assets').mkdir()
    (tmp_path/'assets/benchmark_config.json').write_text(json.dumps(dict(models=['base.1', 'quant.1'], baseline='base.1')))
    for mode in ['off', 'mtp']:
        for model in ['base.1', 'quant.1']:
            dest = tmp_path/mode/model
            dest.mkdir(parents=True)
            meta = dict(status='ok', config=dict(mtp=mode == 'mtp', temperature=.7),
                        config_sha256='same', dataset_sha256='same', tasks=['ifeval'],
                        limit_per_task=1, max_tokens_override=None, environment={}, smoke=True)
            (dest/'run.json').write_text(json.dumps(meta))
            row = dict(id='1', task='ifeval', stratum='ifeval', correct=True, answer='A',
                       prompt_token_ids=[1], image_sha256=[])
            (dest/'questions.jsonl').write_text(json.dumps(row)+'\n')
    args = argparse.Namespace(runs=tmp_path/'mtp', reference_runs=tmp_path/'off',
                              output=tmp_path/'paired', resamples=99, family_size=300)
    compare(args)
    stats = json.loads((args.output/'statistics.json').read_text())
    assert stats['family_size'] == 300 and 'On minus Off' in stats['comparison']
    assert (args.output/'base.1_questions.csv').is_file()
    assert (args.output/'quant.1_questions.csv').is_file()
    # A selected-model configuration must not require unrelated, absent workers.
    args.config = tmp_path/'subset.json'
    args.config.write_bytes((tmp_path/'assets/benchmark_config.json').read_bytes())
    (tmp_path/'assets/benchmark_config.json').write_text(json.dumps(
        dict(models=['base.1', 'quant.1', 'unselected'], baseline='base.1')))
    args.output = tmp_path/'subset_paired'
    compare(args)
    assert json.loads((args.output/'statistics.json').read_text())['analysis_config'] == str(args.config)
    # A mixed On/Off candidate set must fail rather than appear as a quantization effect.
    (tmp_path/'off/quant.1/run.json').write_bytes((tmp_path/'mtp/quant.1/run.json').read_bytes())
    args.runs, args.reference_runs, args.output = tmp_path/'off', None, tmp_path/'mixed'
    with pytest.raises(AssertionError, match='config'):
        compare(args)
