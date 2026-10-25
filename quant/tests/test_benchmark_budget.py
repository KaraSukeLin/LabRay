import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from benchmark_budget import capped_records, pipeline_seconds, schedule, stratified_projection, TASK_COUNTS

ROOT = Path(__file__).resolve().parents[1]


def test_pilot_is_an_unmodified_probability_sample_with_separate_stress():
    source = {r['id']: r for r in json.loads((ROOT/'assets/benchmark/prepared_v2/questions.json').read_text())}
    rows = json.loads((ROOT/'assets/benchmark/pilot_v1/questions.json').read_text())
    manifest = json.loads((ROOT/'assets/benchmark/pilot_v1/manifest.json').read_text())
    assert len(rows) == len({r['id'] for r in rows}) == 160
    assert all(r == source[r['id']] for r in rows)
    for task, count in TASK_COUNTS.items():
        sampled = [r for r in rows if r['task'] == task]
        assert sum(manifest['selection'][r['id']]['weight'] for r in sampled) == count
        if task != 'ifeval':
            assert len({r['stratum'] for r in sampled}) == 30
            assert {manifest['population_features'][r['id']]['difficulty'] for r in sampled} == {'Easy', 'Medium', 'Hard'}


def test_weighted_projection_and_paired_model_bootstrap_ignore_extremes():
    manifest = dict(strata=[dict(task='ifeval', stratum='a', sample=2, population=10),
                           dict(task='ifeval', stratum='b', sample=2, population=2)],
                    selection={f'ifeval/{i}': dict(kind='probability', sampling_stratum='a' if i < 2 else 'b') for i in range(4)})
    records = []
    for scale in [1., 2.]:
        records.append({f'ifeval/{i}': dict(preprocessing_seconds=0., inference_seconds=scale*v, scoring_seconds=0.)
                        for i, v in enumerate([1., 3., 10., 10.])})
        records[-1]['ifeval/extreme'] = dict(preprocessing_seconds=0., inference_seconds=10000., scoring_seconds=0.)
    point, draws = stratified_projection(records, manifest, 'ifeval', resamples=999)
    np.testing.assert_allclose(point, [40., 80.])
    np.testing.assert_allclose(draws[1], draws[0]*2)
    assert draws[0].min() >= 30 and draws[0].max() <= 50
    # The two tasks are independently sampled. Equal toy data must not create
    # artificial cross-task covariance from a repeated random stream.
    other_manifest = dict(strata=[dict(s, task='mmmu_standard') for s in manifest['strata']],
        selection={k.replace('ifeval/', 'mmmu_standard/'): v for k, v in manifest['selection'].items()})
    other_records = [{k.replace('ifeval/', 'mmmu_standard/'): v for k, v in model.items()} for model in records]
    other_point, other_draws = stratified_projection(other_records, other_manifest, 'mmmu_standard', resamples=999)
    np.testing.assert_array_equal(point, other_point)
    assert not np.array_equal(draws, other_draws)
    manifest['strata'][0]['population'] = 2
    point, draws = stratified_projection(records, manifest, 'ifeval', resamples=99)
    np.testing.assert_allclose(draws, np.repeat(point[:, None], 99, axis=1))


def test_additional_ifeval_is_disjoint_and_uses_conditional_population_weights():
    first = json.loads((ROOT/'assets/benchmark/pilot_v1/questions.json').read_text())
    extra = json.loads((ROOT/'assets/benchmark/pilot_if_supplement/questions.json').read_text())
    combined = json.loads((ROOT/'assets/benchmark/pilot_v2/questions.json').read_text())
    manifest = json.loads((ROOT/'assets/benchmark/pilot_v2/manifest.json').read_text())
    known = {r['id'] for r in first if r['task'] == 'ifeval'}
    new = {r['id'] for r in extra}
    assert len(known) == 38 and len(new) == 40 and not known.intersection(new)
    assert len(combined) == len({r['id'] for r in combined}) == 200
    assert all(manifest['selection'][k]['weight'] == 1 for k in known)
    assert all(manifest['selection'][k]['weight'] == 503/40 for k in new)
    assert sum(manifest['selection'][r['id']]['weight'] for r in combined if r['task'] == 'ifeval') == 541


def test_long_projection_replaces_only_censored_mmmu_without_double_counting():
    rows = [dict(id='a', task='ifeval', finish_reason='length', max_tokens=4096),
            dict(id='b', task='mmmu_standard', finish_reason='length', max_tokens=1024),
            dict(id='c', task='mmmu_vision', finish_reason='stop', max_tokens=1024)]
    extra = [dict(id='b', task='mmmu_standard', finish_reason='stop', max_tokens=4096),
             dict(id='c', task='mmmu_vision', finish_reason='stop', max_tokens=4096, control=True)]
    assert capped_records(rows, extra) == rows
    projected = capped_records(rows, extra, cap=4096)
    assert projected == [rows[0], extra[0], rows[2]]
    # A fresh full run already at 4096 needs no saved probe, even when censored.
    direct = dict(rows[1], max_tokens=4096)
    assert capped_records([direct], [], cap=4096) == [direct]
    assert pipeline_seconds(dict(preprocessing_seconds=2, inference_seconds=3, scoring_seconds=1)) == 6
    extra[0]['finish_reason'] = 'length'
    further = dict(id='b', task='mmmu_standard', finish_reason='stop', max_tokens=8192)
    assert capped_records(rows, extra+[further], cap=8192) == [rows[0], further, rows[2]]


def test_model_schedule_preserves_gpu_jobs_and_serial_capacity():
    costs = dict(zip('abcdefghi', [9, 8, 7, 6, 5, 4, 3, 2, 1]))
    wall, jobs = schedule(costs, 5)
    assert sum(r['duration_hours'] for r in jobs) == sum(costs.values())
    assert wall >= sum(costs.values())/5
    assert schedule(costs, 1)[0] == sum(costs.values())
    assert schedule(costs, 5, list(reversed(costs)))[1][0]['model'] == 'i'
    for gpu in range(5):
        chain = [r for r in jobs if r['gpu'] == gpu]
        assert chain[0]['start_hours'] == 0
        assert all(a['end_hours'] == b['start_hours'] for a, b in zip(chain, chain[1:]))
