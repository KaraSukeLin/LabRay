"""Paired question-level intervals for accuracy and answer changes."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import bootstrap

from analyze import export
from fidelity import write_json

METRICS = ['accuracy_delta', 'harmful_flip', 'recovery_flip', 'answer_flip']


def paired_rows(reference, candidate):
    assert len(reference) == len(candidate)
    other = {r['id']: r for r in candidate}
    assert len(other) == len(candidate)
    rows = []
    for a in reference:
        b = other[a['id']]
        for key in ['task', 'stratum', 'prompt_token_ids', 'image_sha256']:
            assert a[key] == b[key], (a['id'], key)
        right_a, right_b = bool(a['correct']), bool(b['correct'])
        rows.append(dict(id=a['id'], task=a['task'], stratum=a['stratum'],
            accuracy_delta=int(right_b)-int(right_a), harmful_flip=int(right_a and not right_b),
            recovery_flip=int(not right_a and right_b), answer_flip=int(a['answer'] != b['answer'])))
    return rows


def interval(rows, confidence=.95, resamples=19999):
    strata = sorted({r['stratum'] for r in rows})
    blocks = tuple(np.array([[r[k] for k in METRICS] for r in rows if r['stratum'] == s],
                           dtype=float).T for s in strata)
    count = len(rows)
    def statistic(*samples, axis=-1):
        return sum(b.sum(axis=axis) for b in samples) / count
    point = statistic(*blocks)
    if any(b.shape[-1] < 2 for b in blocks):
        return point, None  # A smoke subset cannot support a stratified confidence interval.
    fit = bootstrap(blocks, statistic, axis=-1, vectorized=True, paired=False,
        n_resamples=resamples, batch=256, method='percentile', confidence_level=confidence,
        rng=np.random.default_rng(20261016))
    return point, fit.confidence_interval


def compare(args):
    started = time.perf_counter()
    config = json.loads(Path('assets/benchmark_config.json').read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    peers = [args.runs / name for name in config['models'][0 if args.reference_runs else 1:]]
    first = json.loads((peers[0]/'run.json').read_text())
    tasks = first['tasks']
    # All four metrics for every planned candidate and task form one family.
    comparisons = len(peers) * len(tasks) * len(METRICS)
    family_size = args.family_size or comparisons
    assert family_size >= comparisons
    summary = []
    for peer in peers:
        baseline = args.reference_runs / peer.name if args.reference_runs else args.runs / config['baseline']
        reference = [json.loads(s) for s in (baseline/'questions.jsonl').read_text().splitlines()]
        metadata = json.loads((baseline/'run.json').read_text())
        assert metadata['status'] == 'ok'
        info = json.loads((peer/'run.json').read_text())
        assert info['status'] == 'ok'
        for key in ['config_sha256', 'dataset_sha256', 'tasks', 'limit_per_task',
                    'max_tokens_override', 'environment']:
            assert info[key] == metadata[key], (peer, key)
        conditions_a, conditions_b = dict(metadata['config']), dict(info['config'])
        if args.reference_runs:
            assert conditions_a.pop('mtp') is False and conditions_b.pop('mtp') is True
        assert conditions_a == conditions_b, (peer, 'config')
        candidate = [json.loads(s) for s in (peer/'questions.jsonl').read_text().splitlines()]
        paired = paired_rows(reference, candidate)
        # export replaces the final suffix; keep dots in checkpoint names intact.
        export(args.output / (peer.name+'_questions.table'), paired)
        for task in tasks:
            subset = [r for r in paired if r['task'] == task]
            reference_correct = sum(r['correct'] for r in reference if r['task'] == task)
            candidate_correct = sum(r['correct'] for r in candidate if r['task'] == task)
            for confidence in [.95, 1-.05/family_size]:
                point, ci = interval(subset, confidence=confidence, resamples=args.resamples)
                for index, metric in enumerate(METRICS):
                    summary.append(dict(model=peer.name, reference=baseline.name, task=task, metric=metric,
                        estimate=float(point[index]), low=float(ci.low[index]) if ci else '',
                        high=float(ci.high[index]) if ci else '', confidence=confidence,
                        questions=len(subset), baseline_correct=reference_correct,
                        candidate_correct=candidate_correct,
                        baseline_accuracy=reference_correct/len(subset),
                        candidate_accuracy=candidate_correct/len(subset), smoke=metadata['smoke']))
    export(args.output/'paired_comparison', summary)
    write_json(args.output/'statistics.json', dict(method='SciPy percentile paired question bootstrap',
        strata='task separately, fixed subject counts within MMMU-Pro', resamples=args.resamples,
        seed=20261016, family_size=family_size, family_confidence=1-.05/family_size,
        comparison='MTP On minus Off for the same model' if args.reference_runs else 'candidate minus BF16 within one decoding mode',
        harmful_flip='BF16 correct and candidate incorrect, divided by all questions',
        recovery_flip='BF16 incorrect and candidate correct, divided by all questions',
        answer_flip='parsed option differs for MMMU; exact stripped response differs for IFEval',
        extraction='official MMMU random fallback, identically seeded by question ID',
        smoke=metadata['smoke'], analysis_seconds=time.perf_counter()-started))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resamples', type=int, default=19999)
    p.add_argument('--reference-runs', type=Path, help='MTP Off runs, paired with the same models in --runs (MTP On)')
    p.add_argument('--family-size', type=int, help='Predeclared joint family size; use 300 for both modes and all On/Off comparisons')
    compare(p.parse_args())
