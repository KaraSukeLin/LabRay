"""Paired question-level intervals for accuracy and answer changes."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, bootstrap

from analyze import export
from fidelity import write_json

METRICS = ['accuracy_delta', 'harmful_flip', 'recovery_flip', 'answer_flip']


def population_intervals(rows, manifest, task, confidences=(.95,), resamples=19999):
    """Reuse the cost estimator's actual sampling design for paired quality effects."""
    from benchmark_budget import stratified_projection, TASK_COUNTS
    records = [{r['id']: dict(value=r[metric]) for r in rows} for metric in METRICS]
    point, draws = stratified_projection(records, manifest, task, resamples,
                                         value=lambda row: row['value'])
    population = TASK_COUNTS[task]
    limits = {c: np.percentile(draws/population, [(1-c)*50, (1+c)*50], axis=-1)
              for c in confidences}
    return point/population, limits


def paired_accuracy_test(rows, family_size):
    """Exact McNemar sensitivity check on observed paired cases, without weights."""
    harmful = sum(r['harmful_flip'] for r in rows)
    recovery = sum(r['recovery_flip'] for r in rows)
    discordant = harmful+recovery
    p = binomtest(recovery, discordant, .5).pvalue if discordant else 1.
    return dict(harmful_count=harmful, recovery_count=recovery, discordant=discordant,
                exact_mcnemar_p=float(p), bonferroni_p=min(1., float(p)*family_size))


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


def intervals(rows, confidences=(.95,), resamples=19999):
    strata = sorted({r['stratum'] for r in rows})
    blocks = tuple(np.array([[r[k] for k in METRICS] for r in rows if r['stratum'] == s],
                           dtype=float).T for s in strata)
    count = len(rows)
    def statistic(*samples, axis=-1):
        return sum(b.sum(axis=axis) for b in samples) / count
    point = statistic(*blocks)
    if any(b.shape[-1] < 2 for b in blocks):
        return point, dict.fromkeys(confidences)  # Insufficient within-stratum observations.
    limits, fit = {}, None
    for confidence in confidences:
        # Official SciPy reuse recipe: change confidence without drawing again.
        fit = bootstrap(blocks, statistic, axis=-1, vectorized=True, paired=False,
            n_resamples=resamples if fit is None else 0, bootstrap_result=fit,
            batch=256, method='percentile', confidence_level=confidence,
            rng=np.random.default_rng(20261016))
        limits[confidence] = fit.confidence_interval
    return point, limits


def compare(args):
    started = time.perf_counter()
    config_path = getattr(args, 'config', Path('assets/benchmark_config.json'))
    config = json.loads(config_path.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    peers = [args.runs / name for name in config['models'][0 if args.reference_runs else 1:]]
    first = json.loads((peers[0]/'run.json').read_text())
    tasks = first['tasks']
    # All four metrics for every planned candidate and task form one family.
    comparisons = len(peers) * len(tasks) * len(METRICS)
    family_size = args.family_size or comparisons
    assert family_size >= comparisons
    manifest_path = getattr(args, 'manifest', None)
    manifest = json.loads(manifest_path.read_text()) if manifest_path else None
    cap = getattr(args, 'mmmu_cap', 1024)
    summary, weighted_summary, exact_tests = [], [], []
    def records_for(path):
        records = [json.loads(s) for s in (path/'questions.jsonl').read_text().splitlines()]
        if cap > 1024:
            from benchmark_budget import capped_records
            extra = [json.loads(s) for s in (path/'extensions.jsonl').read_text().splitlines()]
            records = capped_records(records, extra, cap)
        return records
    for peer in peers:
        baseline = args.reference_runs / peer.name if args.reference_runs else args.runs / config['baseline']
        reference = records_for(baseline)
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
        candidate = records_for(peer)
        paired = paired_rows(reference, candidate)
        # export replaces the final suffix; keep dots in checkpoint names intact.
        export(args.output / (peer.name+'_questions.table'), paired)
        for task in tasks:
            subset = [r for r in paired if r['task'] == task]
            reference_correct = sum(r['correct'] for r in reference if r['task'] == task)
            candidate_correct = sum(r['correct'] for r in candidate if r['task'] == task)
            exact_tests.append(dict(model=peer.name, reference=baseline.name, task=task,
                mmmu_cap=cap, questions=len(subset), **paired_accuracy_test(subset, family_size)))
            point, limits = intervals(subset, confidences=[.95, 1-.05/family_size], resamples=args.resamples)
            for confidence, ci in limits.items():
                for index, metric in enumerate(METRICS):
                    summary.append(dict(model=peer.name, reference=baseline.name, task=task, metric=metric,
                        estimate=float(point[index]), low=float(ci.low[index]) if ci else '',
                        high=float(ci.high[index]) if ci else '', confidence=confidence,
                        questions=len(subset), baseline_correct=reference_correct,
                        candidate_correct=candidate_correct,
                        baseline_accuracy=reference_correct/len(subset),
                        candidate_accuracy=candidate_correct/len(subset), smoke=metadata['smoke']))
            if manifest:
                from benchmark_budget import TASK_COUNTS
                weights = {key: value['weight'] for key, value in manifest['selection'].items()}
                population = TASK_COUNTS[task]
                point, limits = population_intervals(subset, manifest, task,
                    confidences=[.95, 1-.05/family_size], resamples=args.resamples)
                for confidence, bounds in limits.items():
                    for index, metric in enumerate(METRICS):
                        weighted_summary.append(dict(model=peer.name, reference=baseline.name, task=task,
                            metric=metric, estimate=float(point[index]), low=float(bounds[0, index]),
                            high=float(bounds[1, index]), confidence=confidence, questions=len(subset),
                            population_questions=population,
                            baseline_accuracy=sum(weights[r['id']]*r['correct'] for r in reference if r['task']==task)/population,
                            candidate_accuracy=sum(weights[r['id']]*r['correct'] for r in candidate if r['task']==task)/population))
    export(args.output/'paired_comparison', summary)
    export(args.output/'accuracy_tests', exact_tests)
    if manifest:
        export(args.output/'weighted_paired_comparison', weighted_summary)
    write_json(args.output/'statistics.json', dict(method='SciPy percentile paired question bootstrap',
        strata='task separately, fixed subject counts within MMMU-Pro', resamples=args.resamples,
        seed=20261016, family_size=family_size, family_confidence=1-.05/family_size,
        comparison='MTP On minus Off for the same model' if args.reference_runs else 'candidate minus BF16 within one decoding mode',
        harmful_flip='BF16 correct and candidate incorrect, divided by all questions',
        recovery_flip='BF16 incorrect and candidate correct, divided by all questions',
        answer_flip='parsed option differs for MMMU; exact stripped response differs for IFEval',
        extraction='official MMMU random fallback, identically seeded by question ID',
        mmmu_cap=cap, sampling_manifest=str(manifest_path) if manifest_path else None,
        analysis_config=str(config_path),
        exact_test='Unweighted observed-case exact McNemar; Bonferroni uses the same declared family',
        population_intervals='Same conditional IFEval / subject-stratified design as runtime projection' if manifest else None,
        interval_reuse='Official SciPy bootstrap_result with zero extra resamples for the second confidence level',
        smoke=metadata['smoke'], analysis_seconds=time.perf_counter()-started))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--config', type=Path, default=Path('assets/benchmark_config.json'),
                   help='Model list and BF16 reference; use the selected subset configuration')
    p.add_argument('--resamples', type=int, default=19999)
    p.add_argument('--reference-runs', type=Path, help='MTP Off runs, paired with the same models in --runs (MTP On)')
    p.add_argument('--family-size', type=int, help='Predeclared joint family size; use 300 for both modes and all On/Off comparisons')
    p.add_argument('--manifest', type=Path, help='Pilot manifest for population-weighted paired intervals')
    p.add_argument('--mmmu-cap', type=int, choices=[1024, 4096], default=1024,
                   help='Use saved longer responses for the originally truncated MMMU cases')
    compare(p.parse_args())
