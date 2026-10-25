"""CPU-only paired analysis, with the three seeds clustered inside each question."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.stats import bootstrap

from analyze import export
from fidelity import write_json
from ifeval_modes import read_records, record_key, record_sampling_matches
from ifeval_modes import sha

METRICS = ['accuracy_delta', 'harmful_flip', 'recovery_flip', 'answer_flip',
           'instruction_accuracy_delta', 'instruction_harmful_flip', 'instruction_recovery_flip']


def cluster_interval(numerator, denominator, resamples, seed, confidences=(.95,)):
    """Resample questions once, retaining all seeds and instructions of that question."""
    numerator, denominator = np.asarray(numerator).T, np.asarray(denominator).T
    def statistic(a, b, axis=-1):
        return a.sum(axis=axis) / b.sum(axis=axis)
    point = statistic(numerator, denominator)
    if numerator.shape[-1] < 2:
        return point, {c: None for c in confidences}
    fit, limits = None, {}
    for confidence in confidences:
        fit = bootstrap((numerator, denominator), statistic, paired=True, vectorized=True,
            axis=-1, n_resamples=resamples if fit is None else 0, bootstrap_result=fit,
            confidence_level=confidence, method='percentile', batch=128,
            rng=np.random.default_rng(seed))
        limits[confidence] = fit.confidence_interval
    return point, limits


def paired_question(left, right):
    """One vector per question, averaging paired seed effects before the bootstrap."""
    assert len(left) == len(right)
    values = []
    for a, b in zip(left, right):
        assert (a['id'], a['seed']) == (b['id'], b['seed'])
        x, y = np.asarray(a['strict_instructions'], dtype=bool), np.asarray(b['strict_instructions'], dtype=bool)
        assert len(x) == len(y)
        values.append([int(b['correct'])-int(a['correct']), int(a['correct'] and not b['correct']),
            int(not a['correct'] and b['correct']), int(a['final_answer'].strip() != b['final_answer'].strip()),
            int(y.sum())-int(x.sum()), int((x & ~y).sum()), int((~x & y).sum())])
    return np.mean(values, axis=0), [1, 1, 1, 1, len(x), len(x), len(x)]


def matched_question_ids(ids, models, modes, seeds, tables):
    """Only questions with every model/mode/seed result enter quality comparisons."""
    return [q for q in ids if all((mode, seed, q) in tables[model] for model in models
                                  for mode in modes for seed in seeds)]


def comparison_contrasts(models, modes, baseline):
    contrasts = []
    if baseline in models:
        contrasts += [(f'{model}:{mode} minus BF16:{mode}', (baseline, mode), (model, mode))
                      for mode in modes for model in models if model != baseline]
    if set(modes) == {'off', 'on'}:
        contrasts += [(f'{model}:on minus off', (model, 'off'), (model, 'on')) for model in models]
    fast = next((model for model in models if 'fast' in model.lower()), None)
    if fast:
        contrasts += [(f'Fast:{mode} minus {model}:{mode}', (model, mode), (fast, mode))
                      for mode in modes for model in models if model not in {baseline, fast}]
    return contrasts


def selected_family_size(requested, configured, planned):
    family = requested if requested is not None else max(configured, planned)
    assert family >= planned
    return family


def distribution(values, prefix):
    return {f'{prefix}_{name}': float(value) for name, value in zip(
        ['mean', 'p50', 'p95', 'max'], [np.mean(values), *np.percentile(values, [50, 95]), max(values)])}


def analyze(args):
    started = time.perf_counter()
    experiment = json.loads((args.runs/'experiment.json').read_text())
    ids, seeds, modes, models = (experiment[k] for k in ['question_ids', 'seeds', 'modes', 'models'])
    config = experiment['config']
    baseline = config['baseline']
    tables, summaries, costs, per_seed, attempt_rows = {}, [], [], [], []
    all_complete = True
    for model in models:
        rows = read_records(args.runs/model/'questions.jsonl')
        by_key = {record_key(r): r for r in rows}
        expected = {(mode, seed, q) for mode in modes for seed in seeds for q in ids}
        assert set(by_key) <= expected, 'Unexpected question/mode/seed'
        assert all(r['model'] == model for r in rows)
        all_complete &= set(by_key) == expected
        tables[model] = by_key
        attempts = [json.loads(p.read_text()) for p in sorted((args.runs/model/'attempts').glob('*.json'))]
        attempt_rows.extend(attempts)
        allocated = sum(r.get('wall_seconds', 0.) for r in attempts)/3600
        for mode in modes:
            subset = [r for r in rows if r['mode'] == mode]
            if not subset:
                continue
            counts = {q: len(next(r for r in subset if r['id'] == q)['strict_instructions'])
                      for q in {r['id'] for r in subset}}
            for r in subset:
                assert len(r['strict_instructions']) == counts[r['id']]
                assert r['correct'] == all(r['strict_instructions'])
                assert r['answer'] == r['final_answer'].strip()
                assert r['response_sha256_json'] == r['raw_output_sha256_json']
                assert record_sampling_matches(r, config['modes'][mode], r['seed'])
            summaries.append(dict(model=model, mode=mode, evaluations=len(subset),
                observed_evaluations=len(subset),
                questions=len(counts), complete=all((mode, s, q) in by_key for s in seeds for q in ids),
                prompt_strict_accuracy=np.mean([r['correct'] for r in subset]),
                instruction_strict_accuracy=sum(sum(r['strict_instructions']) for r in subset)/
                                            sum(len(r['strict_instructions']) for r in subset),
                truncated=sum(r['truncated'] for r in subset), truncation_rate=np.mean([r['truncated'] for r in subset]),
                missing_final=sum(r['missing_final'] for r in subset)))
            for seed in seeds:
                group = [r for r in subset if r['seed'] == seed]
                if group:
                    per_seed.append(dict(model=model, mode=mode, seed=seed, evaluations=len(group),
                        prompt_strict_accuracy=np.mean([r['correct'] for r in group]),
                        instruction_strict_accuracy=sum(sum(r['strict_instructions']) for r in group)/
                                                   sum(len(r['strict_instructions']) for r in group)))
            inference = sum(r['inference_seconds'] for r in subset)
            # One shared model load; allocate overhead equally among planned mode/seed/question cells.
            shared = sum(max(0., a.get('wall_seconds', 0.) - a['inference_seconds'] -
                             a['scoring_seconds'] - a['preprocessing_seconds']) for a in attempts)
            mode_wall = sum(r[k] for r in subset for k in
                ['inference_seconds', 'scoring_seconds', 'preprocessing_seconds']) + shared/len(modes)
            costs.append(dict(model=model, mode=mode, evaluations=len(subset),
                inference_seconds=inference, scoring_seconds=sum(r['scoring_seconds'] for r in subset),
                preprocessing_seconds=sum(r['preprocessing_seconds'] for r in subset),
                attributed_wall_seconds=mode_wall, attributed_gpu_hours=mode_wall/3600,
                model_total_gpu_hours=allocated, overhead_allocation='equal across planned modes',
                model_initialization_seconds=sum(a['initialization_seconds'] for a in attempts),
                attempt_cost_lower_bound=any(a['status'] == 'running' for a in attempts),
                input_tokens=sum(r['input_tokens'] for r in subset),
                output_tokens=sum(r['output_tokens'] for r in subset),
                aggregate_output_tokens_per_second=sum(r['output_tokens'] for r in subset)/inference,
                pipeline_output_tokens_per_second=sum(r['output_tokens'] for r in subset)/mode_wall,
                **distribution([sum(r[k] for k in ['preprocessing_seconds', 'inference_seconds', 'scoring_seconds'])
                                for r in subset], 'question_seconds'),
                **distribution([r['input_tokens'] for r in subset], 'input_tokens'),
                **distribution([r['inference_seconds'] for r in subset], 'inference_seconds'),
                **distribution([r['output_tokens'] for r in subset], 'output_tokens'),
                **distribution([r['reasoning_tokens'] for r in subset], 'reasoning_tokens')))
    assert all_complete or args.allow_partial, 'Incomplete matrix; use status or --allow-partial for progress only'
    matched = matched_question_ids(ids, models, modes, seeds, tables)
    matched_set = set(matched)
    for row in summaries:
        subset = [r for r in tables[row['model']].values()
                  if r['mode'] == row['mode'] and r['id'] in matched_set]
        row.update(evaluations=len(subset), questions=len(matched), complete=all_complete,
            prompt_strict_accuracy=float(np.mean([r['correct'] for r in subset])) if subset else None,
            instruction_strict_accuracy=(sum(sum(r['strict_instructions']) for r in subset) /
                sum(len(r['strict_instructions']) for r in subset)) if subset else None,
            truncated=sum(bool(r['truncated']) for r in subset),
            truncation_rate=float(np.mean([r['truncated'] for r in subset])) if subset else None,
            missing_final=sum(bool(r['missing_final']) for r in subset))
    for row in per_seed:
        subset = [r for r in tables[row['model']].values()
                  if r['mode'] == row['mode'] and r['seed'] == row['seed'] and r['id'] in matched_set]
        row.update(evaluations=len(subset),
            prompt_strict_accuracy=float(np.mean([r['correct'] for r in subset])) if subset else None,
            instruction_strict_accuracy=(sum(sum(r['strict_instructions']) for r in subset) /
                sum(len(r['strict_instructions']) for r in subset)) if subset else None)
    args.output.mkdir(parents=True, exist_ok=False)
    export(args.output/'accuracy', summaries)
    export(args.output/'per_seed_accuracy', per_seed)
    export(args.output/'timing', costs)
    export(args.output/'attempts', attempt_rows)
    contrasts = comparison_contrasts(models, modes, baseline)
    planned_comparisons = len(contrasts) * len(METRICS)
    family = selected_family_size(args.family_size, config['comparison_family'], planned_comparisons)
    paired, question_rows, seed_contrasts = [], [], []
    coverage = []
    for label, (left_model, left_mode), (right_model, right_mode) in contrasts:
        left, right = tables[left_model], tables[right_model]
        common = [q for q in matched if all((left_mode, s, q) in left and
                                            (right_mode, s, q) in right for s in seeds)]
        coverage.append(dict(comparison=label, questions=len(common), planned_questions=len(ids),
            status='complete' if len(common) == len(ids) else 'paired_subset' if common else 'no_complete_pairs'))
        if not common:
            continue
        numerators, denominators = [], []
        for q in common:
            a = [left[left_mode, s, q] for s in seeds]
            b = [right[right_mode, s, q] for s in seeds]
            for x, y in zip(a, b):
                if left_mode == right_mode:
                    assert x['prompt_token_ids_sha256_json'] == y['prompt_token_ids_sha256_json'], 'Unpaired prompt tokens'
            assert all(r['prompt_token_ids_sha256_json'] == a[0]['prompt_token_ids_sha256_json'] for r in a)
            assert all(r['prompt_token_ids_sha256_json'] == b[0]['prompt_token_ids_sha256_json'] for r in b)
            numerator, denominator = paired_question(a, b)
            numerators.append(numerator)
            denominators.append(denominator)
            question_rows.append(dict(comparison=label, id=q,
                **dict(zip(METRICS, np.asarray(numerator)/denominator))))
        point, limits = cluster_interval(numerators, denominators, args.resamples,
            config['bootstrap_seed'], [.95, 1-.05/family])
        assert np.isclose(point[0], point[2]-point[1])
        assert np.isclose(point[4], point[6]-point[5])
        for confidence, ci in limits.items():
            for i, metric in enumerate(METRICS):
                paired.append(dict(comparison=label, metric=metric, estimate=float(point[i]),
                    low=float(ci.low[i]) if ci else '', high=float(ci.high[i]) if ci else '',
                    confidence=confidence, questions=len(common), seeds=len(seeds),
                    model=right_model, reference=left_model, mode=right_mode, reference_mode=left_mode))
        for seed in seeds:
            values, denominators = zip(*(paired_question([left[left_mode, seed, q]],
                                       [right[right_mode, seed, q]]) for q in common))
            seed_contrasts.append(dict(comparison=label, seed=seed,
                **dict(zip(METRICS, np.sum(values, axis=0)/np.sum(denominators, axis=0)))))
    export(args.output/'paired_comparison', paired)
    export(args.output/'question_effects', question_rows)
    export(args.output/'per_seed_comparison', seed_contrasts)
    export(args.output/'paired_coverage', coverage)
    launches = [json.loads(p.read_text()) for p in args.runs.glob('launch-*.json')]
    total_gpu_hours = sum(r.get('wall_seconds', 0.) for r in attempt_rows)/3600
    write_json(args.output/'statistics.json', dict(complete=all_complete, full_541=experiment['full_541'],
        experiment_sha256=sha(args.runs/'experiment.json'), analysis_source_sha256=sha(Path(__file__)),
        planned_evaluations=experiment['planned_evaluations'],
        completed_evaluations=sum(len(t) for t in tables.values()),
        matched_complete_questions=len(matched), matched_question_ids=matched,
        method='SciPy percentile paired question cluster bootstrap; fixed seeds retained inside question',
        instruction_accuracy='ratio of total passed instructions to total instructions, averaged over seeds',
        seed_inference='conditional on the three fixed seeds; seeds are not independent question replicates',
        family_size=family,
        family_definition=f'Bonferroni family of {family}; {len(contrasts)} planned contrasts x '
            f'{len(METRICS)} metrics = {planned_comparisons} analyzed; descriptive per-seed values excluded',
        contrasts=len(contrasts), resamples=args.resamples,
        harmful_flip='reference correct, candidate incorrect; denominator all paired cases/instructions',
        recovery_flip='reference incorrect, candidate correct; denominator all paired cases/instructions',
        mode_contrast='On minus Off compares the full recommended recipe, including sampling and output cap',
        allocated_gpu_hours=total_gpu_hours,
        launcher_process_gpu_hours=sum(j['wall_seconds'] for launch in launches for j in launch['jobs'])/3600,
        launch_wall_seconds=sum(l['wall_seconds'] for l in launches),
        reservation_gpu_hours=sum(l['reservation_gpu_hours'] for l in launches),
        attempt_cost_is_lower_bound=any(r['status'] == 'running' for r in attempt_rows),
        analysis_seconds=time.perf_counter()-started))
    print(json.dumps(dict(complete=all_complete, output=str(args.output), allocated_gpu_hours=total_gpu_hours)))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--allow-partial', action='store_true', help='Descriptive progress; no incomplete paired contrasts')
    p.add_argument('--resamples', type=int, default=199999)
    p.add_argument('--family-size', type=int, help='Predeclared joint family size; use 300 for the full dual-mode comparison')
    analyze(p.parse_args())
