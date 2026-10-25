"""Aggregate three fixed Terminal-Bench formal seed runs."""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
import shutil

import numpy as np

from terminal_formal_analysis import (
    OPEN, SCORED, attempt_ledger, integrity_audit, load_rows, resource_analysis, wilson,
)


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError, TypeError):
        return default


def write_csv(path, rows):
    rows = list(rows)
    fields = list(rows[0]) if rows else []
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False)
                             if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})


def clustered_bootstrap(base, candidate, seed, resamples=199999, family=6):
    """Resample tasks as clusters, keeping each task's three seeds together."""
    base = np.asarray(base, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if base.shape != candidate.shape or base.ndim != 2 or not base.shape[0]:
        return []
    effects = np.stack((candidate - base,
        ((base == 1) & (candidate == 0)).astype(float),
        ((base == 0) & (candidate == 1)).astype(float)))
    rng = np.random.default_rng(seed)
    draws = np.empty((3, resamples), dtype=np.float64)
    retention_draws = np.empty(resamples, dtype=np.float64)
    n_tasks = base.shape[0]
    for start in range(0, resamples, 2048):
        size = min(2048, resamples-start)
        indices = rng.integers(0, n_tasks, size=(size, n_tasks))
        sampled_base = base[indices]
        sampled_candidate = candidate[indices]
        draws[:, start:start+size] = effects[:, indices].mean(axis=(2, 3))
        base_rate = sampled_base.mean(axis=(1, 2))
        candidate_rate = sampled_candidate.mean(axis=(1, 2))
        retention_draws[start:start+size] = np.divide(candidate_rate, base_rate,
            out=np.full(size, np.nan), where=base_rate != 0)
    rows = []
    for confidence in (0.95, 1-0.05/family):
        alpha = (1-confidence)/2
        bounds = np.nanquantile(draws, [alpha, 1-alpha], axis=1)
        for index, metric in enumerate(('accuracy_delta', 'harmful_flip', 'recovery_flip')):
            rows.append(dict(metric=metric, estimate=float(effects[index].mean()),
                low=float(bounds[0, index]), high=float(bounds[1, index]),
                confidence=confidence, paired_tasks=n_tasks,
                task_seed_observations=n_tasks*base.shape[1], resamples=resamples, seed=seed))
    defined_draws = retention_draws[np.isfinite(retention_draws)]
    retention_bounds = (np.quantile(defined_draws, [0.025, 0.975])
        if len(defined_draws) else (None, None))
    base_rate = float(base.mean())
    candidate_rate = float(candidate.mean())
    rows.append(dict(metric='retention_ratio', estimate=(candidate_rate/base_rate if base_rate else None),
        low=(float(retention_bounds[0]) if retention_bounds[0] is not None else None),
        high=(float(retention_bounds[1]) if retention_bounds[1] is not None else None), confidence=0.95,
        paired_tasks=n_tasks, task_seed_observations=n_tasks*base.shape[1],
        resamples=resamples, seed=seed))
    return rows


def clustered_mean_interval(values, seed, resamples=199999):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not values.shape[0]:
        return None
    rng = np.random.default_rng(seed)
    draws = np.empty(resamples, dtype=np.float64)
    n_tasks = values.shape[0]
    for start in range(0, resamples, 2048):
        size = min(2048, resamples-start)
        indices = rng.integers(0, n_tasks, size=(size, n_tasks))
        draws[start:start+size] = values[indices].mean(axis=(1, 2))
    low, high = np.quantile(draws, [0.025, 0.975])
    return dict(estimate=float(values.mean()), low=float(low), high=float(high),
        confidence=0.95, paired_tasks=n_tasks, task_seed_observations=n_tasks*values.shape[1],
        resamples=resamples, seed=seed)


def aggregate(run_dirs, protocol_path, output):
    protocol = read(protocol_path)
    seeds = protocol['matrix']['seeds']
    if len(run_dirs) != 3 or len(seeds) != 3:
        raise ValueError('Exactly three seed runs and three frozen seeds are required')
    if len(set(seeds)) != 3:
        raise ValueError('Seed values must be unique')
    loaded, gates_by_seed, all_attempts, resources, integrity = {}, {}, [], {}, {}
    canonical_tasks = canonical_models = None
    all_records, slot_keys = [], set()
    for seed, run in zip(seeds, map(Path, run_dirs)):
        plan, validation, queue_rows, records = load_rows(run)
        if plan['seed'] != seed:
            raise ValueError(f'{run}: plan seed {plan["seed"]} does not match frozen seed {seed}')
        if plan['planned_slots'] != 267:
            raise ValueError(f'{run}: per-seed plan must contain 267 slots')
        tasks, models = plan['tasks'], plan['models']
        if canonical_tasks is None:
            canonical_tasks, canonical_models = tasks, models
        elif tasks != canonical_tasks or models != canonical_models:
            raise ValueError(f'{run}: task or model order differs across seeds')
        gate_status = {task: validation['tasks'][task]['status'] for task in tasks}
        gates_by_seed[seed] = gate_status
        if seed != seeds[0] and gate_status != gates_by_seed[seeds[0]]:
            raise ValueError(f'{run}: environment gate ledger differs from seed {seeds[0]}')
        for record in records:
            record['seed'] = seed
            if record['gate_status'] == 'CONTROLS_PASS' and record['attempt_count'] == 0:
                record['status'] = 'PENDING'
            key = (seed, record['task'], record['model'])
            if key in slot_keys:
                raise ValueError(f'Duplicate matrix slot: {key}')
            slot_keys.add(key)
            if record['status'] in OPEN or record['status'] == 'INTERRUPTED':
                raise ValueError(f'Non-final matrix slot {key}: {record["status"]}')
            all_records.append(record)
        attempts = attempt_ledger(run, queue_rows)
        for attempt in attempts:
            attempt['seed'] = seed
        all_attempts.extend(attempts)
        resources[seed] = resource_analysis(run, records, attempts)
        integrity[seed] = integrity_audit(run, plan, validation, queue_rows, records)
        loaded[seed] = dict(plan=plan, validation=validation, queue=queue_rows, records=records)

    expected_slots = len(canonical_tasks)*len(canonical_models)*len(seeds)
    if expected_slots != 801 or len(slot_keys) != expected_slots:
        raise ValueError(f'Expected all 801 unique slots, found {len(slot_keys)}')
    evaluable_tasks = [task for task in canonical_tasks
        if gates_by_seed[seeds[0]][task] == 'CONTROLS_PASS']
    by_slot = {(row['seed'], row['task'], row['model']): row for row in all_records}
    paired_tasks = [task for task in evaluable_tasks if all(
        by_slot[(seed, task, model)]['status'] in SCORED
        and by_slot[(seed, task, model)]['score'] is not None
        for seed in seeds for model in canonical_models)]

    per_seed, model_rows, difference_rows, intervals = [], [], [], []
    reference = canonical_models[0]
    score_arrays = {}
    for model in canonical_models:
        score_arrays[model] = {
            seed: np.asarray([by_slot[(seed, task, model)]['score'] for task in paired_tasks], dtype=np.float64)
            for seed in seeds}
    mean_rates = {}
    seed_rates = defaultdict(dict)
    for model in canonical_models:
        for seed in seeds:
            values = score_arrays[model][seed]
            rate = float(values.mean()) if len(values) else None
            seed_rates[model][seed] = rate
            low, high = wilson(int(values.sum()), len(values)) if len(values) else (None, None)
            available = [by_slot[(seed, task, model)]['score'] for task in canonical_tasks
                if by_slot[(seed, task, model)]['score'] is not None]
            per_seed.append(dict(seed=seed, model=model, paired_tasks=len(paired_tasks),
                passes=int(values.sum()) if len(values) else None, pass_rate=rate,
                wilson95_low=low, wilson95_high=high,
                all_scored_tasks=len(available), all_scored_passes=int(sum(available)),
                all_scored_pass_rate=(float(np.mean(available)) if available else None),
                scored_slots=sum(by_slot[(seed, task, model)]['score'] is not None for task in evaluable_tasks),
                status_counts=dict(Counter(row['status'] for row in all_records
                    if row['seed'] == seed and row['model'] == model))))
        rates = [seed_rates[model][seed] for seed in seeds]
        mean_rates[model] = float(np.mean(rates)) if all(rate is not None for rate in rates) else None
    base_mean = mean_rates.get(reference)

    for model in canonical_models:
        observed = [row for row in all_records if row['model'] == model]
        model_attempts = [row for row in all_attempts if row['model'] == model]
        available_scores = [item['score'] for item in observed if item['score'] is not None]
        base_rate = base_mean
        model_matrix = np.stack([score_arrays[model][seed] for seed in seeds], axis=1)
        bootstrap_cfg = loaded[seeds[0]]['plan']['config']
        pass_interval = clustered_mean_interval(model_matrix,
            int(bootstrap_cfg['bootstrap_seed'])+10+canonical_models.index(model),
            int(bootstrap_cfg['bootstrap_resamples']))
        row = dict(model=model,
            candidate_label='BF16' if model == reference else next((label for label, name in
                protocol['protocol_config'].get('candidate_models', {}).items() if name == model), model),
            paired_tasks=len(paired_tasks), paired_task_seed_observations=len(paired_tasks)*len(seeds),
            seed_pass_rates={str(seed): seed_rates[model][seed] for seed in seeds},
            mean_pass_rate=mean_rates[model],
            mean_pass_rate_cluster95=pass_interval,
            seed_pass_rate_sd=(float(np.std([seed_rates[model][seed] for seed in seeds], ddof=1))
                if all(seed_rates[model][seed] is not None for seed in seeds) else None),
            seed_pass_rate_min=(min(seed_rates[model][seed] for seed in seeds)
                if all(seed_rates[model][seed] is not None for seed in seeds) else None),
            seed_pass_rate_max=(max(seed_rates[model][seed] for seed in seeds)
                if all(seed_rates[model][seed] is not None for seed in seeds) else None),
            mean_retention_vs_bf16=(mean_rates[model]/base_rate if model != reference and base_rate else
                (1.0 if model == reference and base_rate else None)),
            mean_pass_rate_delta_vs_bf16=(mean_rates[model]-base_rate if model != reference and base_rate is not None else
                (0.0 if model == reference and base_rate is not None else None)),
            all_scored_slots=sum(item['score'] is not None for item in observed),
            all_scored_passes=int(sum(available_scores)),
            all_scored_pass_rate=(float(np.mean(available_scores)) if available_scores else None),
            status_counts=dict(Counter(item['status'] for item in observed)),
            attempts=len(model_attempts),
            cost_usd=sum(item.get('cost_usd') or 0 for item in model_attempts),
            input_tokens=sum(item.get('input_tokens') or 0 for item in model_attempts),
            output_tokens=sum(item.get('output_tokens') or 0 for item in model_attempts),
            tool_calls=sum(item.get('tool_calls') or 0 for item in model_attempts),
            agent_seconds=sum(item.get('agent_seconds') or 0 for item in model_attempts),
            mtp_drafts=sum(item.get('mtp_drafts') or 0 for item in model_attempts),
            mtp_draft_tokens=sum(item.get('mtp_draft_tokens') or 0 for item in model_attempts),
            mtp_accepted_tokens=sum(item.get('mtp_accepted_tokens') or 0 for item in model_attempts),
            gpu_hours=sum(resources[seed]['model_gpu_hours'].get(model, 0) for seed in seeds))
        row['mtp_token_acceptance_rate'] = (row['mtp_accepted_tokens']/row['mtp_draft_tokens']
            if row['mtp_draft_tokens'] else None)
        row['output_tokens_per_agent_second'] = (row['output_tokens']/row['agent_seconds']
            if row['agent_seconds'] else None)
        seed_retention = {str(seed): (seed_rates[model][seed]/seed_rates[reference][seed]
            if seed_rates[reference][seed] else None) for seed in seeds}
        defined_retention = [value for value in seed_retention.values() if value is not None]
        row['seed_retention_vs_bf16'] = seed_retention
        row['mean_seed_retention_vs_bf16'] = (float(np.mean(defined_retention))
            if defined_retention else None)
        row['seed_retention_sd'] = (float(np.std(defined_retention, ddof=1))
            if len(defined_retention) > 1 else None)
        model_rows.append(row)

    for candidate in canonical_models[1:]:
        base_matrix = np.stack([score_arrays[reference][seed] for seed in seeds], axis=1)
        candidate_matrix = np.stack([score_arrays[candidate][seed] for seed in seeds], axis=1)
        for task_index, task in enumerate(paired_tasks):
            for seed_index, seed in enumerate(seeds):
                base_score = int(base_matrix[task_index, seed_index])
                candidate_score = int(candidate_matrix[task_index, seed_index])
                difference_rows.append(dict(task=task, seed=seed, candidate=candidate, reference=reference,
                    reference_score=base_score, candidate_score=candidate_score,
                    delta=candidate_score-base_score,
                    harmful_flip=int(base_score == 1 and candidate_score == 0),
                    recovery_flip=int(base_score == 0 and candidate_score == 1)))
        base_cfg = loaded[seeds[0]]['plan']['config']
        seed_value = int(base_cfg['bootstrap_seed'])+canonical_models.index(candidate)-1
        intervals.extend(clustered_bootstrap(base_matrix, candidate_matrix, seed_value,
            int(base_cfg['bootstrap_resamples']), int(base_cfg['comparison_family'])))
        for interval in intervals:
            if interval['seed'] == seed_value:
                interval.update(candidate=candidate, reference=reference)

    gate_counts = dict(Counter(gates_by_seed[seeds[0]].values()))
    slot_counts = dict(Counter(row['status'] for row in all_records))
    total_gpu_hours = sum(resource['gpu_hours'] for resource in resources.values())
    total_wall_hours = sum(resource['run_wall_hours'] or 0 for resource in resources.values())
    schedule = dict(gpu_hours=total_gpu_hours, sequential_wall_hours=total_wall_hours,
        allocated_gpu_fraction=(total_gpu_hours/(5*total_wall_hours) if total_wall_hours else None),
        per_seed={str(seed): resources[seed] for seed in seeds},
        gpu_resource_samples=sum(item['resource_samples'] for item in resources.values()))
    overall_integrity = dict(passed=all(item['passed'] for item in integrity.values()),
        per_seed={str(seed): integrity[seed] for seed in seeds},
        issues=[dict(seed=seed, **issue) for seed in seeds for issue in integrity[seed]['issues']])
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(Path(__file__), out/'terminal_formal_multiseed_analysis.py')
    runner_audit = read(Path(run_dirs[0])/'RUNNER_AUDIT.json', {}) or {}
    if runner_audit:
        shutil.copy2(Path(run_dirs[0])/'RUNNER_AUDIT.json', out/'RUNNER_AUDIT.json')
    write_csv(out/'matrix_801.csv', all_records)
    write_csv(out/'attempts.csv', all_attempts)
    write_csv(out/'seed_model_summary.csv', per_seed)
    write_csv(out/'model_summary.csv', model_rows)
    write_csv(out/'paired_task_seed_differences.csv', difference_rows)
    write_csv(out/'paired_intervals.csv', intervals)
    (out/'environment_validation.json').write_text(json.dumps(dict(
        denominator=89, counts=gate_counts, tasks=gates_by_seed[seeds[0]]),
        indent=2, ensure_ascii=False)+'\n')
    (out/'integrity_audit.json').write_text(json.dumps(overall_integrity,
        indent=2, ensure_ascii=False)+'\n')
    final = dict(protocol=protocol, expected_slots=801, observed_slots=len(all_records),
        slot_status_counts=slot_counts, environment_gate_counts=gate_counts,
        all_environment_gates_conclusive=not any(status in OPEN for status in gate_counts),
        all_matrix_slots_terminal=not any(status in OPEN or status == 'INTERRUPTED' for status in slot_counts),
        all_evaluable_slots_attempted=all(row['attempt_count'] > 0 for row in all_records
            if row['gate_status'] == 'CONTROLS_PASS'),
        evaluable_tasks=evaluable_tasks, paired_tasks=paired_tasks,
        paired_task_count=len(paired_tasks), model_summary=model_rows,
        seed_model_summary=per_seed, pairwise_intervals=intervals,
        scheduling=schedule, integrity_audit=overall_integrity, runner_audit=runner_audit,
        infrastructure_failures_excluded_from_scores=True,
        analyzed_at=__import__('time').time())
    (out/'analysis.json').write_text(json.dumps(final, indent=2, ensure_ascii=False)+'\n')
    report = ['# Terminal-Bench 2.1 three-seed formal analysis', '',
        f"- Matrix slots: {len(all_records)}/801.",
        f"- Environment decisions: {sum(gate_counts.values())}/89; gate counts `{json.dumps(gate_counts, ensure_ascii=False)}`.",
        f"- All matrix slots terminal: {final['all_matrix_slots_terminal']}; all Gate-passed slots have a Trial attempt: {final['all_evaluable_slots_attempted']}.",
        f"- Common fully scored tasks across all models and all three seeds: {len(paired_tasks)}/{len(evaluable_tasks)} Gate-passed tasks.",
        f"- Five-GPU allocated time: {total_gpu_hours:.2f} GPU-hours; sum of seed-run wall time: {total_wall_hours:.2f} hours; allocated fraction: {schedule['allocated_gpu_fraction']!r}.",
        '', '| Model | Seed pass rates | Three-seed mean | Seed SD | All-scored rate | Mean retention vs BF16 | Mean seed retention | Paired difference | Scored slots | GPU hours | Reported cost USD | Tokens in/out | Tools | MTP accepted/draft |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in model_rows:
        rates = ', '.join(f"{seed}: {row['seed_pass_rates'][str(seed)]:.3f}" if row['seed_pass_rates'][str(seed)] is not None else f'{seed}: NA' for seed in seeds)
        mean_rate = row['mean_pass_rate']
        mean_ci = row['mean_pass_rate_cluster95']
        mean_cell = (f"{mean_rate:.4f} [{mean_ci['low']:.4f}, {mean_ci['high']:.4f}]"
            if mean_rate is not None and mean_ci else 'NA')
        retention = row['mean_retention_vs_bf16']
        delta = row['mean_pass_rate_delta_vs_bf16']
        report.append(f"| {row['candidate_label']} | {rates} | {mean_cell} | {row['seed_pass_rate_sd']!r} | {row['all_scored_pass_rate']!r} | {retention!r} | {row['mean_seed_retention_vs_bf16']!r} | {delta!r} | {row['all_scored_slots']} | {row['gpu_hours']:.2f} | {row['cost_usd']:.4f} | {row['input_tokens']}/{row['output_tokens']} | {row['tool_calls']} | {row['mtp_accepted_tokens']}/{row['mtp_draft_tokens']} |")
    status_names = sorted({status for row in per_seed for status in row['status_counts']})
    report.extend(['', '## Status and failure distribution', '',
        'Each row covers 89 task slots for one model and seed. Shared Gate failures appear for all models and seeds; trial-level verifier and environment failures have no score.',
        '', '| Seed | Model | ' + ' | '.join(status_names) + ' |',
        '|---:|---|' + '---:|'*len(status_names)])
    for row in per_seed:
        report.append('| '+str(row['seed'])+' | '+row['model']+' | '
            +' | '.join(str(row['status_counts'].get(status, 0)) for status in status_names)+' |')
    report.extend(['', '## Paired uncertainty', '',
        'Intervals resample whole tasks and keep each task’s three seeds together while holding the fixed Seed list. Accuracy difference, harmful flips, and recovery flips use the six-comparison family adjustment fixed in the protocol. Retention has a 95% task-cluster bootstrap interval. Seed-to-seed spread is reported in `model_summary.csv`.',
        '', '| Candidate | Measure | Estimate | 95% interval | Family-adjusted interval |',
        '|---|---|---:|---:|---:|'])
    for candidate in canonical_models[1:]:
        for metric in ('accuracy_delta', 'harmful_flip', 'recovery_flip'):
            matching = [item for item in intervals if item.get('candidate') == candidate and item['metric'] == metric]
            normal = next((item for item in matching if item['confidence'] == 0.95), None)
            adjusted = next((item for item in matching if item['confidence'] != 0.95), None)
            if normal and adjusted:
                report.append(f"| {candidate} | {metric} | {normal['estimate']:.4f} | [{normal['low']:.4f}, {normal['high']:.4f}] | [{adjusted['low']:.4f}, {adjusted['high']:.4f}] |")
        retention = next((item for item in intervals if item.get('candidate') == candidate and item['metric'] == 'retention_ratio'), None)
        if retention:
            report.append(f"| {candidate} | retention ratio | {retention['estimate']:.4f} | [{retention['low']:.4f}, {retention['high']:.4f}] | NA |")
    report.extend(['', '## Paired task differences', '',
        'See `paired_task_seed_differences.csv` for each task and seed, including the exact BF16 reward, candidate reward, score delta, harmful flip, and recovery flip.',
        '', '## Status and audit files', '',
        f"Runner audit: live-release simulation {runner_audit.get('schedule_ab', {}).get('live_queue_makespan_seconds')} s vs batch-barrier {runner_audit.get('schedule_ab', {}).get('batch_barrier_makespan_seconds')} s; same-model service batch {runner_audit.get('schedule_ab', {}).get('service_batch', 3)}; {runner_audit.get('targeted_tests', {}).get('passed')} targeted regressions passed. The simulation is scheduler evidence, not measured GPU speed; see `RUNNER_AUDIT.json`.",
        f"Slot outcomes across the 801-cell ledger: `{json.dumps(slot_counts, ensure_ascii=False)}`.",
        f"Integrity checks passed for all seeds: {overall_integrity['passed']}. See `integrity_audit.json` for row-level issues.",
        'Infrastructure, verifier, and PRoot failures remain explicit in `matrix_801.csv`; they have no score and are excluded from paired ability estimates. `attempts.csv` preserves every attempt, including a retry. `paired_intervals.csv` gives the uncertainty intervals.',
        '', 'This is an internal comparison in the shared PRoot environment, not an official Docker leaderboard result.', ''])
    (out/'REPORT.md').write_text('\n'.join(report))
    return final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--runs', type=Path, nargs=3, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = aggregate(args.runs, args.protocol, args.output)
    print(json.dumps(dict(output=str(args.output), observed_slots=result['observed_slots'],
        paired_tasks=result['paired_task_count'], status_counts=result['slot_status_counts']),
        ensure_ascii=False))


if __name__ == '__main__':
    main()
