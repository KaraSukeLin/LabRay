#!/usr/bin/env python3
"""Analyze one saved Terminal-Bench Low/MTP3 formal run."""
import argparse
import csv
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import shutil
import sqlite3
import statistics
import time

import numpy as np


SCORED = {'PASS', 'MODEL_FAIL', 'TIMEOUT'}
OPEN = {'ENV_PENDING', 'VALIDATING', 'VALIDATING_RETRY', 'PENDING', 'RUNNING'}


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError, TypeError):
        return default


def write_csv(path, rows, fields):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})


def iso_epoch(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp()
    except (ValueError, AttributeError):
        return None


def elapsed_seconds(data):
    start, stop = iso_epoch(data.get('started_at')), iso_epoch(data.get('finished_at'))
    return stop-start if start is not None and stop is not None else None


def trajectory_counts(path):
    data = read(path, {}) or {}
    steps = data.get('steps', [])
    return (sum(step.get('source') == 'agent' for step in steps),
            sum(len(step.get('tool_calls') or []) for step in steps))


def mtp_counts(path):
    data = read(path, {}) or {}
    delta = data.get('delta') or {}
    names = ('spec_decode_num_drafts_total', 'spec_decode_num_draft_tokens_total',
             'spec_decode_num_accepted_tokens_total')
    counts = {name.replace('spec_decode_num_', ''): 0.0 for name in names}
    positions = defaultdict(float)
    for metric, value in delta.items():
        try:
            amount = float(value)
        except (TypeError, ValueError):
            continue
        for name in names:
            if name in metric:
                counts[name.replace('spec_decode_num_', '')] += amount
        if 'spec_decode_num_accepted_tokens_per_pos_total' in metric:
            try:
                position = int(metric.split('position="', 1)[1].split('"', 1)[0])
            except (IndexError, ValueError):
                continue
            positions[position] += amount
    return counts, dict(sorted(positions.items()))


def load_rows(run):
    plan = read(run/'PLAN.json')
    validation = read(run/'environment_validation.json')
    if not plan or not validation:
        raise ValueError('PLAN.json or environment_validation.json is missing')
    tasks, models = plan['tasks'], plan['models']
    if len(tasks) != 89 or len(models) != 3:
        raise ValueError(f'Expected the fixed 89-task, 3-model protocol; found {len(tasks)} tasks and {len(models)} models')
    con = sqlite3.connect(run/'matrix.sqlite')
    con.row_factory = sqlite3.Row
    queued = [dict(row) for row in con.execute('SELECT * FROM trials ORDER BY task,model,attempt,started')]
    con.close()
    by_slot = defaultdict(list)
    for row in queued:
        by_slot[(row['task'], row['model'])].append(row)

    records = []
    for task in tasks:
        gate = validation['tasks'][task]['status']
        for model in models:
            attempts = sorted(by_slot[(task, model)], key=lambda row: (row['attempt'], row['started'] or 0))
            chosen = attempts[-1] if attempts else None
            status = chosen['status'] if chosen else gate
            folder = run/'trials'/chosen['id'] if chosen else None
            result_path = Path(chosen['result']) if chosen and chosen.get('result') else None
            data = read(result_path, {}) if result_path else {}
            summary = read(folder/'summary.json', {}) if folder else {}
            agent_result = data.get('agent_result') or {}
            steps, tool_calls = trajectory_counts(folder/'trial/agent/trajectory.json') if folder else (None, None)
            mtp, mtp_positions = mtp_counts(folder/'mtp.json') if folder else ({}, {})
            native = data.get('verifier_result') or {}
            native_reward = (native.get('rewards') or {}).get('reward')
            exception = data.get('exception_info') or summary.get('exception') or {}
            if status in SCORED:
                if native_reward in (0, 0.0, 1, 1.0):
                    score, score_basis = int(native_reward), 'native_verifier_reward'
                elif status == 'TIMEOUT':
                    score, score_basis = 0, 'timeout_without_reward'
                else:
                    score, score_basis = None, 'missing_reward'
            else:
                score, score_basis = None, None
            records.append(dict(task=task, model=model, gate_status=gate, status=status,
                attempt_count=len(attempts), attempt=chosen['attempt'] if chosen else None,
                trial_id=chosen['id'] if chosen else None, gpu=chosen['gpu'] if chosen else None,
                queue_started=chosen['started'] if chosen else None,
                queue_finished=chosen['finished'] if chosen else None,
                trial_started=iso_epoch(data.get('started_at')),
                trial_finished=iso_epoch(data.get('finished_at')),
                elapsed_seconds=(chosen['finished']-chosen['started']) if chosen and chosen['started'] and chosen['finished'] else None,
                reward=native_reward, score=score, score_basis=score_basis,
                summary_status=summary.get('status'), cleanup_clean=summary.get('cleanup_clean'),
                api_identity=(summary.get('api_audit') or {}).get('identity'),
                exception_type=exception.get('exception_type'),
                exception_message=exception.get('exception_message'),
                result=str(result_path) if result_path else None,
                input_tokens=agent_result.get('n_input_tokens'),
                cached_tokens=agent_result.get('n_cache_tokens'),
                output_tokens=agent_result.get('n_output_tokens'),
                cost_usd=agent_result.get('cost_usd'),
                agent_seconds=elapsed_seconds(data.get('agent_execution') or {}),
                wall_seconds=elapsed_seconds(data), steps=steps, tool_calls=tool_calls,
                api_requests=(summary.get('api_audit') or {}).get('requests'),
                api_complete_requests=(summary.get('api_audit') or {}).get('complete_requests'),
                api_incomplete_requests=(summary.get('api_audit') or {}).get('incomplete_requests'),
                mtp_drafts=mtp.get('drafts_total'), mtp_draft_tokens=mtp.get('draft_tokens_total'),
                mtp_accepted_tokens=mtp.get('accepted_tokens_total'),
                mtp_acceptance_rate=(mtp.get('accepted_tokens_total')/mtp.get('draft_tokens_total')
                    if mtp.get('draft_tokens_total') else None), mtp_positions=mtp_positions))
    return plan, validation, queued, records


def attempt_ledger(run, queue_rows):
    attempts = []
    for item in queue_rows:
        folder = run/'trials'/item['id']
        result_path = Path(item['result']) if item.get('result') else None
        data = read(result_path, {}) if result_path else {}
        summary = read(folder/'summary.json', {}) or {}
        failure = read(folder/'infrastructure_failure.json', {}) or {}
        agent_result = data.get('agent_result') or {}
        mtp, positions = mtp_counts(folder/'mtp.json')
        steps, tool_calls = trajectory_counts(folder/'trial/agent/trajectory.json')
        exception = data.get('exception_info') or summary.get('exception') or failure
        attempts.append(dict(id=item['id'], task=item['task'], model=item['model'], mode=item['mode'],
            seed=item['seed'], attempt=item['attempt'], status=item['status'], gpu=item['gpu'],
            started=item['started'], finished=item['finished'],
            trial_started=iso_epoch(data.get('started_at')), trial_finished=iso_epoch(data.get('finished_at')),
            elapsed_seconds=(item['finished']-item['started'])
                if item.get('started') and item.get('finished') else None,
            reward=((data.get('verifier_result') or {}).get('rewards') or {}).get('reward'),
            exception_type=exception.get('exception_type'), exception_message=exception.get('exception_message') or exception.get('error'),
            result=str(result_path) if result_path else None,
            input_tokens=agent_result.get('n_input_tokens'), cached_tokens=agent_result.get('n_cache_tokens'),
            output_tokens=agent_result.get('n_output_tokens'), cost_usd=agent_result.get('cost_usd'),
            agent_seconds=elapsed_seconds(data.get('agent_execution') or {}), wall_seconds=elapsed_seconds(data),
            steps=steps, tool_calls=tool_calls,
            api_requests=(summary.get('api_audit') or {}).get('requests'),
            api_complete_requests=(summary.get('api_audit') or {}).get('complete_requests'),
            api_incomplete_requests=(summary.get('api_audit') or {}).get('incomplete_requests'),
            mtp_drafts=mtp.get('drafts_total'), mtp_draft_tokens=mtp.get('draft_tokens_total'),
            mtp_accepted_tokens=mtp.get('accepted_tokens_total'), mtp_positions=positions))
    return attempts


def paired_bootstrap(base, candidate, seed, resamples=199999, family=6):
    base, candidate = np.asarray(base, dtype=np.int8), np.asarray(candidate, dtype=np.int8)
    if base.shape != candidate.shape or not len(base):
        return []
    effects = np.stack((candidate-base, ((base == 1) & (candidate == 0)).astype(float),
                        ((base == 0) & (candidate == 1)).astype(float)))
    rng = np.random.default_rng(seed)
    draws = np.empty((3, resamples), dtype=np.float64)
    batch = 4096
    for start in range(0, resamples, batch):
        size = min(batch, resamples-start)
        indices = rng.integers(0, len(base), size=(size, len(base)))
        draws[:, start:start+size] = effects[:, indices].mean(axis=2)
    rows = []
    for confidence in (0.95, 1-0.05/family):
        alpha = (1-confidence)/2
        bounds = np.quantile(draws, [alpha, 1-alpha], axis=1)
        for i, metric in enumerate(('accuracy_delta', 'harmful_flip', 'recovery_flip')):
            rows.append(dict(metric=metric, estimate=float(effects[i].mean()), low=float(bounds[0, i]),
                high=float(bounds[1, i]), confidence=confidence, paired_tasks=len(base),
                resamples=resamples, seed=seed))
    return rows


def wilson(successes, n, z=1.959963984540054):
    if not n:
        return None, None
    p = successes/n
    denom = 1+z*z/n
    center = (p+z*z/(2*n))/denom
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/denom
    return center-half, center+half


def resource_analysis(run, records, attempts):
    samples = []
    resource_file = run/'resources.jsonl'
    if resource_file.exists():
        for line in resource_file.read_text().splitlines():
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    gpu_values = defaultdict(lambda: {'util': [], 'used': []})
    pressure = defaultdict(list)
    for sample in samples:
        for index, gpu in (sample.get('gpus') or {}).items():
            gpu_values[str(index)]['util'].append(gpu.get('utilization'))
            gpu_values[str(index)]['used'].append(gpu.get('used_mb'))
        for key in ('cpu_pressure', 'memory_pressure', 'io_pressure'):
            raw = sample.get(key, '')
            try:
                pressure[key].append(float(raw.split('avg10=', 1)[1].split()[0]))
            except (IndexError, ValueError):
                pass
    services = []
    for path in run.glob('services/**/services.json'):
        rows = read(path, []) or []
        services.extend(rows)
    observed_trials = attempts or records
    for service in services:
        service['trials_in_session'] = sum(row.get('gpu') == service.get('gpu')
            and row.get('model') == service.get('model')
            and row.get('trial_started', row.get('started', row.get('queue_started'))) is not None
            and service.get('started_at', 0) <= row.get('trial_started', row.get('started', row.get('queue_started'))) <= service.get('stopped_at', float('inf'))
            for row in observed_trials)
    gpu_summary = {}
    for gpu in range(5):
        values = gpu_values[str(gpu)]
        util = [x for x in values['util'] if x is not None]
        used = [x for x in values['used'] if x is not None]
        gpu_summary[str(gpu)] = dict(samples=len(util), mean_utilization_percent=float(np.mean(util)) if util else None,
            zero_utilization_fraction=float(np.mean(np.asarray(util) == 0)) if util else None,
            mean_memory_used_mb=float(np.mean(used)) if used else None,
            services=sum(row.get('gpu') == gpu for row in services),
            service_gpu_hours=sum(row.get('gpu_hours', 0) for row in services if row.get('gpu') == gpu))
    all_times = [sample.get('time') for sample in samples if sample.get('time')]
    run_hours = (max(all_times)-min(all_times))/3600 if len(all_times) > 1 else None
    total_gpu_hours = sum(row.get('gpu_hours', 0) for row in services)
    return dict(resource_samples=len(samples), run_wall_hours=run_hours, gpu_hours=total_gpu_hours,
        allocated_gpu_fraction=total_gpu_hours/(5*run_hours) if run_hours else None,
        per_gpu=gpu_summary, mean_pressure_avg10={key:float(np.mean(val)) for key,val in pressure.items() if val},
        model_loads=len(services), service_batches=dict(Counter(str(row.get('trials_in_session', 0)) for row in services)),
        model_gpu_hours={model:sum(row.get('gpu_hours', 0) for row in services if row.get('model') == model)
            for model in sorted({row.get('model') for row in services})},
        services=services)


def gate_delay(run, validation, records):
    started_at = (read(run/'STATE.json', {}) or {}).get('started_at')
    delays = []
    by_task = defaultdict(list)
    for row in records:
        start = row.get('trial_started') or row.get('queue_started')
        if start is not None:
            by_task[row['task']].append(start)
    for task, gate in validation['tasks'].items():
        if gate['status'] != 'CONTROLS_PASS' or task not in by_task:
            continue
        finishes = []
        latest = gate.get('attempts', [])[-1] if gate.get('attempts') else {}
        for control in (latest.get('controls') or {}).values():
            result_path = control.get('result') if isinstance(control, dict) else None
            result = read(result_path, {}) if result_path else {}
            stamp = iso_epoch(result.get('finished_at'))
            if stamp is not None:
                finishes.append(stamp)
        if finishes:
            control_finished = max(finishes)
            # Older pre-existing controls were valid gates, but they did not
            # wait for a formal GPU slot during this run.
            if started_at is not None and control_finished < started_at:
                continue
            delay = min(by_task[task])-control_finished
            delays.append(dict(task=task, gate_to_first_trial_seconds=delay))
    vals = [row['gate_to_first_trial_seconds'] for row in delays]
    return dict(tasks_with_measured_release=len(vals), p50_seconds=float(np.median(vals)) if vals else None,
        p90_seconds=float(np.quantile(vals, .9)) if vals else None,
        under_5_minutes=sum(value < 300 for value in vals), rows=delays)


def gate_failures(validation):
    rows = []
    for task, entry in validation['tasks'].items():
        if entry['status'] == 'CONTROLS_PASS':
            continue
        last = entry.get('attempts', [])[-1] if entry.get('attempts') else {}
        checked = last.get('checked') or {}
        raw = last.get('controls') or {}
        row = dict(task=task, status=entry['status'], attempt=last.get('attempt'),
            retryable=last.get('retryable'), gate_error=last.get('error'))
        for agent in ('oracle', 'nop'):
            detail = checked.get(agent) or raw.get(agent) or {}
            exception = detail.get('exception') or {}
            row.update({f'{agent}_status':detail.get('status'), f'{agent}_reward':detail.get('reward'),
                f'{agent}_verifier_tests':detail.get('verifier_tests'),
                f'{agent}_verifier_failed':detail.get('verifier_failed'),
                f'{agent}_cleanup_clean':detail.get('cleanup_clean'),
                f'{agent}_exception_type':exception.get('exception_type') if isinstance(exception,dict) else str(exception),
                f'{agent}_exception_message':exception.get('exception_message') if isinstance(exception,dict) else None,
                f'{agent}_result':detail.get('result')})
        rows.append(row)
    return rows


def integrity_audit(run, plan, validation, queue_rows, records):
    issues = []
    by_slot = defaultdict(list)
    api_complete = defaultdict(int)
    for path in run.glob('services/**/audit/*.json'):
        event = read(path, {}) or {}
        identity = (event.get('identity') or {}).get('x-terminal-identity')
        if identity and event.get('status') == 'complete' and event.get('http_status') == 200 \
                and bool((event.get('response') or {}).get('choices')):
            api_complete[identity] += 1
    for item in queue_rows:
        by_slot[(item['task'], item['model'])].append(item)
    for task in plan['tasks']:
        evaluable = validation['tasks'][task]['status'] == 'CONTROLS_PASS'
        for model in plan['models']:
            attempts = by_slot[(task, model)]
            if evaluable and not attempts:
                issues.append(dict(task=task, model=model, check='missing_queue_slot'))
            if not evaluable and attempts:
                issues.append(dict(task=task, model=model, check='queued_without_passing_gate'))
    for row in records:
        if row['status'] not in SCORED:
            continue
        result = read(row['result'], {}) if row.get('result') else {}
        folder = run/'trials'/row['trial_id']
        cleanup_path = folder/'trial/proot/cleanup.json'
        cleanup = read(cleanup_path, {}) or {}
        if not result.get('finished_at'):
            issues.append(dict(task=row['task'], model=row['model'], check='scored_result_missing_or_incomplete'))
        if row.get('summary_status') != row['status']:
            issues.append(dict(task=row['task'], model=row['model'], check='summary_status_mismatch',
                summary=row.get('summary_status'), queue=row['status']))
        if not cleanup_path.exists() or cleanup.get('after') != [] or row.get('cleanup_clean') is not True:
            issues.append(dict(task=row['task'], model=row['model'], check='unclean_or_unrecorded_proot_cleanup'))
        if row['score'] is None:
            issues.append(dict(task=row['task'], model=row['model'], check='scored_slot_has_no_score'))
        if row.get('reward') is not None:
            try:
                reward = float(row['reward'])
                if reward not in (0.0, 1.0) or row['score'] != int(reward):
                    issues.append(dict(task=row['task'], model=row['model'], check='reward_status_mismatch',
                        reward=row['reward'], status=row['status']))
            except (TypeError, ValueError):
                issues.append(dict(task=row['task'], model=row['model'], check='non_numeric_reward',
                    reward=row['reward']))
        identity = f"formal/{row['model']}/{row['task']}/{row['trial_id']}"
        complete_requests = api_complete.get(identity, 0)
        if row['status'] in {'PASS','MODEL_FAIL'} and not max(row.get('api_complete_requests') or 0, complete_requests):
            issues.append(dict(task=row['task'], model=row['model'], check='scored_trial_has_no_complete_api_request'))
        if row.get('api_identity') not in (None, identity):
            issues.append(dict(task=row['task'], model=row['model'], check='api_audit_identity_mismatch'))
    return dict(passed=not issues, issues=issues,
        scored_slots=sum(row['status'] in SCORED for row in records),
        checked_cleanup_receipts=sum(row['status'] in SCORED for row in records))


def analyze(args):
    run, output = args.run.resolve(), args.output.resolve()
    plan, validation, queue_rows, records = load_rows(run)
    attempts = attempt_ledger(run, queue_rows)
    if output.exists():
        raise FileExistsError(f'Analysis output already exists: {output}')
    output.mkdir(parents=True)
    shutil.copy2(Path(__file__), output/'terminal_formal_analysis.py')
    runner_audit = read(run/'RUNNER_AUDIT.json', {}) or {}
    if runner_audit:
        shutil.copy2(run/'RUNNER_AUDIT.json', output/'RUNNER_AUDIT.json')
    models = plan['models']
    by_task = defaultdict(dict)
    for row in records:
        by_task[row['task']][row['model']] = row
    paired = [task for task in plan['tasks'] if all(by_task[task][model]['score'] is not None for model in models)]
    pairs = []
    summaries = []
    counts = {}
    for row in records:
        counts.setdefault(row['model'], Counter())[row['status']] += 1
    reference = models[0]
    labels = {model: label for label, model in plan['config'].get('candidate_models', {}).items()}
    panels = {model: [by_task[task][model]['score'] for task in paired] for model in models}
    for index, model in enumerate(models):
        observed = [row for row in records if row['model'] == model]
        model_attempts = [row for row in attempts if row['model'] == model]
        scored = [row for row in observed if row['score'] is not None]
        base_values = panels[model]
        passes = sum(base_values)
        lo, hi = wilson(passes, len(base_values))
        entry = dict(model=model, candidate_label='BF16' if model == reference else labels.get(model, model),
            paired_tasks=len(paired), paired_passes=passes, paired_pass_rate=passes/len(paired) if paired else None,
            pass_rate_wilson95_low=lo, pass_rate_wilson95_high=hi,
            all_scored_tasks=len(scored), all_scored_passes=sum(row['score'] for row in scored),
            status_counts=dict(counts[model]),
            attempts=len(model_attempts),
            input_tokens=sum(row['input_tokens'] or 0 for row in model_attempts),
            output_tokens=sum(row['output_tokens'] or 0 for row in model_attempts),
            output_tokens_per_agent_second=(sum(row['output_tokens'] or 0 for row in model_attempts)/
                sum(row['agent_seconds'] or 0 for row in model_attempts)
                if sum(row['agent_seconds'] or 0 for row in model_attempts) else None),
            tool_calls=sum(row['tool_calls'] or 0 for row in model_attempts),
            agent_seconds=sum(row['agent_seconds'] or 0 for row in model_attempts),
            trial_wall_p50_seconds=float(np.median([row['wall_seconds'] for row in model_attempts if row['wall_seconds'] is not None]))
                if any(row['wall_seconds'] is not None for row in model_attempts) else None,
            trial_wall_p95_seconds=float(np.quantile([row['wall_seconds'] for row in model_attempts if row['wall_seconds'] is not None],.95))
                if any(row['wall_seconds'] is not None for row in model_attempts) else None,
            mtp_drafts=sum(row['mtp_drafts'] or 0 for row in model_attempts),
            mtp_draft_tokens=sum(row['mtp_draft_tokens'] or 0 for row in model_attempts),
            mtp_accepted_tokens=sum(row['mtp_accepted_tokens'] or 0 for row in model_attempts))
        entry['mtp_token_acceptance_rate'] = (entry['mtp_accepted_tokens']/entry['mtp_draft_tokens']
            if entry['mtp_draft_tokens'] else None)
        if model != reference:
            base_rate = summaries[0]['paired_pass_rate'] if summaries else None
            entry['retention_vs_bf16'] = entry['paired_pass_rate']/base_rate if base_rate else None
            entry['pass_rate_delta_vs_bf16'] = (entry['paired_pass_rate']-base_rate) if base_rate is not None and entry['paired_pass_rate'] is not None else None
        else:
            entry['retention_vs_bf16'] = 1.0 if entry['paired_pass_rate'] is not None else None
            entry['pass_rate_delta_vs_bf16'] = 0.0 if entry['paired_pass_rate'] is not None else None
        summaries.append(entry)
    for idx, model in enumerate(models[1:]):
        base = panels[reference]
        candidate = panels[model]
        for task, a, b in zip(paired, base, candidate):
            pairs.append(dict(task=task, candidate=model, reference=reference,
                reference_score=a, candidate_score=b, delta=b-a,
                harmful_flip=int(a == 1 and b == 0), recovery_flip=int(a == 0 and b == 1)))
        estimates = paired_bootstrap(base, candidate, int(plan['config']['bootstrap_seed'])+idx,
            int(plan['config']['bootstrap_resamples']), int(plan['config']['comparison_family']))
        for row in estimates:
            row.update(candidate=model, reference=reference)
            yield_pair = row
            # Keep the interval family in its own tidy table.
            args._intervals.append(yield_pair)

    gate_counts = Counter(row['status'] for row in validation['tasks'].values())
    slot_counts = Counter(row['status'] for row in records)
    gate_final = not any(status in OPEN for status in gate_counts)
    slots_final = gate_final and not any(status in OPEN for status in slot_counts)
    all_evaluable_completed = all(by_task[task][model]['status'] in SCORED for task in plan['tasks']
        if validation['tasks'][task]['status'] == 'CONTROLS_PASS' for model in models)
    resources = resource_analysis(run, records, attempts)
    delays = gate_delay(run, validation, records)
    failed_gates = gate_failures(validation)
    integrity = integrity_audit(run, plan, validation, queue_rows, records)
    write_csv(output/'slots.csv', records, list(records[0]))
    write_csv(output/'attempts.csv', attempts, list(attempts[0]) if attempts else [])
    write_csv(output/'paired_task_differences.csv', pairs,
        ['task','candidate','reference','reference_score','candidate_score','delta','harmful_flip','recovery_flip'])
    write_csv(output/'paired_intervals.csv', args._intervals,
        ['candidate','reference','metric','estimate','low','high','confidence','paired_tasks','resamples','seed'])
    write_csv(output/'gate_release_delays.csv', delays['rows'], ['task','gate_to_first_trial_seconds'])
    write_csv(output/'gate_failures.csv', failed_gates, list(failed_gates[0]) if failed_gates else ['task','status','attempt'])
    (output/'environment_validation.json').write_text(json.dumps(dict(denominator=89,counts=dict(gate_counts),tasks=validation['tasks']),indent=2,ensure_ascii=False)+'\n')
    (output/'integrity_audit.json').write_text(json.dumps(integrity,indent=2,ensure_ascii=False)+'\n')
    (output/'resources_and_services.json').write_text(json.dumps(resources,indent=2,ensure_ascii=False)+'\n')
    summary = dict(protocol=plan['config'], expected_tasks=89, planned_matrix_slots=267,
        environment_counts=dict(gate_counts), slot_status_counts=dict(slot_counts),
        paired_task_count=len(paired), paired_tasks=paired, model_summary=summaries,
        pairwise_intervals=args._intervals, gate_release_delays=delays, resources=resources,
        analysis_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        runner_audit=runner_audit,
        all_environment_gates_conclusive=gate_final, all_slots_conclusive=slots_final,
        all_evaluable_trials_scored=all_evaluable_completed,
        integrity_audit=integrity,
        analyzed_at=time.time(), infrastructure_failure_counted_in_score=False,
        native_binary_reward_preferred=True, timeout_without_reward_scored_zero=True,
        denominator_note='Primary ability estimates use the intersection of tasks with a passed Oracle/NOP gate and a scored PASS, MODEL_FAIL, or TIMEOUT result for all three models. Infrastructure or verifier failures remain visible in slots.csv and status counts, and remove that task from every model in the paired estimate.')
    (output/'analysis.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    report = ['# Terminal-Bench 2.1 Low/MTP3 formal analysis', '',
        f"Run: `{run}`", '', 'The primary paired denominator contains tasks with a passing Oracle/NOP gate and one scored result for all three models. Scores use the binary native Verifier reward when Harbor saved one; a TIMEOUT with no reward scores zero. Timeout status remains in the failure distribution even when a Verifier reward exists. Infrastructure and verifier failures stay in the slot ledger and are excluded from every model in the paired estimate.', '',
        f"- Planned slots: {len(plan['tasks'])*len(models)}; conclusive environment gates: {sum(count for status,count in gate_counts.items() if status not in OPEN)}/89.",
        f"- Paired task count: {len(paired)}/89.",
        f"- All gates conclusive: {gate_final}; all slots conclusive: {slots_final}; all evaluable Trials scored: {all_evaluable_completed}.",
        f"- Scored-result, reward, API-audit, and PRoot-cleanup integrity checks: {'PASS' if integrity['passed'] else 'FAIL'} ({len(integrity['issues'])} issues).",
        f"- Five-GPU service time: {resources['gpu_hours']:.2f} GPU-hours across {resources['model_loads']} model loads; measured wall time {resources['run_wall_hours']:.2f} hours." if resources['run_wall_hours'] else '- GPU hours or wall time are unavailable.', '',
        '| Model | Paired passes | Paired pass rate | Retention vs BF16 | Delta vs BF16 | Scored tasks |',
        '|---|---:|---:|---:|---:|---:|']
    for row in summaries:
        report.append(f"| {row['candidate_label']} | {row['paired_passes']}/{row['paired_tasks']} | {fmt(row['paired_pass_rate'])} | {fmt(row['retention_vs_bf16'])} | {fmt(row['pass_rate_delta_vs_bf16'])} | {row['all_scored_tasks']} |")
    statuses = sorted(slot_counts)
    report.extend(['', '### Final status distribution', '',
        '| Model | ' + ' | '.join(statuses) + ' |',
        '|---|' + '---:|'*len(statuses)])
    for model in models:
        report.append('| ' + model + ' | ' + ' | '.join(str(counts[model].get(status, 0)) for status in statuses) + ' |')
    report.extend(['', '### Inference and scheduling cost', '',
        '| Model | Input tokens | Output tokens | Output tok/agent s | Tool calls | Agent hours | GPU hours | MTP accepted/draft tokens |',
        '|---|---:|---:|---:|---:|---:|---:|---:|'])
    for row in summaries:
        rate = row['mtp_token_acceptance_rate']
        report.append(f"| {row['candidate_label']} | {row['input_tokens']} | {row['output_tokens']} | {fmt(row['output_tokens_per_agent_second'])} | {row['tool_calls']} | {row['agent_seconds']/3600:.2f} | {resources['model_gpu_hours'].get(row['model'],0):.2f} | {row['mtp_accepted_tokens']}/{row['mtp_draft_tokens']} ({fmt(rate)}) |")
    report.extend(['', 'Paired bootstrap intervals compare each candidate with BF16 over whole tasks. The table reports the 95% interval and the family-adjusted interval specified by the fixed protocol. See `paired_intervals.csv` for estimates and bounds.', '',
        '| Candidate | Metric | Estimate | 95% interval | Family-adjusted interval |', '|---|---|---:|---:|---:|'])
    for model in models[1:]:
        for metric in ('accuracy_delta','harmful_flip','recovery_flip'):
            found = [r for r in args._intervals if r['candidate'] == model and r['metric'] == metric]
            a = next((r for r in found if r['confidence'] == .95), None)
            b = next((r for r in found if r['confidence'] != .95), None)
            if a and b:
                report.append(f"| {model} | {metric} | {fmt(a['estimate'])} | [{fmt(a['low'])}, {fmt(a['high'])}] | [{fmt(b['low'])}, {fmt(b['high'])}] |")
    report.extend(['', '## Status and cost details', '',
        f"Gate outcomes: `{json.dumps(dict(gate_counts), ensure_ascii=False)}`. GPU utilization is the mean of {resources['resource_samples']} periodic samples across the run. The model-affinity scheduler released each passed task without a matrix-wide barrier; measured gate-to-first-Trial delay was p50 {fmt(delays['p50_seconds'])} s, p90 {fmt(delays['p90_seconds'])} s.", '',
        'See `environment_validation.json` for all 89 gate decisions, `gate_failures.csv` for verifier and cleanup evidence on common non-evaluable tasks, `integrity_audit.json` for scored-result checks, `slots.csv` for final model/task states, `attempts.csv` for every queued Trial attempt, `paired_task_differences.csv` for exact paired outcomes, and `resources_and_services.json` for GPU sampling, service loads, resource pressure, MTP totals, token use, and tool-call totals.', '',
        'PRoot uses shared kernel, PID, and network resources. This is an internal comparison under the fixed shared PRoot environment, not an official Docker leaderboard score.', ''])
    schedule = runner_audit.get('schedule_ab', {})
    tests = runner_audit.get('targeted_tests', {})
    if schedule:
        report.insert(-1, f"Runner A/B evidence: {schedule.get('live_queue_makespan_seconds')} s for the deterministic live-release simulation vs {schedule.get('batch_barrier_makespan_seconds')} s with a batch barrier. This is a semantic simulation, not measured GPU speed. Same-model service batching used {schedule.get('gpu_workers')} GPU workers. Targeted regression: {tests.get('passed')} passed, {tests.get('failed')} failed.")
    (output/'REPORT.md').write_text('\n'.join(report))
    print(json.dumps(dict(output=str(output), paired_tasks=len(paired), gates=gate_counts,
        slots=slot_counts, all_gates_conclusive=gate_final, all_slots_conclusive=slots_final,
        all_evaluable_trials_scored=all_evaluable_completed),ensure_ascii=False))


def fmt(value):
    return 'NA' if value is None else f'{value:.4f}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args._intervals = []
    analyze(args)


if __name__ == '__main__':
    main()
