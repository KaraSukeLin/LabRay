import sys
import csv
import json
import sqlite3
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))

from terminal_formal_multiseed_analysis import aggregate, clustered_bootstrap


def test_cluster_bootstrap_keeps_seed_repeats_within_task():
    base = np.asarray([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
    rows = clustered_bootstrap(base, base.copy(), seed=7, resamples=300, family=6)
    by_metric = {row['metric']: row for row in rows}
    assert by_metric['accuracy_delta']['estimate'] == 0
    assert by_metric['harmful_flip']['estimate'] == 0
    assert by_metric['recovery_flip']['estimate'] == 0
    assert by_metric['retention_ratio']['estimate'] == 1
    assert all(row['paired_tasks'] == 3 and row['task_seed_observations'] == 9 for row in rows)


def test_cluster_bootstrap_reports_paired_flip_rates():
    base = np.asarray([[1, 0, 1], [1, 0, 1]])
    candidate = np.asarray([[0, 0, 1], [0, 1, 1]])
    rows = clustered_bootstrap(base, candidate, seed=8, resamples=300)
    by_metric = {row['metric']: row for row in rows}
    assert by_metric['accuracy_delta']['estimate'] == -1/6
    assert by_metric['harmful_flip']['estimate'] == 1/3
    assert by_metric['recovery_flip']['estimate'] == 1/6
    assert by_metric['retention_ratio']['estimate'] == 0.75


def test_aggregate_writes_all_801_seed_model_task_slots(tmp_path):
    models = ['bf16', 'fast', 'aqua']
    tasks = [f'task-{index:02d}' for index in range(89)]
    seeds = [20261020, 20261021, 20261022]
    runs = []
    for seed in seeds:
        run = tmp_path/f'seed-{seed}'
        run.mkdir()
        runs.append(run)
        config = dict(bootstrap_seed=11, bootstrap_resamples=300, comparison_family=6,
            candidate_models={'NaviQuant': 'fast', 'Aqua': 'aqua'})
        (run/'PLAN.json').write_text(json.dumps(dict(seed=seed, planned_slots=267,
            tasks=tasks, models=models, config=config)))
        validation = {'tasks': {task: {'status': 'CONTROLS_PASS'} for task in tasks}}
        (run/'environment_validation.json').write_text(json.dumps(validation))
        (run/'trials').mkdir()
        con = sqlite3.connect(run/'matrix.sqlite')
        con.execute('''CREATE TABLE trials (
            id TEXT, model TEXT, task TEXT, mode INTEGER, seed INTEGER, attempt INTEGER,
            status TEXT, gpu INTEGER, started REAL, finished REAL, result TEXT)''')
        for task in tasks:
            for gpu, model in enumerate(models):
                trial_id = f'{seed}-{task}-{model}'
                folder = run/'trials'/trial_id
                native = folder/'trial'
                (native/'proot').mkdir(parents=True)
                (native/'proot/cleanup.json').write_text('{"after": []}')
                result_path = native/'result.json'
                result_path.write_text(json.dumps(dict(task_name=task,
                    started_at='2026-10-22T00:00:00+00:00',
                    finished_at='2026-10-22T00:00:01+00:00',
                    verifier_result={'rewards': {'reward': 1}},
                    agent_result={'cost_usd': 0.01 if model == 'bf16' else None})))
                identity = f'formal/{model}/{task}/{trial_id}'
                (folder/'summary.json').write_text(json.dumps(dict(status='PASS',
                    cleanup_clean=True, api_audit={'identity': identity,
                        'requests': 1, 'complete_requests': 1, 'incomplete_requests': 0})))
                con.execute('INSERT INTO trials VALUES (?,?,?,?,?,?,?,?,?,?,?)',
                    (trial_id, model, task, 0, seed, 0, 'PASS', gpu, 1.0, 2.0, str(result_path)))
        con.commit()
        con.close()
    protocol = tmp_path/'protocol.json'
    protocol.write_text(json.dumps(dict(matrix={'seeds': seeds}, protocol_config={
        'candidate_models': {'NaviQuant': 'fast', 'Aqua': 'aqua'}})))
    output = tmp_path/'analysis'
    final = aggregate(runs, protocol, output)
    assert final['expected_slots'] == final['observed_slots'] == 801
    assert final['paired_task_count'] == 89
    assert final['slot_status_counts'] == {'PASS': 801}
    assert all(row['all_scored_pass_rate'] == 1 for row in final['model_summary'])
    assert all(row['seed_pass_rate_sd'] == 0 for row in final['model_summary'])
    assert all(row['mean_pass_rate_cluster95']['low'] == row['mean_pass_rate_cluster95']['high'] == 1
        for row in final['model_summary'])
    assert final['model_summary'][1]['mean_seed_retention_vs_bf16'] == 1
    costs = {row['model']: row['cost_usd'] for row in final['model_summary']}
    assert round(costs['bf16'], 2) == 2.67
    assert costs['fast'] is None and costs['aqua'] is None
    assert json.loads((output/'integrity_audit.json').read_text())['passed']
    assert len((output/'matrix_801.csv').read_text().splitlines()) == 802
    csv_costs = {row['model']: row['cost_usd']
        for row in csv.DictReader((output/'model_summary.csv').open())}
    assert csv_costs['fast'] == csv_costs['aqua'] == ''
    report = (output/'REPORT.md').read_text()
    assert '| NA |' in report
    assert 'USD cost is `NA` when any attempt has no reported amount' in report
