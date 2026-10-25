from collections import Counter
import json
from pathlib import Path
import shutil
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))

import terminal_formal_scheduler as formal
from terminal_formal_scheduler import (GATE_FINAL, MODELS, SERVICE_BATCH,
    enqueue_infrastructure_retries, matrix_snapshot, read_mtp_metrics,
    save_mtp_metrics, save_source_snapshot, set_gate_state, sync_coverage)
from terminal_queue import Queue


def test_formal_matrix_has_one_primary_cell_per_model_and_task():
    assert len(MODELS) == 3
    assert SERVICE_BATCH == 3
    assert 89*len(MODELS) == 267
    assert 'CONTROLS_PASS' in GATE_FINAL


def test_gate_failure_is_shared_across_all_models():
    validation = {'tasks': {
        'ready': {'status': 'CONTROLS_PASS', 'attempts': [], 'attempt': 0},
        'bad-proot': {'status': 'PROOT_INCOMPATIBLE', 'attempts': [], 'attempt': 0},
    }}
    matrix = matrix_snapshot(['ready', 'bad-proot'], MODELS, validation, [])
    assert matrix['planned_slots'] == 6
    assert {slot['status'] for slot in matrix['tasks']['bad-proot']['slots'].values()} == {'PROOT_INCOMPATIBLE'}
    assert {slot['status'] for slot in matrix['tasks']['ready']['slots'].values()} == {'PENDING'}


def test_passed_gate_enqueues_immediately_and_deduplicates(tmp_path):
    queue = Queue(tmp_path/'matrix.sqlite', {'schema': 1})
    validation = {'tasks': {'new-task': {'attempts': [], 'attempt': 0, 'status': 'VALIDATING'}}}
    coverage = {'tasks': {'new-task': {}}}
    result = dict(task='new-task', attempt=0, status='CONTROLS_PASS', retryable=False)
    assert not set_gate_state('new-task', result, validation, coverage)
    queue.enqueue(MODELS, ['new-task'], 42, modes=[False])
    queue.enqueue(MODELS, ['new-task'], 42, modes=[False])
    rows = queue.snapshot()
    assert len(rows) == 3
    assert {row['status'] for row in rows} == {'PENDING'}
    assert {row['task'] for row in rows} == {'new-task'}


def test_matrix_accepts_later_gate_results_without_rebuilding_queue(tmp_path):
    queue = Queue(tmp_path/'matrix.sqlite', {'schema': 1})
    queue.enqueue(MODELS, ['early'], 42, modes=[False])
    first = queue.claim(0, 101, 'a', list(MODELS))
    assert first is not None and first['status'] == 'RUNNING'
    queue.finish(first['id'], 'PASS', 'early/result.json')
    queue.enqueue(MODELS, ['later'], 42, modes=[False])
    assert sum(row['status'] == 'PENDING' for row in queue.snapshot()) == 5
    next_row = queue.claim(0, 101, 'a', list(MODELS))
    assert next_row is not None and next_row['task'] in {'early', 'later'}


def test_five_gpu_claims_balance_models_and_cap_loaded_batches(tmp_path):
    queue = Queue(tmp_path/'matrix.sqlite', {'schema': 1})
    queue.enqueue(MODELS, [f'task-{index}' for index in range(8)], 42, modes=[False])
    first = [queue.claim(gpu, 100+gpu, str(gpu), list(MODELS)) for gpu in range(5)]
    model_counts = Counter(row['model'] for row in first)
    assert set(model_counts) == set(MODELS)
    assert max(model_counts.values())-min(model_counts.values()) <= 1

    service_trials = 0
    service_loads = 0
    for gpu, first_row in enumerate(first):
        service_loads += 1
        model = first_row['model']
        row = first_row
        for _ in range(SERVICE_BATCH):
            queue.finish(row['id'], 'PASS', 'result.json')
            service_trials += 1
            if service_trials >= 15:
                break
            row = queue.claim(gpu, 100+gpu, str(gpu), [model], preferred=model)
            if row is None:
                break
    assert service_trials == 15
    assert service_loads == 5
    assert service_loads < service_trials


def test_retry_attempt_is_distinct_and_does_not_overwrite_failure(tmp_path):
    queue = Queue(tmp_path/'matrix.sqlite', {'schema': 1})
    queue.enqueue(['bf16'], ['task'], 42, modes=[False])
    first = queue.claim(0, 10, 'one', ['bf16'])
    queue.finish(first['id'], 'INTERRUPTED', 'attempt0/interrupted.json')
    queue.enqueue(['bf16'], ['task'], 42, modes=[False], attempt=1)
    rows = queue.snapshot()
    assert len(rows) == 2
    assert [(row['attempt'], row['status']) for row in rows] == [(0, 'INTERRUPTED'), (1, 'PENDING')]
    with pytest.raises(RuntimeError, match='overwritten'):
        queue.finish(first['id'], 'PASS', 'replacement.json')


def test_only_clean_infrastructure_failures_get_one_retry(tmp_path):
    run = tmp_path/'run'
    run.mkdir()
    queue = Queue(run/'matrix.sqlite', {'schema': 1})
    tasks = ['api', 'model-fail', 'timeout', 'dirty']
    for task in tasks:
        queue.enqueue(['bf16'], [task], 42, modes=[False])
    statuses = {'api': 'API_FAILURE', 'model-fail': 'MODEL_FAIL',
                'timeout': 'TIMEOUT', 'dirty': 'ENVIRONMENT_FAIL'}
    for _ in tasks:
        row = queue.claim(0, 10, str(_), ['bf16'])
        cleanup = run/'trials'/row['id']/'trial/proot/cleanup.json'
        cleanup.parent.mkdir(parents=True)
        cleanup.write_text('{"after": []}' if row['task'] != 'dirty' else '{"after": [{"pid": 99}]}')
        queue.finish(row['id'], statuses[row['task']], 'failure.json')
    added = enqueue_infrastructure_retries(queue, run)
    assert {row['task'] for row in added} == {'api'}
    attempts = Counter((row['task'], row['attempt']) for row in queue.snapshot())
    assert attempts[('api', 1)] == 1
    assert attempts[('model-fail', 1)] == attempts[('timeout', 1)] == attempts[('dirty', 1)] == 0


def test_clean_pre_agent_tmux_failure_with_native_result_gets_one_retry(tmp_path):
    import json

    run = tmp_path/'run'
    run.mkdir()
    queue = Queue(run/'matrix.sqlite', {'schema': 1})
    queue.enqueue(['bf16'], ['task'], 42, modes=[False])
    row = queue.claim(0, 10, 'one', ['bf16'])
    trial = run/'trials'/row['id']/'trial'
    cleanup = trial/'proot/cleanup.json'
    cleanup.parent.mkdir(parents=True)
    cleanup.write_text('{"after": []}')
    result = trial/'result.json'
    result.parent.mkdir(parents=True, exist_ok=True)
    result.write_text(json.dumps({
        'finished_at': '2026-10-22T00:00:00Z',
        'exception_info': {
            'exception_type': 'RuntimeError',
            'exception_message': 'Failed to start tmux session. Error: tmux: command not found',
        },
        'agent_result': None,
        'agent_execution': None,
    }))
    queue.finish(row['id'], 'ENVIRONMENT_FAIL', str(result))

    added = enqueue_infrastructure_retries(queue, run)

    assert len(added) == 1
    assert added[0]['task'] == 'task' and added[0]['attempt'] == 1
    assert [(r['attempt'], r['status']) for r in queue.snapshot()] == [
        (0, 'ENVIRONMENT_FAIL'), (1, 'PENDING')]
    review = json.loads((run/'retry-review'/f"{row['id']}.json").read_text())
    assert review['status'] == 'retry_enqueued'
    assert enqueue_infrastructure_retries(queue, run) == []
    assert len(queue.snapshot()) == 2


def test_completed_model_generated_environment_error_is_not_retried(tmp_path):
    import json

    run = tmp_path/'run'
    run.mkdir()
    queue = Queue(run/'matrix.sqlite', {'schema': 1})
    queue.enqueue(['bf16'], ['task'], 42, modes=[False])
    row = queue.claim(0, 10, 'one', ['bf16'])
    trial = run/'trials'/row['id']/'trial'
    cleanup = trial/'proot/cleanup.json'
    cleanup.parent.mkdir(parents=True)
    cleanup.write_text('{"after": []}')
    result = trial/'result.json'
    result.write_text(json.dumps({
        'finished_at': '2026-10-22T00:00:00Z',
        'exception_info': {
            'exception_type': 'ValueError',
            'exception_message': 'embedded null byte',
        },
        'agent_result': {'n_output_tokens': 10},
        'agent_execution': {'started_at': '2026-10-22T00:00:00Z'},
    }))
    queue.finish(row['id'], 'ENVIRONMENT_FAIL', str(result))

    added = enqueue_infrastructure_retries(queue, run)

    assert added == []
    assert len(queue.snapshot()) == 1


def test_mtp_metrics_are_observational_and_keep_scrape_errors(tmp_path):
    values = {'vllm:spec_decode_num_accepted_tokens_total': 4.0}
    before, before_error = read_mtp_metrics(lambda gpu: values, 0)
    assert before == values and before_error is None

    def unavailable(gpu):
        raise OSError('metrics endpoint unavailable')

    missing, error = read_mtp_metrics(unavailable, 0)
    assert missing is None and 'metrics endpoint unavailable' in error
    folder = tmp_path/'trial'
    save_mtp_metrics(folder, 0, before, before_error, unavailable)
    record = json.loads((folder/'mtp.json').read_text())
    assert record['before'] == values and record['after'] is None
    assert record['delta'] is None
    assert 'metrics endpoint unavailable' in record['after_error']


def test_resume_restores_formal_gate_results_to_coverage():
    validation = {'tasks': {
        'ready': {'status': 'CONTROLS_PASS', 'attempts': [
            {'status': 'CONTROLS_PASS', 'controls': {'oracle': 'oracle-result'}}]},
        'pending': {'status': 'ENV_PENDING', 'attempts': []},
    }}
    coverage = {'tasks': {'ready': {'status': 'UNVALIDATED'},
                          'pending': {'status': 'UNVALIDATED'}}}
    sync_coverage(validation, coverage)
    assert coverage['tasks']['ready']['status'] == 'CONTROLS_PASS'
    assert coverage['tasks']['ready']['controls'] == {'oracle': 'oracle-result'}
    assert coverage['tasks']['pending']['status'] == 'UNVALIDATED'


def test_source_snapshot_creates_prior_evidence_directory(tmp_path):
    run_dir = formal.ROOT/'build/terminal_bench'/('snapshot-test-'+tmp_path.name)
    try:
        snapshot = save_source_snapshot(run_dir)
        copied = run_dir/'source_snapshot/prior_evidence/GOAL.json'
        assert copied.read_bytes() == (formal.LOW_OUTPUT/'GOAL.json').read_bytes()
        assert snapshot['prior_evidence']['GOAL.json']
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


def test_live_gate_scheduling_overlaps_cpu_and_gpu_work():
    """A small A/B simulation: live queue release overlaps work with later gates."""
    gate_ready = [0]*5 + [60]*84
    trial_seconds = 10
    workers = 5

    def makespan(barrier):
        jobs = []
        for gate in gate_ready:
            ready = max(gate_ready) if barrier else gate
            jobs.extend([(ready, trial_seconds)]*len(MODELS))
        remaining = sorted(jobs)
        running = []
        now = 0
        while remaining or running:
            while len(running) < workers and remaining and remaining[0][0] <= now:
                _, duration = remaining.pop(0)
                running.append(now+duration)
            if running:
                now = min(running)
                running = [end for end in running if end != now]
            elif remaining:
                now = remaining[0][0]
        return now

    dynamic = makespan(False)
    batch_barrier = makespan(True)
    assert dynamic < batch_barrier
    assert dynamic == 570 and batch_barrier == 600
