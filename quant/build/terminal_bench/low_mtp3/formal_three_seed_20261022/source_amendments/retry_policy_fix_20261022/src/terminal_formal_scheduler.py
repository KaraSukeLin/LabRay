"""Run the authorized 89-task Low/MTP3 matrix from live PRoot gates.

Environment controls and model trials progress together. A task is made
claimable only after its original Oracle/NOP/Verifier controls pass.
"""
import argparse
import asyncio
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback
from unittest.mock import patch

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))

import terminal_bench as serving
from terminal_bench import ROOT, read, save, sha, prepare
from terminal_low import CONFIG, OUTPUT as LOW_OUTPUT, classify, control
from terminal_queue import Queue
from terminal_smoke_scheduler import identity as smoke_identity

RUN_DEFAULT = LOW_OUTPUT/'formal_20261021'
MODELS = ('Qwen-Qwen3.8-27B', 'eNeural-Qwen3.8-27B-fast',
          'rdtand-Qwen3.8-27B-PrismaAQUA-5.5bit-vllm')
GATE_FINAL = {'CONTROLS_PASS', 'PROOT_INCOMPATIBLE', 'CONTROL_TIMEOUT',
              'ORACLE_CONTROL_FAIL', 'SCORING_GATE_FAIL', 'CONTROL_INFRA_FAILURE', 'IMAGE_PREP_FAILED'}
SERVICE_BATCH = 3
IDLE_POLL_SECONDS = 5


class ProtocolMismatch(RuntimeError):
    pass


def formal_identity():
    cfg = read(CONFIG)
    base = smoke_identity()
    return dict(schema=1, base=base, primary_mode=False,
        seed=cfg['seeds'][0], service_batch=SERVICE_BATCH,
        runner_sha256=sha(Path(__file__)))


def verify_smoke_and_models():
    cfg = read(CONFIG)
    assert cfg['models'] == list(MODELS)
    assert cfg['interleaved_thinking'] is False and cfg['mtp']
    assert cfg['num_speculative_tokens'] == 3 and cfg['kv_cache_dtype'] == 'bfloat16'
    assert cfg['reasoning_effort'] == 'low' and cfg['timeout_multiplier'] == 1.0
    summary = read(LOW_OUTPUT/'SMOKE_SUMMARY.json')
    audit = read(LOW_OUTPUT/'SMOKE_AUDIT.json')
    completion = read(LOW_OUTPUT/'COMPLETION_AUDIT.json')
    assert summary['final'] and summary['completed'] == 30 and summary['formal_trials'] == 0
    assert sum(summary['status_counts'].values()) == 30
    assert audit['passed'] and completion['passed'] and completion['smoke_trials'] == 30
    models = []
    for model in MODELS:
        folder = LOW_OUTPUT/'preflight'/model
        assert read(folder/'AUDIT.json')['passed'], model
        runtime = read(folder/'RUNTIME.json')
        assert runtime['passed'] and all(row['mtp'] == 3 and row['tp'] == 1
            and row['kv_dtype'] == 'bfloat16' for row in runtime['services']), model
        services = read(folder/'services/services.json')
        assert services and all(row.get('stopped_at') for row in services), model
        models.append(model)
    return dict(smoke_trials=30, smoke_status_counts=summary['status_counts'],
        preflight_models=models, nonstandard_environment='PRoot; internal comparison only')


def verify_control(task, agent, row):
    """Check native result, official verifier, fixed timeout and clean PRoot exit."""
    assert row and row.get('result'), f'{task}/{agent}: missing saved result'
    result_path = Path(row['result'])
    native = read(result_path)
    assert native.get('finished_at')
    assert native['task_name'].split('/')[-1] == task
    config = native['config']
    assert config['agent']['name'] == agent
    assert Path(config['task']['path']).name == task
    assert config['timeout_multiplier'] == 1.0
    assert config['environment']['import_path'] == 'terminal_proot:PRootEnvironment'
    reward = (native.get('verifier_result') or {}).get('rewards', {}).get('reward')
    cleanup_path = result_path.parent/'proot/cleanup.json'
    cleanup = read(cleanup_path) if cleanup_path.exists() else None
    clean = cleanup is not None and cleanup.get('after') == [] and row.get('cleanup_clean')
    error = native.get('exception_info') or {}
    if error or not clean:
        return dict(agent=agent, status=row['status'], reward=reward,
            exception=error, cleanup_clean=bool(clean), result=str(result_path))
    ctrf_path = result_path.parent/'verifier/ctrf.json'
    ctrf = read(ctrf_path)['results']['summary']
    assert ctrf['tests'] > 0
    if agent == 'oracle' and row['status'] == 'PASS':
        assert ctrf.get('failed', 0) == 0
    if agent == 'nop' and row['status'] == 'MODEL_FAIL':
        assert ctrf.get('failed', 0) > 0
    return dict(agent=agent, status=row['status'], reward=reward,
        verifier_tests=ctrf['tests'], verifier_failed=ctrf.get('failed', 0),
        cleanup_clean=True, exception=error, result=str(result_path))


def proot_failure(rows):
    pattern = re.compile(r'proot|rootfs|ptrace|mount namespace|user namespace|operation not permitted', re.I)
    for row in rows.values():
        error = row.get('exception') or {}
        text = ' '.join([str(error.get('exception_type', '')), str(error.get('exception_message', ''))])
        if pattern.search(text):
            return True
        if row.get('cleanup_clean') is False:
            return True
    return False


def validate_one(task, attempt):
    """Process worker: pinned image, then original Oracle and NOP controls."""
    from terminal_proot_images import prepare_image
    try:
        prepare_image(task)
    except Exception as exc:
        return dict(task=task, attempt=attempt, status='IMAGE_PREP_FAILED', retryable=True,
            error=type(exc).__name__+': '+str(exc), traceback=traceback.format_exc())
    rows = {}
    try:
        for agent in ('oracle', 'nop'):
            rows[agent] = asyncio.run(control(task, agent, attempt))
    except Exception as exc:
        return dict(task=task, attempt=attempt, status='CONTROL_INFRA_FAILURE', retryable=True,
            controls=rows, error=type(exc).__name__+': '+str(exc), traceback=traceback.format_exc())
    checked = {}
    try:
        for agent in ('oracle', 'nop'):
            checked[agent] = verify_control(task, agent, rows[agent])
    except Exception as exc:
        return dict(task=task, attempt=attempt, status='SCORING_GATE_FAIL', retryable=False,
            controls=rows, error=type(exc).__name__+': '+str(exc), traceback=traceback.format_exc())
    if any(not item.get('cleanup_clean', False) for item in checked.values()) and proot_failure(rows):
        status = 'PROOT_INCOMPATIBLE'
        retryable = False
    elif any(item['status'] in {'API_FAILURE', 'ENVIRONMENT_FAIL'} for item in checked.values()):
        status = 'CONTROL_INFRA_FAILURE'
        retryable = attempt == 0
    elif any(item['status'] == 'TIMEOUT' for item in checked.values()):
        status = 'CONTROL_TIMEOUT'
        retryable = False
    elif checked['oracle']['status'] != 'PASS' or checked['oracle']['reward'] != 1:
        status = 'PROOT_INCOMPATIBLE' if proot_failure(rows) else 'ORACLE_CONTROL_FAIL'
        retryable = False
    elif (checked['oracle']['status'] == 'PASS' and checked['oracle']['reward'] == 1
        and checked['nop']['status'] == 'MODEL_FAIL' and checked['nop']['reward'] == 0):
        return dict(task=task, attempt=attempt, status='CONTROLS_PASS', retryable=False,
            controls=rows, checked=checked)
    elif checked['nop']['status'] == 'VERIFIER_FAIL':
        status = 'SCORING_GATE_FAIL'
        retryable = False
    elif proot_failure(rows):
        status = 'PROOT_INCOMPATIBLE'
    else:
        status = 'SCORING_GATE_FAIL'
        retryable = False
    return dict(task=task, attempt=attempt, status=status, retryable=retryable,
        controls=rows, checked=checked)


def initial_validation(task_names, coverage, run_dir):
    path = run_dir/'environment_validation.json'
    if path.exists():
        value = read(path)
        assert value['tasks'] and set(value['tasks']) == set(task_names)
        return value
    tasks = {}
    for task in task_names:
        source = coverage['tasks'][task]
        if source['status'] == 'CONTROLS_PASS':
            result = validate_one_from_existing(task, source['controls'])
            tasks[task] = dict(status=result['status'], attempts=[result], attempt=0)
        else:
            tasks[task] = dict(status='ENV_PENDING', attempts=[], attempt=0)
    value = dict(denominator=len(task_names), completed=0, tasks=tasks)
    save(path, value)
    return value


def validate_one_from_existing(task, controls):
    try:
        checked = {agent: verify_control(task, agent, controls[agent]) for agent in ('oracle', 'nop')}
        passed = (checked['oracle']['status'] == 'PASS' and checked['oracle']['reward'] == 1
            and checked['nop']['status'] == 'MODEL_FAIL' and checked['nop']['reward'] == 0)
        return dict(task=task, attempt=0,
            status='CONTROLS_PASS' if passed else 'SCORING_GATE_FAIL', retryable=False,
            controls=controls, checked=checked)
    except Exception as exc:
        return dict(task=task, attempt=0, status='SCORING_GATE_FAIL', retryable=False,
            controls=controls, error=type(exc).__name__+': '+str(exc), traceback=traceback.format_exc())


def set_gate_state(task, result, validation, coverage):
    row = validation['tasks'][task]
    row['attempts'].append(result)
    row['attempt'] = result['attempt']
    if result.get('retryable') and result['attempt'] == 0:
        row['status'] = 'VALIDATING_RETRY'
        return True
    row['status'] = result['status']
    if task in coverage['tasks']:
        coverage['tasks'][task].update(status=result['status'], controls=result.get('controls'))
    return False


def sync_coverage(validation, coverage):
    """Restore the public gate ledger from the durable formal checkpoint."""
    for task, entry in validation['tasks'].items():
        if entry['status'] not in GATE_FINAL:
            continue
        attempts = entry.get('attempts', [])
        latest = attempts[-1] if attempts else {}
        coverage['tasks'][task].update(status=entry['status'],
            controls=latest.get('controls', coverage['tasks'][task].get('controls')))
    return coverage


def matrix_snapshot(task_names, models, validation, queue_rows):
    by_slot = defaultdict(list)
    for row in queue_rows:
        by_slot[(row['task'], row['model'])].append(row)
    tasks = {}
    for task in task_names:
        gate = validation['tasks'][task]['status']
        slots = {}
        for model in models:
            attempts = sorted(by_slot[(task, model)], key=lambda row: row['attempt'])
            if attempts:
                last = attempts[-1]
                slots[model] = dict(status=last['status'], attempts=attempts)
            elif gate == 'CONTROLS_PASS':
                slots[model] = dict(status='PENDING', attempts=[])
            else:
                slots[model] = dict(status=gate, attempts=[])
        tasks[task] = dict(environment_status=gate, slots=slots)
    counts = Counter(slot['status'] for task in tasks.values() for slot in task['slots'].values())
    return dict(planned_slots=len(task_names)*len(models), tasks=tasks,
        slot_status_counts=dict(counts), environment_status_counts=dict(Counter(
            row['status'] for row in validation['tasks'].values())))


def audit_trial_config(row, data):
    cfg = read(CONFIG)
    try:
        config = data['config']
        agent = config['agent']
        assert data['task_name'].split('/')[-1] == row['task']
        assert Path(config['task']['path']).name == row['task']
        assert agent['model_name'] == 'hosted_vllm/'+row['model']
        kwargs = agent['kwargs']
        assert kwargs['reasoning_effort'] == 'low'
        assert kwargs['interleaved_thinking'] is False
        assert kwargs['llm_call_kwargs']['seed'] == row['seed'] == cfg['seeds'][0]
        assert kwargs['llm_call_kwargs']['temperature'] == cfg['sampling']['temperature']
        assert kwargs['llm_call_kwargs']['top_p'] == cfg['sampling']['top_p']
        assert kwargs['llm_call_kwargs']['max_tokens'] == cfg['sampling']['max_tokens']
        assert kwargs['llm_call_kwargs']['presence_penalty'] == cfg['sampling']['presence_penalty']
        call = kwargs['llm_call_kwargs']
        extra = call['extra_body']
        assert extra['top_k'] == cfg['sampling']['top_k']
        assert extra['min_p'] == cfg['sampling']['min_p']
        assert extra['repetition_penalty'] == cfg['sampling']['repetition_penalty']
        assert extra['chat_template_kwargs'] == cfg['chat_template_kwargs']
        assert extra['return_token_ids'] is True
        assert config['timeout_multiplier'] == 1.0
        assert config['environment']['import_path'] == 'terminal_proot:PRootEnvironment'
    except (AssertionError, KeyError, TypeError) as exc:
        raise ProtocolMismatch(f'Trial config differs from frozen protocol: {exc}') from exc


def audit_wire_trial(audit_dir, row):
    cfg = read(CONFIG)
    identity = f'formal/{row["model"]}/{row["task"]}/{row["id"]}'
    events = []
    for path in Path(audit_dir).glob('*.json'):
        event = read(path)
        if event.get('identity', {}).get('x-terminal-identity') == identity:
            events.append((path, event))
    completed = 0
    for path, event in events:
        request = event.get('request') or {}
        response = event.get('response') or {}
        if event.get('model') != row['model'] or request.get('model') != row['model']:
            raise ProtocolMismatch(f'Wrong model in API audit: {path}')
        if request.get('reasoning_effort') != 'low' or request.get('chat_template_kwargs') != cfg['chat_template_kwargs']:
            raise ProtocolMismatch(f'Wrong thinking settings in API audit: {path}')
        if request.get('seed') != row['seed']:
            raise ProtocolMismatch(f'Wrong seed in API audit: {path}')
        if event.get('status') != 'complete' or not response.get('choices'):
            continue
        expected = dict(cfg['sampling'], seed=row['seed'])
        if event.get('engine_sampling') != expected:
            raise ProtocolMismatch(f'Engine sampling differs from protocol: {path}')
        if event.get('http_status') != 200 or not response.get('prompt_token_ids'):
            raise ProtocolMismatch(f'Incomplete successful API record: {path}')
        if not response['choices'][0].get('token_ids'):
            raise ProtocolMismatch(f'Missing output token IDs: {path}')
        completed += 1
    return dict(identity=identity, requests=len(events), complete_requests=completed,
        incomplete_requests=len(events)-completed)


def read_mtp_metrics(metrics_fn, gpu):
    """Metrics are observational; a scrape error must not suppress a Trial."""
    try:
        return metrics_fn(gpu), None
    except Exception as exc:
        return None, type(exc).__name__+': '+str(exc)


def save_mtp_metrics(folder, gpu, before, before_error, metrics_fn):
    after, after_error = read_mtp_metrics(metrics_fn, gpu)
    delta = None
    if before is not None and after is not None:
        delta = {key: after[key]-before[key] for key in before.keys() & after.keys()}
    save(folder/'mtp.json', dict(before=before, after=after, delta=delta,
        before_error=before_error, after_error=after_error))


def audit_service(service, model):
    argv = service['command']
    try:
        assert argv[argv.index('--tensor-parallel-size')+1] == '1'
        assert json.loads(argv[argv.index('--speculative-config')+1]) == dict(
            method='mtp', num_speculative_tokens=3)
        log = (Path(service['folder'])/'server.log').read_text()
        expected_quant = 'None' if model == MODELS[0] else 'compressed-tensors'
        assert 'num_spec_tokens=3' in log and 'tensor_parallel_size=1' in log
        assert 'kv_cache_dtype=bfloat16' in log and 'quantization='+expected_quant in log
    except (AssertionError, KeyError, ValueError, OSError) as exc:
        raise ProtocolMismatch(f'vLLM service differs from frozen protocol for {model}: {exc}') from exc


def pause_protocol(run_dir, queue, gpu, row, exc):
    folder = run_dir/'trials'/row['id']
    failure = folder/'protocol_failure.json'
    save(failure, dict(time=time.time(), gpu=gpu, model=row['model'], task=row['task'],
        attempt=row['attempt'], error=type(exc).__name__+': '+str(exc), traceback=traceback.format_exc()))
    current = next((item for item in queue.snapshot() if item['id'] == row['id']), None)
    if current and current['status'] == 'RUNNING':
        queue.finish(row['id'], 'PROTOCOL_FAIL', failure)
    pause_dir = run_dir/'PAUSE'
    pause_dir.mkdir(parents=True, exist_ok=True)
    save(pause_dir/f'gpu{gpu}-{row["id"]}.json', dict(
        time=time.time(), reason='major protocol mismatch; stop model trials', result=str(failure)))


def recover_complete_result(queue, run_dir, row):
    """Reuse a complete native result after interruption; never rerun it."""
    folder = run_dir/'trials'/row['id']
    result_path = folder/'trial/result.json'
    cleanup_path = folder/'trial/proot/cleanup.json'
    if not result_path.exists() or not cleanup_path.exists():
        return False
    data = read(result_path)
    if not data.get('finished_at') or read(cleanup_path).get('after') != []:
        return False
    audit_trial_config(row, data)
    info = classify(data, folder/'trial')
    info.update(identity=read(folder/'identity.json'), agent_execution=data.get('agent_execution'),
        native_reward=(data.get('verifier_result') or {}).get('rewards'), cleanup_clean=True,
        result=str(result_path))
    summary_path = folder/'summary.json'
    if summary_path.exists():
        old = read(summary_path)
        assert old['status'] == info['status'] and old['result'] == info['result']
    else:
        save(summary_path, info)
    queue.finish(row['id'], info['status'], result_path)
    return True


async def run_trial(row, run_dir, gpu, audit_dir):
    from harbor.models.trial.config import TrialConfig
    from harbor.trial.trial import Trial
    folder = run_dir/'trials'/row['id']
    folder.mkdir(parents=True, exist_ok=False)
    save(folder/'identity.json', row)
    cfg = read(CONFIG)
    agent = serving.agent_config(cfg, row['model'], gpu, row['seed'], False,
        f'formal/{row["model"]}/{row["task"]}/{row["id"]}')
    agent.pop('n_concurrent')
    trial_cfg = TrialConfig.model_validate(dict(
        task=dict(path=str(serving.SOURCE/'terminal-bench-2-1/tasks'/row['task'])),
        trial_name='trial', trials_dir=str(folder), agent=agent,
        environment=dict(type=None, import_path='terminal_proot:PRootEnvironment', delete=True),
        timeout_multiplier=1.0))
    instance = await Trial.create(trial_cfg)
    try:
        result = await instance.run()
        data = result.model_dump(mode='json')
        audit_trial_config(row, data)
        info = classify(data, folder/'trial')
        wire = audit_wire_trial(audit_dir, row)
        if info['status'] in {'PASS', 'MODEL_FAIL'} and wire['complete_requests'] == 0:
            raise ProtocolMismatch(f'No successful audited API call for scored trial {row["id"]}')
        cleanup_path = folder/'trial/proot/cleanup.json'
        cleanup = read(cleanup_path) if cleanup_path.exists() else None
        clean = cleanup is not None and cleanup.get('after') == []
        if cleanup is not None and not clean:
            info.update(status='ENVIRONMENT_FAIL', cleanup_failure=True)
        info.update(identity=row, agent_execution=data.get('agent_execution'),
            native_reward=(data.get('verifier_result') or {}).get('rewards'),
            cleanup_clean=clean, api_audit=wire, result=str(folder/'trial/result.json'))
        save(folder/'summary.json', info)
        return info
    finally:
        if instance.agent_environment.started:
            await instance.agent_environment.stop(True)


def update_worker_state(run_dir, gpu, **values):
    path = run_dir/'gpu_workers'/f'gpu{gpu}.json'
    current = read(path) if path.exists() else dict(gpu=gpu)
    current.update(time=time.time(), **values)
    save(path, current)


def worker(gpu, run_path):
    run_dir = Path(run_path)
    cfg = read(CONFIG)
    identity = read(run_dir/'PLAN.json')['queue_identity']
    queue = Queue(run_dir/'matrix.sqlite', identity)
    owner_start = Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19]
    import terminal_low_preflight as preflight
    update_worker_state(run_dir, gpu, pid=os.getpid(), owner_start=owner_start, status='IDLE')
    while True:
        if any((run_dir/'PAUSE').glob('*.json')):
            update_worker_state(run_dir, gpu, status='PAUSED_PROTOCOL')
            return
        row = queue.claim(gpu, os.getpid(), owner_start, cfg['models'], preferred=None)
        if row is None:
            rows = queue.snapshot()
            state = read(run_dir/'STATE.json')
            if state.get('environment_validation_complete') and not any(
                r['status'] in {'PENDING', 'RUNNING'} for r in rows):
                update_worker_state(run_dir, gpu, status='COMPLETE')
                return
            update_worker_state(run_dir, gpu, status='IDLE', pending=sum(r['status']=='PENDING' for r in rows),
                running=sum(r['status']=='RUNNING' for r in rows))
            time.sleep(IDLE_POLL_SECONDS)
            continue
        model = row['model']
        original = serving.server_command
        def command(config, checkpoint, index):
            return original(config, checkpoint, index)+['--speculative-config',
                json.dumps(dict(method='mtp', num_speculative_tokens=3))]
        service_dir = run_dir/'services'/f'gpu{gpu}'/str(time.time_ns())
        interrupted = False
        with patch.object(serving, 'server_command', command):
            try:
                with serving.services(cfg, model, [gpu], service_dir):
                    service = read(service_dir/'services.json')[-1]
                    audit_service(service, model)
                    for service_index in range(SERVICE_BATCH):
                        update_worker_state(run_dir, gpu, status='RUNNING', model=model,
                            trial_id=row['id'], task=row['task'], service_index=service_index+1)
                        folder = run_dir/'trials'/row['id']
                        try:
                            before, before_error = read_mtp_metrics(preflight.metrics, gpu)
                            info = asyncio.run(run_trial(row, run_dir, gpu,
                                Path(service['folder'])/'audit'))
                        except KeyboardInterrupt:
                            if not recover_complete_result(queue, run_dir, row):
                                save(folder/'interrupted.json', dict(time=time.time(), gpu=gpu, model=model,
                                    task=row['task'], attempt=row['attempt'],
                                    reason='worker interrupted; preserve original attempt'))
                                queue.finish(row['id'], 'INTERRUPTED', folder/'interrupted.json')
                            interrupted = True
                            break
                        except ProtocolMismatch as exc:
                            pause_protocol(run_dir, queue, gpu, row, exc)
                            update_worker_state(run_dir, gpu, status='PAUSED_PROTOCOL',
                                model=model, trial_id=row['id'])
                            interrupted = True
                            break
                        except httpx.HTTPError as exc:
                            failure = dict(time=time.time(), gpu=gpu, model=model, task=row['task'],
                                attempt=row['attempt'], error=type(exc).__name__+': '+str(exc),
                                traceback=traceback.format_exc())
                            save(folder/'infrastructure_failure.json', failure)
                            queue.finish(row['id'], 'API_FAILURE', folder/'infrastructure_failure.json')
                            print(gpu, row['id'], 'API_FAILURE', failure['error'], flush=True)
                        except Exception as exc:
                            failure = dict(time=time.time(), gpu=gpu, model=model, task=row['task'],
                                attempt=row['attempt'], error=type(exc).__name__+': '+str(exc),
                                traceback=traceback.format_exc())
                            save(folder/'infrastructure_failure.json', failure)
                            queue.finish(row['id'], 'ENVIRONMENT_FAIL', folder/'infrastructure_failure.json')
                            print(gpu, row['id'], 'ENVIRONMENT_FAIL', failure['error'], flush=True)
                        else:
                            queue.finish(row['id'], info['status'], info['result'])
                            print(gpu, row['id'], info['status'], flush=True)
                        finally:
                            save_mtp_metrics(folder, gpu, before, before_error,
                                preflight.metrics)
                        if service_index+1 >= SERVICE_BATCH:
                            break
                        row = queue.claim(gpu, os.getpid(), owner_start, [model], preferred=model)
                        if row is None:
                            break
                    update_worker_state(run_dir, gpu, status='IDLE', model=None, trial_id=None)
            except KeyboardInterrupt:
                # The services context stops its vLLM child and PRoot cleanup.
                if row and queue.snapshot():
                    current = next((item for item in queue.snapshot() if item['id']==row['id']), None)
                    if current and current['status']=='RUNNING':
                        if not recover_complete_result(queue, run_dir, row):
                            interrupted = run_dir/'trials'/row['id']/'interrupted.json'
                            save(interrupted, dict(time=time.time(), gpu=gpu, model=model,
                                task=row['task'], attempt=row['attempt'], reason='service interrupted'))
                            queue.finish(row['id'], 'INTERRUPTED', interrupted)
                update_worker_state(run_dir, gpu, status='INTERRUPTED')
                return
            except ProtocolMismatch as exc:
                pause_protocol(run_dir, queue, gpu, row, exc)
                update_worker_state(run_dir, gpu, status='PAUSED_PROTOCOL', model=model,
                    trial_id=row['id'])
                return
            except Exception as exc:
                failure = run_dir/'gpu_workers'/f'gpu{gpu}-failure-{time.time_ns()}.json'
                save(failure, dict(time=time.time(), gpu=gpu, model=model,
                    trial_id=row['id'] if row else None, error=type(exc).__name__+': '+str(exc),
                    traceback=traceback.format_exc()))
                current = next((item for item in queue.snapshot() if row and item['id']==row['id']), None)
                if current and current['status']=='RUNNING':
                    queue.finish(row['id'], 'ENVIRONMENT_FAIL', failure)
                update_worker_state(run_dir, gpu, status='SERVICE_FAILURE', error=str(exc))
                preferred = None
        if interrupted:
            return


def resource_sample(run_dir):
    sample = dict(gpus=serving.gpu_inventory(), load_average=Path('/proc/loadavg').read_text().strip(),
        disk_free_bytes=shutil.disk_usage(run_dir).free)
    for name in ('cpu', 'memory', 'io'):
        path = Path('/proc/pressure')/name
        if path.exists(): sample[name+'_pressure'] = path.read_text()
    for name in ('cpu.max', 'cpu.stat', 'memory.max', 'memory.current', 'memory.events', 'io.stat'):
        path = Path('/sys/fs/cgroup')/name
        if path.exists(): sample[name] = path.read_text()
    return sample


def write_progress(run_dir, task_names, cfg, validation, queue, validation_running, stage_started,
                   paused_protocol=False):
    rows = queue.snapshot()
    matrix = matrix_snapshot(task_names, cfg['models'], validation, rows)
    save(run_dir/'matrix.json', dict(updated_at=time.time(), **matrix))
    pending = sum(row['status']=='PENDING' for row in rows)
    running = sum(row['status']=='RUNNING' for row in rows)
    gate_counts = Counter(row['status'] for row in validation['tasks'].values())
    sample = resource_sample(run_dir)
    gpu_workers = {}
    for gpu in range(5):
        path = run_dir/'gpu_workers'/f'gpu{gpu}.json'
        gpu_workers[str(gpu)] = read(path) if path.exists() else dict(gpu=gpu, status='NOT_STARTED')
    state = dict(state='PAUSED_PROTOCOL' if paused_protocol else 'RUNNING_FORMAL',
        stage='environment_and_trials', started_at=stage_started,
        updated_at=time.time(), planned_slots=267, queue_rows=len(rows), pending_trials=pending,
        running_trials=running, terminal_trials=sum(r['status'] not in {'PENDING','RUNNING'} for r in rows),
        environment_validation_complete=not validation_running,
        environment_completed=sum(gate_counts[s] for s in GATE_FINAL),
        environment_counts=dict(gate_counts), slot_status_counts=matrix['slot_status_counts'],
        active_gpu_workers=gpu_workers, resources=sample)
    save(run_dir/'STATE.json', state)
    with (run_dir/'resources.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(time=state['updated_at'], **sample))+'\n')
    update_goal_records(run_dir, state)
    return state


def update_goal_records(run_dir, state):
    """Keep formal progress beside the preserved Smoke evidence."""
    snapshot = dict(state, run_directory=str(run_dir))
    save(LOW_OUTPUT/'FORMAL_STATE.json', snapshot)
    goal_path = LOW_OUTPUT/'GOAL.json'
    goal = read(goal_path) if goal_path.exists() else dict(
        objective='Complete Low/MTP3 three-model Terminal-Bench 2.1 internal comparison',
        formal_authorized=True, previous_smoke_preserved=True, fidelity_after_matrix=True)
    goal.update(objective='Complete Low/MTP3 three-model Terminal-Bench 2.1 internal comparison',
        formal_authorized=True, previous_smoke_preserved=True, fidelity_after_matrix=True,
        state=state['state'], run_directory=str(run_dir), planned_slots=267,
        environment_complete=state.get('environment_completed', state.get('environment_complete', 0)),
        environment_validation_complete=state.get('environment_validation_complete',
            state.get('gate_conclusions_complete', False)),
        trial_attempts=state.get('terminal_trials', state.get('terminal_trial_attempts', 0)),
        slot_status_counts=state.get('slot_status_counts', {}), updated_at=state['updated_at'])
    save(goal_path, goal)


def create_plan(run_dir, cfg, task_names, preflight):
    protocol = formal_identity()
    snapshot = save_source_snapshot(run_dir)
    smoke_goal = LOW_OUTPUT/'GOAL.json'
    archived_smoke_goal = LOW_OUTPUT/'GOAL.smoke.json'
    if smoke_goal.exists() and not archived_smoke_goal.exists():
        shutil.copy2(smoke_goal, archived_smoke_goal)
    plan = dict(config=cfg, source_snapshot=snapshot, queue_identity=protocol,
        tasks=task_names, models=cfg['models'],
        primary_mode=False, seed=cfg['seeds'][0], planned_slots=len(task_names)*len(cfg['models']),
        preflight=preflight, environment='PRoot internal comparison; no Docker leaderboard claim',
        retry_policy=dict(model_fail=0, timeout=0, api_or_environment_trial=0,
            interrupted_incomplete_trial=1, environment_control_infrastructure=1),
        authorization=dict(source='current user /goal instruction', recorded_at=time.time(),
            scope='89-task environment gates and all 267 formal model/task cells'))
    path = run_dir/'PLAN.json'
    if path.exists():
        existing = read(path)
        assert existing['queue_identity'] == protocol, 'Formal resume identity changed'
        assert existing['planned_slots'] == len(task_names)*len(cfg['models'])
        previous_state = read(run_dir/'STATE.json') if (run_dir/'STATE.json').exists() else {}
        previous_state.update(state='RUNNING_FORMAL', updated_at=time.time())
        update_goal_records(run_dir, previous_state)
        return existing
    save(path, plan)
    save(run_dir/'GOAL.json', dict(objective='Complete Low/MTP3 three-model Terminal-Bench 2.1 internal comparison',
        state='RUNNING', tasks=len(task_names), matrix_slots=len(task_names)*len(cfg['models']),
        previous_smoke_preserved=True, formal_authorized=True, fidelity_after_matrix=True))
    update_goal_records(run_dir, dict(state='RUNNING_FORMAL', updated_at=time.time(),
        environment_completed=0, environment_validation_complete=False,
        terminal_trials=0, slot_status_counts={}))
    return plan


def save_source_snapshot(run_dir):
    fixed = [
        ROOT/'src/terminal_formal_scheduler.py', ROOT/'src/terminal_queue.py',
        ROOT/'src/terminal_low.py', ROOT/'src/terminal_bench.py',
        ROOT/'src/terminal_smoke_scheduler.py', ROOT/'src/terminal_low_preflight.py',
        ROOT/'src/terminal_proot.py', ROOT/'src/terminal_proot_images.py',
        ROOT/'src/terminal_observer.py', ROOT/'assets/terminal_bench_low_mtp3.json',
        ROOT/'assets/terminal_bench_manifest.json', ROOT/'assets/terminal_bench_images.json',
        ROOT/'tests/test_terminal_formal_scheduler.py', ROOT/'tests/test_terminal_queue.py',
        ROOT/'tests/test_terminal_proot_images.py',
        LOW_OUTPUT/'model-audit.json']
    snapshot = run_dir/'source_snapshot'
    fixed_hashes = {}
    for source in fixed:
        relative = source.relative_to(ROOT)
        destination = snapshot/relative
        digest = sha(source)
        if destination.exists():
            assert sha(destination) == digest, f'Formal source snapshot changed: {relative}'
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        fixed_hashes[str(relative)] = digest
    prior_hashes = {}
    prior = ('GOAL.json', 'REPORT.md', 'SMOKE_SUMMARY.json', 'SMOKE_AUDIT.json',
        'COMPLETION_AUDIT.json', 'coverage.json', 'preparation-audit.json',
        'FAILURES.md', 'STATE.json', 'STATUS.json', 'PROGRESS_AUDIT.json',
        'PROGRESS_SUMMARY.json', 'TERMINUS_PARSE_AUDIT.json')
    for name in prior:
        source = LOW_OUTPUT/name
        if not source.exists():
            continue
        destination = snapshot/'prior_evidence'/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copy2(source, destination)
        prior_hashes[name] = sha(destination)
    result = dict(fixed_sources=fixed_hashes, prior_evidence=prior_hashes)
    path = snapshot/'MANIFEST.json'
    if path.exists():
        assert read(path) == result, 'Formal source snapshot identity changed'
    else:
        save(path, result)
    return result


def reconcile_orphans(queue, run_dir, cfg):
    """Recover complete native results; retry only a dead, incomplete clean attempt once."""
    recovered, retried, held = [], [], []
    for row in queue.interrupted_owners():
        folder = run_dir/'trials'/row['id']
        native = folder/'trial'
        result_path = native/'result.json'
        cleanup_path = native/'proot/cleanup.json'
        identity_path = folder/'identity.json'
        if not identity_path.exists():
            held.append(dict(id=row['id'], reason='missing saved Trial identity'))
            continue
        saved = read(identity_path)
        for key in ('id', 'model', 'task', 'mode', 'seed', 'attempt', 'protocol'):
            assert saved[key] == row[key], f'Formal resume identity mismatch: {key}'
        if result_path.exists() and cleanup_path.exists():
            data = read(result_path)
            if data.get('finished_at') and read(cleanup_path).get('after') == []:
                audit_trial_config(row, data)
                info = classify(data, native)
                info.update(identity=saved, agent_execution=data.get('agent_execution'),
                    native_reward=(data.get('verifier_result') or {}).get('rewards'),
                    cleanup_clean=True, result=str(result_path))
                summary_path = folder/'summary.json'
                if summary_path.exists():
                    previous = read(summary_path)
                    assert previous['status'] == info['status'] and previous['result'] == info['result']
                else:
                    save(summary_path, info)
                queue.finish(row['id'], info['status'], result_path)
                recovered.append(row['id'])
                continue
        cleanup_clean = cleanup_path.exists() and read(cleanup_path).get('after') == []
        interruption = folder/'interrupted.json'
        save(interruption, dict(time=time.time(), reason='owner exited before a complete clean result',
            cleanup_clean=cleanup_clean, previous_result=str(result_path) if result_path.exists() else None))
        queue.finish(row['id'], 'INTERRUPTED', interruption)
        if cleanup_clean and row['attempt'] == 0:
            queue.enqueue([row['model']], [row['task']], row['seed'], modes=[bool(row['mode'])], attempt=1)
            retried.append(row['id'])
        else:
            held.append(dict(id=row['id'], reason='incomplete attempt lacks a clean PRoot cleanup receipt or retry already used'))
    return dict(recovered=recovered, retried=retried, held=held)


def enqueue_infrastructure_retries(queue, run_dir):
    """Retry only clean API/environment errors, once; never retry model failures."""
    added = []
    rows = queue.snapshot()
    existing = {(row['model'], row['task'], row['mode'], row['seed'], row['attempt']) for row in rows}
    for row in rows:
        if row['attempt'] != 0 or row['status'] not in {'API_FAILURE', 'ENVIRONMENT_FAIL'}:
            continue
        cleanup = run_dir/'trials'/row['id']/'trial/proot/cleanup.json'
        result_path = run_dir/'trials'/row['id']/'trial/result.json'
        native = (read(result_path) or {}) if result_path.exists() else {}
        complete_clean_result = (result_path.exists() and cleanup.exists()
            and native.get('finished_at') and read(cleanup).get('after') == [])
        exception = native.get('exception_info') or {}
        message = str(exception.get('exception_message') or '')
        pre_agent_tmux_failure = (row['status'] == 'ENVIRONMENT_FAIL'
            and exception.get('exception_type') == 'RuntimeError'
            and 'Failed to start tmux session' in message
            and 'tmux: command not found' in message
            and native.get('agent_result') is None
            and native.get('agent_execution') is None)
        if complete_clean_result and not pre_agent_tmux_failure:
            review = run_dir/'retry-review'/f'{row["id"]}.json'
            if not review.exists():
                save(review, dict(status='not_retried', reason='complete native result was retained as interrupted; manual reconciliation required',
                    result=str(result_path)))
            continue
        if not cleanup.exists() or read(cleanup).get('after') != []:
            review = run_dir/'retry-review'/f'{row["id"]}.json'
            if not review.exists():
                save(review, dict(status='not_retried', reason='no clean PRoot cleanup receipt',
                    result=row['result']))
            continue
        key = (row['model'], row['task'], row['mode'], row['seed'], 1)
        if key in existing:
            continue
        queue.enqueue([row['model']], [row['task']], row['seed'],
            modes=[bool(row['mode'])], attempt=1)
        existing.add(key)
        added.append(dict(previous=row['id'], model=row['model'], task=row['task'], attempt=1))
        if pre_agent_tmux_failure:
            save(run_dir/'retry-review'/f'{row["id"]}.json', dict(status='retry_enqueued',
                reason='clean pre-agent tmux startup failure; one infrastructure retry queued',
                result=str(result_path), attempt=1))
    return added


def enqueue_interrupted_retries(queue, run_dir):
    """Resume a saved interruption only if cleanup is confirmed and attempt 1 is unused."""
    added = []
    rows = queue.snapshot()
    existing = {(row['model'], row['task'], row['mode'], row['seed'], row['attempt']) for row in rows}
    for row in rows:
        if row['attempt'] != 0 or row['status'] != 'INTERRUPTED':
            continue
        key = (row['model'], row['task'], row['mode'], row['seed'], 1)
        if key in existing:
            continue
        cleanup = run_dir/'trials'/row['id']/'trial/proot/cleanup.json'
        result_path = run_dir/'trials'/row['id']/'trial/result.json'
        if result_path.exists() and cleanup.exists() and read(result_path).get('finished_at') \
                and read(cleanup).get('after') == []:
            review = run_dir/'retry-review'/f'{row["id"]}.json'
            if not review.exists():
                save(review, dict(status='not_retried', reason='complete native result was retained as interrupted; manual reconciliation required',
                    result=str(result_path)))
            continue
        if not cleanup.exists() or read(cleanup).get('after') != []:
            review = run_dir/'retry-review'/f'{row["id"]}.json'
            if not review.exists():
                save(review, dict(status='not_retried', reason='interrupted attempt has no clean PRoot receipt',
                    result=row['result']))
            continue
        queue.enqueue([row['model']], [row['task']], row['seed'],
            modes=[bool(row['mode'])], attempt=1)
        existing.add(key)
        added.append(dict(previous=row['id'], model=row['model'], task=row['task'], attempt=1))
    return added


def process_ticks(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def cleanup_owned_workers(run_dir):
    """Stop only orphan workers whose saved PID, start time and argv match this run."""
    cleaned = []
    for gpu in range(5):
        path = run_dir/'gpu_workers'/f'gpu{gpu}.json'
        if not path.exists():
            continue
        saved = read(path)
        pid = saved.get('pid')
        if not pid or process_ticks(pid) != saved.get('owner_start'):
            continue
        try:
            argv = Path(f'/proc/{pid}/cmdline').read_bytes().decode(errors='replace').split('\0')
        except FileNotFoundError:
            continue
        expected = str(Path(__file__).resolve())
        if expected not in argv or 'worker' not in argv or str(run_dir.resolve()) not in argv:
            raise RuntimeError(f'GPU {gpu} has a live PID recorded by this run, but argv does not match')
        os.killpg(pid, signal.SIGINT)
        deadline = time.monotonic()+30
        while process_ticks(pid) == saved['owner_start'] and time.monotonic() < deadline:
            time.sleep(1)
        if process_ticks(pid) == saved['owner_start']:
            os.killpg(pid, signal.SIGTERM)
            deadline = time.monotonic()+5
            while process_ticks(pid) == saved['owner_start'] and time.monotonic() < deadline:
                time.sleep(.2)
        if process_ticks(pid) == saved['owner_start']:
            raise RuntimeError(f'Owned GPU {gpu} worker did not stop cleanly; preserve its state for review')
        cleaned.append(dict(gpu=gpu, pid=pid, stopped_at=time.time()))
    if cleaned:
        save(run_dir/f'orphan-workers-cleaned-{time.time_ns()}.json', dict(workers=cleaned))
    return cleaned


def launch_worker(gpu, run_dir, logs):
    argv = [str(ROOT/'.harbor-venv/bin/python'), str(Path(__file__).resolve()), 'worker',
        '--gpu', str(gpu), '--run', str(run_dir)]
    env = dict(os.environ, OPENAI_API_KEY='local-only', LITELLM_LOCAL_MODEL_COST_MAP='True',
        PYTHONDONTWRITEBYTECODE='1')
    child = subprocess.Popen(argv, env=env, stdout=logs[gpu], stderr=subprocess.STDOUT,
        start_new_session=True)
    ticks = Path(f'/proc/{child.pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
    return child, dict(pid=child.pid, start_ticks=ticks, gpu=gpu, command=argv, started_at=time.time())


def run(args):
    run_dir = Path(args.output).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir/'gpu_workers').mkdir(exist_ok=True)
    with (run_dir/'launcher.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        cfg = read(CONFIG)
        preflight = verify_smoke_and_models()
        manifest = prepare(cfg)
        task_names = sorted(manifest['tasks'])
        assert len(task_names) == cfg['dataset']['tasks'] == 89
        plan = create_plan(run_dir, cfg, task_names, preflight)
        cleanup_owned_workers(run_dir)
        for gpu, facts in serving.gpu_inventory().items():
            assert facts['used_mb'] < 500, f'GPU {gpu} occupied; do not preempt another job'
        queue = Queue(run_dir/'matrix.sqlite', plan['queue_identity'])
        coverage = read(LOW_OUTPUT/'coverage.json')
        validation = initial_validation(task_names, coverage, run_dir)
        sync_coverage(validation, coverage)
        for task in task_names:
            entry = validation['tasks'][task]
            if entry['status'] == 'CONTROLS_PASS':
                queue.enqueue(cfg['models'], [task], cfg['seeds'][0], modes=[False])
        save(run_dir/'environment_validation.json', validation)
        save(LOW_OUTPUT/'coverage.json', coverage)
        reconcile = reconcile_orphans(queue, run_dir, cfg)
        save(run_dir/f'recovery-{time.time_ns()}.json', reconcile)
        gpu_paused = any((run_dir/'PAUSE').glob('*.json'))
        interrupted_retries = [] if gpu_paused else enqueue_interrupted_retries(queue, run_dir)
        if interrupted_retries:
            save(run_dir/f'interrupted-retries-{time.time_ns()}.json', dict(retries=interrupted_retries))
        if reconcile['held']:
            save(run_dir/'RECOVERY_REVIEW.json', reconcile)
        workers, records, logs = {}, {}, {}
        for gpu in range(5):
            logs[gpu] = (run_dir/f'gpu{gpu}.log').open('a')
        stage_started = time.time()
        save(run_dir/'STATE.json', dict(state='RUNNING_FORMAL', stage='environment_and_trials',
            started_at=stage_started, planned_slots=267, environment_validation_complete=False))
        write_progress(run_dir, task_names, cfg, validation, queue, True, stage_started, gpu_paused)
        executor = ProcessPoolExecutor(max_workers=args.cpus)
        future_tasks = {}
        try:
            if not gpu_paused:
                for gpu in range(5):
                    workers[gpu], records[gpu] = launch_worker(gpu, run_dir, logs)
            for task, entry in validation['tasks'].items():
                if entry['status'] in {'ENV_PENDING', 'VALIDATING', 'VALIDATING_RETRY'}:
                    attempt = entry.get('attempt', 0)
                    entry['status'] = 'VALIDATING'
                    future = executor.submit(validate_one, task, attempt)
                    future_tasks[future] = (task, attempt)
            save(run_dir/'environment_validation.json', validation)
            save(run_dir/'GPU_WORKERS.json', records)
            next_checkpoint = 0.0
            controller_interrupted = False
            while True:
                done, _ = wait(future_tasks, timeout=1, return_when=FIRST_COMPLETED) if future_tasks else (set(), set())
                for future in done:
                    task, attempt = future_tasks.pop(future)
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = dict(task=task, attempt=attempt, status='CONTROL_INFRA_FAILURE',
                            retryable=True, error=type(exc).__name__+': '+str(exc),
                            traceback=traceback.format_exc())
                    retry = set_gate_state(task, result, validation, coverage)
                    if retry:
                        validation['tasks'][task]['attempt'] = 1
                        future_tasks[executor.submit(validate_one, task, 1)] = (task, 1)
                    elif result['status'] == 'CONTROLS_PASS':
                        queue.enqueue(cfg['models'], [task], cfg['seeds'][0], modes=[False])
                    save(run_dir/'environment_validation.json', validation)
                    save(LOW_OUTPUT/'coverage.json', coverage)
                retries = enqueue_infrastructure_retries(queue, run_dir)
                if retries:
                    save(run_dir/f'infrastructure-retries-{time.time_ns()}.json', dict(retries=retries))
                validation_running = bool(future_tasks)
                if not gpu_paused and any((run_dir/'PAUSE').glob('*.json')):
                    gpu_paused = True
                    for gpu, child in list(workers.items()):
                        serving.stop_process(child)
                        update_worker_state(run_dir, gpu, status='STOPPED_PROTOCOL_PAUSE')
                    workers.clear()
                if validation_running:
                    save(run_dir/'STATE.json', dict(state='RUNNING_FORMAL', stage='environment_and_trials',
                        started_at=stage_started, updated_at=time.time(), planned_slots=267,
                        environment_validation_complete=False, validation_running=len(future_tasks),
                        environment_counts=dict(Counter(row['status'] for row in validation['tasks'].values()))))
                else:
                    state_path = run_dir/'STATE.json'
                    state = read(state_path) if state_path.exists() else {}
                    state.update(environment_validation_complete=True,
                        environment_completed=sum(row['status'] in GATE_FINAL for row in validation['tasks'].values()),
                        environment_counts=dict(Counter(row['status'] for row in validation['tasks'].values())))
                    save(state_path, state)
                now = time.time()
                if now >= next_checkpoint:
                    write_progress(run_dir, task_names, cfg, validation, queue,
                        validation_running, stage_started, gpu_paused)
                    next_checkpoint = now+15
                for gpu, child in list(workers.items()):
                    code = child.poll()
                    if code is None:
                        continue
                    if code != 0:
                        save(run_dir/f'gpu{gpu}-exit-{time.time_ns()}.json', dict(gpu=gpu,
                            pid=child.pid, returncode=code, time=time.time()))
                        recovery = reconcile_orphans(queue, run_dir, cfg)
                        save(run_dir/f'recovery-{time.time_ns()}.json', recovery)
                    rows = queue.snapshot()
                    pending = any(row['status'] == 'PENDING' for row in rows)
                    running = any(row['status'] == 'RUNNING' for row in rows)
                    state = read(run_dir/'STATE.json')
                    if state.get('environment_validation_complete') and not pending and not running:
                        workers.pop(gpu)
                        if code != 0:
                            raise RuntimeError(f'GPU worker {gpu} exited {code} after queue completion')
                    else:
                        # A completed worker can be safely replaced after orphan reconciliation.
                        workers[gpu], records[gpu] = launch_worker(gpu, run_dir, logs)
                        save(run_dir/'GPU_WORKERS.json', records)
                rows = queue.snapshot()
                finished = ((gpu_paused and not future_tasks) or
                    (not future_tasks and all(row['status'] in GATE_FINAL for row in validation['tasks'].values())
                    and all(row['status'] not in {'PENDING', 'RUNNING'} for row in rows)
                    and not workers))
                if finished:
                    break
                time.sleep(1)
        except KeyboardInterrupt:
            save(run_dir/'INTERRUPTED.json', dict(time=time.time(), queue=queue.snapshot(),
                environment=validation, reason='controller stopped; completed results retained'))
            controller_interrupted = True
        finally:
            for gpu, child in workers.items():
                serving.stop_process(child)
                update_worker_state(run_dir, gpu, status='STOPPED_BY_CONTROLLER')
            executor.shutdown(wait=True, cancel_futures=True)
            for stream in logs.values():
                stream.close()
        if controller_interrupted:
            state = write_progress(run_dir, task_names, cfg, validation, queue,
                bool(future_tasks), stage_started, gpu_paused)
            state['state'] = 'INTERRUPTED'
            state['updated_at'] = time.time()
            save(run_dir/'STATE.json', state)
            update_goal_records(run_dir, state)
            return
        write_progress(run_dir, task_names, cfg, validation, queue, False, stage_started, gpu_paused)
        finalize(run_dir, cfg, task_names, validation, queue)


def finalize(run_dir, cfg, task_names, validation, queue):
    rows = queue.snapshot()
    matrix = matrix_snapshot(task_names, cfg['models'], validation, rows)
    save(run_dir/'matrix.json', dict(updated_at=time.time(), **matrix))
    conclusive_gates = {'CONTROLS_PASS', 'PROOT_INCOMPATIBLE', 'CONTROL_TIMEOUT',
        'ORACLE_CONTROL_FAIL', 'SCORING_GATE_FAIL'}
    gate_done = all(item['status'] in conclusive_gates for item in validation['tasks'].values())
    slot_statuses = [slot['status'] for task in matrix['tasks'].values() for slot in task['slots'].values()]
    complete_trial_statuses = {'PASS', 'MODEL_FAIL', 'TIMEOUT'}
    all_slots_final = all(status not in {'ENV_PENDING', 'VALIDATING', 'PENDING', 'RUNNING', 'INTERRUPTED'}
        for status in slot_statuses)
    trials_done = all(status in complete_trial_statuses or status in conclusive_gates
        for status in slot_statuses)
    paused = any((run_dir/'PAUSE').glob('*.json'))
    final_status = ('PAUSED_PROTOCOL' if paused else 'TRIALS_COMPLETE'
        if gate_done and trials_done and all_slots_final else 'NEEDS_REVIEW')
    state = dict(state=final_status,
        environment_complete=sum(item['status'] in GATE_FINAL for item in validation['tasks'].values()),
        environment_denominator=len(task_names), planned_slots=matrix['planned_slots'],
        terminal_trial_attempts=sum(row['status'] not in {'PENDING', 'RUNNING'} for row in rows),
        queue_rows=len(rows), slot_status_counts=matrix['slot_status_counts'],
        environment_status_counts=dict(Counter(item['status'] for item in validation['tasks'].values())),
        updated_at=time.time(), gate_conclusions_complete=gate_done, trials_terminal=trials_done,
        protocol_pause=paused)
    save(run_dir/'FINAL_STATE.json', state)
    current = read(run_dir/'GOAL.json')
    current.update(state=state['state'], environment_complete=state['environment_complete'],
        trial_attempts=state['terminal_trial_attempts'], completed_slots=sum(
            count for status, count in matrix['slot_status_counts'].items()
            if status not in {'ENV_PENDING', 'VALIDATING', 'PENDING', 'RUNNING'}))
    save(run_dir/'GOAL.json', current)
    update_goal_records(run_dir, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p_run = sub.add_parser('run')
    p_run.add_argument('--output', type=Path, default=RUN_DEFAULT)
    p_run.add_argument('--cpus', type=int, default=4)
    p_worker = sub.add_parser('worker')
    p_worker.add_argument('--gpu', type=int, required=True)
    p_worker.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'run':
        if args.cpus < 1 or args.cpus > 8:
            parser.error('--cpus must be between 1 and 8')
        signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
        os.environ['OPENAI_API_KEY'] = 'local-only'
        os.environ['LITELLM_LOCAL_MODEL_COST_MAP'] = 'True'
        run(args)
    else:
        signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
        worker(args.gpu, str(args.run))


if __name__ == '__main__':
    main()
