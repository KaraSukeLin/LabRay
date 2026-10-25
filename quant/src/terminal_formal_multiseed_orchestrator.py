"""Resume the fixed three-seed Terminal-Bench experiment to completion."""
import json
import os
from pathlib import Path
import shutil
import signal
import sqlite3
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT/'build/terminal_bench/low_mtp3/formal_three_seed_20261022'
PROTOCOL_PATH = PARENT/'PROTOCOL.json'
CONFIG_PATH = ROOT/'assets/terminal_bench_low_mtp3.json'
FIRST_RUN = ROOT/'build/terminal_bench/low_mtp3/formal_20261021'
SCHEDULER = ROOT/'src/terminal_formal_scheduler.py'
ANALYZER = ROOT/'src/terminal_formal_analysis.py'
AGGREGATOR = ROOT/'src/terminal_formal_multiseed_analysis.py'
FIDELITY = ROOT/'src/agentic_fidelity.py'
GATE_FINAL = {'CONTROLS_PASS', 'PROOT_INCOMPATIBLE', 'CONTROL_TIMEOUT',
              'ORACLE_CONTROL_FAIL', 'SCORING_GATE_FAIL', 'CONTROL_INFRA_FAILURE',
              'IMAGE_PREP_FAILED'}
OPEN = {'PENDING', 'RUNNING', 'INTERRUPTED'}


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return default


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')
    temp.replace(path)


def slot_snapshot(run_dir, seed, gate_tasks):
    db = Path(run_dir)/'matrix.sqlite'
    if not db.exists():
        return dict(status_counts={}, cells={}, open=0, terminal=0, enqueued=0)
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    rows = [dict(row) for row in con.execute('SELECT * FROM trials')]
    con.close()
    counts = {}
    latest = {}
    for row in rows:
        counts[row['status']] = counts.get(row['status'], 0)+1
        if row['seed'] != seed:
            raise ValueError(f'{run_dir}: found unexpected seed {row["seed"]}')
        key = (row['task'], row['model'])
        previous = latest.get(key)
        if previous is None or (row['attempt'], row.get('started') or 0) > (
                previous['attempt'], previous.get('started') or 0):
            latest[key] = row
    expected = {(task, model) for task, status in gate_tasks.items()
                if status == 'CONTROLS_PASS' for model in read(Path(run_dir)/'PLAN.json', {}).get('models', [])}
    open_count = sum(row['status'] in OPEN for row in latest.values())
    terminal_count = sum(row['status'] not in OPEN for row in latest.values())
    return dict(status_counts=counts, cells=latest, open=open_count,
        terminal=terminal_count, enqueued=len(latest), expected=expected)


def run_complete(run_dir, seed, gate_tasks):
    run_dir = Path(run_dir)
    if not (run_dir/'FINAL_STATE.json').exists():
        return False
    validation = read(run_dir/'environment_validation.json')
    if not validation or set(validation.get('tasks', {})) != set(gate_tasks):
        return False
    if any(row['status'] not in GATE_FINAL for row in validation['tasks'].values()):
        return False
    plan = read(run_dir/'PLAN.json') or {}
    snap = slot_snapshot(run_dir, seed, gate_tasks)
    expected = snap.get('expected', set())
    if len(expected) != 180 or len(snap['cells']) != 180 or snap['open']:
        return False
    for key in expected:
        if snap['cells'][key]['status'] in OPEN:
            return False
    return plan.get('seed') == seed and plan.get('planned_slots') == 267


def controller_alive(run_dir):
    run_dir = Path(run_dir).resolve()
    targets = {str(run_dir), str(run_dir.relative_to(ROOT))}
    state = read(run_dir/'STATE.json', {}) or {}
    for row in (state.get('active_gpu_workers') or {}).values():
        pid = row.get('pid')
        if not pid:
            continue
        try:
            cmd = (Path('/proc')/str(pid)/'cmdline').read_bytes().decode(errors='replace').split('\0')
            status = (Path('/proc')/str(pid)/'status').read_text()
        except OSError:
            continue
        parent = next((line.split()[1] for line in status.splitlines() if line.startswith('PPid:')), None)
        if ('terminal_formal_scheduler.py' in ' '.join(cmd) and 'worker' in cmd
                and any(target in ' '.join(cmd) for target in targets)):
            if parent:
                try:
                    parent_cmd = (Path('/proc')/parent/'cmdline').read_bytes().decode(errors='replace').split('\0')
                except OSError:
                    continue
                if ('terminal_formal_scheduler.py' in ' '.join(parent_cmd) and 'run' in parent_cmd
                        and any(target in ' '.join(parent_cmd) for target in targets)):
                    return True
    return False


def refresh_goal(stage, seeds, run_dirs, gate_tasks):
    path = PARENT/'GOAL.json'
    goal = read(path, {})
    progress, cells_total, terminal_total, open_total = {}, set(), 0, 0
    for seed, run_dir in zip(seeds, run_dirs):
        snap = slot_snapshot(run_dir, seed, gate_tasks)
        progress[str(seed)] = dict(status_counts=snap['status_counts'],
            queued_cells=snap['enqueued'], terminal_cells=snap['terminal'],
            open_cells=snap['open'], finalized=run_complete(run_dir, seed, gate_tasks))
        cells_total.update((seed, *key) for key in snap['cells'])
        terminal_total += snap['terminal']
        open_total += snap['open']
    gate_failed = sum(status != 'CONTROLS_PASS' for status in gate_tasks.values())
    gate_slots = gate_failed*3*len(seeds)
    goal['state'] = stage
    goal['seed_progress'] = progress
    goal['matrix_progress'] = dict(total_slots=801,
        common_gate_non_evaluable_slots=gate_slots,
        terminal_status_slots=gate_slots+terminal_total,
        open_trial_slots=open_total,
        not_yet_enqueued_evaluable_slots=540-(len(cells_total)),
        enqueued_evaluable_slots=len(cells_total))
    goal['updated_at'] = time.time()
    save(path, goal)
    return goal['matrix_progress']


def report_progress(stage, seeds, run_dirs, gate_tasks):
    progress = refresh_goal(stage, seeds, run_dirs, gate_tasks)
    print(f"{stage}: {progress['terminal_status_slots']}/801 terminal; "
          f"{progress['open_trial_slots']} active or pending; "
          f"{progress['not_yet_enqueued_evaluable_slots']} evaluable slots not yet queued",
          flush=True)


def run_scheduler(run_dir, seed, seeds, run_dirs, gate_tasks, command_attempt):
    log = PARENT/f'scheduler_seed_{seed}_resume_{command_attempt}.log'
    argv = [sys.executable, str(SCHEDULER), 'run', '--cpus', '4',
            '--output', str(Path(run_dir).resolve())]
    print(f"Launching formal scheduler for seed {seed}: {argv}", flush=True)
    with log.open('a') as stream:
        process = subprocess.Popen(argv, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        last_report = 0.0
        while process.poll() is None:
            now = time.time()
            if now-last_report >= 60:
                report_progress(f'RUNNING_SEED_{seed}', seeds, run_dirs, gate_tasks)
                last_report = now
            time.sleep(5)
        code = process.returncode
    report_progress(f'CHECK_SEED_{seed}', seeds, run_dirs, gate_tasks)
    return code


def ensure_seed_config(seed, config_files):
    config = Path(config_files[str(seed)])
    value = read(config)
    if value is None or value.get('seeds') != [seed]:
        raise ValueError(f'Invalid frozen per-seed config: {config}')
    if CONFIG_PATH.read_bytes() != config.read_bytes():
        shutil.copy2(config, CONFIG_PATH)


def ensure_gate_copy(run_dir, source_validation):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    target = run_dir/'environment_validation.json'
    if not target.exists():
        shutil.copy2(source_validation, target)
    if read(target) != read(source_validation):
        raise ValueError(f'{target}: copied environment Gate ledger differs from seed 20261020')


def ensure_analysis(run_dir, seed):
    run_dir = Path(run_dir)
    output = run_dir/'analysis'
    if read(output/'analysis.json'):
        return
    if output.exists():
        output.rename(run_dir/f'analysis_partial_{time.time_ns()}')
    argv = [str(ROOT/'.vllm-venv/bin/python'), str(ANALYZER),
            '--run', str(run_dir), '--output', str(output)]
    with (PARENT/f'analysis_seed_{seed}.log').open('a') as stream:
        subprocess.run(argv, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)


def run_child_to_completion(seed, run_dir, seeds, run_dirs, gate_tasks, config_files, source_validation):
    ensure_seed_config(seed, config_files)
    ensure_gate_copy(run_dir, source_validation)
    for attempt in range(1, 4):
        if run_complete(run_dir, seed, gate_tasks):
            return
        if Path(run_dir).joinpath('PAUSE').exists():
            raise RuntimeError(f'{run_dir}: protocol pause is active; inspect PAUSE evidence')
        code = run_scheduler(run_dir, seed, seeds, run_dirs, gate_tasks, attempt)
        if run_complete(run_dir, seed, gate_tasks):
            return
        if Path(run_dir).joinpath('PAUSE').exists():
            raise RuntimeError(f'{run_dir}: scheduler paused on a protocol mismatch')
        print(f'seed {seed} controller exited {code}; resuming checkpoint ({attempt}/3)', flush=True)
    raise RuntimeError(f'{run_dir}: three scheduler recoveries did not reach a final matrix')


def wait_for_seed_one(seed, run_dir, seeds, run_dirs, gate_tasks):
    run_dir = Path(run_dir)
    last_report = 0.0
    recoveries = 0
    while not run_complete(run_dir, seed, gate_tasks):
        if run_dir.joinpath('PAUSE').exists():
            raise RuntimeError(f'{run_dir}: protocol pause is active; inspect PAUSE evidence')
        if not controller_alive(run_dir):
            recoveries += 1
            if recoveries > 3:
                raise RuntimeError(f'{run_dir}: controller did not recover after three resume attempts')
            code = run_scheduler(run_dir, seed, seeds, run_dirs, gate_tasks, recoveries)
            print(f'first-seed controller exited {code}; checkpoint resume check {recoveries}/3', flush=True)
            continue
        now = time.time()
        if now-last_report >= 60:
            report_progress('RUNNING_SEED_20261020', seeds, run_dirs, gate_tasks)
            last_report = now
        time.sleep(10)


def safe_output(output):
    output = Path(output)
    if output.exists() and read(output/'analysis.json'):
        return False
    if output.exists():
        output.rename(output.with_name(output.name+f'_partial_{time.time_ns()}'))
    return True


def run_fidelity(first_run):
    prepared = PARENT/'agentic_fidelity_prepared'
    output = PARENT/'agentic_fidelity'
    vpython = str(ROOT/'.vllm-venv/bin/python')
    if not read(prepared/'sampling.json'):
        if prepared.exists():
            prepared.rename(PARENT/f'agentic_fidelity_prepared_partial_{time.time_ns()}')
        argv = [vpython, str(FIDELITY), 'prepare', '--trajectories', str(first_run),
            '--output', str(prepared), '--seed', '20261020', '--windows-per-task', '4',
            '--tokens-per-window', '32', '--max-context', '65536']
        subprocess.run(argv, cwd=ROOT, check=True)
    if read(output/'agentic_summary.json'):
        return output
    if output.exists():
        output.rename(PARENT/f'agentic_fidelity_partial_{time.time_ns()}')
    argv = [vpython, str(FIDELITY), 'run', '--prepared', str(prepared),
        '--output', str(output), '--gpus', '0', '--authorize-full']
    env = dict(os.environ)
    compat = str(ROOT/'build/cuda-compat/cuda-compat')
    env['LD_LIBRARY_PATH'] = compat+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    subprocess.run(argv, cwd=ROOT, env=env, check=True)
    return output


def append_fidelity_report(analysis_dir, fidelity_dir):
    report_path = Path(analysis_dir)/'REPORT.md'
    report = report_path.read_text()
    if '## Agentic target-model Fidelity' in report:
        return
    summary = read(Path(fidelity_dir)/'agentic_summary.json', {}) or {}
    sampling = read(Path(fidelity_dir)/'trajectory_sampling.json', {}) or {}
    metrics = [{key: row.get(key) for key in ('model', 'stratum', 'tasks', 'scoring_positions', 'task_equal_weight', 'task_cluster_95ci')}
        for row in summary.get('results', []) if row.get('stratum') == 'all']
    report += ('\n## Agentic target-model Fidelity\n\n'
        f"- Prepared from first-seed Low BF16 trajectories; {sampling.get('tasks')} tasks, "
        f"{sampling.get('windows')} windows, {sampling.get('tokens')} scored positions.\n"
        '- This teacher-forced full-vocabulary result is a Fidelity check, not a benchmark score.\n\n'
        '```json\n'+json.dumps(metrics, indent=2, ensure_ascii=False)+'\n```\n')
    report_path.write_text(report)


def update_project_goal(seeds, run_dirs, analysis, fidelity, result):
    low_output = ROOT/'build/terminal_bench/low_mtp3'
    state = 'MATRIX_AND_FIDELITY_COMPLETE'
    fields = dict(state=state, objective='Complete the Terminal-Bench 2.1 LabRay three-model capability-retention experiment',
        planned_slots=801, matrix_slots=801, seeds=seeds,
        seed_run_directories={str(seed): str(path) for seed, path in zip(seeds, run_dirs)},
        completed_slots=result['observed_slots'], environment_complete=89,
        environment_validation_complete=True,
        environment_status_counts=result['environment_gate_counts'],
        slot_status_counts=result['slot_status_counts'],
        all_matrix_slots_terminal=result['all_matrix_slots_terminal'],
        all_evaluable_slots_attempted=result['all_evaluable_slots_attempted'],
        trial_attempts=sum(row['attempts'] for row in result['model_summary']),
        paired_task_count=result['paired_task_count'], formal_run_directory=str(PARENT),
        final_analysis=str(analysis), agentic_fidelity=str(fidelity), updated_at=time.time())
    goal_path = low_output/'GOAL.json'
    goal = read(goal_path, {}) or {}
    goal.update(fields, formal_authorized=True, previous_smoke_preserved=True,
        fidelity_after_matrix=True)
    save(goal_path, goal)
    formal_state_path = low_output/'FORMAL_STATE.json'
    formal_state = read(formal_state_path, {}) or {}
    formal_state.update(fields)
    save(formal_state_path, formal_state)
    save(PARENT/'FINAL_STATE.json', fields)


def main():
    protocol = read(PROTOCOL_PATH)
    seeds = protocol['matrix']['seeds']
    run_dirs = [ROOT/path for path in protocol['seed_run_directories'].values()]
    if seeds != [20261020, 20261021, 20261022] or len(run_dirs) != 3:
        raise ValueError('The frozen seed protocol differs from the authorized three-seed matrix')
    config_files = protocol['seed_config_files']
    gate_ledger = FIRST_RUN/'environment_validation.json'
    validation = read(gate_ledger)
    if not validation or len(validation.get('tasks', {})) != 89:
        raise ValueError('First-seed 89-task environment ledger is missing')
    gate_tasks = {task: row['status'] for task, row in validation['tasks'].items()}

    wait_for_seed_one(seeds[0], run_dirs[0], seeds, run_dirs, gate_tasks)
    ensure_analysis(run_dirs[0], seeds[0])
    report_progress('SEED_20261020_COMPLETE', seeds, run_dirs, gate_tasks)

    for seed, run_dir in zip(seeds[1:], run_dirs[1:]):
        run_child_to_completion(seed, run_dir, seeds, run_dirs, gate_tasks, config_files, gate_ledger)
        ensure_analysis(run_dir, seed)
        report_progress(f'SEED_{seed}_COMPLETE', seeds, run_dirs, gate_tasks)

    shutil.copy2(PARENT/'CONFIG_THREE_SEEDS.json', CONFIG_PATH)
    analysis = PARENT/'analysis'
    if safe_output(analysis):
        argv = [str(ROOT/'.vllm-venv/bin/python'), str(AGGREGATOR),
            '--protocol', str(PROTOCOL_PATH), '--runs', *[str(path) for path in run_dirs],
            '--output', str(analysis)]
        with (PARENT/'analysis.log').open('a') as stream:
            subprocess.run(argv, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
    result = read(analysis/'analysis.json')
    if not result or result.get('observed_slots') != 801 or not result.get('all_matrix_slots_terminal'):
        raise RuntimeError('Final analysis did not verify all 801 terminal matrix slots')
    report_progress('MATRIX_ANALYZED_801', seeds, run_dirs, gate_tasks)

    fidelity = run_fidelity(run_dirs[0])
    append_fidelity_report(analysis, fidelity)
    update_project_goal(seeds, run_dirs, analysis, fidelity, result)
    goal = read(PARENT/'GOAL.json', {})
    goal.update(state='MATRIX_AND_FIDELITY_COMPLETE', matrix_slots=801,
        final_analysis=str(analysis), agentic_fidelity=str(fidelity), updated_at=time.time())
    save(PARENT/'GOAL.json', goal)
    print(f"Complete: 801 slots, {result['paired_task_count']} common paired tasks; "
          f"report at {analysis/'REPORT.md'}", flush=True)


if __name__ == '__main__':
    def handle_signal(signum, _frame):
        raise KeyboardInterrupt(f'orchestrator received signal {signum}')
    signal.signal(signal.SIGTERM, handle_signal)
    try:
        main()
    except KeyboardInterrupt as exc:
        print(str(exc), flush=True)
        raise
    except BaseException as exc:
        marker = PARENT/'ORCHESTRATOR_FAILURE.json'
        save(marker, dict(time=time.time(), error=type(exc).__name__+': '+str(exc)))
        goal = read(PARENT/'GOAL.json', {})
        goal.update(state='RECOVERY_REQUIRED', recovery_condition=str(exc), updated_at=time.time())
        save(PARENT/'GOAL.json', goal)
        raise
