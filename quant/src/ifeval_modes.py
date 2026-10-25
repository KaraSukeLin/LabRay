"""Resumable IFEval matrix; reuses benchmark engines and Google's official scorer."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from threading import Event
import time

from benchmark import BASE, ROOT, TOOLS, create_llm, score, scorers
from fidelity import write_json

CONFIG = ROOT / 'assets/ifeval_modes_config.json'
SAMPLING_KEYS = ['temperature', 'top_p', 'top_k', 'min_p', 'presence_penalty',
                 'repetition_penalty', 'max_tokens']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record_key(row):
    return row['mode'], row['seed'], row['id']


def pending_tasks(experiment, completed, modes, question_ids, seeds):
    return [(row, mode, seed) for row in experiment['questions']
            if row['id'] in question_ids for mode in modes
            for seed in experiment['seeds'] if seed in seeds
            if (mode, seed, row['id']) not in completed]


def write_attempt(path, attempt):
    """A kill during a timing update must leave the previous snapshot readable."""
    temporary = path.with_suffix('.tmp')
    write_json(temporary, attempt)
    temporary.replace(path)


def read_records(path, repair=False, allow_partial_tail=False):
    """Only an incomplete final line can be discarded after an interrupted write."""
    if not path.exists():
        return []
    raw = path.read_bytes()
    if raw and not raw.endswith(b'\n'):
        if allow_partial_tail:
            raw = raw[:raw.rfind(b'\n') + 1]
        else:
            assert repair, f'Incomplete final JSONL line: {path}; resume the worker first'
            end = raw.rfind(b'\n') + 1
            path.with_name(path.name + f'.partial-{time.time_ns()}').write_bytes(raw[end:])
            path.write_bytes(raw[:end])
            raw = raw[:end]
    rows = [json.loads(line) for line in raw.splitlines()]
    assert len({record_key(r) for r in rows}) == len(rows), 'Duplicate completed records'
    return rows


def read_model_records(folder, repair_paths=(), allow_partial_tail=False):
    paths = [folder/'questions.jsonl', *sorted(folder.glob('questions.shard-*-of-*.jsonl'))]
    rows, seen = [], {}
    for path in paths:
        chunk = read_records(path, repair=path in repair_paths,
            allow_partial_tail=allow_partial_tail and path not in repair_paths)
        for row in chunk:
            key = record_key(row)
            if key in seen:
                assert seen[key] == row, f'Conflicting duplicate record across shard files: {key}'
                continue
            seen[key] = row
            rows.append(row)
    return rows


def shard_question_ids(experiment, index, count):
    assert count >= 1 and 0 <= index < count
    return {row['id'] for position, row in enumerate(experiment['questions'])
            if position % count == index}


def sampling_values(settings, seed):
    return dict({key: settings[key] for key in SAMPLING_KEYS}, seed=seed)


def record_sampling_matches(record, settings, seed):
    expected = sampling_values(settings, seed)
    actual = record['sampling']
    return (actual == record['engine_sampling'] and
        all(actual[key] == value for key, value in expected.items() if key != 'max_tokens') and
        actual['max_tokens'] in settings.get('max_tokens_history', [expected['max_tokens']]))


def template_kwargs(settings):
    values = dict(enable_thinking=settings['enable_thinking'], preserve_thinking=True)
    if settings['enable_thinking']:
        values['reasoning_effort'] = settings['reasoning_effort']
    return values


def render_prompt(processor, row, settings):
    prompt = processor.apply_chat_template(
        [dict(role='user', content=[dict(type='text', text=row['prompt'])])],
        tokenize=False, add_generation_prompt=True, **template_kwargs(settings))
    if settings['enable_thinking']:
        assert prompt.endswith('<think>\n')
        if settings['reasoning_effort'] == 'xhigh':
            assert 'Reasoning effort is set to xhigh.' in prompt
    else:
        assert prompt.endswith('<think>\n\n</think>\n\n')
        assert 'Reasoning effort is set to ' not in prompt
    return prompt


def extract_answer(tokenizer, generated, prompt_ids, settings):
    """Use the same registered qwen3 parser as vLLM's official serving recipe."""
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.reasoning import ReasoningParserManager
    kwargs = template_kwargs(settings)
    parser = ReasoningParserManager.get_reasoning_parser('qwen3')(
        tokenizer, chat_template_kwargs=kwargs)
    parser.adjust_initial_state_from_prompt(prompt_ids)
    request = ChatCompletionRequest(model='shared-tokenizer',
        messages=[dict(role='user', content='')], chat_template_kwargs=kwargs)
    reasoning, final = parser.extract_reasoning(generated.text, request)
    # A thinking-only truncated response has no final answer. Never score the trace.
    return dict(reasoning=reasoning or '', final_answer=final or '',
        missing_final=not bool(final and final.strip()),
        reasoning_tokens=parser.count_reasoning_tokens(generated.token_ids),
        parser='vllm/qwen3')


def make_experiment(args):
    config = json.loads(args.config.read_text())
    rows = [r for r in json.loads((args.data / 'questions.json').read_text())
            if r['task'] == 'ifeval']
    canonical = [json.loads(line) for line in
        (ROOT / 'assets/benchmark/ifeval/input_data_google_repository.jsonl').read_text().splitlines()]
    assert len(rows) == 541 and [r['example'] for r in rows] == canonical
    if args.question_ids:
        ids = set(args.question_ids)
        selected = [r for r in rows if r['id'] in ids]
        assert len(selected) == len(ids), 'Unknown question ID'
    else:
        stop = args.stop if args.stop is not None else (541 if args.full else 2)
        assert 0 <= args.start < stop <= 541
        selected = rows[args.start:stop]
    assert len(selected) < 541 or args.full, 'All 541 questions require explicit --full'
    models, modes, seeds = args.models or config['models'], args.modes, args.seeds or config['seeds']
    assert set(models) <= set(config['models']) and len(set(models)) == len(models)
    assert set(modes) <= set(config['modes']) and len(set(modes)) == len(modes)
    assert len(set(seeds)) == len(seeds)
    assert config['mtp'] is False and config['kv_cache_dtype'] == 'bfloat16'
    inventory = {r['model']: r for r in json.loads(
        (ROOT / 'build/autoquant/checkpoints/inventory.json').read_text())}
    model_versions = {}
    for model in models:
        folder = Path('/root/model') / model
        model_versions[model] = dict(revision=inventory[model]['revision'],
            config_sha256=sha(folder / 'config.json'),
            weights=[dict(file=p.name, bytes=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)
                     for p in sorted(folder.glob('*.safetensors'))])
    files = [BASE / name for name in ['tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja']]
    files += [Path(__file__), ROOT / 'src/benchmark.py']
    files += [TOOLS / 'google-research/instruction_following_eval' / name for name in
              ['evaluation_lib.py', 'instructions.py', 'instructions_registry.py', 'instructions_util.py']]
    return dict(schema=1, config=config, config_sha256=sha(args.config),
        models=models, modes=modes, seeds=seeds, question_ids=[r['id'] for r in selected],
        questions=selected, full_541=len(selected) == 541,
        planned_evaluations=len(models) * len(modes) * len(seeds) * len(selected),
        dataset_sha256=sha(args.data / 'questions.json'), model_versions=model_versions,
        files_sha256={str(p): sha(p) for p in files},
        environment={p: importlib.metadata.version(p) for p in
                     ['vllm', 'torch', 'transformers', 'scipy', 'numpy', 'langdetect']})


def save_plan(args):
    experiment = make_experiment(args)
    dest = args.output / 'experiment.json'
    if dest.exists():
        assert args.resume, 'Output exists; use --resume with the same experiment settings'
        assert json.loads(dest.read_text()) == experiment, 'Resume settings/data/code/model identity changed'
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        write_json(dest, experiment)
    print(json.dumps(dict(output=str(args.output), planned=experiment['planned_evaluations'],
        questions_per_cell=len(experiment['questions']), models=experiment['models'],
        modes=experiment['modes'], seeds=experiment['seeds'], inference_started=False)), flush=True)
    return experiment


def worker(args):
    started = time.perf_counter()
    experiment = json.loads(args.experiment.read_text())
    for name, digest in experiment['files_sha256'].items():
        assert sha(Path(name)) == digest, f'Experiment source/tokenizer changed: {name}'
    assert args.model in experiment['models']
    for package, version in experiment['environment'].items():
        assert importlib.metadata.version(package) == version, f'Environment changed: {package}'
    folder = Path('/root/model') / args.model
    version = experiment['model_versions'][args.model]
    assert sha(folder/'config.json') == version['config_sha256']
    assert [dict(file=p.name, bytes=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)
            for p in sorted(folder.glob('*.safetensors'))] == version['weights']
    task_modes = args.task_modes or experiment['modes']
    task_question_ids = args.task_question_ids or experiment['question_ids']
    task_seeds = args.task_seeds or experiment['seeds']
    assert set(task_modes) <= set(experiment['modes'])
    assert set(task_question_ids) <= set(experiment['question_ids'])
    assert set(task_seeds) <= set(experiment['seeds'])
    shard_index, shard_count = args.task_shard_index, args.task_shard_count
    assert shard_count >= 1 and 0 <= shard_index < shard_count
    dest = args.experiment.parent / args.model
    dest.mkdir(exist_ok=True)
    sharded = shard_count > 1
    record_path = dest / (f'questions.shard-{shard_index}-of-{shard_count}.jsonl'
                          if sharded else 'questions.jsonl')
    if sharded:
        task_question_ids = set(task_question_ids) & shard_question_ids(
            experiment, shard_index, shard_count)
    # Shards share a model read lock, but each has its own output and exclusive lock.
    import fcntl
    with ExitStack() as locks:
        model_lock = locks.enter_context((dest / '.worker.lock').open('w'))
        mode = fcntl.LOCK_SH if sharded else fcntl.LOCK_EX
        fcntl.flock(model_lock, mode | fcntl.LOCK_NB)
        if sharded:
            shard_lock = locks.enter_context(
                (dest / f'.worker.shard-{shard_index}-of-{shard_count}.lock').open('w'))
            fcntl.flock(shard_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        records = read_model_records(dest,
            repair_paths={dest/'questions.jsonl', record_path}, allow_partial_tail=True)
        completed = {record_key(r) for r in records}
        # Finish every mode/seed for one question before moving on. If a run stops
        # at its wall-time limit, this gives the largest common paired prefix.
        pending = pending_tasks(experiment, completed, task_modes, task_question_ids, task_seeds)
        if not pending:
            print('All records already complete; no model load', flush=True)
            return
        from transformers import AutoProcessor
        from vllm import SamplingParams
        processor = AutoProcessor.from_pretrained(BASE, local_files_only=True)
        ifeval, mmmu = scorers()
        attempts = dest / 'attempts'
        attempts.mkdir(exist_ok=True)
        attempt_path = attempts / f'{time.time_ns()}.json'
        attempt = dict(status='running', model=args.model,
            start_utc=datetime.now(timezone.utc).isoformat(), gpu_count=1,
            cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
            task_shard=dict(index=shard_index, count=shard_count),
            initialization_seconds=0., inference_seconds=0., scoring_seconds=0.,
            preprocessing_seconds=0., shutdown_seconds=0., new_records=0,
            completed_before=len(completed), pending_before=len(pending),
            setup_seconds=time.perf_counter() - started)
        write_attempt(attempt_path, attempt)
        llm = None
        try:
            before = time.perf_counter()
            llm = create_llm(args.model, experiment['config'], experiment['seeds'][0])
            attempt['initialization_seconds'] = time.perf_counter() - before
            effective = llm.llm_engine.vllm_config
            assert effective.speculative_config is None
            assert effective.cache_config.cache_dtype == 'bfloat16'
            assert effective.parallel_config.tensor_parallel_size == 1
            attempt['engine'] = dict(mtp=False, kv_cache_dtype=effective.cache_config.cache_dtype,
                tensor_parallel_size=1, max_num_seqs=effective.scheduler_config.max_num_seqs,
                max_model_len=effective.model_config.max_model_len, enforce_eager=effective.model_config.enforce_eager,
                enable_prefix_caching=effective.cache_config.enable_prefix_caching,
                max_num_batched_tokens=effective.scheduler_config.max_num_batched_tokens)
            # Record the validated request after vLLM applies generation config and
            # clones SamplingParams, immediately before it enters the engine core.
            received = []
            process_inputs = llm.llm_engine.input_processor.process_inputs
            def audited_inputs(*values, **kwargs):
                request = process_inputs(*values, **kwargs)
                received.append({k: getattr(request.sampling_params, k) for k in SAMPLING_KEYS + ['seed']})
                return request
            llm.llm_engine.input_processor.process_inputs = audited_inputs
            with record_path.open('a') as output:
                for row, mode, seed in pending:
                    attempt['current_task'] = dict(question=row['id'], mode=mode, seed=seed)
                    write_attempt(attempt_path, attempt)
                    before = time.perf_counter()
                    settings = experiment['config']['modes'][mode]
                    prompt = render_prompt(processor, row, settings)
                    prompt_ids = processor.tokenizer.encode(prompt, add_special_tokens=False)
                    sampling = sampling_values(settings, seed)
                    params = SamplingParams(**sampling)
                    actual = {k: getattr(params, k) for k in sampling}
                    assert actual == sampling, 'vLLM normalized a requested sampling setting'
                    assert len(prompt_ids) + params.max_tokens <= effective.model_config.max_model_len
                    preprocessing_seconds = time.perf_counter() - before
                    before = time.perf_counter()
                    result = llm.generate([dict(prompt_token_ids=prompt_ids)], params, use_tqdm=False)[0]
                    inference_seconds = time.perf_counter() - before
                    assert len(received) == 1 and received[0] == actual, 'Engine sampling differs from request'
                    engine_sampling = received.pop()
                    assert result.prompt_token_ids == prompt_ids
                    generated = result.outputs[0]
                    before = time.perf_counter()
                    parsed = extract_answer(processor.tokenizer, generated, prompt_ids, settings)
                    graded = score(row, parsed['final_answer'], ifeval, mmmu)
                    scoring_seconds = time.perf_counter() - before
                    record = dict(id=row['id'], task='ifeval', model=args.model, mode=mode, seed=seed,
                        response=generated.text, raw_output=generated.text, token_ids=generated.token_ids,
                        prompt=prompt, prompt_token_ids=prompt_ids, sampling=actual, engine_sampling=engine_sampling,
                        chat_template_kwargs=template_kwargs(settings),
                        finish_reason=generated.finish_reason, stop_reason=generated.stop_reason,
                        truncated=generated.finish_reason == 'length', input_tokens=len(prompt_ids),
                        output_tokens=len(generated.token_ids),
                        final_answer_tokens=len(processor.tokenizer.encode(
                            parsed['final_answer'], add_special_tokens=False)),
                        preprocessing_seconds=preprocessing_seconds, inference_seconds=inference_seconds,
                        scoring_seconds=scoring_seconds, output_tokens_per_second=len(generated.token_ids)/inference_seconds,
                        completed_utc=datetime.now(timezone.utc).isoformat(),
                        attempt=attempt_path.name, **parsed, **graded)
                    output.write(json.dumps(record, ensure_ascii=False) + '\n')
                    output.flush()
                    os.fsync(output.fileno())
                    attempt['new_records'] += 1
                    attempt['current_task'] = None
                    for key in ['preprocessing_seconds', 'inference_seconds', 'scoring_seconds']:
                        attempt[key] += record[key]
                    attempt['wall_seconds'] = time.perf_counter() - started
                    write_attempt(attempt_path, attempt)
                    print(mode, seed, row['id'], 'correct', record['correct'], 'tokens', record['output_tokens'],
                          'finish', record['finish_reason'], 'missing_final', parsed['missing_final'], flush=True)
                    if args.max_new_records and attempt['new_records'] >= args.max_new_records:
                        break
            attempt['status'] = 'ok' if attempt['new_records'] == len(pending) else 'partial'
        except BaseException as error:
            attempt['status'] = 'interrupted' if isinstance(error, (KeyboardInterrupt, SystemExit)) else 'failed'
            attempt['error'] = repr(error)
            raise
        finally:
            if llm is not None:
                before = time.perf_counter()
                llm.llm_engine.engine_core.shutdown()
                attempt['shutdown_seconds'] = time.perf_counter() - before
            attempt.update(wall_seconds=time.perf_counter() - started,
                           end_utc=datetime.now(timezone.utc).isoformat())
            write_attempt(attempt_path, attempt)


def merge_model_shards(output, model):
    folder = output / model
    paths = sorted(folder.glob('questions.shard-*-of-*.jsonl'))
    if not paths:
        return dict(model=model, merged=0, files=[])
    import fcntl
    with (folder / '.worker.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main_path = folder / 'questions.jsonl'
        main = read_records(main_path, repair=True)
        by_key = {record_key(row): row for row in main}
        new_rows = []
        for path in paths:
            for row in read_records(path, repair=True):
                key = record_key(row)
                if key in by_key:
                    assert by_key[key] == row, f'Conflicting duplicate while merging {key}'
                    continue
                by_key[key] = row
                new_rows.append(row)
        experiment = json.loads((output/'experiment.json').read_text())
        order = {qid: i for i, qid in enumerate(experiment['question_ids'])}
        mode_order = {mode: i for i, mode in enumerate(experiment['modes'])}
        seed_order = {seed: i for i, seed in enumerate(experiment['seeds'])}
        new_rows.sort(key=lambda r: (order[r['id']], mode_order[r['mode']], seed_order[r['seed']]))
        with main_path.open('a') as stream:
            for row in new_rows:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
        archived = folder / 'merged-shards'
        archived.mkdir(exist_ok=True)
        for path in paths:
            path.replace(archived/path.name)
        result = dict(model=model, merged=len(new_rows), files=[str(p) for p in paths],
            archived_to=str(archived), completed_records=len(by_key))
        write_json(output/f'merged-shards-{model}.json', result)
        return result


def launch(args):
    experiment = save_plan(args)
    assert len(set(args.gpus)) == len(args.gpus)
    if args.split_model:
        baseline = experiment['config']['baseline']
        expected = {(mode, seed, qid) for mode in experiment['modes']
            for seed in experiment['seeds'] for qid in experiment['question_ids']}
        assert expected <= {record_key(r) for r in read_model_records(args.output/baseline)}, \
            'Split scheduling requires a completed baseline'
        active = [m for m in experiment['models'] if m != baseline]
        assert len(args.gpus) == 5 and len(active) == 4 and args.split_model in active, \
            '--split-model requires five GPUs, one complete baseline and four candidate models'
        others = [m for m in active if m != args.split_model]
        schedule = [(args.gpus[0], [dict(model=args.split_model, shard=(0, 2))]),
                    (args.gpus[1], [dict(model=args.split_model, shard=(1, 2))])]
        schedule += [(gpu, [dict(model=model)]) for gpu, model in zip(args.gpus[2:], others)]
    else:
        schedule = [(gpu, [dict(model=model) for model in experiment['models'][i::len(args.gpus)]])
                    for i, gpu in enumerate(args.gpus)]
    started = time.perf_counter()
    stamp = time.time_ns()
    hardware = subprocess.check_output(['nvidia-smi',
        '--query-gpu=index,uuid,name,driver_version,memory.total', '--format=csv'], text=True)
    (args.output / f'hardware-{stamp}.csv').write_text(hardware)
    gpu_samples = (args.output / f'gpu_samples-{stamp}.csv').open('w')
    monitor = subprocess.Popen(['nvidia-smi',
        '--query-gpu=timestamp,index,uuid,utilization.gpu,memory.used,power.draw',
        '--format=csv', '--loop=10'], stdout=gpu_samples, stderr=subprocess.DEVNULL)
    env = dict(os.environ, OMP_NUM_THREADS='8', VLLM_NO_USAGE_STATS='1', HF_HUB_OFFLINE='1',
        VLLM_USE_V2_MODEL_RUNNER='1', LD_LIBRARY_PATH=str(ROOT/'build/cuda-compat/cuda-compat') +
        (':' + os.environ['LD_LIBRARY_PATH'] if os.environ.get('LD_LIBRARY_PATH') else ''))
    # Keep fixed model-to-GPU assignments across resumes, including when jobs are complete.
    children = []
    def stop(signum, frame):
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGINT)
    prior = {sig: signal.signal(sig, stop) for sig in [signal.SIGINT, signal.SIGTERM]}
    def slot(item):
        gpu, tasks = item
        jobs = []
        for task in tasks:
            model, shard = task['model'], task.get('shard')
            command = [sys.executable, '-u', str(Path(__file__).resolve()), 'worker',
                '--experiment', str((args.output / 'experiment.json').resolve()), '--model', model]
            if shard:
                command += ['--task-shard-index', str(shard[0]), '--task-shard-count', str(shard[1])]
            if args.task_modes:
                command += ['--task-modes', *args.task_modes]
            if args.task_question_ids:
                command += ['--task-question-ids', *args.task_question_ids]
            if args.task_seeds:
                command += ['--task-seeds', *map(str, args.task_seeds)]
            before = time.perf_counter()
            label = f'{model}.shard-{shard[0]}-of-{shard[1]}' if shard else model
            with (args.output / f'{label}.{stamp}.log').open('w') as log:
                child = subprocess.Popen(command, cwd=ROOT, env=dict(env, CUDA_VISIBLE_DEVICES=str(gpu)),
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                children.append(child)
                code = child.wait()
            record = dict(model=model, gpu=gpu, shard=shard, exit_code=code, command=command,
                wall_seconds=time.perf_counter() - before, start_offset_seconds=before-started)
            jobs.append(record)
            print(json.dumps(record), flush=True)
            if code:
                break
        return jobs
    try:
        with ThreadPoolExecutor(max_workers=len(schedule)) as pool:
            jobs = [job for group in pool.map(slot, schedule) for job in group]
    finally:
        monitor.terminate()
        monitor.wait()
        gpu_samples.close()
        for sig, handler in prior.items():
            signal.signal(sig, handler)
    assert all(job['exit_code'] == 0 for job in jobs), 'Worker failed; inspect log and resume explicitly'
    if args.split_model:
        merge_model_shards(args.output, args.split_model)
    write_json(args.output / f'launch-{stamp}.json', dict(jobs=jobs, gpus=args.gpus,
        split_model=args.split_model,
        wall_seconds=time.perf_counter()-started, reservation_gpu_hours=(time.perf_counter()-started)*len(args.gpus)/3600))


def status(args):
    experiment = json.loads((args.output / 'experiment.json').read_text())
    now = datetime.now(timezone.utc)
    by_model = {}
    cell_rows = []
    for model in experiment['models']:
        rows = read_model_records(args.output / model, allow_partial_tail=True)
        records = {record_key(r): r for r in rows}
        attempts_dir = args.output / model / 'attempts'
        attempts = [json.loads(p.read_text()) for p in sorted(attempts_dir.glob('*.json'))] \
            if attempts_dir.exists() else []
        failed = set()
        for attempt in attempts:
            task = attempt.get('current_task')
            if task and attempt.get('status') in ['failed', 'interrupted']:
                key = task['mode'], task['seed'], task['question']
                if key not in records:
                    failed.add(key)
        total_gpu_seconds = 0.
        for attempt in attempts:
            if attempt.get('status') == 'running' and attempt.get('start_utc'):
                started = datetime.fromisoformat(attempt['start_utc'])
                total_gpu_seconds += max(0., (now - started).total_seconds())
            else:
                total_gpu_seconds += attempt.get('wall_seconds', 0.)
        by_model[model] = dict(records=records, attempts=attempts, failed=failed,
            gpu_hours=total_gpu_seconds / 3600)
    matched_questions = [q for q in experiment['question_ids'] if all(
        (mode, seed, q) in by_model[model]['records'] for model in experiment['models']
        for mode in experiment['modes'] for seed in experiment['seeds'])]
    for model in experiment['models']:
        values = by_model[model]
        rows = list(values['records'].values())
        model_cells = []
        for mode in experiment['modes']:
            for seed in experiment['seeds']:
                cell = [r for r in rows if r['mode'] == mode and r['seed'] == seed]
                errors = sum(1 for m, s, q in values['failed'] if m == mode and s == seed)
                seconds = [r['preprocessing_seconds'] + r['inference_seconds'] + r['scoring_seconds']
                           for r in cell]
                mean_seconds = sum(seconds) / len(seconds) if seconds else None
                pending = len(experiment['question_ids']) - len(cell) - errors
                progress = dict(model=model, mode=mode, seed=seed,
                    completed=len(cell), target=len(experiment['question_ids']),
                    correct=sum(bool(r['correct']) for r in cell), incorrect=sum(not r['correct'] for r in cell),
                    failed=errors, pending=pending,
                    truncated=sum(bool(r['truncated']) for r in cell),
                    missing_final=sum(bool(r['missing_final']) for r in cell),
                    input_tokens=sum(r['input_tokens'] for r in cell),
                    output_tokens=sum(r['output_tokens'] for r in cell),
                    reasoning_tokens=sum(r['reasoning_tokens'] for r in cell),
                    final_answer_tokens=sum(r.get('final_answer_tokens', 0) for r in cell),
                    inference_seconds=sum(r['inference_seconds'] for r in cell),
                    mean_seconds_per_completed=mean_seconds,
                    eta_seconds=pending * mean_seconds if mean_seconds is not None else None)
                cell_rows.append(progress)
                model_cells.append(progress)
        print(json.dumps(dict(model=model, completed=len(rows), target=len(experiment['question_ids']) *
            len(experiment['modes']) * len(experiment['seeds']), gpu_hours=values['gpu_hours'],
            attempts=len(values['attempts']), truncated=sum(r['truncated'] for r in rows),
            missing_final=sum(r['missing_final'] for r in rows), cells=model_cells)))
    args.output.mkdir(parents=True, exist_ok=True)
    progress_path = args.output / 'progress.json'
    write_attempt(progress_path, dict(updated_utc=now.isoformat(),
        planned_evaluations=experiment['planned_evaluations'],
        completed_evaluations=sum(len(v['records']) for v in by_model.values()),
        completed_paired_questions=len(matched_questions), paired_question_ids=matched_questions,
        gpu_hours=sum(v['gpu_hours'] for v in by_model.values()), cells=cell_rows))
    csv_path = args.output / 'progress.csv'
    temporary = csv_path.with_suffix('.tmp')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(cell_rows[0]))
        writer.writeheader()
        writer.writerows(cell_rows)
    temporary.replace(csv_path)


def verify(args):
    """Replay saved finals with the unmodified Google CLI; no model inference."""
    from transformers import AutoProcessor
    experiment = json.loads((args.runs / 'experiment.json').read_text())
    processor = AutoProcessor.from_pretrained(BASE, local_files_only=True)
    by_id = {r['id']: r for r in experiment['questions']}
    args.output.mkdir(parents=True, exist_ok=False)
    checked, prompt_ids = 0, {}
    for model in experiment['models']:
        rows = read_model_records(args.runs / model)
        assert len(rows) == len(by_id) * len(experiment['modes']) * len(experiment['seeds'])
        for mode in experiment['modes']:
            for seed in experiment['seeds']:
                subset = [r for r in rows if r['mode'] == mode and r['seed'] == seed]
                dest = args.output / model / mode / str(seed)
                dest.mkdir(parents=True)
                inputs, responses = dest / 'inputs.jsonl', dest / 'responses.jsonl'
                inputs.write_text(''.join(json.dumps(by_id[r['id']]['example']) + '\n' for r in subset))
                responses.write_text(''.join(json.dumps(dict(prompt=by_id[r['id']]['prompt'],
                    response=r['final_answer'])) + '\n' for r in subset))
                with (dest / 'google_cli.log').open('w') as log:
                    subprocess.run([sys.executable, '-m', 'instruction_following_eval.evaluation_main',
                        '--input_data', str(inputs.resolve()), '--input_response_data', str(responses.resolve()),
                        '--output_dir', str(dest.resolve())], cwd=TOOLS/'google-research', check=True,
                        env=dict(os.environ, NLTK_DATA=str(ROOT/'assets/benchmark/nltk_data')),
                        stdout=log, stderr=subprocess.STDOUT)
                for grading in ['strict', 'loose']:
                    grades = [json.loads(line) for line in (dest/f'eval_results_{grading}.jsonl').read_text().splitlines()]
                    assert len(grades) == len(subset)
                    for row, grade in zip(subset, grades):
                        assert grade['prompt'] == by_id[row['id']]['prompt']
                        assert grade['follow_all_instructions'] == row['correct' if grading == 'strict' else 'loose_correct']
                        assert grade['follow_instruction_list'] == row[grading+'_instructions']
                for row in subset:
                    from types import SimpleNamespace
                    settings = experiment['config']['modes'][mode]
                    parsed = extract_answer(processor.tokenizer,
                        SimpleNamespace(text=row['raw_output'], token_ids=row['token_ids']),
                        row['prompt_token_ids'], settings)
                    assert all(parsed[k] == row[k] for k in parsed)
                    assert record_sampling_matches(row, settings, seed)
                    assert processor.tokenizer.decode(row['token_ids'], skip_special_tokens=True) == row['raw_output']
                    prompt = render_prompt(processor, by_id[row['id']], settings)
                    ids = processor.tokenizer.encode(prompt, add_special_tokens=False)
                    assert ids == row['prompt_token_ids']
                    key = mode, row['id']
                    assert prompt_ids.setdefault(key, ids) == ids, 'Prompts differ across model/seed'
                    checked += 1
    write_json(args.output / 'audit.json', dict(passed=True, official_cli_responses=checked,
        parser_replays=checked, engine_sampling_matches=checked,
        prompt_tokens_identical_across_models_and_seeds=True,
        google_revision=experiment['config']['google_ifeval_revision'], inference_started=False))
    print(f'Official CLI, parser, sampling and token audits passed: {checked}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    for name in ['plan', 'run']:
        command = sub.add_parser(name)
        command.add_argument('--config', type=Path, default=CONFIG)
        command.add_argument('--data', type=Path, default=ROOT/'assets/benchmark/prepared_v2')
        command.add_argument('--output', type=Path, required=True)
        command.add_argument('--models', nargs='+')
        command.add_argument('--modes', nargs='+', choices=['off', 'on'], default=['off', 'on'])
        command.add_argument('--seeds', type=int, nargs='+')
        command.add_argument('--start', type=int, default=0)
        command.add_argument('--stop', type=int)
        command.add_argument('--question-ids', nargs='+')
        command.add_argument('--full', action='store_true', help='Explicit full 541-question selection')
        command.add_argument('--resume', action='store_true')
        if name == 'run':
            command.add_argument('--gpus', type=int, nargs='+', default=[0, 1, 2, 3, 4])
            command.add_argument('--split-model', help='Split one candidate across GPUs 0 and 1 after baseline completion')
    command = sub.add_parser('worker')
    command.add_argument('--experiment', type=Path, required=True)
    command.add_argument('--model', required=True)
    command.add_argument('--max-new-records', type=int, help='Bounded partial run for resume validation')
    command.add_argument('--task-modes', nargs='+', choices=['off', 'on'])
    command.add_argument('--task-question-ids', nargs='+')
    command.add_argument('--task-seeds', nargs='+', type=int)
    command.add_argument('--task-shard-index', type=int, default=0)
    command.add_argument('--task-shard-count', type=int, default=1)
    command = sub.add_parser('status')
    command.add_argument('--output', type=Path, required=True)
    sub.choices['run'].add_argument('--task-modes', nargs='+', choices=['off', 'on'])
    sub.choices['run'].add_argument('--task-question-ids', nargs='+')
    sub.choices['run'].add_argument('--task-seeds', nargs='+', type=int)
    command = sub.add_parser('verify')
    command.add_argument('--runs', type=Path, required=True)
    command.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    {'plan': save_plan, 'run': launch, 'worker': worker, 'status': status, 'verify': verify}[args.command](args)


if __name__ == '__main__':
    main()
