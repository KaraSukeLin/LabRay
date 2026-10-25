"""Thin vLLM runner using the official Google IFEval and MMMU-Pro scorers."""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import os
import sys
import time

from fidelity import write_json

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / 'build/autoquant/tools'
BASE = Path('/root/model/Qwen-Qwen3.8-27B')
CONFIG = ROOT / 'assets/benchmark_config.json'


def official_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def scorers():
    sys.path.insert(0, str(TOOLS / 'google-research'))
    from instruction_following_eval import evaluation_lib
    import langdetect
    import nltk
    langdetect.DetectorFactory.seed = 0
    nltk.data.path.insert(0, str(ROOT / 'assets/benchmark/nltk_data'))
    mmmu = official_module('official_mmmu', TOOLS / 'MMMU/mmmu-pro/evaluate.py')
    return evaluation_lib, mmmu


def score(row, response, ifeval, mmmu):
    # Official MMMU uses a random guess when extraction fails. Seed it per question,
    # identically for every model; preserve and report that official behavior.
    random.seed(int(hashlib.sha256(row['id'].encode()).hexdigest()[:16], 16))
    if row['task'] == 'ifeval':
        item = ifeval.InputExample(**row['example'])
        strict = ifeval.test_instruction_following_strict(item, {item.prompt: response})
        loose = ifeval.test_instruction_following_loose(item, {item.prompt: response})
        return dict(correct=bool(strict.follow_all_instructions),
                    strict_instructions=strict.follow_instruction_list,
                    loose_correct=bool(loose.follow_all_instructions),
                    loose_instructions=loose.follow_instruction_list,
                    answer=response.strip())
    choices, letters = mmmu.get_multi_choice_info(ast.literal_eval(str(row['options'])))
    used_fallback = False
    choice = random.choice
    def record_choice(values):
        nonlocal used_fallback
        used_fallback = True
        return choice(values)
    random.choice = record_choice
    try:
        answer = mmmu.parse_multi_choice_response(response, letters, choices)
    finally:
        random.choice = choice
    return dict(correct=bool(mmmu.eval_multi_choice(row['gold'], answer)), answer=answer,
                extraction_fallback=used_fallback)


def prepare(args):
    import io
    import pyarrow.parquet as pq
    from PIL import Image
    import yaml
    config = json.loads(CONFIG.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    ifeval = ROOT / 'assets/benchmark/ifeval/input_data_google_repository.jsonl'
    for line in ifeval.read_text().splitlines():
        item = json.loads(line)
        rows.append(dict(id=f'ifeval/{item["key"]}', task='ifeval', stratum='ifeval',
                         prompt=item['prompt'], images=[], example=item))
    helper = official_module('official_mmmu_prompts', TOOLS / 'MMMU/mmmu-pro/infer/infer_transformers.py')
    helper.prompt_config = yaml.safe_load((TOOLS / 'MMMU/mmmu-pro/prompts.yaml').read_text())['direct']
    sources = []
    for variant, task in [('standard (10 options)', 'mmmu_standard'), ('vision', 'mmmu_vision')]:
        files = sorted((ROOT / 'build/benchmark/sources/MMMU_Pro' / variant).glob('*.parquet'))
        assert files, variant
        for path in files:
            sources.append(dict(file=str(path.relative_to(ROOT)),
                sha256=hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()))
            for batch in pq.ParquetFile(path).iter_batches(batch_size=8):
                for item in batch.to_pylist():
                    for key in list(item):
                        if key.startswith('image') and item[key] is not None:
                            image = item[key]
                            item[key] = Image.open(io.BytesIO(image['bytes'])).convert('RGB')
                    prompt, images = helper.process_prompt(item, variant)
                    paths = []
                    for index, image in enumerate(images):
                        destination = args.output / 'images' / f'{task}_{item["id"]}_{index}.png'
                        destination.parent.mkdir(exist_ok=True)
                        image.save(destination)
                        paths.append(str(destination.resolve()))
                    rows.append(dict(id=f'{task}/{item["id"]}', task=task,
                        stratum=item['subject'],
                        prompt=prompt, images=paths, options=item['options'], gold=item['answer']))
    assert len({r['id'] for r in rows}) == len(rows)
    for row in rows:
        row['image_sha256'] = [hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in row['images']]
    write_json(args.output / 'questions.json', rows)
    write_json(args.output / 'manifest.json', dict(source_revisions={
        k: v for k, v in config.items() if k.endswith('_revision')}, prompt_mode='direct', source_files=sources,
        ifeval_canonical_source=dict(revision=config['google_ifeval_revision'],
            sha256=hashlib.sha256(ifeval.read_bytes()).hexdigest(),
            url=f'https://raw.githubusercontent.com/google-research/google-research/{config["google_ifeval_revision"]}/instruction_following_eval/data/input_data.jsonl'),
        questions_sha256=hashlib.sha256((args.output / 'questions.json').read_bytes()).hexdigest(),
        counts={task: sum(r['task'] == task for r in rows) for task in config['max_tokens']},
        tools={name: str(path) for name, path in [('google', TOOLS/'google-research/revision.json'),
                                                 ('mmmu', TOOLS/'MMMU')]}))


def create_llm(model, config, seed, mtp=False):
    """Shared, validated TP=1 / BF16-KV engine settings for benchmark workers."""
    from vllm import LLM
    return LLM(model=str(Path('/root/model') / model), tokenizer=str(BASE), dtype='bfloat16',
        tensor_parallel_size=1, seed=seed, max_model_len=config['max_model_len'],
        max_num_seqs=1, max_num_batched_tokens=4096, gpu_memory_utilization=.85,
        enforce_eager=True, enable_prefix_caching=False,
        speculative_config=dict(method='mtp', num_speculative_tokens=2) if mtp else None,
        disable_log_stats=False, kv_cache_dtype='bfloat16', generation_config='vllm',
        limit_mm_per_prompt={'image': config['max_images'], 'video': 0}, skip_mm_profiling=True,
        mm_processor_kwargs={k: config[k] for k in ['min_pixels', 'max_pixels']},
        kernel_config={'linear_backend_per_quant': {'fp8_block_w8a8': 'triton'}})


def run(args):
    run_started = time.perf_counter()
    import importlib.metadata
    from PIL import Image
    from transformers import AutoProcessor
    from vllm import SamplingParams
    config = json.loads(args.config.read_text())
    config['mtp'] = args.mtp
    assert args.model in config['models']
    args.output.mkdir(parents=True, exist_ok=False)
    rows = json.loads((args.data / 'questions.json').read_text())
    selected = [r for r in rows if r['task'] in args.tasks]
    if args.limit:
        selected = [r for task in args.tasks
                    for r in [x for x in selected if x['task'] == task][:args.limit]]
    processor = AutoProcessor.from_pretrained(BASE, local_files_only=True)
    ifeval, mmmu = scorers()
    setup_seconds = time.perf_counter() - run_started
    extension_ids = set(json.loads(args.long_ids.read_text())) if args.long_ids else set()
    started = time.perf_counter()
    speculative = dict(method='mtp', num_speculative_tokens=2) if args.mtp else None
    llm = create_llm(args.model, config, config['seed'], args.mtp)
    initialization_seconds = time.perf_counter() - started
    timing = dict(preprocessing_seconds=0., inference_seconds=0., scoring_seconds=0.)
    with (args.output / 'questions.jsonl').open('w') as output, \
            (args.output / 'extensions.jsonl').open('w') as extensions:
        for row in selected:
            preprocessing_started = time.perf_counter()
            images = [Image.open(p).convert('RGB') for p in row['images']]
            assert row['image_sha256'] == [hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in row['images']]
            # Match the official transformers runner's text and recorded image order.
            content = [{'type': 'text', 'text': row['prompt']}]
            content.extend({'type': 'image', 'image': image} for image in images)
            prompt = processor.apply_chat_template([{'role': 'user', 'content': content}],
                tokenize=False, add_generation_prompt=True, enable_thinking=config['enable_thinking'])
            request = {'prompt': prompt}
            if images:
                request['multi_modal_data'] = {'image': images}
            preprocessing_seconds = time.perf_counter() - preprocessing_started
            timing['preprocessing_seconds'] += preprocessing_seconds
            sampling = dict(temperature=config['temperature'], top_p=config['top_p'],
                top_k=config['top_k'], seed=config['seed'],
                min_p=config.get('min_p', 0.), presence_penalty=config.get('presence_penalty', 0.),
                repetition_penalty=config.get('repetition_penalty', 1.),
                max_tokens=args.max_tokens or config['max_tokens'][row['task']])
            # Long-cap probes use the same prompt/seed and official scoring. Keep
            # them separate so the main paired comparison always has one cap.
            def generate_record(cap):
                inference_started = time.perf_counter()
                result = llm.generate([request], SamplingParams(**dict(sampling, max_tokens=cap)), use_tqdm=False)[0]
                inference_seconds = time.perf_counter() - inference_started
                generated = result.outputs[0]
                scoring_started = time.perf_counter()
                graded = score(row, generated.text, ifeval, mmmu)
                scoring_seconds = time.perf_counter() - scoring_started
                timing['inference_seconds'] += inference_seconds
                timing['scoring_seconds'] += scoring_seconds
                return dict(id=row['id'], task=row['task'], stratum=row['stratum'],
                    response=generated.text, token_ids=generated.token_ids,
                    prompt_token_ids=result.prompt_token_ids, finish_reason=generated.finish_reason,
                    stop_reason=generated.stop_reason, max_tokens=cap,
                    inference_seconds=inference_seconds, scoring_seconds=scoring_seconds,
                    preprocessing_seconds=preprocessing_seconds,
                    input_tokens=len(result.prompt_token_ids), output_tokens=len(generated.token_ids),
                    output_tokens_per_second=len(generated.token_ids)/inference_seconds,
                    image_count=len(images), image_sha256=row['image_sha256'], **graded)
            record = generate_record(sampling['max_tokens'])
            output.write(json.dumps(record, ensure_ascii=False) + '\n')
            output.flush()
            print(row['id'], 'correct', record['correct'], 'tokens', record['output_tokens'],
                  'finish', record['finish_reason'], flush=True)
            if args.long_max_tokens and row['task'] != 'ifeval' and (
                    record['finish_reason'] == 'length' or row['id'] in extension_ids):
                extended = generate_record(args.long_max_tokens)
                extended['probe_reason'] = 'truncated' if record['finish_reason'] == 'length' else 'fixed_control'
                extended['main_prefix_equal'] = extended['token_ids'][:record['output_tokens']] == record['token_ids']
                extensions.write(json.dumps(extended, ensure_ascii=False) + '\n')
                extensions.flush()
                print(row['id'], 'long_cap', extended['output_tokens'], extended['finish_reason'], flush=True)
    speculative_metrics = [vars(metric) for metric in llm.get_metrics()
                           if metric.name.startswith('vllm:spec_decode_')]
    shutdown_started = time.perf_counter()
    llm.llm_engine.engine_core.shutdown()
    shutdown_seconds = time.perf_counter() - shutdown_started
    write_json(args.output / 'run.json', dict(status='ok', model=args.model,
        smoke=bool(args.limit or len(selected) < 4001), limit_per_task=args.limit, max_tokens_override=args.max_tokens,
        tasks=args.tasks, questions=len(selected), seconds=time.perf_counter()-started,
        run_wall_seconds=time.perf_counter()-run_started, setup_seconds=setup_seconds,
        shutdown_seconds=shutdown_seconds, **timing,
        long_max_tokens=args.long_max_tokens, long_ids=sorted(extension_ids),
        execution=dict(gpu_count=1, tensor_parallel_size=1, max_num_seqs=1,
            max_num_batched_tokens=4096, enforce_eager=True, enable_prefix_caching=False,
            cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
            omp_num_threads=os.environ.get('OMP_NUM_THREADS')),
        initialization_seconds=initialization_seconds, speculative_config=speculative,
        speculative_metrics=speculative_metrics,
        config=config, config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
        dataset_sha256=hashlib.sha256((args.data / 'questions.json').read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        environment={p: importlib.metadata.version(p) for p in ['vllm', 'torch', 'transformers']}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--output', type=Path, default=ROOT/'assets/benchmark/prepared_v2')
    inference = sub.add_parser('run')
    inference.add_argument('--model', required=True)
    inference.add_argument('--config', type=Path, default=CONFIG)
    inference.add_argument('--mtp', action='store_true', help='Native MTP with two speculative tokens')
    inference.add_argument('--data', type=Path, default=ROOT/'assets/benchmark/prepared_v2')
    inference.add_argument('--output', type=Path, required=True)
    inference.add_argument('--tasks', nargs='+', choices=['ifeval', 'mmmu_standard', 'mmmu_vision'],
                           default=['ifeval', 'mmmu_standard', 'mmmu_vision'])
    inference.add_argument('--limit', type=int, help='Smoke test questions per task; omit only for a full authorized run')
    inference.add_argument('--max-tokens', type=int, help='Smoke test override')
    inference.add_argument('--long-max-tokens', type=int, help='Separate MMMU long-cap probes on truncated and fixed-control questions')
    inference.add_argument('--long-ids', type=Path, help='JSON list of fixed-control question IDs for long-cap probes')
    args = p.parse_args()
    (prepare if args.command == 'prepare' else run)(args)
