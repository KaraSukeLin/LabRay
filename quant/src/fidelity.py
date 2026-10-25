#!/usr/bin/env python3
"""Full-vocabulary, teacher-forced fidelity evaluation with vLLM."""
import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS = Path('/root/model')
BASELINE = 'Qwen-Qwen3.8-27B'


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def full_logprobs(flat, token_ids, vocab_size):
    """vLLM prompt position j already predicts input token j from prefix [:j]."""
    if len(flat.start_indices) != len(token_ids):
        raise ValueError('Prompt logprob positions do not match the input tokens')
    if flat.end_indices[0] != flat.start_indices[0]:
        raise ValueError('The first token must have no prompt logprob')
    result = np.full((len(token_ids) - 1, vocab_size), np.nan, dtype=np.float32)
    for j in range(1, len(token_ids)):
        start, end = flat.start_indices[j], flat.end_indices[j]
        # The target may appear twice; both entries carry the same logprob.
        result[j - 1, flat.token_ids[start:end]] = flat.logprobs[start:end]
    return validate_logprobs(result)


def validate_logprobs(result):
    if np.isnan(result).any():
        raise ValueError('incomplete vocabulary or NaN log probabilities; full KL unavailable')
    mass = np.exp(result.astype(np.float64)).sum(axis=1)
    if not np.allclose(mass, 1.0, atol=2e-5, rtol=0):
        raise ValueError('Full-vocabulary probabilities do not sum to one')
    return result


def compare_arrays(reference, candidate, token_ids):
    targets = np.asarray(token_ids[1:], dtype=np.int64)
    if reference.shape != candidate.shape or reference.shape[0] != len(targets):
        raise ValueError('Distribution or target shape mismatch')
    p_log, q_log = reference.astype(np.float64), candidate.astype(np.float64)
    p = np.exp(p_log)
    # Padded / masked vocab entries can have logprob -inf in both runs.
    active = p > 0
    kl = np.sum(p[active] * (p_log[active] - q_log[active]))
    nll = -candidate[np.arange(len(targets)), targets].astype(np.float64).sum()
    ref_nll = -reference[np.arange(len(targets)), targets].astype(np.float64).sum()
    flips = np.count_nonzero(reference.argmax(axis=1) != candidate.argmax(axis=1))
    return dict(tokens=len(targets), nll=float(nll), ref_nll=float(ref_nll),
                kl_sum=float(kl), flips=int(flips))


def aggregate(parts):
    count = sum(p['tokens'] for p in parts)
    ppl = math.exp(sum(p['nll'] for p in parts) / count)
    ref_ppl = math.exp(sum(p['ref_nll'] for p in parts) / count)
    return dict(tokens=count, ppl=ppl, reference_ppl=ref_ppl,
                ppl_delta=ppl - ref_ppl,
                kl_bf16_to_candidate=sum(p['kl_sum'] for p in parts) / count,
                top1_flip_rate=sum(p['flips'] for p in parts) / count)


def sanity_passed(metrics):
    return (abs(metrics['kl_bf16_to_candidate']) <= 1e-6
            and metrics['top1_flip_rate'] == 0
            and abs(metrics['ppl_delta']) <= 1e-6)


def capture(args):
    import torch
    from vllm import LLM, SamplingParams

    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    tokens = json.loads(args.tokens.read_text())
    config = json.loads((args.model / 'config.json').read_text())
    text_config = config.get('text_config', config)
    vocab_size = text_config['vocab_size']
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    record = dict(model=args.model.name, model_path=str(args.model), status='failed',
                  environment={p: importlib.metadata.version(p) for p in
                               ['vllm', 'torch', 'transformers', 'compressed-tensors']},
                  gpu=torch.cuda.get_device_name(0), cuda=torch.version.cuda,
                  mtp=False, kv_cache_dtype='bfloat16', quant_method=config.get(
                      'quantization_config', {}).get('quant_method', 'none'))
    try:
        llm = LLM(model=str(args.model), tokenizer=tokens['tokenizer'],
                  dtype='bfloat16', tensor_parallel_size=1, seed=0,
                  max_model_len=tokens.get('max_model_len', 512), max_num_seqs=1,
                  max_num_batched_tokens=tokens.get('max_model_len', 512),
                  gpu_memory_utilization=0.85, enforce_eager=True,
                  enable_prefix_caching=False, enable_chunked_prefill=True,
                  speculative_config=None, kv_cache_dtype='bfloat16',
                  language_model_only=True, max_logprobs=-1,
                  kernel_config={'linear_backend_per_quant': {'fp8_block_w8a8': 'triton'}},
                  logprobs_mode='raw_logprobs', generation_config='vllm')
        record['loaded'] = True
        params = SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=-1,
                                flat_logprobs=True, detokenize=False, seed=0)
        parts, hashes, generated = [], [], []
        starts = tokens.get('score_starts', [1] * len(tokens['input_ids']))
        dense = args.logprob_format == 'dense'
        vocab_ids = list(range(vocab_size)) if dense else None
        record.update(logprob_format=args.logprob_format,
                      tokens_sha256=hashlib.sha256(args.tokens.read_bytes()).hexdigest(),
                      source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      v2_model_runner=os.getenv('VLLM_USE_V2_MODEL_RUNNER'))
        for i, ids in enumerate(tokens['input_ids']):
            start = starts[i]
            if dense:
                params = SamplingParams(temperature=0, max_tokens=1, detokenize=False, seed=0,
                    prompt_logprob_token_ids=vocab_ids, prompt_logprob_start=start - 1)
            output = llm.generate([{'prompt_token_ids': ids}], params,
                                  use_tqdm=False)[0]
            if output.prompt_token_ids != ids:
                raise ValueError('The engine changed the shared input token IDs')
            logp = (validate_logprobs(output.prompt_token_id_logprobs) if dense else
                    full_logprobs(output.prompt_logprobs, ids, vocab_size)[start - 1:])
            if logp.shape != (len(ids) - start, vocab_size):
                raise ValueError('Scored distribution shape mismatch')
            if args.reference is None:
                np.save(args.output / f'{i}.npy', logp)
            reference = np.load(args.reference / f'{i}.npy', mmap_mode='r') if args.reference else logp
            part = compare_arrays(reference, logp, ids[start - 1:])
            if 'samples' in tokens:
                part.update({k: tokens['samples'][i][k] for k in
                             ['id', 'document_id', 'domain', 'language', 'context']})
            parts.append(part)
            hashes.append(hashlib.sha256(logp.tobytes()).hexdigest())
            generated.append(output.outputs[0].token_ids)
            print(f'Captured sample {i + 1}/{len(tokens["input_ids"])}', flush=True)
            del output, reference, logp
        llm.llm_engine.engine_core.shutdown()
        record.update(aggregate(parts), status='ok', vocab_size=vocab_size,
                      distribution_sha256=hashes, basic_inference_token_ids=generated,
                      documents=parts)
    except Exception as exc:
        record['reason'] = f'{type(exc).__name__}: {exc}'
        traceback.print_exc()
    record['elapsed_seconds'] = time.time() - started
    write_json(args.output / 'result.json', record)
    return 0 if record['status'] == 'ok' else 1


def report(output, rows):
    columns = ['model', 'status', 'ppl', 'ppl_delta', 'kl_bf16_to_candidate',
               'top1_flip_rate', 'tokens', 'reason']
    with (output / 'comparison.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    lines = ['| Model | Status | PPL | PPL delta | KL (BF16 → candidate) | Top-1 flip rate | Tokens | Reason |',
             '|---|---|---:|---:|---:|---:|---:|---|']
    for row in rows:
        cells = []
        for key in columns:
            value = row.get(key, '')
            if isinstance(value, float):
                value = f'{value:.10g}'
            cells.append(str(value).replace('|', '\\|').replace('\n', ' '))
        lines.append('| ' + ' | '.join(cells) + ' |')
    (output / 'comparison.md').write_text('\n'.join(lines) + '\n')


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    baseline = args.baseline.resolve()
    args.model_root = args.model_root.resolve()
    if args.tokens:
        tokens = json.loads(args.tokens.read_text())
        tokens['tokenizer'] = str(baseline)
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(baseline), local_files_only=True)
        texts = json.loads(args.texts.read_text())
        ids = [tokenizer.encode(t, add_special_tokens=False)[:args.max_tokens] for t in texts]
        if not 2 <= args.max_tokens <= 511 or any(len(x) < 2 for x in ids):
            raise ValueError('Each sample needs 2..511 tokens')
        tokens = dict(tokenizer=str(baseline), input_ids=ids, max_tokens=args.max_tokens,
                      text_sha256=hashlib.sha256(args.texts.read_bytes()).hexdigest())
    write_json(args.output / 'tokens.json', tokens)
    models = (sorted(p for p in args.model_root.iterdir()
                     if p.is_dir() and not p.name.startswith('.')) if args.all else
              [p.resolve() if p.is_dir() else (args.model_root / p).resolve()
               for p in args.models])
    inventory = []
    for path in sorted(set([baseline] + models)):
        config_path = path / 'config.json'
        item = dict(model=path.name, path=str(path), config_exists=config_path.exists())
        index = path / 'model.safetensors.index.json'
        if index.exists():
            shards = sorted(set(json.loads(index.read_text())['weight_map'].values()))
            item.update(shards=shards, missing_shards=[s for s in shards if not (path / s).is_file()])
        if config_path.exists():
            c = json.loads(config_path.read_text())
            item.update(architectures=c.get('architectures'), quantization_config=c.get('quantization_config'),
                        config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest())
        inventory.append(item)
    write_json(args.output / 'inventory.json', inventory)
    rows = []

    def execute(model, name, reference=None, gpu=None):
        dest = args.output / name
        dest.mkdir()
        command = [sys.executable, str(Path(__file__).resolve()), 'capture', '--model',
                   str(model), '--tokens', str((args.output / 'tokens.json').resolve()),
                   '--output', str(dest.resolve()), '--logprob-format', args.logprob_format]
        if reference:
            command += ['--reference', str(reference.resolve())]
        print(f'Running {name}: {model}', flush=True)
        with (dest / 'run.log').open('w') as log:
            env = os.environ.copy()
            if gpu is not None:
                env['CUDA_VISIBLE_DEVICES'] = str(gpu)
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        result_path = dest / 'result.json'
        result = json.loads(result_path.read_text()) if result_path.exists() else dict(
            model=model.name, status='failed', reason=f'Worker exit {completed.returncode}; see {name}/run.log')
        result['run'] = name
        result['worker_exit_code'] = completed.returncode
        if completed.returncode != 0:
            result['status'] = 'failed'
            result['reason'] = f'Worker exit {completed.returncode}; see {name}/run.log'
        print(json.dumps({k: v for k, v in result.items() if k in
                          ['model', 'status', 'ppl', 'kl_bf16_to_candidate', 'top1_flip_rate', 'reason']}), flush=True)
        return result

    first = execute(baseline, 'bf16_reference', gpu=args.gpus[0])
    rows.append(first)
    report(args.output, rows)
    if first['status'] != 'ok':
        raise RuntimeError('BF16 reference failed; inspect its run.log')
    second = execute(baseline, 'bf16_repeat', args.output / 'bf16_reference', args.gpus[-1])
    rows.append(second)
    report(args.output, rows)
    passed = second['status'] == 'ok' and sanity_passed(second)
    write_json(args.output / 'sanity.json', dict(passed=passed, metrics=second))
    if not passed:
        raise RuntimeError('BF16 sanity check failed; candidate comparisons were not started')
    candidates = [model for model in models if model != baseline]
    for offset in range(0, len(candidates), len(args.gpus)):
        batch = candidates[offset:offset + len(args.gpus)]
        with ThreadPoolExecutor(max_workers=len(batch)) as pool:
            jobs = [pool.submit(execute, model, model.name, args.output / 'bf16_reference', gpu)
                    for model, gpu in zip(batch, args.gpus)]
            rows.extend(job.result() for job in jobs)
        report(args.output, rows)
    write_json(args.output / 'results.json', rows)
    return 0 if all(r['status'] == 'ok' for r in rows) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    run_parser = sub.add_parser('run', help='Run independent BF16 sanity, then candidates')
    run_parser.add_argument('--model-root', type=Path, default=DEFAULT_MODELS)
    run_parser.add_argument('--baseline', type=Path, default=DEFAULT_MODELS / BASELINE)
    select = run_parser.add_mutually_exclusive_group(required=True)
    select.add_argument('--all', action='store_true')
    select.add_argument('--models', type=Path, nargs='+')
    run_parser.add_argument('--texts', type=Path, default=ROOT / 'assets/fidelity_texts.json')
    run_parser.add_argument('--tokens', type=Path, help='Prepared shared token manifest')
    run_parser.add_argument('--gpus', nargs='+', default=[os.getenv('CUDA_VISIBLE_DEVICES', '0')],
                            help='One independent model per GPU, after sanity passes')
    run_parser.add_argument('--logprob-format', choices=['flat', 'dense'], default='flat')
    run_parser.add_argument('--max-tokens', type=int, default=64)
    run_parser.add_argument('--output', type=Path, required=True)
    cap = sub.add_parser('capture', help='Capture one model in a fresh process')
    cap.add_argument('--model', type=Path, required=True)
    cap.add_argument('--tokens', type=Path, required=True)
    cap.add_argument('--reference', type=Path)
    cap.add_argument('--logprob-format', choices=['flat', 'dense'], default='flat')
    cap.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    return run(args) if args.command == 'run' else capture(args)


if __name__ == '__main__':
    sys.exit(main())
