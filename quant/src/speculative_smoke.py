"""Isolated startup and generation checks for native MTP and the public DSpark draft."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from fidelity import write_json


def worker(args):
    import torch
    from vllm import LLM, SamplingParams
    record = dict(model=args.models[0], mode=args.modes[0], status='failed',
        gpu=torch.cuda.get_device_name(0), kv_cache_dtype='bfloat16',
        packages={p: importlib.metadata.version(p) for p in ['vllm', 'torch', 'transformers']})
    started = time.perf_counter()
    try:
        speculative = dict(method=record['mode'], num_speculative_tokens=2)
        if record['mode'] == 'dspark':
            speculative.update(model='/root/model/.dspark-Qwen3.8-27B',
                               num_speculative_tokens=7, enforce_eager=True)
        record['speculative_config'] = speculative
        llm = LLM(model='/root/model/'+record['model'],
            tokenizer='/root/model/Qwen-Qwen3.8-27B', dtype='bfloat16', seed=0,
            max_model_len=2048, max_num_batched_tokens=2048, max_num_seqs=1,
            gpu_memory_utilization=.85, enforce_eager=True, enable_prefix_caching=False,
            language_model_only=True, kv_cache_dtype='bfloat16',
            speculative_config=speculative, generation_config='vllm',
            kernel_config={'linear_backend_per_quant': {'fp8_block_w8a8': 'triton'}})
        prompts = ['The sum of 17 and 25 is', 'def fibonacci(n):\n',
                   'Explain why the sky appears blue in one sentence.']
        results = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=32, seed=0), use_tqdm=False)
        record['outputs'] = [dict(prompt=prompt, text=r.outputs[0].text,
                                 token_ids=r.outputs[0].token_ids) for prompt, r in zip(prompts, results)]
        assert all(r['token_ids'] for r in record['outputs'])
        llm.llm_engine.engine_core.shutdown()
        record['status'] = 'ok'
    except Exception as exc:
        traceback.print_exc()
        record['reason'] = f'{type(exc).__name__}: {exc}'
    record['seconds'] = time.perf_counter()-started
    write_json(args.output/'result.json', record)
    return int(record['status'] != 'ok')


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    tasks = [(model, mode) for model in args.models for mode in args.modes]
    def execute(task, gpu):
        model, mode = task
        destination = args.output / (model+'_'+mode)
        destination.mkdir()
        command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--models',
                   model, '--modes', mode, '--output', str(destination.resolve())]
        with (destination/'run.log').open('w') as log:
            code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), VLLM_USE_V2_MODEL_RUNNER='1')).returncode
        path = destination/'result.json'
        result = json.loads(path.read_text()) if path.exists() else dict(
            model=model, mode=mode, status='failed', reason='Worker exited before result; inspect run.log')
        result['worker_exit_code'] = code
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return result
    results = []
    for offset in range(0, len(tasks), len(args.gpus)):
        with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
            futures = [pool.submit(execute, task, gpu)
                       for task, gpu in zip(tasks[offset:offset+len(args.gpus)], args.gpus)]
            results.extend(f.result() for f in futures)
        write_json(args.output/'results.json', results)
    return 0


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models', nargs='+', required=True)
    p.add_argument('--modes', nargs='+', choices=['mtp', 'dspark'], default=['mtp', 'dspark'])
    p.add_argument('--gpus', type=int, nargs='+', default=[2, 3])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--worker', action='store_true')
    a = p.parse_args()
    sys.exit(worker(a) if a.worker else run(a))
