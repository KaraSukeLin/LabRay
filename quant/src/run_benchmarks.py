"""Run unchanged benchmark workers, one model per GPU, with a process-time ledger."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
import sys
import time

from benchmark import CONFIG, ROOT
from fidelity import write_json


def main(args):
    config = json.loads(args.config.read_text())
    models = args.models or config['models']
    counts = {model: len(json.loads(((args.data/model if args.per_model_data else args.data)/'questions.json').read_text())) for model in models}
    assert max(counts.values()) < 4001 or args.full, 'Full inference requires an explicit --full invocation'
    assert len(set(args.gpus)) == len(args.gpus)
    args.output.mkdir(parents=True, exist_ok=False)
    hardware = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,driver_version,memory.total,power.limit',
                                       '--format=csv'], text=True)
    (args.output/'hardware.csv').write_text(hardware)
    (args.output/'cpu.txt').write_text(subprocess.check_output(['lscpu'], text=True))
    jobs = Queue()
    for model in models:
        assert model in config['models']
        jobs.put(model)
    started = time.perf_counter()
    start_utc = datetime.now(timezone.utc).isoformat()
    env = dict(os.environ, OMP_NUM_THREADS='8', VLLM_NO_USAGE_STATS='1', HF_HUB_OFFLINE='1',
               VLLM_USE_V2_MODEL_RUNNER='1', LD_LIBRARY_PATH=str(ROOT/'build/cuda-compat/cuda-compat')+
               (':'+os.environ['LD_LIBRARY_PATH'] if os.environ.get('LD_LIBRARY_PATH') else ''))

    def worker(gpu):
        records = []
        while not jobs.empty():
            try:
                model = jobs.get_nowait()
            except Empty:
                break
            dest = args.output/model
            data = args.data/model if args.per_model_data else args.data
            command = [sys.executable, '-u', str(ROOT/'src/benchmark.py'), 'run', '--model', model,
                       '--config', str(args.config.resolve()), '--data', str(data.resolve()), '--output', str(dest.resolve())]
            if args.long_max_tokens:
                command += ['--long-max-tokens', str(args.long_max_tokens), '--long-ids', str((args.data/'long_ids.json').resolve())]
            before = time.perf_counter()
            with Path(str(dest)+'.log').open('w') as log:
                proc = subprocess.run(command, cwd=ROOT, env=dict(env, CUDA_VISIBLE_DEVICES=str(gpu)),
                                      stdout=log, stderr=subprocess.STDOUT)
            record = dict(model=model, gpu=gpu, command=command, exit_code=proc.returncode,
                          start_offset_seconds=before-started, process_wall_seconds=time.perf_counter()-before)
            write_json(args.output/(model+'.process.json'), record)
            records.append(record)
            print(json.dumps(record), flush=True)
        return records

    with (args.output/'gpu_samples.csv').open('w') as samples:
        monitor = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw,temperature.gpu',
                                    '--format=csv', '-l', '10'], stdout=samples, stderr=subprocess.STDOUT)
        try:
            with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
                records = [r for batch in pool.map(worker, args.gpus) for r in batch]
        finally:
            monitor.terminate()
            monitor.wait()
    report = dict(start_utc=start_utc, end_utc=datetime.now(timezone.utc).isoformat(),
                  wall_seconds=time.perf_counter()-started, gpus=args.gpus, mtp=False,
                  questions_per_model=counts if args.per_model_data else next(iter(counts.values())), models=models, jobs=records,
                  source='one TP=1 worker per physical GPU, sequential questions, shared filesystem')
    write_json(args.output/'launch.json', report)
    assert all(r['exit_code'] == 0 for r in records), 'Inspect failed worker logs; no automatic retries or mixed outputs'


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=ROOT/'assets/benchmark/pilot_v1')
    p.add_argument('--config', type=Path, default=CONFIG)
    p.add_argument('--gpus', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    p.add_argument('--models', nargs='+')
    p.add_argument('--per-model-data', action='store_true', help='Read DATA/MODEL/questions.json for adaptive long-cap probes')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--long-max-tokens', type=int)
    p.add_argument('--full', action='store_true', help='Explicitly authorize the complete input set')
    main(p.parse_args())
