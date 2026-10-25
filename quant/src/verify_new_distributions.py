"""Repeat one document per stratum and independently audit raw full-vocabulary metrics."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from fidelity import write_json


if __name__ == '__main__':
    output = Path('build/autoquant/distribution_audit')
    output.mkdir(exist_ok=False)
    tokens = json.loads(Path('assets/expanded_v2/tokens.json').read_text())
    strata = {}
    for i, sample in enumerate(tokens['samples']):
        strata.setdefault((sample['domain'], sample['context']), i)
    indices = list(strata.values())
    assert len(indices) == 12
    for key in ['input_ids', 'score_starts', 'samples']:
        tokens[key] = [tokens[key][i] for i in indices]
    write_json(output/'tokens.json', tokens)
    models = json.loads(Path('assets/benchmark_config.json').read_text())['models'][-2:]
    def capture(model, gpu):
        destination = output/model
        destination.mkdir()
        with (destination/'run.log').open('w') as log:
            subprocess.run([sys.executable, 'src/fidelity.py', 'capture', '--model',
                '/root/model/'+model, '--tokens', str(output/'tokens.json'), '--output',
                str(destination), '--logprob-format', 'dense'], check=True,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), VLLM_USE_V2_MODEL_RUNNER='1'),
                stdout=log, stderr=subprocess.STDOUT)
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(capture, model, gpu) for model, gpu in zip(models, [0, 1])]
        for job in jobs:
            job.result()
    complete = {r['model']: r for r in json.loads(Path('build/autoquant/combined/results.json').read_text())}
    checks = []
    for model in models:
        fresh = json.loads((output/model/'result.json').read_text())
        assert fresh['status'] == 'ok'
        for local, original in enumerate(indices):
            q = np.load(output/model/f'{local}.npy', mmap_mode='r')
            p = np.load(Path('build/expanded/full/bf16_reference')/f'{original}.npy', mmap_mode='r')
            expected = complete[model]['documents'][original]
            digest = hashlib.sha256(q.tobytes()).hexdigest()
            assert digest == complete[model]['distribution_sha256'][original]
            ids, start = tokens['input_ids'][local], tokens['score_starts'][local]
            targets = np.array(ids[start:])
            nll = -q[np.arange(len(targets)), targets].astype(np.float64).sum()
            ref_nll = -p[np.arange(len(targets)), targets].astype(np.float64).sum()
            kl = 0.
            for a, b in zip(p, q):
                a, b = a.astype(np.float64), b.astype(np.float64)
                probability = np.exp(a)
                active = probability > 0
                kl += np.dot(probability[active], a[active]-b[active])
            flips = int((p.argmax(axis=1) != q.argmax(axis=1)).sum())
            errors = {key: float(value-expected[key]) for key, value in
                      [('nll', nll), ('ref_nll', ref_nll), ('kl_sum', kl), ('flips', flips)]}
            assert max(abs(e) for e in errors.values()) < 1e-7, errors
            checks.append(dict(model=model, document_id=expected['document_id'],
                               identical_distribution=True, errors=errors))
    write_json(output/'audit.json', dict(passed=True, documents_per_model=12,
        tokens_per_model=2304, full_vocabulary=248320, checks=checks))
