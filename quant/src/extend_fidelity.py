"""Evaluate new checkpoints against the preserved v2.1 reference and a fresh BF16 gate."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from fidelity import BASELINE, DEFAULT_MODELS, report, sanity_passed, write_json


def extend(args):
    old = args.reference_run.resolve()
    tokens = old / 'tokens.json'
    original = json.loads((old / 'bf16_reference/result.json').read_text())
    assert tokens.read_bytes() == Path('assets/expanded_v2/tokens.json').read_bytes()
    assert original['tokens_sha256'] == hashlib.sha256(tokens.read_bytes()).hexdigest()
    assert json.loads((old / 'sanity.json').read_text())['passed']
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'tokens.json').write_bytes(tokens.read_bytes())

    def capture(model, name):
        destination = args.output / name
        destination.mkdir()
        command = [sys.executable, str(Path(__file__).with_name('fidelity.py')), 'capture',
                   '--model', str(model), '--tokens', str(tokens), '--reference',
                   str(old / 'bf16_reference'), '--logprob-format', 'dense',
                   '--output', str(destination.resolve())]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), VLLM_USE_V2_MODEL_RUNNER='1')
        with (destination / 'run.log').open('w') as log:
            completed = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        result = json.loads((destination / 'result.json').read_text())
        result.update(run=name, worker_exit_code=completed.returncode)
        assert completed.returncode == 0 and result['status'] == 'ok', destination
        for key in ['tokens_sha256', 'environment', 'mtp', 'kv_cache_dtype', 'gpu',
                    'cuda', 'vocab_size', 'logprob_format', 'v2_model_runner']:
            assert result[key] == original[key], (key, result[key], original[key])
        assert [(d['id'], d['tokens'], d['ref_nll']) for d in result['documents']] == [
            (d['id'], d['tokens'], d['ref_nll']) for d in original['documents']]
        print(json.dumps({k: result[k] for k in ['model', 'ppl', 'kl_bf16_to_candidate',
                                               'top1_flip_rate', 'elapsed_seconds']}), flush=True)
        return result

    if args.sanity_run:
        gate = json.loads((args.sanity_run / 'sanity.json').read_text())
        assert gate['reference_run'] == str(old)
        assert gate['tokens_sha256'] == original['tokens_sha256']
    else:
        repeated = capture(DEFAULT_MODELS / BASELINE, 'bf16_repeat')
        gate = dict(passed=sanity_passed(repeated), metrics=repeated,
                    identical_distributions=repeated['distribution_sha256'] == original['distribution_sha256'],
                    reference_run=str(old), tokens_sha256=original['tokens_sha256'])
    assert gate['passed'] and gate['identical_distributions']
    write_json(args.output / 'sanity.json', gate)
    rows = []
    for name in args.models:
        rows.append(capture(DEFAULT_MODELS / name, name))
        write_json(args.output / 'results.json', rows)
        report(args.output, rows)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-run', type=Path, default=Path('build/expanded/full'))
    p.add_argument('--sanity-run', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--models', nargs='+', required=True)
    p.add_argument('--gpu', type=int, default=0)
    extend(p.parse_args())
