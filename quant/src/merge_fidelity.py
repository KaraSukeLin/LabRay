"""Merge completed extension results after verifying the shared experiment conditions."""
import argparse
import hashlib
import json
from pathlib import Path

from fidelity import report, write_json


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--extensions', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    original = Path('build/expanded/full')
    tokens = (original/'tokens.json').read_bytes()
    rows = json.loads((original/'results.json').read_text())
    baseline = rows[0]
    provenance = []
    for run in args.extensions:
        assert (run/'tokens.json').read_bytes() == tokens
        gate = json.loads((run/'sanity.json').read_text())
        assert gate['passed'] and gate['identical_distributions']
        added = json.loads((run/'results.json').read_text())
        for result in added:
            assert result['status'] == 'ok' and result['worker_exit_code'] == 0
            for key in ['environment', 'gpu', 'cuda', 'mtp', 'kv_cache_dtype',
                        'tokens_sha256', 'vocab_size', 'logprob_format', 'v2_model_runner']:
                assert result[key] == baseline[key], (run, key)
            assert [(d['id'], d['tokens'], d['ref_nll']) for d in result['documents']] == [
                (d['id'], d['tokens'], d['ref_nll']) for d in baseline['documents']]
        rows.extend(added)
        provenance.append(dict(path=str(run), results_sha256=hashlib.sha256((run/'results.json').read_bytes()).hexdigest()))
    models = json.loads(Path('assets/benchmark_config.json').read_text())['models']
    by_name = {r['model']: r for r in rows if r['run'] != 'bf16_repeat'}
    assert len(by_name) == len(rows)-1  # no repeated candidate runs in a final table
    selected = [by_name[name] for name in models]
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'tokens.json').write_bytes(tokens)
    write_json(args.output/'results.json', selected)
    write_json(args.output/'sanity.json', gate)
    write_json(args.output/'provenance.json', dict(original=str(original),
        original_results_sha256=hashlib.sha256((original/'results.json').read_bytes()).hexdigest(),
        extensions=provenance, formal_models=models,
        excluded_historical_model='RadixArk-Qwen3.8-27B-NVFP4'))
    report(args.output, selected)
