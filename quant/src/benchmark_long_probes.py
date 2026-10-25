"""Select unresolved 4096-token cases for 8192 probes and replay official grading."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time

from benchmark import CONFIG, ROOT, TOOLS, scorers
from fidelity import write_json


def prepare(args):
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(CONFIG.read_text())
    canonical = {r['id']: r for r in json.loads((ROOT/'assets/benchmark/prepared_v2/questions.json').read_text())}
    fixed = json.loads((ROOT/'assets/benchmark/pilot_v1/long_ids.json').read_text())
    controls = [next(k for k in fixed if k.startswith(task+'/')) for task in ['mmmu_standard', 'mmmu_vision']]
    models = {}
    for model in config['models']:
        source = args.runs/model/'extensions.jsonl'
        records = [json.loads(s) for s in source.read_text().splitlines()]
        tails = {r['id'] for r in records if r['max_tokens'] == 4096 and r['finish_reason'] == 'length'}
        ids = tails.union(controls)
        dest = args.output/model
        dest.mkdir()
        write_json(dest/'questions.json', [canonical[k] for k in sorted(ids)])
        models[model] = dict(truncated_4096=sorted(tails), controls=controls,
                            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                            questions_sha256=hashlib.sha256((dest/'questions.json').read_bytes()).hexdigest())
    config['max_tokens'].update(mmmu_standard=8192, mmmu_vision=8192)
    if args.config_output.exists():
        assert json.loads(args.config_output.read_text()) == config
    else:
        write_json(args.config_output, config)
    write_json(args.output/'manifest.json', dict(method='All unresolved 4096 cases, plus two common prespecified controls',
        source_runs=str(args.runs.resolve()), models=models, max_tokens=8192, mtp=False))
    print(json.dumps({model: len(set(r['truncated_4096']+r['controls'])) for model, r in models.items()}))


def audit(args):
    started = time.perf_counter()
    _, mmmu = scorers()
    config = json.loads(CONFIG.read_text())
    expected = json.loads((args.data/'manifest.json').read_text())
    prior = json.loads(args.source_audit.read_text())
    assert prior['passed']
    official = TOOLS/'MMMU/mmmu-pro/evaluate.py'
    sha = hashlib.sha256(official.read_bytes()).hexdigest()
    assert sha == prior['upstream_source_sha256'][str(official.relative_to(ROOT))]
    checked = 0
    for model in config['models']:
        info = json.loads((args.runs/model/'run.json').read_text())
        process = json.loads((args.runs/(model+'.process.json')).read_text())
        assert info['status'] == 'ok' and process['exit_code'] == 0 and info['config']['mtp'] is False
        assert info['dataset_sha256'] == expected['models'][model]['questions_sha256']
        inputs = {r['id']: r for r in json.loads((args.data/model/'questions.json').read_text())}
        records = [json.loads(s) for s in (args.runs/model/'questions.jsonl').read_text().splitlines()]
        assert len(records) == len(inputs) and {r['id'] for r in records} == set(inputs)
        for record in records:
            assert record['max_tokens'] == 8192 and record['task'] != 'ifeval'
            row = inputs[record['id']]
            random.seed(int(hashlib.sha256(row['id'].encode()).hexdigest()[:16], 16))
            result = mmmu.mmmu_process_results(dict(id=row['id'], options=row['options'], answer=row['gold'], response=record['response']))
            assert result['pred_indexs'] == record['answer'] and result['if_right'] == record['correct']
            checked += 1
    write_json(args.output, dict(passed=True, official_mmmu_regraded=checked, max_tokens=8192,
        official_source_sha256=sha, source_data_audit=str(args.source_audit), seconds=time.perf_counter()-started))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--runs', type=Path, required=True)
    prep.add_argument('--output', type=Path, required=True)
    prep.add_argument('--config-output', type=Path, default=ROOT/'assets/benchmark_8192_config.json')
    a = sub.add_parser('audit')
    a.add_argument('--runs', type=Path, required=True)
    a.add_argument('--data', type=Path, required=True)
    a.add_argument('--source-audit', type=Path, required=True)
    a.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    (prepare if args.command == 'prepare' else audit)(args)
