"""Add an independent IFEval sample, then merge actual, traceable pilot phases."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

from benchmark import CONFIG, ROOT
from fidelity import write_json


def prepare(args):
    previous = json.loads((args.original/'questions.json').read_text())
    manifest = json.loads((args.original/'manifest.json').read_text())
    canonical = json.loads((ROOT/'assets/benchmark/prepared_v2/questions.json').read_text())
    known = {r['id'] for r in previous if r['task'] == 'ifeval'}
    remaining = [r for r in canonical if r['task'] == 'ifeval' and r['id'] not in known]
    rng = np.random.default_rng(20261018)
    sampled = rng.choice(len(remaining), 40, replace=False)
    additional = [remaining[int(i)] for i in sorted(sampled)]
    args.supplement.mkdir(parents=True, exist_ok=False)
    args.combined.mkdir(parents=True, exist_ok=False)
    write_json(args.supplement/'questions.json', additional)
    union = {r['id'] for r in previous+additional}
    write_json(args.combined/'questions.json', [r for r in canonical if r['id'] in union])
    # Conditional on phase 1, its 38 observed IFEval questions are known exactly.
    # Estimate only the remaining population from the independent new SRS.
    for key in known:
        manifest['selection'][key] = dict(kind='probability', sampling_stratum='phase1_known', weight=1.)
    for row in additional:
        manifest['selection'][row['id']] = dict(kind='probability', sampling_stratum='phase2_remaining', weight=len(remaining)/len(additional))
    manifest['method'] = 'Conditional IFEval known sum plus remaining-population SRS; MMMU subject-stratified SRS'
    manifest['phase1_questions_sha256'] = manifest['questions_sha256']
    manifest['phase1_data'] = str(args.original.resolve())
    manifest['strata'] = [s for s in manifest['strata'] if s['task'] != 'ifeval']+[
        dict(task='ifeval', stratum='phase1_known', population=len(known), sample=len(known)),
        dict(task='ifeval', stratum='phase2_remaining', population=len(remaining), sample=len(additional))]
    manifest['questions_sha256'] = hashlib.sha256((args.combined/'questions.json').read_bytes()).hexdigest()
    manifest['counts']['ifeval'] += len(additional)
    manifest['probability_counts']['ifeval'] = len(known)+len(additional)
    manifest['ifeval_extension'] = dict(seed=20261018, known=len(known), remaining=len(remaining), sample=len(additional),
        method='Condition on all phase-1 observations; exact known sum plus remaining-population SRS mean',
        reason='BF16 phase-1 IFEval weighted projection 3.136 h; 95% bootstrap range 0.936 to 6.727 h was too wide')
    write_json(args.combined/'manifest.json', manifest)
    print(json.dumps(manifest['ifeval_extension']))


def merge(args):
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(CONFIG.read_text())
    sha = hashlib.sha256((args.data/'questions.json').read_bytes()).hexdigest()
    expected = {r['id'] for r in json.loads((args.data/'questions.json').read_text())}
    launches = [json.loads((phase/'launch.json').read_text()) for phase in args.phases]
    process_records = []
    for model in config['models']:
        paths = [phase/model for phase in args.phases]
        infos = [json.loads((path/'run.json').read_text()) for path in paths]
        info = dict(infos[0])
        for index, item in enumerate(infos):
            assert item['status'] == 'ok'
            keys = ['config', 'config_sha256', 'environment', 'source_sha256', 'max_tokens_override']
            if args.last_is_probe and index == len(infos)-1:
                # The last phase raises only the MMMU cap; primary conditions stay fixed.
                conditions = dict(item['config'], max_tokens=info['config']['max_tokens'])
                assert conditions == info['config']
                assert item['config']['max_tokens'] == dict(info['config']['max_tokens'], mmmu_standard=8192, mmmu_vision=8192)
                keys = [k for k in keys if k not in ['config', 'config_sha256']]
            for key in keys:
                assert item[key] == info[key], (model, key)
        dest = args.output/model
        dest.mkdir()
        primary_paths = paths[:-1] if args.last_is_probe else paths
        for filename in ['questions.jsonl', 'extensions.jsonl']:
            data = b''.join((path/filename).read_bytes() for path in primary_paths)
            (dest/filename).write_bytes(data)
        if args.last_is_probe:
            main = {r['id']: r for r in map(json.loads, (dest/'questions.jsonl').read_text().splitlines())}
            previous = {r['id']: r for r in map(json.loads, (dest/'extensions.jsonl').read_text().splitlines())}
            with (dest/'extensions.jsonl').open('a') as output:
                for row in map(json.loads, (paths[-1]/'questions.jsonl').read_text().splitlines()):
                    before = previous[row['id']]
                    assert row['max_tokens'] == 8192
                    assert row['prompt_token_ids'] == main[row['id']]['prompt_token_ids']
                    assert row['image_sha256'] == main[row['id']]['image_sha256']
                    row.update(probe_reason='truncated_4096' if before['finish_reason'] == 'length' else 'fixed_control_8192',
                        main_prefix_equal=row['token_ids'][:len(main[row['id']]['token_ids'])] == main[row['id']]['token_ids'],
                        previous_cap_prefix_equal=row['token_ids'][:len(before['token_ids'])] == before['token_ids'],
                        shared_preprocessing=False, source_run=str(paths[-1].resolve()))
                    output.write(json.dumps(row, ensure_ascii=False)+'\n')
        rows = [json.loads(s) for s in (dest/'questions.jsonl').read_text().splitlines()]
        assert len(rows) == len(expected) and {r['id'] for r in rows} == expected
        for key in ['seconds', 'run_wall_seconds', 'setup_seconds', 'shutdown_seconds', 'initialization_seconds',
                    'preprocessing_seconds', 'inference_seconds', 'scoring_seconds']:
            info[key] = sum(item[key] for item in infos)
        info.update(questions=len(rows), dataset_sha256=sha,
                    run_kind='Derived union of actual pilot phases; responses copied byte for byte',
                    component_runs=[str(path.resolve()) for path in paths], model_initializations=len(paths))
        info['component_config_sha256'] = [item['config_sha256'] for item in infos]
        info['probe_caps'] = [4096, 8192] if args.last_is_probe else [4096]
        info['execution'] = dict(info['execution'],
            cuda_visible_devices=[item['execution']['cuda_visible_devices'] for item in infos])
        write_json(dest/'run.json', info)
        processes = [json.loads((phase/(model+'.process.json')).read_text()) for phase in args.phases]
        assert all(p['exit_code'] == 0 for p in processes)
        process = dict(model=model, gpu=[p['gpu'] for p in processes], exit_code=0,
            process_wall_seconds=sum(p['process_wall_seconds'] for p in processes),
            component_processes=processes, derived=True)
        write_json(args.output/(model+'.process.json'), process)
        process_records.append(process)
        Path(str(dest)+'.log').write_bytes(b''.join(Path(str(path)+'.log').read_bytes() for path in paths))
    elapsed = (datetime.fromisoformat(launches[-1]['end_utc'])-datetime.fromisoformat(launches[0]['start_utc'])).total_seconds()
    write_json(args.output/'launch.json', dict(start_utc=launches[0]['start_utc'], end_utc=launches[-1]['end_utc'],
        wall_seconds=elapsed, gpus=launches[0]['gpus'], mtp=False, questions_per_model=len(expected),
        models=config['models'], jobs=process_records, component_launches=launches, derived=True,
        inference_phase_wall_seconds=sum(r['wall_seconds'] for r in launches)))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--original', type=Path, default=ROOT/'assets/benchmark/pilot_v1')
    prep.add_argument('--supplement', type=Path, default=ROOT/'assets/benchmark/pilot_if_supplement')
    prep.add_argument('--combined', type=Path, default=ROOT/'assets/benchmark/pilot_v2')
    m = sub.add_parser('merge')
    m.add_argument('--phases', type=Path, nargs='+', required=True)
    m.add_argument('--data', type=Path, default=ROOT/'assets/benchmark/pilot_v2')
    m.add_argument('--output', type=Path, required=True)
    m.add_argument('--last-is-probe', action='store_true', help='Append final 8192 phase as probes, preserving primary responses')
    args = p.parse_args()
    (prepare if args.command == 'prepare' else merge)(args)
