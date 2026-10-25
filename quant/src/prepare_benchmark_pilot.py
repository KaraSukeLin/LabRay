"""Select a reproducible probability sample and separate extreme-case probes."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from benchmark import BASE, ROOT
from fidelity import write_json


def main(args):
    import pyarrow.parquet as pq
    from PIL import Image
    from transformers import AutoTokenizer
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

    args.output.mkdir(parents=True, exist_ok=False)
    rows = json.loads((args.data/'questions.json').read_text())
    config = json.loads((ROOT/'assets/benchmark_config.json').read_text())
    difficulty = {}
    for path in (ROOT/'build/benchmark/sources/MMMU_Pro/standard (10 options)').glob('*.parquet'):
        for row in pq.read_table(path, columns=['id', 'topic_difficulty']).to_pylist():
            difficulty[row['id']] = row['topic_difficulty']
    tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    features, groups = {}, defaultdict(list)
    for row in rows:
        words = max([int(k.get('num_words') or 0) for k in row.get('example', {}).get('kwargs', [])]+[0])
        visual_tokens = 0
        for path in row['images']:
            with Image.open(path) as image:
                h, w = smart_resize(image.height, image.width, factor=32,
                    min_pixels=config['min_pixels'], max_pixels=config['max_pixels'])
            visual_tokens += h*w//1024
        f = dict(text_tokens=len(tokenizer.encode(row['prompt'], add_special_tokens=False)),
                 visual_tokens=visual_tokens, image_count=len(row['images']), requested_words=words)
        if row['task'] == 'ifeval':
            f['constraint_count'] = len(row['example']['kwargs'])
            f['difficulty'] = 'instruction-count proxy; no official difficulty labels'
            length = 'long' if words >= 300 else 'short' if words else 'unspecified'
            group = f'constraints_{min(f["constraint_count"], 3)}_words_{length}'
        else:
            f['difficulty'] = difficulty[row['id'].split('/', 1)[1]]
            group = row['stratum']
        f['input_proxy_tokens'] = f['text_tokens'] + visual_tokens
        f['sampling_stratum'] = group
        features[row['id']] = f
        groups[(row['task'], group)].append(row)
    rng = np.random.default_rng(args.seed)
    selection = {}
    strata = []
    for (task, group), population in sorted(groups.items()):
        n = min(4 if task == 'ifeval' else 2, len(population))
        chosen = rng.choice(len(population), n, replace=False)
        strata.append(dict(task=task, stratum=group, population=len(population), sample=n))
        for index in chosen:
            row = population[int(index)]
            selection[row['id']] = dict(kind='probability', sampling_stratum=group,
                                        weight=len(population)/n)
    # Extremes validate coverage; never add them to the probability-weighted mean.
    for task in config['max_tokens']:
        population = [r for r in rows if r['task'] == task]
        keys = ['requested_words', 'text_tokens'] if task == 'ifeval' else ['image_count', 'input_proxy_tokens']
        for key in keys:
            if len({features[r['id']][key] for r in population}) == 1:
                continue
            row = max(population, key=lambda r: features[r['id']][key])
            if row['id'] not in selection:
                selection[row['id']] = dict(kind='stress', reason=f'maximum {key}', weight=0.)
    if 'ifeval/2785' not in selection:
        selection['ifeval/2785'] = dict(kind='stress', reason='canonical-source regression', weight=0.)
    selected = [r for r in rows if r['id'] in selection]
    controls = []
    for task in ['mmmu_standard', 'mmmu_vision']:
        for level in ['Easy', 'Medium', 'Hard']:
            candidates = [r['id'] for r in selected if r['task'] == task
                          and selection[r['id']]['kind'] == 'probability'
                          and features[r['id']]['difficulty'] == level]
            controls.extend(rng.choice(candidates, min(2, len(candidates)), replace=False).tolist())
    write_json(args.output/'questions.json', selected)
    write_json(args.output/'long_ids.json', controls)
    write_json(args.output/'manifest.json', dict(seed=args.seed,
        method='SRS without replacement within IFEval constraint/word-demand or MMMU subject strata',
        source_data=str(args.data), source_sha256=hashlib.sha256((args.data/'questions.json').read_bytes()).hexdigest(),
        questions_sha256=hashlib.sha256((args.output/'questions.json').read_bytes()).hexdigest(),
        selection=selection, strata=strata, population_features=features,
        counts={task: sum(r['task'] == task for r in selected) for task in config['max_tokens']},
        probability_counts={task: sum(r['task'] == task and selection[r['id']]['kind'] == 'probability'
                            for r in selected) for task in config['max_tokens']},
        limitations=['Two questions per MMMU subject cannot give precise subject-specific latency or quality estimates',
                     'Instruction count and requested words are IFEval cost proxies, not measured difficulty',
                     'Generation length is an outcome; audit its distribution after inference']))
    print(json.dumps(dict(selected=len(selected), probability=sum(s['kind'] == 'probability' for s in selection.values()),
                          long_controls=len(controls))))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=ROOT/'assets/benchmark/prepared_v2')
    p.add_argument('--output', type=Path, default=ROOT/'assets/benchmark/pilot_v1')
    p.add_argument('--seed', type=int, default=20261017)
    main(p.parse_args())
