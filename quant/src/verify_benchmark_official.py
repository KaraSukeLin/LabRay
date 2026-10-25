"""CPU-only audit of prepared questions, upstream source, and saved official scores."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import urllib.request

from benchmark import CONFIG, ROOT, TOOLS, official_module, scorers
from fidelity import write_json


def main(args):
    started = time.perf_counter()
    import pyarrow.parquet as pq
    from PIL import Image
    import yaml

    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(CONFIG.read_text())
    provenance = json.loads((TOOLS/'provenance.json').read_text())
    google_revision = config['google_ifeval_revision']
    mmmu_revision = provenance['repositories']['MMMU'][1]
    source_hashes = {}
    sources = [(TOOLS/'google-research/instruction_following_eval'/f,
                f'https://raw.githubusercontent.com/google-research/google-research/{google_revision}/instruction_following_eval/{f}')
               for f in ['evaluation_main.py', 'evaluation_lib.py', 'instructions.py', 'instructions_registry.py', 'instructions_util.py']]
    sources += [(TOOLS/'MMMU/mmmu-pro'/f,
                 f'https://raw.githubusercontent.com/MMMU-Benchmark/MMMU/{mmmu_revision}/mmmu-pro/{f}')
                for f in ['evaluate.py', 'prompts.yaml', 'infer/infer_transformers.py']]
    for path, url in sources:
        data = urllib.request.urlopen(url).read()
        assert data == path.read_bytes(), path
        source_hashes[str(path.relative_to(ROOT))] = hashlib.sha256(data).hexdigest()
    original = urllib.request.urlopen(
        f'https://raw.githubusercontent.com/google-research/google-research/{google_revision}/instruction_following_eval/data/input_data.jsonl').read()
    examples = [json.loads(line) for line in original.splitlines()]
    local = ROOT/'assets/benchmark/ifeval/input_data_google_repository.jsonl'
    assert examples == [json.loads(line) for line in local.read_text().splitlines()]
    prepared = json.loads((args.data/'questions.json').read_text())
    by_id = {r['id']: r for r in prepared}
    assert len(by_id) == len(prepared) == 4001
    for example in examples:
        row = by_id[f'ifeval/{example["key"]}']
        assert row['example'] == example and row['prompt'] == example['prompt'] and row['images'] == []
    helper = official_module('audit_official_prompt', TOOLS/'MMMU/mmmu-pro/infer/infer_transformers.py')
    helper.prompt_config = yaml.safe_load((TOOLS/'MMMU/mmmu-pro/prompts.yaml').read_text())['direct']
    checked_images = checked_mmmu = 0
    manifest = json.loads((args.data/'manifest.json').read_text())
    for source in manifest['source_files']:
        path = ROOT/source['file']
        assert hashlib.file_digest(path.open('rb'), 'sha256').hexdigest() == source['sha256']
        variant = path.parent.name
        task = 'mmmu_vision' if variant == 'vision' else 'mmmu_standard'
        for batch in pq.ParquetFile(path).iter_batches(batch_size=8):
            for item in batch.to_pylist():
                for key in list(item):
                    if key.startswith('image') and item[key] is not None:
                        item[key] = Image.open(io.BytesIO(item[key]['bytes'])).convert('RGB')
                prompt, images = helper.process_prompt(item, variant)
                row = by_id[f'{task}/{item["id"]}']
                assert (row['prompt'], row['gold'], row['options'], row['stratum']) == (
                    prompt, item['answer'], item['options'], item['subject'])
                assert len(images) == len(row['images'])
                for image, dest, sha in zip(images, row['images'], row['image_sha256']):
                    assert hashlib.sha256(Path(dest).read_bytes()).hexdigest() == sha
                    saved = Image.open(dest).convert('RGB')
                    assert image.size == saved.size and image.tobytes() == saved.tobytes()
                    checked_images += 1
                checked_mmmu += 1
    ifeval, mmmu = scorers()
    data_audit_seconds = time.perf_counter()-started
    scoring_started = time.perf_counter()
    checked_ifeval_scores = checked_mmmu_scores = 0
    for run in sorted(args.runs.glob('*/questions.jsonl')):
        records = [json.loads(line) for line in run.read_text().splitlines()]
        if args.include_extensions:
            records.extend(json.loads(line) for line in (run.parent/'extensions.jsonl').read_text().splitlines())
        dest = args.output/run.parent.name
        dest.mkdir()
        subset = [r for r in records if r['task'] == 'ifeval']
        inputs = dest/'inputs.jsonl'
        responses = dest/'responses.jsonl'
        inputs.write_text(''.join(json.dumps(by_id[r['id']]['example'])+'\n' for r in subset))
        responses.write_text(''.join(json.dumps(dict(prompt=by_id[r['id']]['prompt'], response=r['response']))+'\n' for r in subset))
        with (dest/'google_cli.log').open('w') as log:
            subprocess.run([sys.executable, '-m', 'instruction_following_eval.evaluation_main',
                '--input_data', str(inputs.resolve()), '--input_response_data', str(responses.resolve()),
                '--output_dir', str(dest.resolve())], cwd=TOOLS/'google-research', check=True,
                env=dict(os.environ, NLTK_DATA=str(ROOT/'assets/benchmark/nltk_data')),
                stdout=log, stderr=subprocess.STDOUT)
        for mode in ['strict', 'loose']:
            scores = [json.loads(line) for line in (dest/f'eval_results_{mode}.jsonl').read_text().splitlines()]
            for record, result in zip(subset, scores):
                assert result['prompt'] == by_id[record['id']]['prompt']
                assert result['follow_all_instructions'] == record['correct' if mode == 'strict' else 'loose_correct']
                assert result['follow_instruction_list'] == record[mode+'_instructions']
        checked_ifeval_scores += len(subset)
        for record in records:
            if record['task'] == 'ifeval':
                continue
            row = by_id[record['id']]
            random.seed(int(hashlib.sha256(row['id'].encode()).hexdigest()[:16], 16))
            result = mmmu.mmmu_process_results(dict(id=row['id'], options=row['options'],
                                                  answer=row['gold'], response=record['response']))
            assert result['pred_indexs'] == record['answer'] and result['if_right'] == record['correct']
            checked_mmmu_scores += 1
    report = dict(passed=True, upstream_source_sha256=source_hashes,
        ifeval_questions_equal_to_google_repository=len(examples),
        mmmu_prompts_options_gold_subjects_equal=checked_mmmu, images_pixels_order_sha256_equal=checked_images,
        google_cli_strict_and_loose_regraded=checked_ifeval_scores,
        official_mmmu_process_results_regraded=checked_mmmu_scores,
        inference_run=False, runs=str(args.runs), data_audit_seconds=data_audit_seconds,
        scoring_replay_seconds=time.perf_counter()-scoring_started,
        total_seconds=time.perf_counter()-started)
    write_json(args.output/'audit.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', type=Path, default=ROOT/'build/benchmark/smoke')
    p.add_argument('--data', type=Path, default=ROOT/'assets/benchmark/prepared_v2')
    p.add_argument('--include-extensions', action='store_true', help='Also replay saved long-cap responses')
    p.add_argument('--output', type=Path, required=True)
    main(p.parse_args())
