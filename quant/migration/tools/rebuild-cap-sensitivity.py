"""Recreate the quality-only IFEval sensitivity matrix from three saved replacements."""
import argparse
import json
from pathlib import Path
import shutil

root=Path(__file__).resolve().parents[2]
primary=root/'build/benchmark/ifeval_modes_formal_20261017'
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,required=True,help='New output directory')
args=parser.parse_args()
args.output.mkdir(parents=True,exist_ok=False)
patch=[json.loads(line) for line in (primary/'cap_recovery_sensitivity_20261020/replacements.jsonl').open()]
assert len(patch)==3
key=lambda r:(r['model'],r['mode'],r['seed'],r['id'])
replacements={key(r):r for r in patch}
used=set()
experiment_path=primary/'cap_recovery_sensitivity_20261020/experiment.json'
experiment=json.loads(experiment_path.read_text())
shutil.copy2(experiment_path,args.output/'experiment.json')
for model in experiment['models']:
    folder=args.output/model;folder.mkdir()
    with (primary/model/'questions.jsonl').open() as source,(folder/'questions.jsonl').open('w') as destination:
        for line in source:
            row=json.loads(line);identity=key(row)
            if identity in replacements:
                assert row['truncated']
                replacement=replacements[identity]
                assert row['prompt_token_ids_sha256_json']==replacement['prompt_token_ids_sha256_json']
                row=replacement;used.add(identity)
            destination.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')
    (folder/'attempts').symlink_to(primary/model/'attempts',target_is_directory=True)
for p in primary.glob('launch-*.json'):
    (args.output/p.name).symlink_to(p)
assert used==set(replacements)
print(json.dumps(dict(rows_replaced=3,quality_only=True,timing_is_not_a_new_execution_cost=True)))
