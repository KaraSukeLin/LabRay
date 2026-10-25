"""Restore small model assets from content hashes to an explicit new model-root."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True, help='New or empty directory; contains no model weights')
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
assert not any(args.output.iterdir()), 'Output must be empty'
entries = json.loads((root/'metadata/model-metadata/index.json').read_text())
for row in entries:
    source = root/'metadata'/row['blob']
    assert hashlib.sha256(source.read_bytes()).hexdigest() == row['sha256']
    destination = args.output/Path(row['source']).relative_to('/root/model')
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source,destination)
print(json.dumps(dict(restored_files=len(entries),weights=False)))
