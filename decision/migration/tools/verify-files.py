"""Verify this project's saved files without loading models."""
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parents[2]
checked=0
for line in (root/'migration/SHA256SUMS').read_text().splitlines():
    expected,name=line.split('  ',1)
    h=hashlib.sha256()
    with (root/name).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    assert h.hexdigest()==expected,name
    checked+=1
print(json.dumps(dict(project='decision',verified_files=checked,models_loaded=False)))
