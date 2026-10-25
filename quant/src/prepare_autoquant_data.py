"""Freeze CNN/DailyMail calibration and validation, separate from Fidelity v2.1."""
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer

from audit_calibration import grams, words
from fidelity import write_json


if __name__ == '__main__':
    output = Path('assets/autoquant')
    output.mkdir(exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained('/root/model/Qwen-Qwen3.8-27B', local_files_only=True)
    source = Path('build/expanded/calibration/cnn0.parquet')
    evaluation = json.loads(Path('assets/expanded_v2/tokens.json').read_text())
    evaluation_grams = set()
    for ids in evaluation['input_ids']:
        evaluation_grams.update(grams(words(tokenizer.decode(ids)), 13))
    selected = []
    for batch in pq.ParquetFile(source).iter_batches(batch_size=256):
        for row in batch.to_pylist():
            text = row['article']
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) < 512 or evaluation_grams.intersection(grams(words(text), 13)):
                continue
            selected.append(dict(document_id=row['id'], text=text, input_ids=ids[:512]))
            if len(selected) == 160:
                break
        if len(selected) == 160:
            break
    assert len({s['document_id'] for s in selected}) == 160
    write_json(output / 'calibration.json', selected[:128])
    write_json(output / 'validation.json', selected[128:])
    write_json(output / 'manifest.json', dict(
        source='CNN/DailyMail 3.0.0 train', source_metadata=json.loads(Path('assets/calibration_sources.json').read_text())[0],
        source_file_sha256=hashlib.file_digest(source.open('rb'), 'sha256').hexdigest(),
        selection='first 160 documents with >=512 tokens and no normalized 13-word overlap with any Fidelity input',
        calibration_documents=128, validation_documents=32, tokens_per_document=512,
        evaluation_overlap_13word=0, calibration_validation_document_overlap=0,
        sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.glob('*.json')}))
