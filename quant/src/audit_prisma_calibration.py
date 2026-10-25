"""Screen fixed Fidelity windows against the public WikiText source named by Prisma."""
import hashlib
import json
from pathlib import Path
import unicodedata

from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
from transformers import AutoTokenizer

from audit_calibration import grams, words
from fidelity import write_json


if __name__ == '__main__':
    revision = 'b08601e04326c79dfdd32d625aee71d232d685c3'
    tokens = json.loads(Path('assets/expanded_v2/tokens.json').read_text())
    tokenizer = AutoTokenizer.from_pretrained(tokens['tokenizer'], local_files_only=True)
    index, zh_index = {}, {}
    for sample in tokens['samples']:
        text = tokenizer.decode(sample['input_ids'])
        for gram in grams(words(text), 13):
            index.setdefault(gram, set()).add(sample['id'])
        if sample['language'] == 'zh':
            for gram in grams(''.join(unicodedata.normalize('NFKC', text).casefold().split()), 32):
                zh_index.setdefault(gram, set()).add(sample['id'])
    results = []
    for split in ['train', 'validation', 'test']:
        path = Path(hf_hub_download('Salesforce/wikitext',
            f'wikitext-2-raw-v1/{split}-00000-of-00001.parquet', repo_type='dataset',
            revision=revision, local_dir='build/autoquant/calibration_sources'))
        text = '\n\n'.join(pq.read_table(path)['text'].to_pylist())
        hits = set()
        for gram in grams(words(text), 13):
            hits.update(index.get(gram, ()))
        for gram in grams(''.join(unicodedata.normalize('NFKC', text).casefold().split()), 32):
            hits.update(zh_index.get(gram, ()))
        results.append(dict(split=split, matching_fidelity_windows=sorted(hits),
            source_sha256=hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()))
    write_json(Path('build/autoquant/prisma_calibration_overlap.json'), dict(
        repository='Salesforce/wikitext', config='wikitext-2-raw-v1', revision=revision,
        method='NFKC/casefold 13-word and Chinese whitespace-free 32-character grams; entire Fidelity input windows',
        results=results, limitation='Revision comes from the published Prisma PPL contract. Actual calibration window IDs and train revision are not fully published; this is a source screen, not proof of their exact calibration data.'))
