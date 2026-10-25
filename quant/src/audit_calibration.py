#!/usr/bin/env python3
"""Audit fixed evaluation windows against available public calibration sources."""
import argparse
import json
from pathlib import Path
import re
import unicodedata
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from fidelity import write_json
from prepare_dataset import source_file


def words(text):
    return re.findall(r'\w+', unicodedata.normalize('NFKC', text).casefold())


def grams(sequence, n):
    for j in range(len(sequence) - n + 1):
        yield tuple(sequence[j:j + n]) if isinstance(sequence, list) else sequence[j:j + n]


def audit(tokens_path, calibration, output):
    calibration.mkdir(parents=True, exist_ok=True)
    sources = json.loads((Path(__file__).resolve().parents[1] / 'assets/calibration_sources.json').read_text())
    for source in sources:
        source_file(source, calibration)
    tokens = json.loads(tokens_path.read_text())
    tokenizer = AutoTokenizer.from_pretrained(tokens['tokenizer'], local_files_only=True)
    word_index, zh_index = {}, {}
    for s in tokens['samples']:
        text = tokenizer.decode(s['input_ids'], skip_special_tokens=False)
        for g in grams(words(text), 13):
            word_index.setdefault(g, set()).add(s['id'])
        if s['language'] == 'zh':
            text = ''.join(unicodedata.normalize('NFKC', text).casefold().split())
            for g in grams(text, 32):
                zh_index.setdefault(g, set()).add(s['id'])
    results = []
    specifications = [('CNN/DailyMail complete train', sorted(calibration.glob('cnn*.parquet')), None),
                      ('PerfectBlend train[:512]', [calibration / 'perfectblend0.parquet'], 512)]
    for name, paths, limit in specifications:
        counts, hits = 0, []
        for path in paths:
            for batch in pq.ParquetFile(path).iter_batches(batch_size=512):
                for row in batch.to_pylist():
                    if limit is not None and counts >= limit:
                        break
                    counts += 1
                    text = ('\n'.join(x['value'] for x in row['conversations']) if 'conversations' in row
                            else row['article'] + '\n' + row['highlights'])
                    matches = set()
                    for g in grams(words(text), 13):
                        matches.update(word_index.get(g, ()))
                    if sum('\u4e00' <= ch <= '\u9fff' for ch in text) >= 32:
                        normalized = ''.join(unicodedata.normalize('NFKC', text).casefold().split())
                        for g in grams(normalized, 32):
                            matches.update(zh_index.get(g, ()))
                    if matches:
                        hits.append(dict(row=counts - 1, file=path.name, sample_ids=sorted(matches)))
                if limit is not None and counts >= limit:
                    break
        results.append(dict(source=name, checked_documents=counts, matching_documents=len(hits), hits=hits))
        print(name, counts, 'matching documents', len(hits), flush=True)
    write_json(output, dict(method='NFKC/casefold word 13-grams; Chinese whitespace-free 32-character grams; entire input windows',
        sources=sources, results=results,
        limitation='Exact normalized overlap only. PerfectBlend mapping/version may differ from conversion; NVIDIA selected Nemotron samples and Fast/Unsloth/Qwen calibration text unavailable. Does not prove absence of paraphrases or pretraining overlap.'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tokens', type=Path, default=Path('assets/expanded_v2/tokens.json'))
    p.add_argument('--calibration', type=Path, default=Path('build/expanded/calibration'))
    p.add_argument('--output', type=Path, default=Path('build/expanded/calibration_overlap.json'))
    a = p.parse_args()
    audit(a.tokens, a.calibration, a.output)
