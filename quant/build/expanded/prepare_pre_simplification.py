#!/usr/bin/env python3
"""Build fixed, licensed public document windows without model-based selection."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import requests
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from fidelity import write_json

ROOT = Path(__file__).resolve().parents[1]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def documents(path, source):
    if source['type'] == 'wikipedia':
        for batch in pq.ParquetFile(path).iter_batches(batch_size=1024):
            for row in batch.to_pylist():
                yield row['id'], row['title'], row['url'], row['text']
    else:
        with tarfile.open(path) as archive:
            for member in archive:
                name = member.name.partition('/')[2]
                if member.isfile() and name.startswith(source['prefix']) and name.endswith(source['suffix']):
                    text = archive.extractfile(member).read().decode('utf-8', errors='replace')
                    repo = 'leanprover-community/mathlib4' if source['language'] == 'en-Lean' else 'python/cpython'
                    url = f'https://github.com/{repo}/blob/{source["revision"]}/{name}'
                    yield name, name, url, text


def prepare(config_path, output, cache):
    config = json.loads(config_path.read_text())
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer'], local_files_only=True)
    cache.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=False)
    rows, samples, source_records = [], [], {}
    seen_texts = set()
    for domain, source in config['sources'].items():
        path = cache / source['file']
        if not path.exists():
            response = requests.get(source['url'], timeout=300)
            response.raise_for_status()
            path.write_bytes(response.content)
        digest = sha(path.read_bytes())
        if source.get('sha256', digest) != digest:
            raise ValueError(f'Source hash mismatch: {path}')
        source_records[domain] = dict(source, sha256=digest)
        # Fixed hash ordering is independent of source order and model outputs.
        pool = sorted(documents(path, source), key=lambda d: sha(f'{config["seed"]}/{domain}/{d[0]}'.encode()))
        used = set()
        for context in sorted(config['contexts'], reverse=True):
            count = 0
            for doc_id, title, url, text in pool:
                text_hash = sha(text.encode())
                if (doc_id in used or text_hash in seen_texts
                        or f'{domain}:{doc_id}' in config.get('exclude_document_ids', [])):
                    continue
                # Cheap length filter before tokenizing long collections.
                if len(text) < context + config['scored_tokens_per_document']:
                    continue
                ids = tokenizer.encode(text, add_special_tokens=False)
                size = context + config['scored_tokens_per_document']
                if len(ids) < size:
                    continue
                offset = int(sha(f'{config["seed"]}/{domain}/{doc_id}/offset'.encode()), 16) % (len(ids) - size + 1)
                window = ids[offset:offset + size]
                sample_id = f'{domain}-{context}-{count:02d}'
                metadata = dict(id=sample_id, document_id=f'{domain}:{doc_id}', domain=domain,
                                language=source['language'], context=context, score_start=context,
                                source_url=url, title=title, license=source['license'],
                                source_revision=source['revision'], document_sha256=text_hash,
                                token_offset=offset, source_tokens=len(ids))
                rows.append(dict(metadata, text=text))
                samples.append(dict(metadata, input_ids=window))
                used.add(doc_id)
                seen_texts.add(text_hash)
                count += 1
                if count == config['documents_per_cell']:
                    break
            if count != config['documents_per_cell']:
                raise ValueError(f'Not enough documents for {domain}/{context}: {count}')
            print(f'{domain}/{context}: {count} documents', flush=True)
    samples.sort(key=lambda x: x['id'])
    rows.sort(key=lambda x: x['id'])
    (output / 'documents.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    tokens = dict(version=config['version'], tokenizer=config['tokenizer'], samples=samples,
                  input_ids=[s['input_ids'] for s in samples], max_model_len=2304,
                  score_starts=[s['score_start'] for s in samples],
                  scored_tokens=sum(len(s['input_ids']) - s['score_start'] for s in samples),
                  config_sha256=sha(config_path.read_bytes()))
    write_json(output / 'tokens.json', tokens)
    write_json(output / 'manifest.json', dict(config=config, sources=source_records,
               documents=len(rows), scored_tokens=tokens['scored_tokens'],
               tokens_sha256=sha((output / 'tokens.json').read_bytes()),
               documents_sha256=sha((output / 'documents.jsonl').read_bytes()),
               selection='SHA256 order, longest context first, unique source documents, seeded token offset; no model outputs'))
    print(f'Total: {len(rows)} documents, {tokens["scored_tokens"]} scored tokens', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT / 'assets/expanded_config.json')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cache', type=Path, default=ROOT / 'build/expanded/sources')
    a = p.parse_args()
    prepare(a.config, a.output, a.cache)
