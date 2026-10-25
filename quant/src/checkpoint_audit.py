"""Measure actual checkpoint bytes and precision from safetensors, without loading a model."""
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import struct

from analyze import export
from fidelity import write_json

MODEL_ROOT = Path('/root/model')


def headers(path):
    result = {}
    for shard in sorted(path.glob('*.safetensors')):
        with shard.open('rb') as f:
            size = struct.unpack('<Q', f.read(8))[0]
            header = json.loads(f.read(size))
        for name, value in header.items():
            if name == '__metadata__':
                continue
            assert name not in result, (path, name)
            start, end = value['data_offsets']
            result[name] = dict(value, bytes=end-start, shard=shard, offset=8+size+start)
    return result


def component(name):
    if name.startswith('mtp.') or '.mtp.' in name:
        return 'mtp'
    if '.visual.' in name or '.vision_tower.' in name:
        return 'vision'
    if 'embed_tokens' in name:
        return 'embedding'
    if name.startswith('lm_head.'):
        return 'lm_head'
    return 'language_body'


def tensor_hash(item):
    with item['shard'].open('rb') as f:
        f.seek(item['offset'])
        return hashlib.sha256(f.read(item['bytes'])).hexdigest()


def audit():
    output = Path('build/autoquant/checkpoints')
    output.mkdir(parents=True, exist_ok=False)
    names = json.loads(Path('assets/benchmark_config.json').read_text())['models']
    original = headers(MODEL_ROOT / names[0])
    original_hashes = {n: tensor_hash(v) for n, v in original.items() if component(n) != 'language_body'}
    fingerprints = json.loads(Path('build/expanded/model_fingerprints.json').read_text())
    history = []
    for entry in fingerprints:
        if entry['model'] not in names:
            continue
        path = MODEL_ROOT / entry['model'] / entry['file']
        digest = hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()
        assert path.stat().st_size == entry['bytes'] and digest == entry['sha256'], path
        history.append(dict(entry, reverified=True))
    write_json(output/'historical_fingerprints_verified.json', history)
    summaries = []
    for name in names:
        path = MODEL_ROOT / name
        tensors = headers(path)
        assert tensors, name
        allocation, component_bytes = [], defaultdict(int)
        aligned = Counter()
        for key, item in tensors.items():
            part = component(key)
            component_bytes[part] += item['bytes']
            if key in original_hashes:
                aligned[part+'_total'] += 1
                if item['dtype'] == original[key]['dtype'] and item['shape'] == original[key]['shape']:
                    aligned[part+'_identical'] += int(tensor_hash(item) == original_hashes[key])
        for key, source in original.items():
            if not (key.startswith('model.language_model.layers.') and key.endswith('.weight') and len(source['shape']) == 2):
                continue
            module = key.removesuffix('.weight')
            weight = tensors.get(key, tensors.get(module+'.weight_packed'))
            assert weight is not None, (name, key)
            dtype = weight['dtype']
            precision = 'NVFP4' if dtype == 'U8' else ('FP8' if dtype.startswith('F8') else dtype)
            serialized = sum(t['bytes'] for n, t in tensors.items() if n.startswith(module+'.'))
            allocation.append(dict(module=module, format=precision, parameters=math.prod(source['shape']),
                                   weight_bytes=weight['bytes'], module_bytes=serialized))
        params = sum(r['parameters'] for r in allocation)
        data_bytes = sum(v['bytes'] for v in tensors.values())
        weights = sorted(path.glob('*.safetensors'))
        metadata = path/'.cache/huggingface/download/config.json.metadata'
        row = dict(model=name, weight_bytes=sum(p.stat().st_size for p in weights),
            tensor_data_bytes=data_bytes, decimal_GB=sum(p.stat().st_size for p in weights)/1e9,
            GiB=sum(p.stat().st_size for p in weights)/2**30,
            body_linear_parameters=params, body_linear_bytes=sum(r['module_bytes'] for r in allocation),
            body_linear_effective_bits=8*sum(r['module_bytes'] for r in allocation)/params,
            allocation_counts=dict(Counter(r['format'] for r in allocation)),
            allocation_parameters={k: sum(r['parameters'] for r in allocation if r['format'] == k)
                                   for k in sorted({r['format'] for r in allocation})},
            component_bytes=dict(component_bytes), source_precision_alignment=dict(aligned),
            source_weight_keys_missing=sorted(k for k in original if k not in tensors and not (
                k.endswith('.weight') and k.removesuffix('.weight')+'.weight_packed' in tensors)),
            mtp_tensors=sum(component(k) == 'mtp' for k in tensors),
            revision=metadata.read_text().splitlines()[0] if metadata.exists() else 'locally generated',
            config_sha256=hashlib.sha256((path/'config.json').read_bytes()).hexdigest())
        summaries.append(row)
        write_json(output/(name+'.json'), dict(summary=row, body_linears=allocation))
        print(name, row['decimal_GB'], row['allocation_counts'], flush=True)
    write_json(output/'inventory.json', summaries)
    export(output/'capacity', [dict(model=r['model'], GB=r['decimal_GB'], GiB=r['GiB'],
        body_effective_bits=r['body_linear_effective_bits'], mtp_tensors=r['mtp_tensors'],
        **r['allocation_counts']) for r in summaries])


if __name__ == '__main__':
    audit()
