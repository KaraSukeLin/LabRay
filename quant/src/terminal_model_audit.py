"""Read-only checkpoint completeness audit; aliases require user confirmation."""
import json
import hashlib
from pathlib import Path
import struct
import time
import zlib

from terminal_bench import ROOT, read, save, sha
from terminal_low import OUTPUT


def audit(model):
    path=Path('/root/model')/model
    inventory=next(row for row in read(ROOT/'build/autoquant/checkpoints/inventory.json') if row['model']==model)
    cfg=read(path/'config.json')
    assert sha(path/'config.json')==inventory['config_sha256']
    tensors={}
    files=[]
    for file in sorted(path.glob('*.safetensors')):
        with file.open('rb') as stream:
            size=struct.unpack('<Q',stream.read(8))[0]
            header=json.loads(stream.read(size))
        data_size=file.stat().st_size-size-8
        for key,value in header.items():
            if key=='__metadata__': continue
            start,end=value['data_offsets']
            assert 0<=start<=end<=data_size,(file,key)
            assert key not in tensors,(file,key)
            tensors[key]=dict(file=file.name,dtype=value['dtype'],shape=value['shape'])
        files.append(dict(name=file.name,bytes=file.stat().st_size,mtime_ns=file.stat().st_mtime_ns))
    index=read(path/'model.safetensors.index.json')['weight_map']
    assert all(key in tensors and tensors[key]['file']==file for key,file in index.items())
    assert sum(row['bytes'] for row in files)==inventory['weight_bytes']
    crc=[]
    if (path/'crc32.txt').exists():
        for line in (path/'crc32.txt').read_text().splitlines():
            if not line.strip(): continue
            expected,name=line.split(maxsplit=1)
            file=path/name.lstrip('*')
            value=0
            with file.open('rb') as stream:
                while block:=stream.read(8*1024*1024): value=zlib.crc32(block,value)
            crc.append(dict(file=file.name,expected_crc32=expected,actual_crc32=f'{value:08x}',matches=f'{value:08x}'==expected.lower()))
    historical={row['file']:row for row in read(ROOT/'build/autoquant/checkpoints/historical_fingerprints_verified.json') if row['model']==model}
    for row in files:
        digest=hashlib.sha256()
        with (path/row['name']).open('rb') as stream:
            while block:=stream.read(8*1024*1024): digest.update(block)
        row['sha256']=digest.hexdigest()
        reference=historical.get(row['name'])
        if reference is not None:
            row['fingerprint_source']='historical_verified'
            row['historical_match']=reference['sha256']==row['sha256']
            assert row['historical_match'],row['name']
        else:
            metadata=path/'.cache/huggingface/download'/(row['name']+'.metadata')
            revision,etag,*_=metadata.read_text().splitlines()
            row['fingerprint_source']='local_huggingface_download_lfs_sha256'
            assert revision==inventory['revision'] and etag==row['sha256'],row['name']
    shared=Path('/root/model/Qwen-Qwen3.8-27B')
    manifest=read(ROOT/'assets/terminal_bench_manifest.json')['tokenizer']
    assert all(sha(shared/name)==value for name,value in manifest.items())
    return dict(model=model,path=str(path),revision=inventory['revision'],config_sha256=sha(path/'config.json'),
        quantization_config=cfg.get('quantization_config'),mtp_hidden_layers=cfg.get('text_config',{}).get('mtp_num_hidden_layers'),
        mtp_tensors=[key for key in tensors if 'mtp' in key],files=files,indexed_tensors=len(index),
        crc32_checked=crc,shared_tokenizer_matches_fixed_manifest=True,
        source_crc_mismatches=[row for row in crc if not row['matches']],structural_integrity_passed=True,
        runtime_quantization_verified=False,requested_alias_confirmed=model=='Qwen-Qwen3.8-27B',time=time.time())


if __name__=='__main__':
    OUTPUT.mkdir(parents=True,exist_ok=True)
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model')
    args=parser.parse_args()
    rows=read(OUTPUT/'model-audit.json') if (OUTPUT/'model-audit.json').exists() else []
    for model in ['Qwen-Qwen3.8-27B','eNeural-Qwen3.8-27B-fast','rdtand-Qwen3.8-27B-PrismaAQUA-5.5bit-vllm']:
        if args.model and args.model!=model: continue
        result=audit(model)
        rows=[row for row in rows if row['model']!=model]+[result]
        save(OUTPUT/'model-audit.json',rows)
        print(model,'STRUCTURAL PASS; runtime and alias checks separate',flush=True)
