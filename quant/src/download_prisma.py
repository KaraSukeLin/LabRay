"""Download the public PrismaAQUA checkpoint at an immutable revision."""
import hashlib
import json
from pathlib import Path
import urllib.request

from huggingface_hub import snapshot_download


if __name__ == '__main__':
    repository = 'rdtand/Qwen3.8-27B-PrismaAQUA-5.5bit-vllm'
    revision = '6dc090346f4f32acc320e48d9c413ea96a98d4c6'
    metadata_path = Path('build/autoquant/prisma_hub_metadata.json')
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    if not metadata_path.exists():
        url = f'https://huggingface.co/api/models/{repository}/revision/{revision}?blobs=true'
        metadata_path.write_text(json.dumps(json.load(urllib.request.urlopen(url)), indent=2))
    metadata = json.loads(metadata_path.read_text())
    assert metadata['id'] == repository and metadata['sha'] == revision
    destination = Path('/root/model/rdtand-Qwen3.8-27B-PrismaAQUA-5.5bit-vllm')
    snapshot_download(metadata['id'], revision=metadata['sha'], local_dir=destination,
                      max_workers=5)
    files = []
    for entry in metadata['siblings']:
        path = destination / entry['rfilename']
        if not path.is_file():
            raise FileNotFoundError(path)
        item = dict(file=entry['rfilename'], bytes=path.stat().st_size)
        if 'lfs' in entry:
            digest = hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()
            assert digest == entry['lfs']['sha256'], path
            item['sha256'] = digest
        files.append(item)
    Path('build/autoquant/prisma_download.json').write_text(json.dumps(dict(
        repository=metadata['id'], revision=metadata['sha'], files=files,
        weight_bytes=sum(f['bytes'] for f in files if f['file'].endswith('.safetensors')),
    ), indent=2) + '\n')
