"""Fetch immutable upstream tools and official benchmark data; do not run inference."""
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request

from huggingface_hub import hf_hub_download, snapshot_download

from fidelity import write_json

TOOLS = Path('build/autoquant/tools')
REPOSITORIES = {
    'Model-Optimizer': ('NVIDIA/Model-Optimizer', 'ed66a9e0a33db681f0d550f65e7f47f817f6e3c2'),
    'MMMU': ('MMMU-Benchmark/MMMU', '268471d0d488258990025331c7528359c324aa25'),
}


if __name__ == '__main__':
    TOOLS.mkdir(parents=True, exist_ok=True)
    for directory, (repository, revision) in REPOSITORIES.items():
        path = TOOLS/directory
        if not path.exists():
            subprocess.run(['git', 'init', str(path)], check=True)
            subprocess.run(['git', '-C', str(path), 'remote', 'add', 'origin', 'https://github.com/'+repository+'.git'], check=True)
            subprocess.run(['git', '-C', str(path), 'fetch', '--depth', '1', 'origin', revision], check=True)
            subprocess.run(['git', '-C', str(path), 'checkout', '--detach', 'FETCH_HEAD'], check=True)
        assert subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip() == revision
    config = json.loads(Path('assets/benchmark_config.json').read_text())
    revision = config['google_ifeval_revision']
    google = TOOLS/'google-research'
    destination = google/'instruction_following_eval'
    destination.mkdir(parents=True, exist_ok=True)
    url = f'https://api.github.com/repos/google-research/google-research/contents/instruction_following_eval?ref={revision}'
    files = json.load(urllib.request.urlopen(url))
    for entry in files:
        if entry['type'] == 'file':
            (destination/entry['name']).write_bytes(urllib.request.urlopen(entry['download_url']).read())
    (google/'LICENSE').write_bytes(urllib.request.urlopen(
        f'https://raw.githubusercontent.com/google-research/google-research/{revision}/LICENSE').read())
    write_json(google/'revision.json', dict(repository='google-research/google-research', revision=revision))
    hf_hub_download('google/IFEval', 'ifeval_input_data.jsonl', repo_type='dataset',
        revision=config['ifeval_hf_reference_revision'], local_dir='assets/benchmark/ifeval')
    # The repository README uses this canonical input. The Hub copy differs on key 2785.
    Path('assets/benchmark/ifeval/input_data_google_repository.jsonl').write_bytes(
        urllib.request.urlopen(f'https://raw.githubusercontent.com/google-research/google-research/{revision}/instruction_following_eval/data/input_data.jsonl').read())
    snapshot_download('MMMU/MMMU_Pro', repo_type='dataset', revision=config['mmmu_dataset_revision'],
        local_dir='build/benchmark/sources/MMMU_Pro', max_workers=6,
        allow_patterns=['README.md', 'eval.yaml', 'standard (10 options)/*.parquet', 'vision/*.parquet'])
    snapshot_download('RadixArk/Qwen3.8-27B-DSpark', revision='b9a5dbdf03bc999c6c73c426b19c2d9041cea393',
        local_dir='/root/model/.dspark-Qwen3.8-27B', max_workers=3)
    write_json(TOOLS/'provenance.json', dict(repositories=REPOSITORIES,
        google_revision=revision, source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [destination/'evaluation_lib.py', destination/'instructions.py',
                      TOOLS/'MMMU/mmmu-pro/evaluate.py', TOOLS/'MMMU/mmmu-pro/infer/infer_transformers.py']}))
