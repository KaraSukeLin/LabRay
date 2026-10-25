import argparse
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import extend_benchmark_pilot


def test_merge_preserves_primary_outputs_and_accounts_for_probe_phase(tmp_path, monkeypatch):
    def save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value)+'\n')

    config = tmp_path/'config.json'
    save(config, dict(models=['base']))
    monkeypatch.setattr(extend_benchmark_pilot, 'CONFIG', config)
    data = tmp_path/'data'
    save(data/'questions.json', [dict(id=k) for k in ['if1', 'mm1', 'if2']])
    base_config = dict(mtp=False, temperature=.7, max_tokens=dict(ifeval=4096, mmmu_standard=1024, mmmu_vision=1024))
    phases = [tmp_path/name for name in ['phase1', 'supplement', 'long']]
    mm = dict(id='mm1', task='mmmu_standard', prompt_token_ids=[10], image_sha256=['same'],
              token_ids=[1], max_tokens=1024, finish_reason='length', correct=False)
    main = [dict(id='if1'), mm]
    first_long = dict(mm, token_ids=[1, 2], max_tokens=4096, correct=True)
    last_long = dict(mm, token_ids=[1, 2, 3], max_tokens=8192, finish_reason='stop', correct=True)
    for i, phase in enumerate(phases):
        folder = phase/'base'
        folder.mkdir(parents=True)
        rows = [main, [dict(id='if2')], [last_long]][i]
        (folder/'questions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        (folder/'extensions.jsonl').write_text(json.dumps(first_long)+'\n' if i == 0 else '')
        conditions = dict(base_config)
        if i == 2:
            conditions['max_tokens'] = dict(base_config['max_tokens'], mmmu_standard=8192, mmmu_vision=8192)
        info = dict(status='ok', config=conditions, config_sha256='baseline' if i < 2 else 'long',
                    environment={}, source_sha256='same-code', max_tokens_override=None,
                    execution=dict(cuda_visible_devices=str(i)))
        for key in ['seconds', 'run_wall_seconds', 'setup_seconds', 'shutdown_seconds', 'initialization_seconds',
                    'preprocessing_seconds', 'inference_seconds', 'scoring_seconds']:
            info[key] = 1.
        save(folder/'run.json', info)
        save(phase/'base.process.json', dict(gpu=i, exit_code=0, process_wall_seconds=60.))
        Path(str(folder)+'.log').write_text(f'phase {i}\n')
        save(phase/'launch.json', dict(start_utc=f'2026-10-17T00:0{i}:00+00:00',
                                     end_utc=f'2026-10-17T00:0{i+1}:00+00:00', wall_seconds=60., gpus=[0, 1, 2]))
    args = argparse.Namespace(phases=phases, data=data, output=tmp_path/'merged', last_is_probe=True)
    extend_benchmark_pilot.merge(args)
    out = args.output/'base'
    assert (out/'questions.jsonl').read_bytes() == b''.join((p/'base/questions.jsonl').read_bytes() for p in phases[:2])
    extra = [json.loads(s) for s in (out/'extensions.jsonl').read_text().splitlines()]
    assert extra[0] == first_long
    assert all(extra[1][k] == v for k, v in last_long.items())
    assert extra[1]['main_prefix_equal'] and extra[1]['previous_cap_prefix_equal']
    assert extra[1]['shared_preprocessing'] is False
    info = json.loads((out/'run.json').read_text())
    assert info['config'] == base_config and info['model_initializations'] == 3
    assert info['initialization_seconds'] == 3 and info['questions'] == 3
    assert info['execution']['cuda_visible_devices'] == ['0', '1', '2']
    process = json.loads((args.output/'base.process.json').read_text())
    assert process['process_wall_seconds'] == 180
    # Raising the cap must not silently change sampling.
    path = phases[-1]/'base/run.json'
    info = json.loads(path.read_text())
    info['config']['temperature'] = 0.
    save(path, info)
    args.output = tmp_path/'bad'
    with pytest.raises(AssertionError):
        extend_benchmark_pilot.merge(args)
