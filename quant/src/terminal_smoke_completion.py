"""Read-only completion audit for the user-requested smoke stop boundary."""
import itertools
import json
from pathlib import Path
import sqlite3
import time

from terminal_bench import read,save,sha,prepare,gpu_inventory
from terminal_low import CONFIG,OUTPUT
from terminal_smoke_scheduler import identity


def main():
    cfg=read(CONFIG)
    prepare(cfg)
    ready=read(OUTPUT/'smoke-readiness.json')
    assert identity()==ready['identity'],'Frozen experiment identity changed'
    for name,digest in ready['identity']['files'].items():
        assert sha(OUTPUT/'source_snapshot'/name)==digest,name
    connection=sqlite3.connect('file:'+str(OUTPUT/'smoke.sqlite')+'?mode=ro',uri=True)
    connection.row_factory=sqlite3.Row
    rows=[dict(row) for row in connection.execute('SELECT * FROM trials')]
    expected=set(itertools.product(cfg['models'],cfg['smoke_tasks'],[0,1],[cfg['seeds'][0]],[0]))
    actual={(row['model'],row['task'],row['mode'],row['seed'],row['attempt']) for row in rows}
    assert len(rows)==len(actual)==len(expected)==30 and actual==expected
    assert all(row['status'] in {'PASS','MODEL_FAIL','TIMEOUT'} for row in rows)
    native_ids=set()
    owned_still_present=[]
    for row in rows:
        folder=OUTPUT/'smoke'/row['id']/'trial'
        assert row['result']==str(folder/'result.json')
        result=read(folder/'result.json')
        assert result.get('finished_at')
        native_ids.add(result['id'])
        config=result['config']
        assert Path(config['task']['path']).name==row['task']
        assert config['agent']['model_name']=='hosted_vllm/'+row['model']
        kwargs=config['agent']['kwargs']
        assert kwargs['reasoning_effort']=='low' and kwargs['interleaved_thinking']==bool(row['mode'])
        assert kwargs['llm_call_kwargs']['seed']==row['seed']
        assert config['timeout_multiplier']==1.0
        assert config['environment']['import_path']=='terminal_proot:PRootEnvironment'
        assert config['agent'].get('override_timeout_sec') is None
        assert (folder/'agent/trajectory.json').exists()
        cast=folder/'agent/recording.cast'
        assert cast.exists() and cast.stat().st_size>0
        with cast.open() as stream:assert json.loads(stream.readline())['version']==2
        cleanup=read(folder/'proot/cleanup.json')
        assert not cleanup['after']
        for process in cleanup['before']:
            path=Path(f'/proc/{process["pid"]}/stat')
            try:current=path.read_text().rsplit(')',1)[1].split()[19]
            except (FileNotFoundError,ProcessLookupError):continue
            if current==process['start_ticks']:owned_still_present.append(process)
        counters=read(OUTPUT/'smoke'/row['id']/'mtp.json')['delta']
        assert sum(value for key,value in counters.items() if key.startswith('vllm:spec_decode_num_draft_tokens_total'))>0
    assert len(native_ids)==30
    assert not owned_still_present,owned_still_present
    audit=read(OUTPUT/'SMOKE_AUDIT.json')
    assert audit['passed'] and {row['id'] for row in audit['trials']}=={row['id'] for row in rows}
    for width in [1,2,5]:
        gate=read(OUTPUT/f'concurrency-{width}.json')
        assert gate['passed'] and gate['actual_peak_concurrent_agents']>=width
    services=[]
    for path in (OUTPUT/'services').rglob('services.json'):
        for row in read(path):
            assert row.get('stopped_at') and row.get('gpu_hours') is not None
            services.append(row)
    state=read(OUTPUT/'STATE.json')
    assert state['state']=='PAUSED_AWAITING_USER_CONFIRMATION' and state['formal_trials_started']==0
    summary=read(OUTPUT/'SMOKE_SUMMARY.json')
    assert summary['final'] and summary['completed']==30 and summary['formal_trials']==0
    baseline=read(OUTPUT.parent/'proot_pilot/inference_diagnostics/baseline-hashes.json')
    assert all(sha(Path(name))==digest for name,digest in baseline.items())
    save(OUTPUT/'COMPLETION_AUDIT.json',dict(passed=True,time=time.time(),smoke_trials=30,unique_native_trials=30,
        original_baseline_unchanged=True,full_goal_complete=False,formal_trials_started=0,stop_after_smoke=True,
        services_stopped=len(services),owned_cleanup_processes_still_present=owned_still_present,gpus=gpu_inventory(),
        scope='Thirty smoke trials, protocol/Verifier/recording/cleanup and immutable inputs only. Formal evaluation and full-vocabulary Fidelity remain pending user confirmation.'))
    print('PASS: 30 unique smoke trials, frozen protocol, native recording/verifier/token audits, cleanup and mandatory stop')


if __name__=='__main__':main()
