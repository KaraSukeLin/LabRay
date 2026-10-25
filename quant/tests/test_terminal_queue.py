from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from terminal_queue import Queue


def test_resume_keeps_terminal_failure_and_prevents_duplicate_gpu_claim(tmp_path):
    path=tmp_path/'queue.sqlite'
    q=Queue(path,{'effort':'low','mtp':3})
    q.enqueue(['bf16','candidate'],['task'],20261020)
    row=q.claim(0,123,'0',['bf16','candidate'])
    with pytest.raises(RuntimeError,match='already'):q.claim(0,124,'1',['bf16','candidate'])
    q.finish(row['id'],'MODEL_FAIL','original-result.json')
    with pytest.raises(RuntimeError,match='overwritten'):q.finish(row['id'],'PASS','replacement.json')
    resumed=Queue(path,{'effort':'low','mtp':3})
    resumed.enqueue(['bf16','candidate'],['task'],20261020)
    assert len(resumed.snapshot())==4
    assert next(r for r in resumed.snapshot() if r['id']==row['id'])['status']=='MODEL_FAIL'
    assert resumed.claim(0,124,'1',['bf16','candidate'])['id']!=row['id']
    with pytest.raises(ValueError,match='identity changed'): Queue(path,{'effort':'xhigh','mtp':0})


def test_ready_model_gate_and_model_affinity(tmp_path):
    q=Queue(tmp_path/'queue.sqlite',{'mtp':3})
    q.enqueue(['bf16','unvalidated'],['one','two'],42)
    assert q.claim(0,1,'0',[]) is None
    first=q.claim(0,1,'0',['bf16'])
    second=q.claim(1,2,'0',['bf16'],preferred='bf16')
    assert first['model']==second['model']=='bf16' and first['id']!=second['id']
    assert all(row['status']=='PENDING' for row in q.snapshot() if row['model']=='unvalidated')


def test_scheduler_refuses_unconfirmed_checkpoint_aliases():
    from terminal_smoke_scheduler import gates
    from terminal_low import CONFIG
    from terminal_bench import read
    with pytest.raises(AssertionError,match='checkpoint mapping'):
        config=read(CONFIG)
        config['candidate_models']={'NaviQuant':None,'Aqua':None}
        gates(config)


def test_concurrency_gate_uses_actual_agent_intervals(tmp_path,monkeypatch):
    import json
    import terminal_smoke_scheduler as scheduler
    monkeypatch.setattr(scheduler,'OUTPUT',tmp_path)
    for key,start,end in [('one','00:00:00','00:00:20'),('two','00:00:10','00:00:30')]:
        folder=tmp_path/'smoke'/key
        folder.mkdir(parents=True)
        (folder/'summary.json').write_text(json.dumps(dict(agent_execution=dict(
            started_at='2026-10-21T'+start+'Z',finished_at='2026-10-21T'+end+'Z'))))
    assert scheduler.overlap([{'id':'one'},{'id':'two'}],2)==2
    with pytest.raises(AssertionError,match='concurrent'):
        scheduler.overlap([{'id':'one'},{'id':'two'}],5)


def test_resume_recovers_completed_native_result_without_new_attempt(tmp_path,monkeypatch):
    import json
    import terminal_smoke_scheduler as scheduler
    monkeypatch.setattr(scheduler,'OUTPUT',tmp_path)
    queue=Queue(tmp_path/'queue.sqlite',{'effort':'low','mtp':3})
    queue.enqueue(['bf16'],['one'],42,modes=[False])
    row=queue.claim(0,999999999,'missing',['bf16'])
    folder=tmp_path/'smoke'/row['id']
    native=folder/'trial'
    (native/'proot').mkdir(parents=True)
    (native/'verifier').mkdir()
    (folder/'identity.json').write_text(json.dumps(row))
    (native/'proot/cleanup.json').write_text(json.dumps({'before':[],'after':[]}))
    (native/'verifier/ctrf.json').write_text('{}')
    result=dict(finished_at='2026-10-21T10:00:00Z',task_name='terminal-bench/one',
        config=dict(agent=dict(model_name='hosted_vllm/bf16',kwargs=dict(
            interleaved_thinking=False,llm_call_kwargs=dict(seed=42)))),
        verifier_result=dict(rewards=dict(reward=0)),agent_execution=None)
    (native/'result.json').write_text(json.dumps(result))
    result['config']['agent']['kwargs']['llm_call_kwargs']['seed']=43
    (native/'result.json').write_text(json.dumps(result))
    with pytest.raises(AssertionError): scheduler.reconcile_finished_trials(queue)
    assert queue.snapshot()[0]['status']=='RUNNING'
    result['config']['agent']['kwargs']['llm_call_kwargs']['seed']=42
    (native/'result.json').write_text(json.dumps(result))
    assert scheduler.reconcile_finished_trials(queue)==[row['id']]
    assert queue.snapshot()[0]['status']=='MODEL_FAIL'
    assert len(queue.snapshot())==1
    assert scheduler.reconcile_finished_trials(queue)==[]
    assert queue.claim(0,1,'other',['bf16']) is None


def test_resume_does_not_recover_live_or_unclean_trial(tmp_path,monkeypatch):
    import os,json
    import terminal_smoke_scheduler as scheduler
    monkeypatch.setattr(scheduler,'OUTPUT',tmp_path)
    queue=Queue(tmp_path/'queue.sqlite',{'mtp':3})
    queue.enqueue(['bf16'],['one'],42,modes=[False])
    ticks=Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')',1)[1].split()[19]
    row=queue.claim(0,os.getpid(),ticks,['bf16'])
    assert scheduler.reconcile_finished_trials(queue)==[]
    assert queue.snapshot()[0]['status']=='RUNNING'
    monkeypatch.setattr(queue,'interrupted_owners',lambda:[row])
    folder=tmp_path/'smoke'/row['id']
    (folder/'trial/proot').mkdir(parents=True)
    (folder/'identity.json').write_text(json.dumps(row))
    (folder/'trial/result.json').write_text(json.dumps({'finished_at':'2026-10-21T10:00:00Z'}))
    (folder/'trial/proot/cleanup.json').write_text(json.dumps({'after':[{'pid':1}]}))
    assert scheduler.reconcile_finished_trials(queue)==[]
    assert queue.snapshot()[0]['status']=='RUNNING'
