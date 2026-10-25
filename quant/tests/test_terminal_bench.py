from pathlib import Path
import json
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from terminal_bench import CONFIG, agent_config, archive_incomplete, completed_job, execute, partition_tasks, planned_cells, read
from terminal_analysis import outcome, paired_intervals, task_effects


def test_matrix_gate_is_reused_and_smoke_is_separate():
    config=read(CONFIG)
    # Fixed manifest is CPU-only and contains all unmodified task resource hashes.
    gate=planned_cells(config,'gate')
    rest=planned_cells(config,'matrix')
    assert len(gate)==1 and len(rest)==8
    def keys(cells):
        return {(c['model'],c['repeat'],t) for c in cells for t in c['tasks']}
    assert len(keys(gate))==89 and len(keys(rest))==712
    assert not keys(gate)&keys(rest)
    smoke=planned_cells(config,'smoke')
    assert sum(len(c['tasks']) for c in smoke)==10
    assert [c['interleaved_thinking'] for c in smoke]==[False,True]
    assert all(c['seed']==config['seeds'][0] and c['model']==config['models'][0] for c in smoke)


def test_formal_run_requires_explicit_new_authorization(tmp_path):
    with pytest.raises(AssertionError,match='NEW user instruction'):
        execute(read(CONFIG),tmp_path,'gate',[0,1,2,3,4])
    assert not list(tmp_path.rglob('services.json'))


def test_all_sampling_and_session_fields_are_forwarded():
    c=read(CONFIG)
    a=agent_config(c,c['models'][0],3,c['seeds'][2],False,'matrix/example/r2-thinking-0/gpu3')
    k=a['kwargs'];q=k['llm_call_kwargs'];extra=q['extra_body']
    assert q['seed']==c['seeds'][2]
    assert k['session_id_headers']==['X-Terminal-Trial']
    assert k['enable_summarize'] is True and k['proactive_summarization_threshold']==8000
    assert k['api_base']=='http://127.0.0.1:18103/v1'
    assert extra['chat_template_kwargs']==c['chat_template_kwargs']
    assert {**{k:q[k] for k in c['sampling'] if k in q},
            **{k:extra[k] for k in c['sampling'] if k in extra}}==c['sampling']
    assert a.get('override_timeout_sec') is None
    assert k['model_info']['max_input_tokens']+q['max_tokens']==c['max_model_len']


def test_resume_avoids_completed_job_gpu_startup(tmp_path):
    job=tmp_path/'gpu1.json'
    job.write_text(json.dumps(dict(jobs_dir=str(tmp_path/'jobs'),job_name='gpu1',
        tasks=[dict(path='/p/fix-git'),dict(path='/p/regex-log')])))
    assert not completed_job(job)
    for task in ['fix-git','regex-log']:
        path=tmp_path/'jobs/gpu1'/f'{task}__id'
        path.mkdir(parents=True)
        (path/'result.json').write_text(json.dumps(dict(task_name='terminal-bench/'+task,
            finished_at='2026-10-20T00:00:00+00:00')))
    assert completed_job(job)
    duplicate=tmp_path/'jobs/gpu1/fix-git__other'
    duplicate.mkdir()
    (duplicate/'result.json').write_text((tmp_path/'jobs/gpu1/fix-git__id/result.json').read_text())
    with pytest.raises(AssertionError,match='Duplicate'): completed_job(job)


def test_official_mean_keeps_failed_verifier_and_timeout_in_denominator():
    base=dict(verifier_result=None,exception_info=dict(exception_type='AgentTimeoutError'))
    assert outcome(base)['success']==0 and outcome(base)['status']=='agent_timeout'
    base['exception_info']['exception_type']='VerifierTimeoutError'
    assert outcome(base)['success']==0 and outcome(base)['status']=='verifier_failed'
    base['exception_info']['exception_type']='CancelledError'
    assert outcome(base)['status']=='aborted'
    good=dict(verifier_result=dict(rewards={'reward':1.0}))
    assert outcome(good)['success']==1
    good['exception_info']=dict(exception_type='AgentTimeoutError')
    assert outcome(good)['success']==1 and outcome(good)['status']=='agent_timeout'


def test_three_repeats_stay_inside_task_clusters():
    a=np.array([[1,1,1],[0,0,0]])
    b=1-a
    effects=task_effects(a,b)
    np.testing.assert_array_equal(effects, [[-1,1],[1,0],[0,1]])
    fit=paired_intervals(a,b,resamples=999,family=6)
    delta=next(r for r in fit if r['metric']=='accuracy_delta' and r['confidence']==.95)
    assert delta['tasks']==2 and delta['low']==-1 and delta['high']==1


def test_incomplete_harbor_trial_logs_are_archived_before_native_resume(tmp_path):
    job=tmp_path/'gpu1.json'
    job.write_text(json.dumps(dict(jobs_dir=str(tmp_path/'jobs'),job_name='gpu1',
        tasks=[dict(path='/p/fix-git')])))
    trial=tmp_path/'jobs/gpu1/fix-git__interrupted'
    trial.mkdir(parents=True)
    (trial/'config.json').write_text('{}')
    (trial/'agent.log').write_bytes(b'partial terminal history\n')
    archive_incomplete(job)
    assert not trial.exists()
    saved=list((tmp_path/'interrupted').glob('*/fix-git__interrupted/agent.log'))
    assert len(saved)==1 and saved[0].read_bytes()==b'partial terminal history\n'


def test_static_timeout_partition_covers_all_tasks_once():
    from terminal_bench import MANIFEST
    tasks=read(MANIFEST)['tasks']
    shards=partition_tasks(list(tasks),5)
    flat=[task for shard in shards for task in shard]
    assert len(flat)==len(set(flat))==89 and set(flat)==set(tasks)
    assert shards==partition_tasks(list(reversed(tasks)),5)
    loads=[sum(tasks[t]['config']['agent']['timeout_sec'] for t in shard) for shard in shards]
    old=[sum(tasks[t]['config']['agent']['timeout_sec'] for t in sorted(tasks)[slot::5]) for slot in range(5)]
    assert max(loads)<max(old)
