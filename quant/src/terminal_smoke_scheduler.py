"""Staged 1 -> 2 -> 5 GPU smoke scheduler. No formal-launch command exists."""
import argparse
import asyncio
from contextlib import ExitStack
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from unittest.mock import patch

import terminal_bench as serving
from terminal_bench import ROOT, read, save, sha
from terminal_low import CONFIG, OUTPUT, classify
from terminal_queue import Queue

INFRA={'API_FAILURE','ENVIRONMENT_FAIL','VERIFIER_FAIL','INTERRUPTED'}


def identity():
    files=[CONFIG,ROOT/'assets/terminal_bench_manifest.json',ROOT/'assets/terminal_bench_images.json',
        *[ROOT/'src'/name for name in ['terminal_smoke_scheduler.py','terminal_queue.py','terminal_low.py',
        'terminal_proot.py','terminal_proot_images.py','terminal_observer.py','terminal_bench.py']]]
    cfg=read(CONFIG)
    audits={row['model']:row for row in read(OUTPUT/'model-audit.json')}
    models={}
    for model in cfg['models']:
        audit=audits[model]
        checkpoint=Path('/root/model')/model
        assert sha(checkpoint/'config.json')==audit['config_sha256']
        for weight in audit['files']:
            stat=(checkpoint/weight['name']).stat()
            assert (stat.st_size,stat.st_mtime_ns)==(weight['bytes'],weight['mtime_ns']), 'Checkpoint changed since integrity audit'
        models[model]={key:audit[key] for key in ['revision','config_sha256','files']}
    return dict(config=cfg,models=models,files={str(path.relative_to(ROOT)):sha(path) for path in files})


def gates(cfg):
    assert cfg['stop_after_smoke'] and cfg['formal_requires_new_user_confirmation']
    mapped=cfg['candidate_models']
    assert set(mapped)=={'NaviQuant','Aqua'} and all(mapped.values()),'User checkpoint mapping is required'
    assert len(cfg['models'])==len(set(cfg['models']))==3
    assert set(cfg['models'])=={'Qwen-Qwen3.8-27B',*mapped.values()}
    coverage=read(OUTPUT/'coverage.json')
    assert coverage['denominator']==89
    for task in cfg['smoke_tasks']:
        assert coverage['tasks'][task]['status']=='CONTROLS_PASS',task
    for model in cfg['models']:
        pre=OUTPUT/'preflight'/model
        assert read(pre/'AUDIT.json')['passed'],model
        services=read(pre/'services/services.json')
        assert services and all(row.get('stopped_at') for row in services)
        for row in services:
            argv=row['command']
            assert argv[argv.index('--tensor-parallel-size')+1]=='1'
            assert json.loads(argv[argv.index('--speculative-config')+1])==dict(method='mtp',num_speculative_tokens=3)
        # Single-GPU preflight success is mandatory; no fallback or forced unsupported TP1.
    return list(cfg['smoke_tasks'])


async def trial(row,cfg,gpu):
    from harbor.models.trial.config import TrialConfig
    from harbor.trial.trial import Trial
    folder=OUTPUT/'smoke'/row['id']
    folder.mkdir(parents=True,exist_ok=False)
    save(folder/'identity.json',row)
    ac=serving.agent_config(cfg,row['model'],gpu,row['seed'],bool(row['mode']),
        f'smoke/{row["model"]}/thinking-{row["mode"]}/{row["id"]}')
    ac.pop('n_concurrent')
    tc=TrialConfig.model_validate(dict(task=dict(path=str(serving.SOURCE/'terminal-bench-2-1/tasks'/row['task'])),
        trial_name='trial',trials_dir=str(folder),agent=ac,
        environment=dict(type=None,import_path='terminal_proot:PRootEnvironment',delete=True),timeout_multiplier=1.0))
    instance=await Trial.create(tc)
    try:
        result=await instance.run()
        data=result.model_dump(mode='json')
        summary=classify(data,folder/'trial')
        cleanup=read(folder/'trial/proot/cleanup.json')
        if cleanup['after']: summary.update(status='ENVIRONMENT_FAIL',cleanup_failure=True)
        summary.update(identity=row,agent_execution=data.get('agent_execution'),
            native_reward=(data.get('verifier_result') or {}).get('rewards'),
            result=str(folder/'trial/result.json'))
        save(folder/'summary.json',summary)
        return summary
    finally:
        if instance.agent_environment.started: await instance.agent_environment.stop(True)


def worker(gpu,limit,preferred):
    import terminal_low_preflight as preflight
    cfg=read(CONFIG)
    queue=Queue(OUTPUT/'smoke.sqlite',identity())
    owner_start=Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')',1)[1].split()[19]
    completed=0
    while limit==0 or completed<limit:
        row=queue.claim(gpu,os.getpid(),owner_start,cfg['models'],preferred)
        if row is None: return
        model=row['model']
        original=serving.server_command
        def command(config,checkpoint,index):
            return original(config,checkpoint,index)+['--speculative-config',json.dumps(dict(method='mtp',num_speculative_tokens=3))]
        service_dir=OUTPUT/'services'/f'gpu{gpu}'/str(time.time_ns())
        with patch.object(serving,'server_command',command):
            try:
                with serving.services(cfg,model,[gpu],service_dir):
                    while row:
                        before=preflight.metrics(gpu)
                        summary=asyncio.run(trial(row,cfg,gpu))
                        after=preflight.metrics(gpu)
                        save(OUTPUT/'smoke'/row['id']/'mtp.json',dict(before=before,after=after,
                            delta={key:value-before.get(key,0) for key,value in after.items()}))
                        queue.finish(row['id'],summary['status'],summary['result'])
                        completed+=1
                        print(gpu,row['id'],summary['status'],flush=True)
                        if summary['status'] in INFRA: raise RuntimeError('Infrastructure gate stopped this wave')
                        if limit and completed>=limit: return
                        # Keep the loaded model while work remains. Do not claim a different model before unloading.
                        row=queue.claim(gpu,os.getpid(),owner_start,[model],model)
            except BaseException:
                # Preserve a reservation and partial artifacts; controller reports it for review.
                raise
        preferred=None


def overlap(rows,required):
    intervals=[]
    for row in rows:
        summary=read(OUTPUT/'smoke'/row['id']/'summary.json')
        timing=summary['agent_execution']
        if not timing or not timing.get('finished_at'): continue
        intervals.append((datetime.fromisoformat(timing['started_at'].replace('Z','+00:00')).timestamp(),1))
        intervals.append((datetime.fromisoformat(timing['finished_at'].replace('Z','+00:00')).timestamp(),-1))
    active=peak=0
    for _,change in sorted(intervals,key=lambda item:(item[0],item[1])):
        active+=change
        peak=max(peak,active)
    assert peak>=required,f'Actual concurrent Agent executions {peak} < {required}'
    return peak



def resource_sample():
    result=dict(gpus=serving.gpu_inventory(),load_average=Path('/proc/loadavg').read_text().strip(),
        disk_free_bytes=__import__('shutil').disk_usage(OUTPUT).free)
    for name in ['cpu','memory','io']:
        path=Path('/proc/pressure')/name
        if path.exists(): result[name+'_pressure']=path.read_text()
    for name in ['cpu.max','cpu.stat','memory.max','memory.current','memory.events','io.stat']:
        path=Path('/sys/fs/cgroup')/name
        if path.exists(): result[name]=path.read_text()
    return result


def launch_wave(queue,width,limit,label):
    checkpoint=OUTPUT/f'{label}-start.json'
    if checkpoint.exists() and limit:
        before=read(checkpoint)
        changed=[row for row in queue.snapshot() if before[row['id']]=='PENDING' and row['status']!='PENDING']
        if changed:
            assert len(changed)==width and all(row['status'] not in INFRA|{'RUNNING'} for row in changed), 'Interrupted concurrency wave requires review; do not consume replacement trials'
            peak=overlap(changed,width)
            save(OUTPUT/f'{label}.json',dict(passed=True,trial_ids=[row['id'] for row in changed],actual_peak_concurrent_agents=peak,formal_trials=0,recovered_from_checkpoint=True))
            return
    before={row['id']:row['status'] for row in queue.snapshot()}
    save(checkpoint,before)
    children=[]
    with ExitStack() as stack:
        try:
            for gpu in range(width):
                argv=[str(ROOT/'.harbor-venv/bin/python'),str(Path(__file__).resolve()),'worker',
                    '--gpu',str(gpu),'--limit',str(limit)]
                if width==1:argv+=['--preferred','Qwen-Qwen3.8-27B']
                log=stack.enter_context((OUTPUT/f'{label}-gpu{gpu}.log').open('a'))
                children.append(subprocess.Popen(argv,env=dict(os.environ,OPENAI_API_KEY='local-only',
                    LITELLM_LOCAL_MODEL_COST_MAP='True'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
            while any(child.poll() is None for child in children):
                failed=[child for child in children if child.poll() not in (None,0)]
                if failed:raise RuntimeError('Worker stopped; inspect per-GPU logs and original evidence')
                rows=queue.snapshot()
                save(OUTPUT/'STATUS.json',dict(stage=label,time=time.time(),trials=rows,
                    pending=sum(row['status']=='PENDING' for row in rows),
                    running=sum(row['status']=='RUNNING' for row in rows),
                    completed=sum(row['status'] not in {'PENDING','RUNNING'} for row in rows),
                    resources=resource_sample(),
                    formal_trials_started=0,stop_after_smoke=True))
                with (OUTPUT/'resources.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(time=time.time(),stage=label,**resource_sample()))+'\n')
                time.sleep(5)
            assert all(child.returncode==0 for child in children)
        finally:
            for child in children: serving.stop_process(child)
    changed=[row for row in queue.snapshot() if before[row['id']]=='PENDING' and row['status'] not in {'PENDING','RUNNING'}]
    assert not any(row['status'] in INFRA for row in changed)
    peak=overlap(changed,width) if limit else None
    save(OUTPUT/f'{label}.json',dict(passed=True,trial_ids=[row['id'] for row in changed],
        actual_peak_concurrent_agents=peak,formal_trials=0))



def reconcile_finished_trials(queue):
    """Recover only a dead owner's complete native result; never launch a retry."""
    recovered=[]
    for row in queue.interrupted_owners():
        folder=OUTPUT/'smoke'/row['id']
        native=folder/'trial'
        result_path=native/'result.json'
        cleanup_path=native/'proot/cleanup.json'
        if not result_path.exists() or not cleanup_path.exists(): continue
        data=read(result_path)
        if not data.get('finished_at') or read(cleanup_path).get('after')!=[]: continue
        saved=read(folder/'identity.json')
        for key in ['id','model','task','mode','seed','attempt','protocol']:
            assert saved[key]==row[key],f'Resume trial identity mismatch: {key}'
        assert data['task_name'].split('/')[-1]==row['task']
        agent=data['config']['agent']
        assert agent['model_name']=='hosted_vllm/'+row['model']
        assert agent['kwargs']['interleaved_thinking']==bool(row['mode'])
        assert agent['kwargs']['llm_call_kwargs']['seed']==row['seed']
        summary=classify(data,native)
        summary.update(identity=saved,agent_execution=data.get('agent_execution'),
            native_reward=(data.get('verifier_result') or {}).get('rewards'),result=str(result_path))
        summary_path=folder/'summary.json'
        if summary_path.exists():
            previous=read(summary_path)
            assert previous['status']==summary['status'] and previous['result']==summary['result']
        else: save(summary_path,summary)
        queue.finish(row['id'],summary['status'],result_path)
        recovered.append(row['id'])
    return recovered


def launch():
    cfg=read(CONFIG)
    tasks=gates(cfg)
    with (OUTPUT/'scheduler.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        queue=Queue(OUTPUT/'smoke.sqlite',identity())
        queue.enqueue(cfg['models'],tasks,cfg['seeds'][0])
        assert len(queue.snapshot())==30
        recovered=reconcile_finished_trials(queue)
        if recovered: save(OUTPUT/f'recovered-{time.time_ns()}.json',dict(trial_ids=recovered,new_attempts=0))
        assert not any(row['status']=='RUNNING' for row in queue.snapshot()),'Reconcile existing live/interrupted owners first'
        assert not any(row['status'] in INFRA for row in queue.snapshot()),'Infrastructure result requires review; no automatic retry'
        for width in [1,2,5]:
            label=f'concurrency-{width}'
            if (OUTPUT/f'{label}.json').exists():
                assert read(OUTPUT/f'{label}.json')['passed']
                continue
            launch_wave(queue,width,1,label)
        launch_wave(queue,5,0,'smoke-remaining')
        rows=queue.snapshot()
        assert len(rows)==30 and all(row['status'] in {'PASS','MODEL_FAIL','TIMEOUT'} for row in rows)
        save(OUTPUT/'STATE.json',dict(state='PAUSED_AWAITING_USER_CONFIRMATION',smoke_completed=30,
            formal_trials_started=0,stop_after_smoke=True,note='Review smoke trajectories, prompt audits and stability before formal authorization.'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['smoke','worker'])
    parser.add_argument('--gpu',type=int)
    parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--preferred')
    args=parser.parse_args()
    signal.signal(signal.SIGTERM,lambda signum,frame: (_ for _ in ()).throw(KeyboardInterrupt()))
    if args.command=='smoke': launch()
    else:worker(args.gpu,args.limit,args.preferred)
