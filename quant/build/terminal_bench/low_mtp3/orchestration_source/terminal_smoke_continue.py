"""Continue eligible smoke work after observed five-Agent overlap, without retries.

The original wave coordinator is held while its current trial keeps running.
Idle GPUs use the unchanged worker/protocol. The coordinator resumes only after
these workers drain, so two controllers cannot claim the same GPU concurrently.
"""
from contextlib import ExitStack
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import terminal_bench as serving
from terminal_bench import ROOT,read,save,sha
from terminal_low import OUTPUT
from terminal_queue import Queue
from terminal_smoke_scheduler import identity,resource_sample,INFRA


def ticks(pid):
    return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]


def proof(queue):
    before=read(OUTPUT/'concurrency-5-start.json')
    rows=[row for row in queue.snapshot() if before[row['id']]=='PENDING' and row['status']!='PENDING']
    assert len(rows)==5
    intervals=[]
    for row in rows:
        result=OUTPUT/'smoke'/row['id']/'trial/result.json'
        if result.exists():
            timing=read(result)['agent_execution']
            start,end=[datetime.fromisoformat(timing[k].replace('Z','+00:00')).timestamp() for k in ['started_at','finished_at']]
            assert not read(result.parent/'proot/cleanup.json')['after']
            evidence=[str(result)]
        else:
            assert ticks(row['owner_pid'])==row['owner_start']
            events=[]
            for path in (OUTPUT/'services').rglob('audit/*.json'):
                event=read(path)
                if event.get('identity',{}).get('x-terminal-identity','').endswith('/'+row['id']) and event.get('status')=='complete':
                    events.append((event,path))
            assert events
            start=min(event['started_at'] for event,_ in events)
            end=max(event['finished_at'] for event,_ in events)
            evidence=[str(path) for _,path in events]
        intervals.append(dict(trial=row['id'],start=start,end=end,evidence=evidence))
    overlap=min(item['end'] for item in intervals)-max(item['start'] for item in intervals)
    assert overlap>0,'No observed five-Agent overlap'
    assert not any(row['status'] in INFRA for row in rows)
    return dict(passed=True,actual_concurrent_agents=5,conservative_overlap_seconds=overlap,
        intervals=intervals,observed_at=time.time(),note='Complete native timings plus a conservative interval bounded by complete API records for the ongoing trial.')


def launch(parent_pid):
    with (OUTPUT/'continuation.lock').open('w') as lock,ExitStack() as stack:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert read(OUTPUT/'STATUS.json')['stage']=='concurrency-5'
        parent_ticks=ticks(parent_pid)
        assert b'terminal_smoke_scheduler.py\x00smoke' in Path(f'/proc/{parent_pid}/cmdline').read_bytes()
        queue=Queue(OUTPUT/'smoke.sqlite',identity())
        children={}
        held=False
        try:
            os.kill(parent_pid,signal.SIGSTOP)
            held=True
            rows=queue.snapshot()
            active=[row for row in rows if row['status']=='RUNNING']
            assert active
            for row in active:
                args=Path(f'/proc/{row["owner_pid"]}/cmdline').read_bytes().split(b'\x00')
                assert args[args.index(b'--limit')+1]==b'1','Main phase already started; do not attach'
            observed=proof(queue)
            save(OUTPUT/'concurrency-5-observed.json',observed)
            save(OUTPUT/'continuation-controller.json',dict(pid=os.getpid(),start_ticks=ticks(os.getpid()),
                parent_pid=parent_pid,parent_start_ticks=parent_ticks,original_trial_ids=[row['id'] for row in active],
                protocol_identity=identity(),source_sha256=sha(Path(__file__)),started_at=time.time()))
            while True:
                rows=queue.snapshot()
                if any(row['status'] in INFRA for row in rows):raise RuntimeError('Infrastructure result: stop affected wave')
                for gpu,child in children.items():
                    code=child.poll()
                    if code not in (None,0):raise RuntimeError(f'Continuation worker GPU {gpu} failed: {code}')
                pending=sum(row['status']=='PENDING' for row in rows)
                running=[row for row in rows if row['status']=='RUNNING']
                if pending:
                    hardware=serving.gpu_inventory()
                    busy={row['gpu'] for row in running}
                    for gpu in range(5):
                        if gpu in children or gpu in busy or hardware[gpu]['used_mb']>=500:continue
                        argv=[str(ROOT/'.harbor-venv/bin/python'),str(ROOT/'src/terminal_smoke_scheduler.py'),
                            'worker','--gpu',str(gpu),'--limit','0']
                        log=stack.enter_context((OUTPUT/f'continuation-gpu{gpu}.log').open('a'))
                        children[gpu]=subprocess.Popen(argv,env=dict(os.environ,OPENAI_API_KEY='local-only',
                            LITELLM_LOCAL_MODEL_COST_MAP='True',PYTHONDONTWRITEBYTECODE='1'),
                            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                        print('START continuation GPU',gpu,'PID',children[gpu].pid,flush=True)
                sampled=resource_sample()
                now=time.time()
                save(OUTPUT/'STATUS.json',dict(stage='smoke-work-conserving',time=now,trials=rows,pending=pending,
                    running=len(running),completed=sum(row['status'] not in {'PENDING','RUNNING'} for row in rows),
                    resources=sampled,formal_trials_started=0,stop_after_smoke=True))
                with (OUTPUT/'resources.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(time=now,stage='smoke-work-conserving',**sampled))+'\n')
                if not pending and not running and all(child.poll()==0 for child in children.values()):break
                time.sleep(5)
            save(OUTPUT/'continuation-complete.json',dict(finished=True,time=time.time(),formal_trials=0,
                completed=sum(row['status'] not in {'PENDING','RUNNING'} for row in queue.snapshot())))
        finally:
            for child in children.values():serving.stop_process(child)
            if held and ticks(parent_pid)==parent_ticks:
                os.kill(parent_pid,signal.SIGCONT)
                print('RESUME original coordinator',parent_pid,flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('parent_pid',type=int)
    signal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    launch(parser.parse_args().parent_pid)
