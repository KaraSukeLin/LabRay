"""Low/MTP3 preparation and original-task controls; formal launch is disabled."""
import argparse
import asyncio
import fcntl
import json
from pathlib import Path
import re
import time

from terminal_bench import ROOT, SOURCE, MANIFEST, read, save, sha, prepare
from terminal_proot_images import prepare_image

OUTPUT = ROOT/'build/terminal_bench/low_mtp3'
CONFIG = ROOT/'assets/terminal_bench_low_mtp3.json'


def inventory():
    cfg = read(CONFIG)
    manifest = prepare(cfg)
    tasks = {}
    for name, entry in manifest['tasks'].items():
        task = SOURCE/'terminal-bench-2-1/tasks'/name
        docker = (task/'environment/Dockerfile').read_text()
        tests = (task/'tests/test.sh').read_text()
        tasks[name] = dict(environment=entry['config']['environment'],
            agent=entry['config']['agent'], verifier=entry['config']['verifier'],
            category=entry['config']['metadata'].get('category'),
            dockerfile=docker, verifier_script=tests,
            special_dependency_lines=[line for line in (docker+'\n'+tests).splitlines()
                if re.search(r'apt|pip |uvx|npm|cargo|cuda|systemctl|docker|mount|/proc|/sys|curl|wget',line,re.I)],
            files=entry['files'], status='UNVALIDATED', controls=None)
    old=read(OUTPUT/'coverage.json')['tasks'] if (OUTPUT/'coverage.json').exists() else {}
    for name in tasks:
        if name in old:
            tasks[name]['status']=old[name]['status']
            tasks[name]['controls']=old[name]['controls']
    save(OUTPUT/'coverage.json',dict(denominator=89,nonstandard_proot=True,tasks=tasks))
    save(OUTPUT/'GOAL.json',dict(protocol=cfg,smoke_planned=30,formal_planned=267,
        stop_after_smoke=True,formal_authorized=False,
        candidate_identity='Awaiting user mapping; eNeural and Prisma not assumed equivalent',
        requirements=['89-task original environment inventory','five Oracle/NOP gates',
        'three model Low/MTP3 API checks','1 -> 2 -> 5 real concurrency gates',
        '30 distinct smoke trials','pause for user confirmation']))
    print(json.dumps(dict(tasks=len(tasks),stop_after_smoke=True)),flush=True)


def classify(data, folder):
    error=data.get('exception_info') or {}
    kind=error.get('exception_type','')
    reward=(data.get('verifier_result') or {}).get('rewards',{}).get('reward')
    if 'Verifier' in kind or 'Reward' in kind or 'AddTests' in kind:
        status='VERIFIER_FAIL'
    elif 'Timeout' in kind: status='TIMEOUT'
    elif any(s in kind for s in ['API','Provider','LLM','Authentication','RateLimit']): status='API_FAILURE'
    elif error: status='ENVIRONMENT_FAIL'
    elif not (folder/'verifier/ctrf.json').exists(): status='VERIFIER_FAIL'
    else: status='PASS' if reward==1 else 'MODEL_FAIL'
    return dict(status=status,reward=reward,exception=error)


async def control(task, agent, attempt=0):
    from harbor.models.trial.config import TrialConfig
    from harbor.trial.trial import Trial
    folder=OUTPUT/'controls'/task/agent/f'attempt-{attempt}'
    summary=folder/'summary.json'
    if summary.exists(): return read(summary)
    # Interrupted attempts remain immutable; never silently re-run them.
    if folder.exists(): raise RuntimeError(f'Incomplete control needs review: {folder}')
    folder.mkdir(parents=True)
    identity=dict(task=task,agent=agent,manifest=sha(MANIFEST),attempt=attempt)
    save(folder/'reservation.json',identity)
    cfg=TrialConfig.model_validate(dict(task=dict(path=str(SOURCE/'terminal-bench-2-1/tasks'/task)),
        trial_name='trial',trials_dir=str(folder),agent=dict(name=agent),
        environment=dict(type=None,import_path='terminal_proot:PRootEnvironment',delete=True),
        timeout_multiplier=1.0))
    trial=await Trial.create(cfg)
    began=time.time()
    try:
        result=await trial.run()
        data=result.model_dump(mode='json')
        info=classify(data,folder/'trial')
        cleanup=read(folder/'trial/proot/cleanup.json')
        info.update(identity=identity,wall_seconds=time.time()-began,
            result=str(folder/'trial/result.json'),cleanup_clean=not cleanup['after'])
        save(summary,info)
        print(task,agent,info['status'],info['reward'],flush=True)
        return info
    finally:
        if trial.agent_environment.started: await trial.agent_environment.stop(True)


async def controls(tasks, attempt=0):
    for task in tasks:
        print('PREPARE',task,flush=True)
        await asyncio.to_thread(prepare_image,task)
        rows={}
        for agent in ['oracle','nop']:
            rows[agent]=await control(task,agent,attempt)
        passed=(rows['oracle']['status']=='PASS' and rows['oracle']['reward']==1
            and rows['nop']['status']=='MODEL_FAIL' and rows['nop']['reward']==0
            and all(row['cleanup_clean'] for row in rows.values()))
        coverage=read(OUTPUT/'coverage.json')
        coverage['tasks'][task].update(status='CONTROLS_PASS' if passed else 'CONTROLS_FAIL',controls=rows)
        save(OUTPUT/'coverage.json',coverage)
        if not passed: raise RuntimeError(f'Environment gate failed: {task}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['inventory','controls'])
    parser.add_argument('--tasks',nargs='+')
    parser.add_argument('--attempt',type=int,default=0)
    args=parser.parse_args()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    with (OUTPUT/'preparation.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.command=='inventory': inventory()
        else: asyncio.run(controls(args.tasks or list(read(CONFIG)['smoke_tasks']),args.attempt))


if __name__=='__main__': main()
