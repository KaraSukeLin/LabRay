#!/usr/bin/env python3
"""Pinned Harbor jobs, one active Agent per TP=1 replica, with explicit gates."""
import argparse
from contextlib import contextmanager, ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from threading import Event, Thread
import tomllib
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT/'assets/terminal_bench_config.json'
SOURCE = ROOT/'build/terminal_bench/sources'
BASE = Path('/root/model/Qwen-Qwen3.8-27B')
MANIFEST = ROOT/'assets/terminal_bench_manifest.json'
IMAGES = ROOT/'assets/terminal_bench_images.json'


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_command(argv, **kwargs):
    return subprocess.check_output([str(x) for x in argv], text=True, **kwargs).strip()


def prepare(config):
    for name, key in [('harbor','harbor'), ('terminal-bench-2-1','dataset')]:
        repo = SOURCE/name
        assert run_command(['git','-C',repo,'rev-parse','HEAD']) == config[key]['revision']
        assert not run_command(['git','-C',repo,'status','--porcelain','--untracked-files=no'])
    tasks = {}
    for path in sorted((SOURCE/'terminal-bench-2-1/tasks').glob('*/task.toml')):
        data = tomllib.loads(path.read_text())
        tasks[path.parent.name] = dict(config=data, files={str(p.relative_to(path.parent)):sha(p)
            for p in sorted(path.parent.rglob('*')) if p.is_file()})
    assert len(tasks) == config['dataset']['tasks'] == 89
    for name, timeout in config['smoke_tasks'].items():
        assert tasks[name]['config']['agent']['timeout_sec'] == timeout, name
    result = dict(dataset=config['dataset'], harbor=config['harbor'], tasks=tasks,
        tokenizer={p:sha(BASE/p) for p in ['tokenizer.json','tokenizer_config.json','chat_template.jinja']})
    if MANIFEST.exists():
        assert read(MANIFEST) == result, 'Pinned task/tokenizer contents changed'
    else:
        save(MANIFEST,result)
    print(json.dumps(dict(tasks=len(tasks), smoke_timeouts=config['smoke_tasks'], manifest=str(MANIFEST))))
    return result


def versions(python, packages):
    code='import importlib.metadata as m,json;print(json.dumps({p:m.version(p) for p in '+repr(packages)+'}))'
    return json.loads(run_command([python,'-c',code]))


def gpu_inventory():
    raw = run_command(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu',
                       '--format=csv,noheader,nounits'])
    return {int(v[0]):dict(index=int(v[0]),uuid=v[1],used_mb=int(v[2]),total_mb=int(v[3]),
                          utilization=int(v[4]))
            for line in raw.splitlines() if (v:=[s.strip() for s in line.split(',')])}


def preflight(output, container_probe=False):
    facts = dict(time=time.time(), gpus=gpu_inventory(), docker_cli=shutil.which('docker'),
        docker_host=os.getenv('DOCKER_HOST'),
        cgroups=[line for line in Path('/proc/mounts').read_text().splitlines() if 'cgroup' in line])
    errors = []
    if facts['docker_cli']:
        for name,argv in [('info',['docker','info','--format','{{json .}}']),
                          ('compose',['docker','compose','version','--short'])]:
            try:
                value = run_command(argv,stderr=subprocess.STDOUT,timeout=30)
                facts[name] = json.loads(value) if name=='info' else value
            except (subprocess.SubprocessError,ValueError) as exc:
                errors.append(f'{name}: {exc}')
        info = facts.get('info',{})
        for field in ['MemoryLimit','CpuCfsQuota']:
            if not info.get(field): errors.append('Docker does not enforce '+field)
    else:
        errors.append('Docker CLI/Engine/Compose unavailable; official task containers cannot start')
    if not errors and container_probe:
        name='quant-tb-preflight-'+str(time.time_ns())
        probe=output.parent/(name+'.txt')
        probe.write_text(name+'\n')
        try:
            pinned=read(IMAGES)['fix-git']['digest']
            text=run_command(['docker','run','--rm','--name',name,'--network','none',
                '--cpus','1','--memory','2g','--mount',f'type=bind,src={probe.resolve()},dst=/probe,readonly',
                '--entrypoint','/bin/sh',pinned,'-c',
                'cat /probe; cat /sys/fs/cgroup/cpu.max; cat /sys/fs/cgroup/memory.max'],
                stderr=subprocess.STDOUT,timeout=60)
            lines=text.splitlines()
            assert lines[0]==name, 'Docker daemon cannot see the same workspace path'
            quota,period=map(int,lines[1].split())
            assert quota/period==1 and int(lines[2])==2*1024**3, 'CPU/RAM limits not enforced'
            facts['container_probe']=dict(passed=True,image=pinned,cpu_max=lines[1],memory_max=lines[2])
        except (subprocess.SubprocessError,ValueError,AssertionError) as exc:
            errors.append('Container bind/resource probe failed: '+str(exc))
        finally:
            subprocess.run(['docker','rm','-f',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            probe.unlink()
    facts.update(ready=not errors,errors=errors)
    save(output,facts)
    print(json.dumps(facts,ensure_ascii=False))
    return facts


def call_kwargs(config, seed, identity):
    sampling = dict(config['sampling'],seed=seed)
    extra = {key:sampling.pop(key) for key in ['top_k','min_p','repetition_penalty']}
    extra.update(chat_template_kwargs=config['chat_template_kwargs'],return_token_ids=True)
    return dict(sampling, extra_body=extra,
                extra_headers={'X-Terminal-Identity':identity})


def agent_config(config, model, gpu, seed, interleaved, identity):
    return dict(name='terminus-2', model_name='hosted_vllm/'+model,
        n_concurrent=1, kwargs=dict(api_base=f'http://127.0.0.1:{config["port_base"]+gpu}/v1',
        temperature=config['sampling']['temperature'], reasoning_effort=config['reasoning_effort'],
        interleaved_thinking=interleaved,enable_summarize=config['enable_summarize'],
        proactive_summarization_threshold=config['proactive_summarization_threshold'],
        collect_rollout_details=True,store_all_messages=True,
        session_id_headers=['X-Terminal-Trial'],
        model_info=dict(max_input_tokens=config['harbor_max_input_tokens'],
            max_output_tokens=config['sampling']['max_tokens'],input_cost_per_token=0.0,
            output_cost_per_token=0.0,litellm_provider='hosted_vllm',supports_reasoning=True),
        llm_kwargs=dict(api_key='local-only'),llm_call_kwargs=call_kwargs(config,seed,identity)))


def planned_cells(config, stage):
    tasks = sorted(read(MANIFEST)['tasks'])
    if stage=='smoke':
        return [dict(model=config['models'][0],repeat=0,seed=config['seeds'][0],
            interleaved_thinking=mode,tasks=list(config['smoke_tasks'])) for mode in [False,True]]
    return [dict(model=model,repeat=repeat,seed=seed,
                 interleaved_thinking=config['interleaved_thinking'],tasks=tasks)
        for model in config['models'] for repeat,seed in enumerate(config['seeds'])
        if (model==config['models'][0] and repeat==0)==(stage=='gate')]


def partition_tasks(tasks, slots):
    """Static greedy balance by declared Agent timeout; Harbor owns each shard."""
    info=read(MANIFEST)['tasks']
    shards=[[] for _ in range(slots)]
    loads=[0.0]*slots
    for task in sorted(tasks,key=lambda t:(-info[t]['config']['agent']['timeout_sec'],t)):
        slot=min(range(slots),key=lambda i:(loads[i],i))
        shards[slot].append(task)
        loads[slot]+=info[task]['config']['agent']['timeout_sec']
    return shards


def write_plan(config, output, gpus):
    prepare(config)
    assert len(gpus)==len(set(gpus)) and all(0<=g<=4 for g in gpus)
    # A fixed slot assignment survives safe Harbor resume without changing endpoints.
    inventory={r['model']:r for r in read(ROOT/'build/autoquant/checkpoints/inventory.json')}
    plan = dict(config=config,manifest_sha256=sha(MANIFEST),images_sha256=sha(IMAGES),gpus=gpus,
        files={str(p.relative_to(ROOT)):sha(p) for p in
               [CONFIG, Path(__file__), ROOT/'src/terminal_observer.py']},
        environment=dict(inference=versions(ROOT/'.modelopt-venv/bin/python',
            ['vllm','torch','transformers','compressed-tensors']),harness=versions(
            ROOT/'.harbor-venv/bin/python',['harbor','litellm','openai','pydantic'])),
        models={name:dict(revision=inventory[name]['revision'],
            config_sha256=sha(Path('/root/model')/name/'config.json'),
            weights=[dict(name=p.name,bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns)
                     for p in sorted((Path('/root/model')/name).glob('*.safetensors'))])
                for name in config['models']},
        cells={stage:planned_cells(config,stage) for stage in ['smoke','gate','matrix']})
    dest = output/'plan.json'
    if dest.exists(): assert read(dest)==plan, 'Resume identity changed (code/config/model/task/GPU)'
    else: save(dest,plan)
    for stage,cells in plan['cells'].items():
        for cell in cells:
            key=f'{stage}/{cell["model"]}/r{cell["repeat"]}-thinking-{int(cell["interleaved_thinking"])}'
            shards=([cell['tasks'][slot::len(gpus)] for slot in range(len(gpus))]
                    if stage=='smoke' else partition_tasks(cell['tasks'],len(gpus)))
            for slot,gpu in enumerate(gpus):
                tasks=shards[slot]
                if not tasks: continue
                identity=key+f'/gpu{gpu}'
                job=dict(job_name=f'gpu{gpu}',jobs_dir=str((output/key/'jobs').resolve()),
                    n_attempts=1,n_concurrent_trials=1,timeout_multiplier=1.0,
                    retry=dict(max_retries=0),environment=dict(type='docker',delete=True,
                        cpu_enforcement_policy='limit',memory_enforcement_policy='limit'),
                    agents=[agent_config(config,cell['model'],gpu,cell['seed'],
                                         cell['interleaved_thinking'],identity)],
                    tasks=[dict(path=str(SOURCE/'terminal-bench-2-1/tasks'/t)) for t in tasks])
                save(output/key/f'gpu{gpu}.json',job)
    if not (output/'STATE.json').exists():
        save(output/'STATE.json',dict(state='PAUSED',formal_trials_started=False,
                                     smoke_planned=10,gate_planned=89,total_formal_planned=801))
    print(json.dumps(dict(plan=str(dest),formal_matrix=801,inference_started=False)))
    return plan


def server_command(config, model, gpu):
    return [str(ROOT/'.modelopt-venv/bin/python'),'-m','vllm.entrypoints.cli.main','serve',
        str(Path('/root/model')/model),'--tokenizer',str(BASE),'--served-model-name',model,
        '--host','127.0.0.1','--port',str(config['port_base']+gpu),'--dtype','bfloat16',
        '--tensor-parallel-size','1','--seed','0','--max-model-len',str(config['max_model_len']),
        '--max-num-seqs','1','--max-num-batched-tokens','4096','--gpu-memory-utilization',
        str(config['gpu_memory_utilization']),'--enforce-eager','--no-enable-prefix-caching',
        '--kv-cache-dtype','bfloat16','--generation-config','vllm','--language-model-only',
        '--reasoning-parser','qwen3','--default-chat-template-kwargs',
        json.dumps(config['chat_template_kwargs']),'--kernel-config',
        json.dumps({'linear_backend_per_quant':{'fp8_block_w8a8':'triton'}}),
        '--middleware','terminal_observer.AuditMiddleware']


def stop_process(process):
    # Address our process group, including workers if the API parent exited early.
    try: os.killpg(process.pid,signal.SIGINT)
    except ProcessLookupError: pass
    try: process.wait(timeout=30)
    except subprocess.TimeoutExpired: pass
    try: os.killpg(process.pid,signal.SIGKILL)
    except ProcessLookupError: pass
    process.wait()


@contextmanager
def services(config, model, gpus, output):
    output.mkdir(parents=True,exist_ok=True)
    processes, ledger = [], []
    old=read(output/'services.json') if (output/'services.json').exists() else []
    halt=Event()
    def monitor():
        with (output/'resources.jsonl').open('a') as f:
            while not halt.is_set():
                row=dict(time=time.time(),gpus=gpu_inventory(),
                    load_average=Path('/proc/loadavg').read_text().strip(),
                    memory=Path('/proc/meminfo').read_text(),
                    disk_free_bytes=shutil.disk_usage(output).free)
                f.write(json.dumps(row)+'\n');f.flush()
                halt.wait(5)
    observer=Thread(target=monitor,daemon=True)
    observer.start()
    with ExitStack() as stack:
        try:
            for gpu in gpus:
                lock=stack.enter_context((ROOT/f'build/terminal_bench/gpu{gpu}.lock').open('w'))
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                facts=gpu_inventory()[gpu]
                assert facts['used_mb']<500, f'GPU {gpu} occupied; never preempt another experiment'
                folder=output/f'gpu{gpu}'/str(time.time_ns())
                folder.mkdir(parents=True)
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='8',
                    VLLM_NO_USAGE_STATS='1',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
                    PYTHONPATH=str(ROOT/'src')+':'+os.getenv('PYTHONPATH',''),
                    LD_LIBRARY_PATH=str(ROOT/'build/cuda-compat/cuda-compat')+':'+os.getenv('LD_LIBRARY_PATH',''),
                    TERMINAL_AUDIT_DIR=str(folder/'audit'),TERMINAL_MODEL=model)
                argv=server_command(config,model,gpu)
                log=stack.enter_context((folder/'server.log').open('w'))
                process=subprocess.Popen(argv,env=env,stdout=log,stderr=subprocess.STDOUT,
                                         start_new_session=True)
                processes.append(process)
                ledger.append(dict(gpu=gpu,hardware=facts,model=model,pid=process.pid,
                    started_at=time.time(),command=argv,folder=str(folder)))
            save(output/'services.json',old+ledger)
            deadline=time.monotonic()+600
            for process,row in zip(processes,ledger):
                url=f'http://127.0.0.1:{config["port_base"]+row["gpu"]}/health'
                while True:
                    assert process.poll() is None, f'Server exited: {row["folder"]}/server.log'
                    try:
                        with urllib.request.urlopen(url,timeout=2) as response:
                            if response.status==200: break
                    except OSError: pass
                    assert time.monotonic()<deadline, 'Service readiness timeout (outside Agent timeout)'
                    time.sleep(1)
                row['ready_at']=time.time()
                save(output/'services.json',old+ledger)
            yield ledger
        finally:
            halt.set()
            observer.join(timeout=10)
            for process,row in zip(processes,ledger):
                stop_process(process)
                row.update(stopped_at=time.time(),exit_code=process.returncode)
                row['gpu_hours']=(row['stopped_at']-row['started_at'])/3600
            save(output/'services.json',old+ledger)


def completed_job(path):
    job=read(path)
    expected={Path(task['path']).name for task in job['tasks']}
    folder=Path(job['jobs_dir'])/job['job_name']
    results=[read(p) for p in folder.glob('*/result.json')]
    names=[r['task_name'].split('/')[-1] for r in results]
    assert len(names)==len(set(names)), 'Duplicate saved trials'
    return set(names)==expected and all(r.get('finished_at') for r in results)


def archive_incomplete(path):
    """Harbor removes unfinished trial folders on resume; preserve them first."""
    job=read(path)
    folder=Path(job['jobs_dir'])/job['job_name']
    if not folder.exists(): return
    for trial in folder.iterdir():
        if trial.is_dir() and (trial/'config.json').exists() and not (trial/'result.json').exists():
            destination=path.parent/'interrupted'/str(time.time_ns())/trial.name
            destination.parent.mkdir(parents=True)
            shutil.move(str(trial),str(destination))


def check_trial_errors(folder):
    for path in folder.glob('*/*/result.json'):
        error=(read(path).get('exception_info') or {}).get('exception_type')
        if error and error not in {'AgentTimeoutError','ContextLengthExceededError',
                'OutputLengthExceededError','AgentSafetyRefusalError'}:
            raise RuntimeError(f'System gate stopped: {error}; {path}')


def execute(config, output, stage, gpus, authorize_full=False):
    output.mkdir(parents=True,exist_ok=True)
    with (output/'launcher.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return execute_stage(config,output,stage,gpus,authorize_full)


def execute_stage(config, output, stage, gpus, authorize_full=False):
    if stage!='smoke':
        assert authorize_full, 'Formal execution requires a NEW user instruction and --authorize-full'
        ready=read(output/'SMOKE_READY.json')
        assert ready['passed'], 'Actual official smoke and protocol evidence must pass first'
    plan=write_plan(config,output,gpus)
    check=preflight(output/'preflight.json',container_probe=True)
    assert check['ready'], '; '.join(check['errors'])
    if stage=='matrix':
        assert read(output/'GATE_READY.json')['passed'], 'Complete BF16 gate must pass first'
    # Keep official published images locally immutable; refuse to pull mutable tags in a job.
    images=read(output/'images.json')
    for task in {t for c in plan['cells'][stage] for t in c['tasks']}:
        item=images[task]
        actual=json.loads(run_command(['docker','image','inspect',item['tag']]))[0]
        assert item['digest'] in actual['RepoDigests'], 'Official image identity changed'
    save(output/'STATE.json',dict(state='RUNNING',stage=stage,started_at=time.time()))
    try:
        for cell in plan['cells'][stage]:
            key=f'{stage}/{cell["model"]}/r{cell["repeat"]}-thinking-{int(cell["interleaved_thinking"])}'
            configs=[p for p in (output/key).glob('gpu*.json') if not completed_job(p)]
            if not configs: continue  # No service startup or GPU cost for completed jobs.
            active_gpus=[int(p.stem.removeprefix('gpu')) for p in configs]
            # Harbor alone decides which completed trials are reusable.
            with services(config,cell['model'],active_gpus,output/key/'services'):
                children=[]
                with ExitStack() as stack:
                    try:
                        for path in sorted(configs):
                            archive_incomplete(path)
                            log=stack.enter_context(path.with_suffix('.log').open('a'))
                            env=dict(os.environ,OPENAI_API_KEY='local-only',LITELLM_LOCAL_MODEL_COST_MAP='True')
                            p=subprocess.Popen([str(ROOT/'.harbor-venv/bin/harbor'),'run','--config',str(path)],
                                env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                            children.append(p)
                        while any(p.poll() is None for p in children):
                            check_trial_errors(output/key/'jobs')
                            time.sleep(1)
                        assert all(p.returncode==0 for p in children), 'Harbor job failed; inspect logs'
                        check_trial_errors(output/key/'jobs')
                    finally:
                        for p in children: stop_process(p)
        save(output/f'{stage.upper()}_EXECUTION.json',dict(finished=True,time=time.time()))
    finally:
        save(output/'STATE.json',dict(state='PAUSED',stage=stage,stopped_at=time.time(),
            note='No automatic continuation. Run audit and review before the next stage.'))


def resolve_images(config):
    """Pin the official published Docker Hub tags without starting containers."""
    tasks=read(MANIFEST)['tasks']
    records=read(IMAGES) if IMAGES.exists() else {}
    from concurrent.futures import ThreadPoolExecutor
    def resolve(name):
        tag=tasks[name]['config']['environment']['docker_image']
        repository,version=tag.rsplit(':',1)
        url='https://auth.docker.io/token?service=registry.docker.io&scope=repository:'+repository+':pull'
        with urllib.request.urlopen(url,timeout=30) as response:
            token=json.load(response)['token']
        request=urllib.request.Request('https://registry-1.docker.io/v2/'+repository+'/manifests/'+version,
            method='HEAD',headers={'Authorization':'Bearer '+token,'Accept':
                'application/vnd.oci.image.index.v1+json, application/vnd.docker.distribution.manifest.list.v2+json, '
                'application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.v2+json'})
        with urllib.request.urlopen(request,timeout=30) as response:
            digest=response.headers['Docker-Content-Digest']
        assert digest.startswith('sha256:') and len(digest)==71
        return name,dict(tag=tag,digest=repository+'@'+digest,resolved_at=time.time())
    with ThreadPoolExecutor(max_workers=4) as pool:
        for name,value in pool.map(resolve,[t for t in sorted(tasks) if t not in records]):
            records[name]=value
            save(IMAGES,records)
    print(json.dumps(dict(pinned_images=len(records),containers_started=0)))


def lock_images(config, output, stage):
    tasks=read(MANIFEST)['tasks']
    names=list(config['smoke_tasks']) if stage=='smoke' else sorted(tasks)
    dest=output/'images.json'
    records=read(dest) if dest.exists() else {}
    for name in names:
        tag=tasks[name]['config']['environment']['docker_image']
        pinned=read(IMAGES)[name]
        assert pinned['tag']==tag
        if name in records: continue
        subprocess.run(['docker','pull',pinned['digest']],check=True)
        subprocess.run(['docker','tag',pinned['digest'],tag],check=True)
        item=json.loads(run_command(['docker','image','inspect',tag]))[0]
        digests=item['RepoDigests']
        assert pinned['digest'] in digests and item['Architecture']=='amd64' and item['Os']=='linux'
        records[name]=dict(tag=tag,digest=pinned['digest'],image_id=item['Id'],
                           architecture=item['Architecture'],locked_at=time.time())
        save(dest,records)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','preflight','plan','run','resolve-images','lock-images'])
    p.add_argument('--config',type=Path,default=CONFIG)
    p.add_argument('--output',type=Path,default=ROOT/'build/terminal_bench/run')
    p.add_argument('--gpus',type=int,nargs='+',default=[0,1,2,3,4])
    p.add_argument('--stage',choices=['smoke','gate','matrix'],default='smoke')
    p.add_argument('--authorize-full',action='store_true')
    p.add_argument('--container-probe',action='store_true')
    args=p.parse_args()
    def terminate(signum,frame):
        raise KeyboardInterrupt('Termination requested; cleaning up owned services/jobs')
    signal.signal(signal.SIGTERM,terminate)
    config=read(args.config)
    assert not config['mtp'] and config['kv_cache_dtype']=='bfloat16'
    assert config['timeout_multiplier']==1.0
    assert config['harbor_max_input_tokens']+config['sampling']['max_tokens']<=config['max_model_len']
    if args.command=='prepare': prepare(config)
    elif args.command=='preflight': return 0 if preflight(args.output/'preflight.json',args.container_probe)['ready'] else 1
    elif args.command=='plan': write_plan(config,args.output.resolve(),args.gpus)
    elif args.command=='resolve-images': resolve_images(config)
    elif args.command=='lock-images': lock_images(config,args.output,args.stage)
    else: execute(config,args.output.resolve(),args.stage,args.gpus,args.authorize_full)
    return 0


if __name__=='__main__':
    sys.exit(main())
