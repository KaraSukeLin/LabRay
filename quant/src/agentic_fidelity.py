#!/usr/bin/env python3
"""Teacher forcing on fixed BF16 API trajectories, reusing fidelity.py unchanged."""
import argparse
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import re
import signal
import subprocess
import sys

import numpy as np
from fidelity import BASELINE, aggregate, write_json
from terminal_bench import BASE, ROOT, gpu_inventory, read, save, sha, stop_process

LABELS=('reasoning','terminal_command','code','after_tool')
LOW_CONFIG=ROOT/'assets/terminal_bench_low_mtp3.json'


def validate_source_request(request):
    expected=read(LOW_CONFIG)['chat_template_kwargs']
    if request.get('chat_template_kwargs')!=expected:
        raise ValueError('Source request differs from the pinned Low/MTP3 chat template')


def completed_payloads(event):
    """Return usable request/response payloads, or skip an incomplete audit event."""
    request=event.get('request') or {}
    response=event.get('response') or {}
    choices=response.get('choices') or []
    if event.get('status')!='complete' or event.get('http_status')!=200 or not choices:
        return None
    return request,response,choices


def task_from_identity(identity, session_id, sessions, smoke=False):
    """Resolve the task from the per-trial identity, not the reused agent session ID."""
    known_tasks=set(sessions.get(session_id, ()))
    parts=identity.split('/')
    if len(parts)>=4 and parts[0]=='formal':
        task=parts[2]
        if known_tasks and task not in known_tasks:
            raise ValueError(f'Identity task {task!r} is absent from session {session_id!r}')
        return task
    if len(known_tasks)==1:
        return next(iter(known_tasks))
    if smoke:
        return identity.split('/thinking-')[0]
    raise ValueError(f'Official source has no unambiguous task identity: {identity!r}')


def render_ids(tokenizer, request):
    value=tokenizer.apply_chat_template(request['messages'],tokenize=True,
        add_generation_prompt=True,**request['chat_template_kwargs'])
    return value['input_ids'] if isinstance(value,Mapping) else value


def generated_ranges(tokenizer, ids, messages):
    """Labels describe positions; the input IDs are never re-tokenized or trimmed."""
    text=tokenizer.decode(ids,skip_special_tokens=False)
    close=tokenizer.convert_tokens_to_ids('</think>')
    end=ids.index(close) if close in ids else len(ids)
    ranges={'reasoning':[(0,end)],'terminal_command':[],'code':[],'after_tool':[]}
    if any(m['role']=='assistant' for m in messages):
        ranges['after_tool']=[(0,min(32,len(ids)))]

    def position(offset):
        # Decode only O(log N) prefixes per boundary. Offsets are annotations only.
        lo,hi=0,len(ids)
        while lo<hi:
            mid=(lo+hi)//2
            if len(tokenizer.decode(ids[:mid],skip_special_tokens=False))<offset: lo=mid+1
            else: hi=mid
        return lo

    for match in re.finditer(r'"keystrokes"\s*:\s*"((?:[^"\\]|\\.)*)"',text):
        left,right=position(match.start(1)),position(match.end(1))
        if left>=right: continue
        label='code' if re.search(r'python|def |import |node|function |<<',match[1]) else 'terminal_command'
        ranges[label].append((left,right))
    return ranges


def prepare(args):
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(BASE,local_files_only=True)
    sessions={}
    for trajectory in sorted(args.trajectories.glob('trials/*/trial/agent/trajectory.json')):
        data=read(trajectory)
        trial_dir=trajectory.parent.parent
        config_path=trial_dir/'config.json'
        if config_path.exists() and data.get('session_id'):
            trial_config=read(config_path)
            task=Path(trial_config['task']['path']).name
            sessions.setdefault(data['session_id'],set()).add(task)
    candidates={}
    source={}
    excluded=dict(non_bf16=0,synthetic=0,incomplete=0,too_long=0)
    for path in sorted(args.trajectories.glob('services/gpu*/**/audit/*.json')):
        event=read(path)
        if event.get('model')!=BASELINE:
            excluded['non_bf16']+=1
            continue
        identity=event.get('identity',{}).get('x-terminal-identity','')
        if identity.startswith('synthetic/') and not args.smoke:
            excluded['synthetic']+=1
            continue
        payload=completed_payloads(event)
        if payload is None:
            excluded['incomplete']+=1
            continue
        request,response,choices=payload
        prompt=response.get('prompt_token_ids')
        generated=(choices[0] or {}).get('token_ids')
        if not prompt or not generated:
            raise ValueError(f'Actual engine IDs missing: {path}')
        # Fixed request headers identify the Harbor session/trial, with ATIF linked by it.
        headers=event.get('identity') or {}
        trial=headers.get('x-terminal-trial') or identity
        assert trial, f'No source trial identity: {path}'
        task=task_from_identity(identity,trial,sessions,smoke=args.smoke)
        validate_source_request(request)
        expected=render_ids(tokenizer,request)
        assert expected==prompt, f'Saved messages do not render to actual model prefix: {path}'
        source[str(path.resolve())]=sha(path)
        for label,ranges in generated_ranges(tokenizer,generated,request['messages']).items():
            for left,right in ranges:
                right=min(right,len(generated),args.max_context-len(prompt))
                if right-left<args.tokens_per_window:
                    excluded['too_long']+=1
                    continue
                candidates.setdefault(task,{}).setdefault(label,[]).append(
                    dict(path=str(path.resolve()),trial=trial,prompt=prompt,generated=generated,
                         left=left,right=right,event_id=event['id']))
    rng=np.random.default_rng(args.seed)
    tasks=sorted(candidates)
    if args.smoke: tasks=tasks[:1]
    assert tasks, 'No eligible BF16 trajectories; synthetic data require explicit --smoke'
    inputs,starts,samples=[],[],[]
    for task in tasks:
        labels=[k for k in LABELS if k in candidates[task]]
        for slot in range(args.windows_per_task):
            label=labels[slot%len(labels)]
            pool=candidates[task][label]
            item=pool[int(rng.integers(len(pool)))]
            begin=int(rng.integers(item['left'],item['right']-args.tokens_per_window+1))
            start=len(item['prompt'])+begin
            ids=item['prompt']+item['generated'][:begin+args.tokens_per_window]
            key=hashlib.sha256(json.dumps([item['event_id'],begin,args.tokens_per_window]).encode()).hexdigest()
            if any(s['window_key']==key for s in samples):
                # No repeated positions from a tiny source masquerade as independent windows.
                continue
            inputs.append(ids)
            starts.append(start)
            samples.append(dict(id=f'{task}/{slot}',document_id=task,domain=label,language='agentic',
                context=start,source=item['path'],source_sha256=source[item['path']],
                trial=item['trial'],event_id=item['event_id'],generated_start=begin,
                window_key=key,tokens=args.tokens_per_window))
    assert inputs
    if args.smoke:
        assert sum(len(ids)-start for ids,start in zip(inputs,starts))<=32, 'Smoke is limited to 32 positions'
    args.output.mkdir(parents=True,exist_ok=False)
    manifest=dict(tokenizer=str(BASE),input_ids=inputs,score_starts=starts,samples=samples,
        max_model_len=max(len(ids) for ids in inputs)+1)
    save(args.output/'tokens.json',manifest)
    save(args.output/'sampling.json',dict(benchmark_score=False,smoke=args.smoke,seed=args.seed,
        windows_per_task=args.windows_per_task,tokens_per_window=args.tokens_per_window,
        tasks=len(tasks),windows=len(inputs),tokens=sum(s['tokens'] for s in samples),
        context_eligibility_limit=args.max_context,prefix_truncation=False,teacher_forced=True,
        labels='reasoning marker; JSON keystrokes lexical code/command; first 32 positions after feedback',
        sampling='fixed RNG; equal maximum window quota per task; cycle available labels; uniform event/span/start',
        unavailable_labels={t:[k for k in LABELS if k not in candidates[t]] for t in tasks},
        excluded=excluded,sources=source,
        atif_sessions={sid:sorted(task_set) for sid,task_set in sessions.items()},
        source_sha256=sha(Path(__file__)),
        fidelity_sha256=sha(ROOT/'src/fidelity.py'),low_config_sha256=sha(LOW_CONFIG),
        tokenizer_sha256={str(p):sha(BASE/p) for p in
            ['tokenizer.json','tokenizer_config.json','chat_template.jinja']}))
    print(json.dumps(dict(output=str(args.output),tasks=len(tasks),windows=len(inputs),
        tokens=sum(s['tokens'] for s in samples))))


def run(args):
    manifest=read(args.prepared/'sampling.json')
    tokens=read(args.prepared/'tokens.json')
    assert sha(ROOT/'src/fidelity.py')==manifest['fidelity_sha256']
    assert sha(LOW_CONFIG)==manifest['low_config_sha256']
    for source,digest in manifest['sources'].items(): assert sha(source)==digest
    for name,digest in manifest['tokenizer_sha256'].items(): assert sha(BASE/name)==digest
    assert manifest['smoke'] or args.authorize_full, 'Formal Agentic KL requires a NEW instruction and --authorize-full'
    argv=[sys.executable,str(ROOT/'src/fidelity.py'),'run','--tokens',str(args.prepared/'tokens.json'),
          '--logprob-format','dense','--models',*read(LOW_CONFIG)['models'][1:],
          '--output',str(args.output),'--gpus',*[str(g) for g in args.gpus]]
    env=dict(__import__('os').environ,VLLM_USE_V2_MODEL_RUNNER='1',OMP_NUM_THREADS='8',
             HF_HUB_OFFLINE='1',VLLM_NO_USAGE_STATS='1')
    env['LD_LIBRARY_PATH']=str(ROOT/'build/cuda-compat/cuda-compat')+':'+env.get('LD_LIBRARY_PATH','')
    assert all(gpu_inventory()[g]['used_mb']<500 for g in args.gpus), 'GPU occupied; never preempt another experiment'
    process=subprocess.Popen(argv,env=env,start_new_session=True)
    try:
        assert process.wait()==0, 'Fidelity worker failed; inspect run logs'
    finally:
        stop_process(process)
    save(args.output/'trajectory_sampling.json',manifest)
    analyze(args.output)


def task_parts(parts):
    # Normalize each complete task to weight 1 before existing SciPy bootstrap.
    result=[]
    for task in sorted({p['document_id'] for p in parts}):
        chunk=[p for p in parts if p['document_id']==task]
        count=sum(p['tokens'] for p in chunk)
        row={key:sum(p[key] for p in chunk)/count for key in ['nll','ref_nll','kl_sum','flips']}
        result.append(dict(row,tokens=1,document_id=task,domain='agentic',context='all'))
    return result


def analyze(run_path):
    from analyze import cluster_interval, METRICS
    rows=read(run_path/'results.json')
    assert read(run_path/'sanity.json')['passed']
    summaries=[]
    for row in rows:
        assert row['status']=='ok'
        if row['run']=='bf16_repeat': continue
        parts=row['documents']
        for label in ['all',*sorted({p['domain'] for p in parts})]:
            chunk=parts if label=='all' else [p for p in parts if p['domain']==label]
            macro=task_parts(chunk)
            macro_metrics=aggregate(macro)
            macro_metrics.pop('tokens')  # Unit task weights are not a token count.
            entry=dict(model=row['model'],stratum=label,tasks=len(macro),
                scoring_positions=sum(p['tokens'] for p in chunk),task_equal_weight=macro_metrics,
                token_weighted=aggregate(chunk),mean_nll=sum(p['nll'] for p in macro)/len(macro),
                gpu_seconds=row['elapsed_seconds'],full_vocabulary=row['vocab_size'])
            if len(macro)>=2:
                point,ci=cluster_interval(macro,resamples=19999)
                entry['task_cluster_95ci']={k:[float(a),float(b)] for k,a,b in zip(METRICS,ci.low,ci.high)}
            else: entry['task_cluster_95ci']=None
            summaries.append(entry)
    save(run_path/'agentic_summary.json',dict(benchmark_score=False,full_vocabulary_kl=True,
        direction='BF16 to candidate',cluster='complete task, all selected windows jointly',
        results=summaries,note='Small smoke data provide correctness evidence, not a model ranking.'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare')
    q.add_argument('--trajectories',type=Path,required=True)
    q.add_argument('--output',type=Path,required=True)
    q.add_argument('--smoke',action='store_true')
    q.add_argument('--seed',type=int,default=20261020)
    q.add_argument('--windows-per-task',type=int,default=4)
    q.add_argument('--tokens-per-window',type=int,default=32)
    q.add_argument('--max-context',type=int,default=65536)
    q=sub.add_parser('run')
    q.add_argument('--prepared',type=Path,required=True)
    q.add_argument('--output',type=Path,required=True)
    q.add_argument('--gpus',type=int,nargs='+',default=[0])
    q.add_argument('--authorize-full',action='store_true')
    q=sub.add_parser('analyze')
    q.add_argument('--run',type=Path,required=True)
    a=p.parse_args()
    def terminate(signum,frame): raise KeyboardInterrupt('Fidelity interrupted')
    signal.signal(signal.SIGTERM,terminate)
    if a.command=='prepare': prepare(a)
    elif a.command=='run': run(a)
    else: analyze(a.run)
