#!/usr/bin/env python3
"""Official Verifier outcomes and paired whole-task bootstrap; no trial exclusion."""
import argparse
from collections import Counter
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
from scipy.stats import bootstrap
from analyze import export
from terminal_bench import SOURCE, read, save

from benchmark import official_module

_METRICS=official_module('terminal_metric_base',SOURCE/'harbor/src/harbor/metrics/base.py')
_POLICY=official_module('terminal_error_policy',SOURCE/'harbor/src/harbor/models/trial/error_policy.py')
policy_for=_POLICY.policy_for


def outcome(result):
    error=(result.get('exception_info') or {}).get('exception_type')
    reward=(result.get('verifier_result') or {}).get('rewards')
    if reward is not None:
        assert len(reward)==1 and next(iter(reward.values())) in [0,1], 'Non-binary official reward'
    score=_METRICS.aggregate_reward_dicts([reward],metric_name='mean',
        aggregate=lambda values:sum(values)/len(values))['mean']
    if error=='CancelledError': status='aborted'
    elif error=='AgentTimeoutError': status='agent_timeout'
    elif error: status=str(policy_for(error).cause)
    elif reward is None: status='verifier_failed'
    else: status='success' if score==1 else 'model_task_failure'
    return dict(success=int(score==1),status=status,error=error,official_reward=reward)


def task_effects(baseline,candidate):
    a,b=np.asarray(baseline),np.asarray(candidate)
    assert a.shape==b.shape and a.ndim==2 and a.shape[1]==3
    delta=(b-a).mean(axis=1)
    harmful=((a==1)&(b==0)).mean(axis=1)
    recovery=((a==0)&(b==1)).mean(axis=1)
    np.testing.assert_allclose(delta,recovery-harmful,atol=1e-15)
    return np.stack([delta,harmful,recovery])


def paired_intervals(baseline,candidate,resamples=199999,family=6,seed=20261020):
    effects=task_effects(baseline,candidate)
    rows=[]
    for confidence in [.95,1-.05/family]:
        fit=bootstrap((effects,),lambda x,axis=-1:np.mean(x,axis=axis),
            axis=-1,vectorized=True,paired=True,n_resamples=resamples,batch=512,
            method='percentile',confidence_level=confidence,rng=np.random.default_rng(seed))
        for metric,value,lo,hi in zip(['accuracy_delta','harmful_flip','recovery_flip'],
                                      effects.mean(axis=1),fit.confidence_interval.low,fit.confidence_interval.high):
            rows.append(dict(metric=metric,estimate=float(value),low=float(lo),high=float(hi),
                             confidence=confidence,tasks=len(baseline)))
    return rows


def elapsed(info):
    if not info or not info.get('started_at') or not info.get('finished_at'): return None
    return (datetime.fromisoformat(info['finished_at'])-datetime.fromisoformat(info['started_at'])).total_seconds()


def collect(runs,stage):
    plan=read(runs/'plan.json')
    stages=['gate','matrix'] if stage=='formal' else ['smoke']
    records=[]
    for stage in stages:
        for cell in plan['cells'][stage]:
            key=f'{stage}/{cell["model"]}/r{cell["repeat"]}-thinking-{int(cell["interleaved_thinking"])}'
            observed={}
            for path in (runs/key/'jobs').glob('*/*/result.json'):
                result=read(path)
                task=result['task_name'].split('/')[-1]
                assert task not in observed, 'Duplicate task/repetition result'
                trajectory=path.parent/'agent/trajectory.json'
                atif=read(trajectory) if trajectory.exists() else {}
                context=result.get('agent_result') or {}
                steps=atif.get('steps',[])
                row=dict(model=cell['model'],repeat=cell['repeat'],seed=cell['seed'],task=task,
                    interleaved_thinking=cell['interleaved_thinking'],result=str(path),
                    **outcome(result),agent_seconds=elapsed(result.get('agent_execution')),
                    wall_seconds=elapsed(result),verifier_seconds=elapsed(result.get('verifier')),
                    steps=sum(s.get('source')=='agent' for s in steps),
                    tool_calls=sum(len(s.get('tool_calls') or []) for s in steps),
                    prompt_tokens=context.get('n_input_tokens'),output_tokens=context.get('n_output_tokens'),
                    session_id=atif.get('session_id'),reasoning_tokens=0,api_seconds=0.0,
                    summarizations=(context.get('metadata') or {}).get('summarization_count',0),
                    parse_errors=sum('Previous response had parsing errors:' in str(s.get('observation') or '')
                                     for s in steps))
                observed[task]=row
            for task in cell['tasks']:
                records.append(observed.get(task,dict(model=cell['model'],repeat=cell['repeat'],
                    seed=cell['seed'],task=task,interleaved_thinking=cell['interleaved_thinking'],
                    status='pending',success=None)))
    by_session={r['session_id']:r for r in records if r.get('session_id')}
    for path in runs.rglob('audit/*.json'):
        event=read(path)
        session=event.get('identity',{}).get('x-terminal-trial')
        if session not in by_session: continue
        row=by_session[session]
        usage=event.get('response',{}).get('usage',{})
        row['reasoning_tokens']+=((usage.get('completion_tokens_details') or {}).get('reasoning_tokens') or 0)
        row['api_seconds']+=event.get('elapsed_seconds',0)
    return records,plan


def analyze(args):
    records,plan=collect(args.runs,args.stage)
    pending=[r for r in records if r['status']=='pending']
    aborted=[r for r in records if r['status']=='aborted']
    assert args.allow_partial or not pending, 'Incomplete matrix; use --allow-partial only for a labelled diagnostic snapshot'
    args.output.mkdir(parents=True,exist_ok=False)
    export(args.output/'trials',records)
    summary=[]
    for model,mode in sorted({(r['model'],r['interleaved_thinking']) for r in records}):
        rows=[r for r in records if r['model']==model and r['interleaved_thinking']==mode and r['success'] is not None]
        if not rows: continue
        tasks=sorted({r['task'] for r in rows})
        complete=[t for t in tasks if sum(r['task']==t for r in rows)==3]
        panel=np.array([[next(r['success'] for r in rows if r['task']==task and r['repeat']==repeat)
                         for repeat in range(3)] for task in complete])
        costs=[r['wall_seconds'] for r in rows if r.get('wall_seconds') is not None]
        api=sum(r.get('api_seconds',0) for r in rows)
        output=sum(r.get('output_tokens') or 0 for r in rows)
        entry=dict(model=model,interleaved_thinking=mode,observed_trials=len(rows),
            success_rate=float(np.mean([r['success'] for r in rows])),complete_three_seed_tasks=len(complete),
            pass_at_3=float(panel.max(axis=1).mean()) if complete else None,
            all_three_success=float(panel.min(axis=1).mean()) if complete else None,
            seed_accuracy=[float(x) for x in panel.mean(axis=0)] if complete else None,
            seed_accuracy_sd=float(panel.mean(axis=0).std(ddof=1)) if complete else None,
            wall_p50=float(np.median(costs)) if costs else None,
            wall_p95=float(np.quantile(costs,.95)) if costs else None,
            output_tokens=output,output_tokens_per_api_second=output/api if api else None,
            diagnostics=dict(Counter(r['status'] for r in rows)))
        summary.append(entry)
    export(args.output/'summary',summary)
    pairs=[]
    config=plan['config']
    if args.stage=='formal':
        # Never silently pair a convenient subset for the primary formal comparison.
        full=not pending and not aborted
        if full:
            tasks=sorted({r['task'] for r in records})
            panels={model:np.array([[next(r['success'] for r in records if
                r['model']==model and r['task']==task and r['repeat']==repeat)
                for repeat in range(3)] for task in tasks]) for model in config['models']}
            for model in config['models'][1:]:
                pairs.extend(dict(model=model,reference=config['models'][0],**r) for r in
                    paired_intervals(panels[config['models'][0]],panels[model],args.resamples,
                                     config['comparison_family'],config['bootstrap_seed']))
            export(args.output/'paired',pairs)
    ledgers=[row for p in args.runs.rglob('services.json') for row in read(p)]
    save(args.output/'statistics.json',dict(official_verifier=True,benchmark_score=True,
        stage=args.stage,expected_trials=len(records),pending=len(pending),aborted=len(aborted),
        complete_primary_matrix=not pending and not aborted,failures_excluded=False,
        gpu_hours=sum(row.get('gpu_hours',0) for row in ledgers),gpu_ledger=ledgers,
        bootstrap=dict(cluster='whole task with all three repeats retained',resamples=args.resamples,
            family=config['comparison_family'],seed=config['bootstrap_seed']),
        note='Pending/aborted matrices are diagnostic snapshots. Infrastructure errors remain in the denominator; they require investigation, not selective exclusion.'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stage',choices=['smoke','formal'],default='formal')
    p.add_argument('--allow-partial',action='store_true')
    p.add_argument('--resamples',type=int,default=199999)
    analyze(p.parse_args())
