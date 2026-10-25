"""Join saved pilot/Fidelity results and schedule explicit model/task choices; no inference."""
import argparse
import csv
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from analyze import export
from benchmark import CONFIG, ROOT
from benchmark_budget import TASK_COUNTS, capped_records, pipeline_seconds, schedule, stratified_projection
from fidelity import write_json

LABELS = ['Qwen BF16', 'Qwen FP8', 'NaviQuant Fast', 'NVIDIA NVFP4', 'RedHatAI NVFP4',
          'Unsloth NVFP4', 'RadixArk BF16-LMHead', 'PrismaAQUA', 'ModelOpt AutoQuantize']
SCENARIOS = dict(optimistic='optimistic_gpu_hours', typical='typical_gpu_hours',
                 conservative='conservative_gpu_hours')
GROUPS = dict(all9=list(range(9)), research4=[0, 2, 7, 8], research5=[0, 1, 2, 7, 8],
              focused5=[0, 2, 5, 7, 8], research6=[0, 1, 2, 5, 7, 8], practical5=[0, 1, 2, 5, 7])
SCOPES = dict(all_tasks=list(TASK_COUNTS), ifeval=['ifeval'], standard=['mmmu_standard'],
              vision=['mmmu_vision'], ifeval_standard=['ifeval', 'mmmu_standard'],
              ifeval_vision=['ifeval', 'mmmu_vision'])


def subset_cost(rows, overhead_seconds, tasks):
    """The original table apportions one load over all tasks; subsets still need one load."""
    value = sum(float(r['typical_gpu_hours']) for r in rows if r['task'] in tasks)
    fraction = sum(TASK_COUNTS[t] for t in tasks)/sum(TASK_COUNTS.values())
    return value+overhead_seconds/3600*(1-fraction)


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    models = json.loads(CONFIG.read_text())['models']
    names = dict(zip(models, LABELS))
    manifest = json.loads((ROOT/'assets/benchmark/pilot_v2/manifest.json').read_text())
    weights = {k: v['weight'] for k, v in manifest['selection'].items()}
    full = list(csv.DictReader((args.pilot/'analysis/full_estimates.csv').open()))
    loads = {r['model']: r for r in csv.DictReader((args.pilot/'analysis/model_load_and_wall.csv').open())}
    fidelity = {r['model']: r for r in csv.DictReader((ROOT/'build/autoquant/comparison.csv').open())}
    quality, caps, joined, reuse = [], [], [], []
    records_by_cap={1024:[],4096:[]}
    for model in models:
        path = args.pilot/'combined_runs'/model
        rows = [json.loads(s) for s in (path/'questions.jsonl').read_text().splitlines()]
        extra = [json.loads(s) for s in (path/'extensions.jsonl').read_text().splitlines()]
        item = dict(model=model, label=names[model], **{k: fidelity[model][k] for k in
            ['weight_GB', 'ppl', 'kl_bf16_to_candidate', 'top1_flip_rate', 'body_effective_bits', 'revision']})
        for cap in [1024, 4096]:
            selected = capped_records(rows, extra, cap)
            records_by_cap[cap].append({r['id']:r for r in selected})
            for task in TASK_COUNTS:
                subset = [r for r in selected if r['task']==task]
                count = sum(r['correct'] for r in subset)
                row = dict(model=model, label=names[model], mmmu_cap=cap, task=task,
                    questions=len(subset), correct=count, accuracy=count/len(subset),
                    population_weighted_accuracy=sum(weights[r['id']]*r['correct'] for r in subset)/TASK_COUNTS[task],
                    truncated=sum(r['finish_reason']=='length' for r in subset),
                    fallback=sum(r.get('extraction_fallback', False) for r in subset),
                    correct_with_fallback=sum(r['correct'] and r.get('extraction_fallback', False) for r in subset),
                    mean_output_tokens=sum(r['output_tokens'] for r in subset)/len(subset),
                    inference_seconds=sum(r['inference_seconds'] for r in subset))
                if task=='ifeval':
                    row.update(loose_accuracy=sum(r['loose_correct'] for r in subset)/len(subset),
                        strict_instruction_accuracy=sum(sum(r['strict_instructions']) for r in subset)/sum(len(r['strict_instructions']) for r in subset),
                        loose_instruction_accuracy=sum(sum(r['loose_instructions']) for r in subset)/sum(len(r['loose_instructions']) for r in subset))
                quality.append(row)
                item[f'{task}_{cap}_correct'] = count
                item[f'{task}_{cap}_weighted_accuracy'] = row['population_weighted_accuracy']
                item[f'{task}_{cap}_truncated'] = row['truncated']
                reuse.append(dict(model=model, cap=cap, task=task, cached_cases=len(subset),
                    saved_pipeline_gpu_hours=sum(map(pipeline_seconds, subset))/3600,
                    remaining_cases=TASK_COUNTS[task]-len(subset)))
            for task in ['mmmu_standard', 'mmmu_vision'] if cap==4096 else []:
                original = {r['id']: r for r in rows if r['task']==task}
                subset = [r for r in selected if r['task']==task]
                caps.append(dict(model=model, label=names[model], task=task, questions=len(subset),
                    main_correct=sum(r['correct'] for r in original.values()),
                    long_correct=sum(r['correct'] for r in subset),
                    harmful=sum(original[r['id']]['correct'] and not r['correct'] for r in subset),
                    recovery=sum(not original[r['id']]['correct'] and r['correct'] for r in subset),
                    answer_changes=sum(original[r['id']]['answer']!=r['answer'] for r in subset),
                    main_truncated=sum(r['finish_reason']=='length' for r in original.values()),
                    long_truncated=sum(r['finish_reason']=='length' for r in subset)))
        for cap in [1024, 4096]:
            item[f'full_{cap}_gpu_hours'] = sum(float(r['typical_gpu_hours']) for r in full
                if r['model']==model and r['profile']==f'mmmu_{cap}')
        joined.append(item)
    export(args.output/'quality_by_cap', quality)
    export(args.output/'cap_sensitivity', caps)
    export(args.output/'integrated_comparison', joined)
    export(args.output/'pilot_reuse_savings', reuse)

    start = datetime.fromisoformat(args.start).astimezone(timezone.utc)
    deadline = datetime.fromisoformat(args.deadline).astimezone(timezone.utc)
    taipei = ZoneInfo('Asia/Taipei')
    budgets, schedules = [], []
    projections={(cap,task):stratified_projection(records_by_cap[cap],manifest,task)
                 for cap in [1024,4096] for task in TASK_COUNTS}
    overhead=np.array([float(loads[m]['once_per_model_overhead_seconds']) for m in models])
    for group, indices in GROUPS.items():
        chosen = [models[i] for i in indices]
        for scope, tasks in SCOPES.items():
            for cap in [1024, 4096]:
                by_model = {m: [r for r in full if r['model']==m and r['profile']==f'mmmu_{cap}'] for m in chosen}
                point=(sum(projections[cap,t][0] for t in tasks)+overhead)/3600
                draws=(sum(projections[cap,t][1] for t in tasks)+overhead[:,None])/3600
                bounds=np.percentile(draws,[2.5,97.5],axis=-1)
                typical={models[i]:float(point[i]) for i in indices}
                for i in indices:
                    assert np.isclose(point[i],subset_cost(by_model[models[i]],overhead[i],tasks),rtol=1e-12)
                order = sorted(chosen, key=typical.get, reverse=True)
                for scenario in SCENARIOS:
                    values={'optimistic':bounds[0],'typical':point,'conservative':bounds[1]*1.25}[scenario]
                    costs={models[i]:float(values[i]) for i in indices}
                    wall, jobs = schedule(costs, len(args.gpus), order)
                    finish = start+timedelta(hours=wall)
                    budgets.append(dict(group=group, scope=scope, mmmu_cap=cap, scenario=scenario,
                        models=len(chosen), cases=len(chosen)*sum(TASK_COUNTS[t] for t in tasks),
                        occupied_gpu_hours=sum(costs.values()), single_gpu_wall_hours=sum(costs.values()),
                        multi_gpu_wall_hours=wall, reserved_gpu_hours=wall*len(args.gpus),
                        longest_model_hours=max(costs.values()), ideal_divisible_wall_lower_bound=sum(costs.values())/len(args.gpus),
                        finish_taipei=finish.astimezone(taipei).isoformat(timespec='minutes'),
                        fits_assumed_deadline=finish<=deadline,
                        leaves_three_hours_for_reporting=finish<=deadline-timedelta(hours=3)))
                    for job in jobs:
                        schedules.append(dict(group=group, scope=scope, mmmu_cap=cap, scenario=scenario,
                            **dict(job, gpu=args.gpus[job['gpu']]), label=names[job['model']],
                            finish_taipei=(start+timedelta(hours=job['end_hours'])).astimezone(taipei).isoformat(timespec='minutes')))
    original=json.loads((args.pilot/'analysis/scenarios.json').read_text())
    for row in budgets:
        if row['group']=='all9' and row['scope']=='all_tasks':
            expected=original[f'mmmu_{row["mmmu_cap"]}'][row['scenario']]
            for key in ['occupied_gpu_hours','multi_gpu_wall_hours']:
                assert np.isclose(row[key],expected[key],rtol=1e-12)
    export(args.output/'subset_budgets', budgets)
    export(args.output/'subset_schedules', schedules)
    write_json(args.output/'planning_assumptions.json', dict(start_utc=start.isoformat(),
        start_taipei=start.astimezone(taipei).isoformat(), deadline_taipei=deadline.astimezone(taipei).isoformat(),
        available_hours=(deadline-start).total_seconds()/3600, gpus=args.gpus,
        reporting_buffer_hours=3,
        groups={g: [names[models[i]] for i in ids] for g, ids in GROUPS.items()},
        budget='Fresh full data, one TP=1 serial-request job per model; one model initialization per task subset',
        reuse='Saved pilot could reduce new inference; subset budgets conservatively charge all cases',
        mtp='Off only; no representative MTP speedup measurement exists',
        uncertainty='Bootstrap sampling bounds and 25% upper contingency, not a deadline guarantee'))

    stages = []
    for stage in ['runs', 'ifeval_supplement_runs']:
        launch = json.loads((args.pilot/stage/'launch.json').read_text())
        infos = [json.loads((args.pilot/stage/m/'run.json').read_text()) for m in models]
        stages.append(dict(stage=stage, cases=sum(r['questions'] for r in infos),
            wall_hours=launch['wall_seconds']/3600,
            occupied_gpu_hours=sum(j['process_wall_seconds'] for j in launch['jobs'])/3600,
            reserved_gpu_hours=launch['wall_seconds']/3600*len(launch['gpus']),
            initialization_seconds=sum(r['initialization_seconds'] for r in infos),
            inference_seconds=sum(r['inference_seconds'] for r in infos),
            scoring_seconds=sum(r['scoring_seconds'] for r in infos),
            all_exit_codes_zero=all(j['exit_code']==0 for j in launch['jobs'])))
    export(args.output/'actual_stage_costs', stages)
    controls = []
    for model in models:
        p=args.pilot/'combined_runs'/model
        main={r['id']:r for r in map(json.loads,(p/'questions.jsonl').read_text().splitlines())}
        for row in map(json.loads,(p/'extensions.jsonl').read_text().splitlines()):
            if row['probe_reason']=='fixed_control':
                original=main[row['id']]
                controls.append(dict(model=model,id=row['id'],same_tokens=row['token_ids']==original['token_ids'],
                    same_answer=row['answer']==original['answer'],
                    inference_delta_seconds=row['inference_seconds']-original['inference_seconds']))
    export(args.output/'uncensored_controls', controls)
    write_json(args.output/'decision_audit.json', dict(main_cases=len(models)*200,
        official_replay=json.loads((args.pilot/'official_replay/audit.json').read_text()),
        matched_uncensored_control_tokens=all(r['same_tokens'] for r in controls),
        matched_uncensored_control_answers=all(r['same_answer'] for r in controls),
        uncensored_controls=len(controls), full_inference_started=False,
        analysis_scripts_launch_inference=False, long_cap_residual_truncation=sum(r['long_truncated'] for r in caps)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pilot',type=Path,default=ROOT/'build/benchmark/pilot_v1')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpus',type=int,nargs='+',default=[0,1,2,3,4])
    p.add_argument('--start',default=datetime.now(timezone.utc).isoformat())
    p.add_argument('--deadline',default='2026-10-19T09:00:00+08:00')
    main(p.parse_args())
