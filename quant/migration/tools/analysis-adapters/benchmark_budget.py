"""Audit a stratified pilot and project serial-request benchmark costs."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np
from scipy.stats import bootstrap

from analyze import export
from benchmark import CONFIG, ROOT
from fidelity import write_json

TASK_COUNTS = dict(ifeval=541, mmmu_standard=1730, mmmu_vision=1730)


def pipeline_seconds(row):
    return sum(row[k] for k in ['preprocessing_seconds', 'inference_seconds', 'scoring_seconds'])


def capped_records(main, extensions, cap=1024):
    """Only censored MMMU records need replacement when projecting a higher cap."""
    extra = {(r['id'], r['max_tokens']): r for r in extensions}
    selected = []
    for row in main:
        for limit in [4096, 8192]:
            if row['task'] != 'ifeval' and row['max_tokens'] < limit <= cap and row['finish_reason'] == 'length':
                row = extra[row['id'], limit]
        selected.append(row)
    return selected


def stratified_projection(records, manifest, task, resamples=19999, value=pipeline_seconds):
    """Models share resampled question indices; strata are independently sampled."""
    blocks, populations = [], []
    constant = np.zeros(len(records))
    for stratum in manifest['strata']:
        if stratum['task'] != task:
            continue
        ids = [k for k, s in manifest['selection'].items() if s['kind'] == 'probability'
               and s['sampling_stratum'] == stratum['stratum']
               and k.startswith(task+'/')]
        assert len(ids) == stratum['sample']
        block = np.array([[value(model[k]) for k in ids] for model in records])
        if stratum['sample'] == stratum['population']:
            constant += block.mean(axis=-1)*stratum['population']
        else:
            blocks.append(block)
            populations.append(stratum['population'])

    if not blocks:
        return constant, np.repeat(constant[:, None], resamples, axis=1)

    def statistic(*samples, axis=-1):
        # Extra final axes are bootstrap replicates supplied by SciPy.
        shape = (len(constant),)+(1,)*(samples[0].ndim-2)
        return constant.reshape(shape)+sum(n*s.mean(axis=axis) for n, s in zip(populations, samples))

    point = statistic(*blocks)
    fit = bootstrap(tuple(blocks), statistic, axis=-1, vectorized=True, paired=False,
                    n_resamples=resamples, batch=128, method='percentile', confidence_level=.95,
                    rng=np.random.default_rng(np.random.SeedSequence([20261017, list(TASK_COUNTS).index(task)])))
    return point, fit.bootstrap_distribution


def schedule(hours, gpus=5, order=None):
    loads = [0.]*gpus
    jobs = []
    order = order or sorted(hours, key=hours.get, reverse=True)
    assert set(order) == set(hours)
    for model in order:
        duration = hours[model]
        gpu = int(np.argmin(loads))
        jobs.append(dict(model=model, gpu=gpu, start_hours=loads[gpu], end_hours=loads[gpu]+duration,
                         duration_hours=duration))
        loads[gpu] += duration
    return max(loads), jobs


def quantiles(values):
    return dict(zip(['mean', 'p50', 'p95', 'max'],
                    map(float, [np.mean(values), *np.percentile(values, [50, 95]), max(values)])))


def write_report(output, models, summary, loads, projections, scenarios, launch, cap_pairs, manifest):
    labels = ['Qwen BF16', 'Qwen FP8', 'NaviQuant Fast', 'NVIDIA NVFP4', 'RedHatAI NVFP4',
              'Unsloth NVFP4', 'RadixArk BF16-LMHead', 'PrismaAQUA 5.5-bit', 'ModelOpt AutoQuantize 5.5-bit']
    names = dict(zip(models, labels))
    counts = manifest['counts']
    extension = manifest.get('ifeval_extension')
    lines = ['# IFEval／MMMU-Pro 代表性 pilot 與正式成本預估', '',
        '本報告只包含 MTP Off、thinking Off 的 pilot；完整 36,009 次能力評估尚未執行。', '',
        f'每模型 {sum(counts.values())} 題，九模型 {sum(counts.values())*len(models)} 次主要推論。',
        f'IFEval {counts["ifeval"]} 題、Standard {counts["mmmu_standard"]} 題、Vision {counts["mmmu_vision"]} 題。',
        'MMMU-Pro 兩種格式各以每學科兩題的隨機樣本覆蓋 30 科；另列極端題。IFEval 依指令數與字數要求分層。',
        ('IFEval 初步區間過寬，因此將已觀測 38 題逐題計入，再以剩餘 503 題的獨立隨機 40 題估算。'
         if extension else 'IFEval 依九個分層的母體題數加權估計。'),
        ('條件估算為已知總秒數＋503×追加樣本平均；bootstrap 將已知 38 題固定，重抽新樣本。'
         if extension else '僅使用機率樣本估計成本，極端題另列。'),
        'MMMU 使用各學科母體權重；兩題極端檢查不加入成本加權平均。', '',
        f'硬體：NVIDIA H100 80GB HBM3、driver 570.86.10；本次使用 {len(launch["gpus"])} 張。TP=1、每 GPU 一個模型、每模型逐題執行。',
        '沿用官方評分器、Qwen non-thinking sampling、BF16 KV、官方影像解析度與 eager kernels。', '',
        '## 實際 pilot', '',
        '分數是這批含極端題的原始正確數，不是完整 Benchmark 的能力排名。時間為主要推論，不含長上限探測。', '',
        '| 模型 | IFEval 正確數；推論分 | Standard 正確數；推論分 | Vision 正確數；推論分 | 載入與初始化合計秒 | 程序總分 |',
        '|---|---:|---:|---:|---:|---:|']
    for model in models:
        parts = [next(r for r in summary if r['model'] == model and r['task'] == task) for task in TASK_COUNTS]
        load = next(r for r in loads if r['model'] == model)
        cells = [f'{r["correct"]}/{r["questions"]}；{r["inference_seconds"]/60:.2f}' for r in parts]
        lines.append(f'| {names[model]} | '+ ' | '.join(cells)+f' | {load["initialization_seconds"]:.1f} | {load["process_wall_seconds"]/60:.2f} |')
    occupied = sum(r['process_wall_seconds'] for r in loads)/3600
    lines += ['', f'九模型程序合計 {occupied:.3f} GPU-hours；{len(launch["gpus"])} GPU 實際整體 wall time {launch["wall_seconds"]/3600:.3f} 小時。',
              '程序總時間包含載入、前處理、主要推論、官方評分、長上限探測及結束程序。', '',
              '## 完整九模型預估', '',
              '表格為一般情境的 GPU-hours；單 GPU 按同設定逐題執行時，其數值也等於 wall hours。載入等固定成本每模型只計一次。']
    for profile in scenarios:
        lines += ['', f'### {profile}（IFEval 維持 4096）', '',
                  '| 模型 | IFEval 541 題 | Standard 1730 題 | Vision 1730 題 | 合計 GPU-hours |', '|---|---:|---:|---:|---:|']
        for model in models:
            parts = [next(r['typical_gpu_hours'] for r in projections if r['profile'] == profile and r['model'] == model and r['task'] == task) for task in TASK_COUNTS]
            lines.append(f'| {names[model]} | '+' | '.join(f'{h:.2f}' for h in parts)+f' | {sum(parts):.2f} |')
        totals = [sum(r['typical_gpu_hours'] for r in projections if r['profile'] == profile and r['task'] == task) for task in TASK_COUNTS]
        lines.append('| 九模型合計 | '+' | '.join(f'{h:.2f}' for h in totals)+f' | {sum(totals):.2f} |')
    lines += ['', '## 排程與不確定性', '',
        '長工作優先，每張 GPU 同時只放一個模型；各模型都連續跑三項 tasks，不增加模型載入次數。',
        '樂觀／一般／保守為規劃情境，不是工期保證。樂觀取每模型 bootstrap 2.5% 邊界，',
        '一般取加權均值，保守取各模型 97.5% 邊界再加 25% 未見長尾、競爭及重試餘量。',
        '共同樣本的 bootstrap 保留九模型配對；模型別邊界的總和不是共同 95% 信賴區間。', '',
        '| MMMU 上限 | 情境 | 單 GPU 小時／占用 GPU-hours | 多 GPU wall 小時 | 整組 GPU 保留成本 GPU-hours |', '|---|---|---:|---:|---:|']
    for profile, data in scenarios.items():
        for scenario in ['optimistic', 'typical', 'conservative']:
            r = data[scenario]
            lines.append(f'| {profile} | {scenario} | {r["occupied_gpu_hours"]:.1f} | {r["multi_gpu_wall_hours"]:.1f} | {r["reserved_cluster_gpu_hours"]:.1f} |')
    lines += ['', '## 截斷與長上限探測', '',
        '| task | 主要回答數 | length 截斷數 | 官方 parser fallback 數 | 平均輸出 tokens |', '|---|---:|---:|---:|---:|']
    for task in TASK_COUNTS:
        s = [r for r in summary if r['task'] == task]
        n = sum(r['questions'] for r in s)
        lines.append(f'| {task} | {n} | {sum(r["truncated"] for r in s)} | {sum(r["extraction_fallback"] for r in s)} | {sum(r["output_tokens"] for r in s)/n:.1f} |')
    for cap in sorted({r['probe_max_tokens'] for r in cap_pairs}):
        group = [r for r in cap_pairs if r['probe_max_tokens'] == cap]
        censored = [r for r in group if r['probe_reason'].startswith('truncated')]
        lines += ['', f'{cap}-token 探測 {len(group)} 次，包含先前截斷 {len(censored)} 次及控制 {len(group)-len(censored)} 次。',
            f'仍截斷 {sum(r["long_truncated"] for r in group)} 次；原主要回答 prefix 相同 {sum(r["main_prefix_equal"] for r in group)}/{len(group)}。',
            f'對前一級上限的正確性改變 {sum(r["previous_cap_correct"] != r["long_correct"] for r in censored)} 次。']
    lines += ['高上限成本逐級替換仍截斷的 MMMU 回答；未截斷回答保留，固定控制核對此假設。',
        '延長探測包含特選截斷例及控制，其比例不能直接當成整體截斷率；母體加權率保存在 full_estimates.csv。',
        '4096 探測共用已開啟圖片及模板，MM processor cache 可能已暖身。'
        + ('8192 探測在新引擎逐題執行，包含其實際前處理。' if any(r['probe_max_tokens']==8192 for r in cap_pairs)
           else '本分析輸入未包含 8192 探測；不能據此估算 8192 或無截斷的成本。'), '',
        '每題延遲與輸出 tokens 的 mean／p50／p95／max、fallback、empty response、題目難度及圖片數均保留於 CSV。',
        '吞吐量為 output tokens / LLM.generate wall time，包含前處理、prefill 與 decode，不能視為純 decode kernel 速度。',
        '每學科只有兩個機率樣本；bootstrap 無法補出未見長尾、不同 seeds 或硬體差異。仍截斷的較長回答會造成成本下界。', '',
        '## 可追溯輸出', '',
        '- `smoke_results.csv`：27 組實際分數、時間與長度分布。',
        '- `question_timings.csv`：所有主要與長上限回答的逐題計時。',
        '- `model_load_and_wall.csv`：權重載入、初始化、推論、評分與程序 wall time。',
        '- `full_estimates.csv`／`full_model_estimates.csv`：各 task／模型的三情境成本。',
        '- `scenarios.json`：多 GPU 逐模型配置、開始／結束時間及整組保留成本。',
        '- `long_cap_pairs.csv`：截斷、答案抽取與正確性變化、token prefix 對照。',
        '- `audit.json`：資料、共同條件、input token IDs 及圖片 SHA 一致性。', '']
    (output/'REPORT.md').write_text('\n'.join(lines))


def main(args):
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(CONFIG.read_text())
    manifest = json.loads((args.data/'manifest.json').read_text())
    models, tasks = config['models'], list(TASK_COUNTS)
    selected = json.loads((args.data/'questions.json').read_text())
    selected_ids = {r['id'] for r in selected}
    metadata, main_rows, extensions, overhead = {}, [], [], []
    summary, timings, loads, cap_pairs = [], [], [], []
    for model in models:
        path = args.runs/model
        info = json.loads((path/'run.json').read_text())
        assert info['status'] == 'ok' and info['config']['mtp'] is False
        assert info['dataset_sha256'] == manifest['questions_sha256']
        records = [json.loads(s) for s in (path/'questions.jsonl').read_text().splitlines()]
        extra = [json.loads(s) for s in (path/'extensions.jsonl').read_text().splitlines()]
        assert len(records) == len(selected_ids) == len({r['id'] for r in records})
        assert {r['id'] for r in records} == selected_ids
        assert all(r['input_tokens'] == r['prompt_token_ids_count'] and r['output_tokens'] == r['token_ids_count'] for r in records+extra)
        assert all(r['input_tokens']+r['max_tokens'] <= info['config']['max_model_len'] for r in records+extra)
        assert all(r['finish_reason'] in ['stop', 'length'] for r in records+extra)
        assert info['speculative_config'] is None
        metadata[model] = info
        main_rows.append({r['id']: r for r in records})
        extensions.append(extra)
        process = json.loads((args.runs/(model+'.process.json')).read_text())
        assert process['exit_code'] == 0
        # In-engine 4096 probes share preprocessing; fresh-engine 8192 probes do not.
        extra_seconds = sum(r['inference_seconds']+r['scoring_seconds']+
                            (0. if r.get('shared_preprocessing', True) else r['preprocessing_seconds']) for r in extra)
        fixed = process['process_wall_seconds']-sum(map(pipeline_seconds, records))-extra_seconds
        assert fixed >= 0
        initializations = info.get('model_initializations', 1)
        overhead.append(fixed/initializations)
        log = Path(str(path)+'.log').read_text()
        load = dict(model=model, gpu=process['gpu'], process_wall_seconds=process['process_wall_seconds'],
            initialization_seconds=info['initialization_seconds'], setup_seconds=info['setup_seconds'],
            main_inference_seconds=sum(r['inference_seconds'] for r in records),
            main_preprocessing_seconds=sum(r['preprocessing_seconds'] for r in records),
            main_scoring_seconds=sum(r['scoring_seconds'] for r in records),
            probe_pipeline_seconds=extra_seconds,
            probe_inference_seconds=sum(r['inference_seconds'] for r in extra),
            probe_scoring_seconds=sum(r['scoring_seconds'] for r in extra),
            probe_preprocessing_seconds=sum(r['preprocessing_seconds'] for r in extra if not r.get('shared_preprocessing', True)),
            shutdown_seconds=info['shutdown_seconds'], initializations=initializations,
            actual_fixed_overhead_seconds=fixed, once_per_model_overhead_seconds=fixed/initializations)
        for key, pattern in [('weight_loading_seconds', r'Loading weights took ([\d.]+) seconds'),
                             ('model_loading_seconds', r'Model loading took .*? and ([\d.]+) seconds'),
                             ('engine_warmup_seconds', r'init engine .*? took ([\d.]+) s')]:
            matches = re.findall(pattern, log)
            load[key] = sum(map(float, matches)) if matches else ''
        loads.append(load)
        for task in tasks:
            subset = [r for r in records if r['task'] == task]
            prob = [r for r in subset if manifest['selection'][r['id']]['kind'] == 'probability']
            weighted_correct = sum(manifest['selection'][r['id']]['weight']*r['correct'] for r in prob)/TASK_COUNTS[task]
            row = dict(model=model, task=task, questions=len(subset), probability_questions=len(prob),
                stress_questions=len(subset)-len(prob), correct=sum(r['correct'] for r in subset),
                accuracy=sum(r['correct'] for r in subset)/len(subset), probability_weighted_accuracy=weighted_correct,
                probability_weighted_truncation_rate=sum(manifest['selection'][r['id']]['weight']*(r['finish_reason'] == 'length') for r in prob)/TASK_COUNTS[task],
                truncated=sum(r['finish_reason'] == 'length' for r in subset),
                probability_truncated=sum(r['finish_reason'] == 'length' for r in prob),
                extraction_fallback=sum(r.get('extraction_fallback', False) for r in subset),
                empty_response=sum(r['response_empty'] for r in subset),
                inference_seconds=sum(r['inference_seconds'] for r in subset),
                preprocessing_seconds=sum(r['preprocessing_seconds'] for r in subset),
                scoring_seconds=sum(r['scoring_seconds'] for r in subset),
                input_tokens=sum(r['input_tokens'] for r in subset), output_tokens=sum(r['output_tokens'] for r in subset),
                output_tokens_per_second=sum(r['output_tokens'] for r in subset)/sum(r['inference_seconds'] for r in subset))
            for field in ['inference_seconds', 'preprocessing_seconds', 'scoring_seconds',
                          'input_tokens', 'output_tokens', 'output_tokens_per_second']:
                row.update({field+'_'+k: v for k, v in quantiles([r[field] for r in subset]).items()})
            if task == 'ifeval':
                row['loose_accuracy'] = sum(r['loose_correct'] for r in subset)/len(subset)
                row['strict_instruction_accuracy'] = sum(sum(r['strict_instructions']) for r in subset)/sum(len(r['strict_instructions']) for r in subset)
            summary.append(row)
        for profile, subset in [('main', records), ('long_probe', extra)]:
            for r in subset:
                feature = manifest['population_features'][r['id']]
                timings.append(dict(model=model, profile=profile, id=r['id'], task=r['task'],
                    sampling_kind=manifest['selection'][r['id']]['kind'], difficulty=feature['difficulty'],
                    sampling_stratum=manifest['selection'][r['id']].get('sampling_stratum', 'stress'),
                    sampling_weight=manifest['selection'][r['id']]['weight'],
                    **{k: r[k] for k in ['input_tokens', 'output_tokens', 'image_count', 'max_tokens', 'finish_reason',
                        'inference_seconds', 'scoring_seconds', 'output_tokens_per_second', 'correct']},
                    preprocessing_seconds=r['preprocessing_seconds'] if profile == 'main' or not r.get('shared_preprocessing', True) else 0.,
                    shared_main_preprocessing_seconds=r['preprocessing_seconds'] if profile == 'long_probe' and r.get('shared_preprocessing', True) else 0.,
                    extraction_fallback=r.get('extraction_fallback', False)))
        extra_by_cap = {(r['id'], r['max_tokens']): r for r in extra}
        for r in extra:
            base = main_rows[-1][r['id']]
            previous = extra_by_cap[r['id'], 4096] if r['max_tokens'] == 8192 else base
            assert base['prompt_token_ids_sha256_json'] == r['prompt_token_ids_sha256_json'] and base['image_sha256'] == r['image_sha256']
            cap_pairs.append(dict(model=model, id=r['id'], task=r['task'], probe_reason=r['probe_reason'],
                probe_max_tokens=r['max_tokens'],
                main_tokens=base['output_tokens'], long_tokens=r['output_tokens'],
                main_truncated=base['finish_reason'] == 'length', long_truncated=r['finish_reason'] == 'length',
                main_correct=base['correct'], long_correct=r['correct'],
                previous_cap_tokens=previous['output_tokens'], previous_cap_correct=previous['correct'],
                previous_cap_fallback=previous.get('extraction_fallback'),
                main_fallback=base.get('extraction_fallback'), long_fallback=r.get('extraction_fallback'),
                main_prefix_equal=r['main_prefix_equal'], previous_cap_prefix_equal=r.get('previous_cap_prefix_equal', ''),
                main_inference_seconds=base['inference_seconds'], long_inference_seconds=r['inference_seconds']))
    # Protect comparisons against a different source, cap, model condition or tokenization.
    first = metadata[models[0]]
    for index, model in enumerate(models):
        for key in ['config', 'config_sha256', 'dataset_sha256', 'environment', 'source_sha256']:
            assert metadata[model][key] == first[key], (model, key)
        for key in selected_ids:
            for field in ['prompt_token_ids_sha256_json', 'image_sha256']:
                assert main_rows[index][key][field] == main_rows[0][key][field], (model, key, field)
    export(args.output/'smoke_results', summary)
    export(args.output/'question_timings', timings)
    export(args.output/'model_load_and_wall', loads)
    export(args.output/'long_cap_pairs', cap_pairs)

    projections, model_projections, scenarios, sensitivity = [], [], {}, []
    caps = [1024, 4096]
    if all(any(r['max_tokens'] == 8192 for r in extra) for extra in extensions):
        caps.append(8192)
    for cap in caps:
        profile = f'mmmu_{cap}'
        records = [{r['id']: r for r in capped_records(list(base.values()), extra, cap)}
                   for base, extra in zip(main_rows, extensions)]
        projected = []
        distributions = []
        for task in tasks:
            point, draws = stratified_projection(records, manifest, task, args.resamples)
            for i, model in enumerate(models):
                subset = [r for r in records[i].values() if r['task'] == task]
                prob = [r for r in subset if manifest['selection'][r['id']]['kind'] == 'probability']
                direct = sum(manifest['selection'][r['id']]['weight']*pipeline_seconds(r) for r in prob)
                assert np.isclose(point[i], direct, rtol=1e-12)
                sensitivity.append(dict(profile=profile, model=model, task=task,
                    weighted_pipeline_gpu_hours=direct/3600,
                    unweighted_probability_mean_gpu_hours=np.mean(list(map(pipeline_seconds, prob)))*TASK_COUNTS[task]/3600,
                    unweighted_with_stress_mean_gpu_hours=np.mean(list(map(pipeline_seconds, subset)))*TASK_COUNTS[task]/3600))
            # Allocate the one model setup by population size; totals charge it once.
            fixed = np.array(overhead)*TASK_COUNTS[task]/sum(TASK_COUNTS.values())
            point, draws = point+fixed, draws+fixed[:, None]
            projected.append(point)
            distributions.append(draws)
            bounds = np.percentile(draws, [2.5, 97.5], axis=-1)
            for i, model in enumerate(models):
                projections.append(dict(profile=profile, model=model, task=task, cases=TASK_COUNTS[task],
                    optimistic_gpu_hours=float(bounds[0, i]/3600), typical_gpu_hours=float(point[i]/3600),
                    conservative_gpu_hours=float(bounds[1, i]*1.25/3600),
                    sampling_ci_low_gpu_hours=float(bounds[0, i]/3600), sampling_ci_high_gpu_hours=float(bounds[1, i]/3600),
                    projected_truncation_rate=sum(manifest['selection'][r['id']]['weight']*(r['finish_reason'] == 'length')
                        for r in records[i].values() if r['task'] == task and manifest['selection'][r['id']]['kind'] == 'probability')/TASK_COUNTS[task]))
        total = np.sum(projected, axis=0)/3600
        draws = np.sum(distributions, axis=0)/3600
        bounds = np.percentile(draws, [2.5, 97.5], axis=-1)
        scenario = {}
        order = [models[int(i)] for i in np.argsort(-total)]
        for name, costs in [('optimistic', bounds[0]), ('typical', total), ('conservative', bounds[1]*1.25)]:
            wall, jobs = schedule(dict(zip(models, map(float, costs))), len(args.gpus), order)
            for job in jobs:
                job['gpu'] = args.gpus[job['gpu']]
            scenario[name] = dict(occupied_gpu_hours=float(sum(costs)), single_gpu_wall_hours=float(sum(costs)),
                multi_gpu_wall_hours=wall, gpus=len(args.gpus), reserved_cluster_gpu_hours=wall*len(args.gpus), schedule=jobs)
        scenario['joint_sampling_interval_gpu_hours'] = np.percentile(draws.sum(axis=0), [2.5, 97.5]).tolist()
        scenario['model_order'] = order
        scenarios[profile] = scenario
        for i, model in enumerate(models):
            model_projections.append(dict(profile=profile, model=model, cases=4001,
                optimistic_gpu_hours=float(bounds[0, i]), typical_gpu_hours=float(total[i]),
                conservative_gpu_hours=float(bounds[1, i]*1.25)))
    export(args.output/'full_estimates', projections)
    export(args.output/'full_model_estimates', model_projections)
    export(args.output/'weighting_sensitivity', sensitivity)
    write_json(args.output/'scenarios.json', scenarios)
    write_report(args.output, models, summary, loads, projections, scenarios,
                 json.loads((args.runs/'launch.json').read_text()), cap_pairs, manifest)
    population = manifest['population_features']
    coverage = {}
    for task in tasks:
        subset = [r for r in selected if r['task'] == task]
        coverage[task] = dict(subjects=len({r['stratum'] for r in subset}),
            difficulty=dict(Counter(population[r['id']]['difficulty'] for r in subset)),
            image_counts=dict(Counter(population[r['id']]['image_count'] for r in subset)),
            sampled_input_proxy_tokens=quantiles([population[r['id']]['input_proxy_tokens'] for r in subset]),
            population_input_proxy_tokens=quantiles([f['input_proxy_tokens'] for k, f in population.items() if k.startswith(task+'/')]))
    write_json(args.output/'audit.json', dict(passed=True, models=len(models), main_answers=len(timings)-len(cap_pairs),
        long_cap_answers=len(cap_pairs), matched_prompt_ids_and_image_sha256=True, matching_conditions=True,
        independent_direct_weighted_sum_equal=True,
        full_benchmark_run=False, coverage=coverage, bootstrap_resamples=args.resamples,
        method='Population-weighted within-stratum means; SciPy paired-across-model question bootstrap, independent strata',
        planning_range='Optimistic: per-model 2.5% bootstrap bound; typical: point; conservative: per-model 97.5% bound plus 25% contingency',
        assumptions=['MTP Off, non-thinking, serial requests, TP=1, one model per H100, same eager kernels and resolution',
            'Bootstrap uncertainty is conditional on sampled questions and observed capped generation; not a schedule guarantee',
            'Stress cases excluded from weighted estimates; census strata fixed; noncensus bootstrap ignores finite population correction',
            'Higher-cap projections recursively replace still-censored MMMU outcomes; uncensored controls audit reuse',
            '25% planning buffer is a judgment allowance for unseen tails, contention and retries, not an empirical confidence limit',
            'Per-model scenario bounds are summed; their sum is not a simultaneous 95% interval',
            'GPU-hours mean GPU allocation including loading, CPU preprocessing and scoring, not GPU-kernel busy time',
            'Fixed-cluster reservation also charges idle GPUs; monetary cost requires an external GPU-hour rate'],
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), analysis_seconds=time.perf_counter()-started))
    print(json.dumps({p: {s: {k: v for k, v in q.items() if k != 'schedule'} for s, q in d.items() if isinstance(q, dict)}
                      for p, d in scenarios.items()}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', type=Path, required=True)
    p.add_argument('--data', type=Path, default=ROOT/'assets/benchmark/pilot_v2')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpus', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    p.add_argument('--resamples', type=int, default=19999)
    main(p.parse_args())
