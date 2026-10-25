"""Audit completion evidence for the automatic-quantization extension."""
import csv
import hashlib
import json
import math
from pathlib import Path

from safetensors import safe_open

from analyze import export
from checkpoint_audit import headers
from fidelity import write_json


def read(path):
    return json.loads(Path(path).read_text())


if __name__ == '__main__':
    root = Path('build/autoquant')
    config = read('assets/benchmark_config.json')
    names = config['models']
    assert len(names) == 9 and 'RadixArk-Qwen3.8-27B-NVFP4' not in names
    rows = read(root/'combined/results.json')
    assert [r['model'] for r in rows] == names
    token_hash = hashlib.sha256(Path('assets/expanded_v2/tokens.json').read_bytes()).hexdigest()
    assert token_hash == read('assets/fidelity_extension_analysis.json')['tokens_sha256']
    for row in rows:
        assert row['status'] == 'ok' and row['worker_exit_code'] == 0
        assert row['tokens'] == 55296 and len(row['documents']) == 288
        assert len({d['document_id'] for d in row['documents']}) == 288
        assert row['tokens_sha256'] == token_hash and not row['mtp']
        assert row['kv_cache_dtype'] == 'bfloat16' and row['vocab_size'] == 248320
        parts = row['documents']
        assert abs(math.exp(sum(d['nll'] for d in parts)/55296)-row['ppl']) < 1e-12
        assert abs(sum(d['kl_sum'] for d in parts)/55296-row['kl_bf16_to_candidate']) < 1e-12
        assert sum(d['flips'] for d in parts)/55296 == row['top1_flip_rate']
    gate = read(root/'fidelity_prisma/sanity.json')
    assert gate['passed'] and gate['identical_distributions']
    assert hashlib.sha256(Path('src/fidelity.py').read_bytes()).hexdigest() == '90fcfa567abee03b3af1c90f6f1afcab5e0bccfa5ed3d09f14ca61198a2eb0dd'
    statistics = read(root/'analysis/statistics.json')
    assert statistics['family_size'] == 18 and statistics['resamples'] == 19999
    assert statistics['family_interval'] == 1-.05/18
    pairs = list(csv.DictReader((root/'analysis/fast_pairwise.csv').open()))
    primary = [r for r in pairs if r['grouping'] == 'all' and float(r['confidence']) > .95
               and r['metric'] in statistics['primary_pair_metrics']]
    assert len(primary) == 18 and all(float(r['low']) <= float(r['estimate']) <= float(r['high']) for r in primary)
    search = read(root/'search_v2/search_audit.json')
    assert search['method'] == 'kl_div' and search['cost_model'] == 'weight'
    assert search['fixed_quantization_config'] is None and not search['module_search_spaces']
    assert search['searchable_groups'] == 256 and search['measured_quantized_candidates'] == 512
    assert search['selected_effective_bits'] <= 5.5
    sensitivity = list(csv.DictReader((root/'search_v2/sensitivities.csv').open()))
    bits = 16*sum(float(r['cost']) for r in sensitivity if r['selected'] == 'True')/search['cost_denominator_parameters']
    assert abs(bits-search['selected_effective_bits']) < 1e-12
    assert read(root/'search_v2/replay_ablation.json')['original_allocation_reproduced']
    raw_audit = read(root/'distribution_audit/audit.json')
    assert raw_audit['passed'] and len(raw_audit['checks']) == 24
    inventory = read(root/'checkpoints/inventory.json')
    capacities = {r['model']: r for r in inventory}
    intervals = {(r['model'], r['metric']): r for r in csv.DictReader(
        (root/'analysis/intervals.csv').open()) if r['grouping'] == 'all'}
    comparison = []
    for row in rows:
        capacity = capacities[row['model']]
        joined = {k: row[k] for k in ['model', 'status', 'ppl', 'ppl_delta',
                  'kl_bf16_to_candidate', 'top1_flip_rate', 'tokens', 'quant_method']}
        joined.update(weight_bytes=capacity['weight_bytes'], weight_GB=capacity['decimal_GB'],
            body_effective_bits=capacity['body_linear_effective_bits'], revision=capacity['revision'])
        for precision in ['NVFP4', 'FP8', 'BF16']:
            joined[precision+'_linears'] = capacity['allocation_counts'].get(precision, 0)
        for metric in ['ppl', 'kl_bf16_to_candidate', 'top1_flip_rate']:
            for bound in ['low', 'high']:
                joined[metric+'_95_'+bound] = float(intervals[row['model'], metric][bound])
        comparison.append(joined)
    export(root/'comparison', comparison)
    header_checks = 0
    for item in inventory:
        model = Path('/root/model')/item['model']
        tensors = headers(model)
        for shard in model.glob('*.safetensors'):
            with safe_open(shard, framework='numpy') as f:
                for key in f.keys():
                    assert list(f.get_slice(key).get_shape()) == tensors[key]['shape']
                    header_checks += 1
        assert not item['source_weight_keys_missing']
        if item['model'] in names[-2:]:
            alignment = item['source_precision_alignment']
            assert sum(v for k, v in alignment.items() if k.endswith('_total')) == 350
            assert all(alignment[k] == alignment[k.replace('_total', '_identical')]
                       for k in alignment if k.endswith('_total'))
    modelopt = Path('/root/model')/names[-1]
    write_json(root/'modelopt_fingerprints.json', [dict(file=p.name, bytes=p.stat().st_size,
        sha256=hashlib.file_digest(p.open('rb'), 'sha256').hexdigest()) for p in sorted(modelopt.glob('*.safetensors'))])
    for name in names:
        run = read(Path('build/benchmark/smoke')/name/'run.json')
        assert run['status'] == 'ok' and run['smoke'] and run['questions'] == 6
        assert run['limit_per_task'] == 2 and run['max_tokens_override'] == 128
    benchmark_stats = read('build/benchmark/smoke_analysis/statistics.json')
    assert benchmark_stats['smoke'] and benchmark_stats['family_size'] == 96
    spec = []
    for directory in ['mtp_smoke', 'dspark_initial', 'dspark_remaining', 'modelopt_speculative']:
        spec.extend(read(Path('build/benchmark')/directory/'results.json'))
    assert len(spec) == 18 and len({(r['model'], r['mode']) for r in spec}) == 18
    assert all(r['status'] == 'ok' and r['worker_exit_code'] == 0 for r in spec)
    matrix = []
    for name in names:
        mtp = next(r for r in spec if r['model'] == name and r['mode'] == 'mtp')
        dspark = next(r for r in spec if r['model'] == name and r['mode'] == 'dspark')
        matrix.append(dict(model=name, mtp_test='pass: 3 prompts x 32 tokens',
            dspark_test='pass: 3 prompts x 32 tokens',
            matching_mtp_dspark_prompts=sum(a['token_ids'] == b['token_ids'] for a,b in zip(mtp['outputs'], dspark['outputs'])),
            mtp_public_evidence='Prisma model card includes MTP command' if name == names[-2] else 'vLLM supports Qwen3.5 family; checkpoint-specific claim not established',
            dspark_public_evidence='draft card explicitly tests FP8 target under SGLang' if name == names[1] else 'exact target not claimed in the draft card'))
    export(Path('build/benchmark/compatibility'), matrix)
    assert '13 passed' in (root/'logs/final_targeted_tests.log').read_text()
    write_json(root/'completion_audit.json', dict(passed=True, formal_models=9,
        full_fidelity_tokens_per_model=55296, fresh_bf16_identical_distributions=288,
        primary_comparisons=18, search=search, independent_raw_documents=24,
        safetensors_header_checks=header_checks, benchmark_smoke_questions=54,
        full_capability_benchmark_run=False, speculative_startup_tests=18,
        cpu_tests=13, source_precision_tensors_identical_per_new_model=350,
        fidelity_implementation_unchanged=True,
        search_gpu_hours=read(root/'search_v2/completed.json')['search_gpu_hours']))
    print((root/'completion_audit.json').read_text())
