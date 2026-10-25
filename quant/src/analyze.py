#!/usr/bin/env python3
"""Document-cluster bootstrap of paired full-vocabulary fidelity measurements."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.stats import bootstrap
from fidelity import aggregate, write_json, report

FAST = 'eNeural-Qwen3.8-27B-fast'
PEERS = ['RadixArk-Qwen3.8-27B-NVFP4-BF16-LMHead', 'nvidia-Qwen3.8-27B-NVFP4',
         'RedHatAI-Qwen3.8-27B-NVFP4', 'unsloth-Qwen3.8-27B-NVFP4']
FIELDS = ['nll', 'ref_nll', 'kl_sum', 'flips', 'tokens']
METRICS = ['ppl', 'ppl_delta', 'delta_nll', 'ppl_ratio', 'kl_bf16_to_candidate', 'top1_flip_rate']
PAIR_METRICS = ['delta_nll', 'ppl_ratio', 'ppl_difference', 'kl_difference', 'flip_difference']


def statistic(*blocks, axis=-1, pair=False):
    total = sum(block.sum(axis=axis) for block in blocks)
    a, b, kl, flip, count = total
    nll_a, nll_b = a / count, b / count
    ppl_a, ppl_b = np.exp(nll_a), np.exp(nll_b)
    if pair:
        return np.stack([nll_a - nll_b, np.exp(nll_a - nll_b), ppl_a - ppl_b, kl / count, flip / count])
    return np.stack([ppl_a, ppl_a - ppl_b, nll_a - nll_b, np.exp(nll_a - nll_b), kl / count, flip / count])


def cluster_interval(parts, *, pair=False, seed=20261016, resamples=19999, confidence=.95):
    # One window per source document. Keep the planned domain/context counts fixed.
    if len({p['document_id'] for p in parts}) != len(parts):
        raise ValueError('Expected one window per document; repeated documents need joint clustering')
    cells = sorted({(p['domain'], p['context']) for p in parts})
    blocks = tuple(np.array([[p[k] for k in FIELDS] for p in parts
                             if (p['domain'], p['context']) == cell], dtype=np.float64).T
                   for cell in cells)
    def stat(*args, axis=-1):
        return statistic(*args, axis=axis, pair=pair)
    fit = bootstrap(blocks, stat, axis=-1, vectorized=True, paired=False,
                    n_resamples=resamples, batch=256, method='percentile',
                    confidence_level=confidence, rng=np.random.default_rng(seed))
    return statistic(*blocks, pair=pair), fit.confidence_interval


def groups(parts):
    yield 'all', 'all', parts
    for key in ['domain', 'context']:
        for value in sorted({p[key] for p in parts}, key=str):
            yield key, str(value), [p for p in parts if p[key] == value]
    for domain, context in sorted({(p['domain'], p['context']) for p in parts}):
        yield 'domain_context', f'{domain}/{context}', [p for p in parts
              if p['domain'] == domain and p['context'] == context]


def export(path, rows):
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with path.with_suffix('.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    lines = ['| ' + ' | '.join(columns) + ' |', '|' + '|'.join(['---'] * len(columns)) + '|']
    for row in rows:
        lines.append('| ' + ' | '.join(f'{row.get(k, ""):.8g}' if isinstance(row.get(k), float)
                    else str(row.get(k, '')).replace('|', '\\|') for k in columns) + ' |')
    path.with_suffix('.md').write_text('\n'.join(lines) + '\n')


def analyze(run, output, first_round, resamples, peers=PEERS):
    output.mkdir(parents=True, exist_ok=False)
    rows = json.loads((run / 'results.json').read_text())
    if not json.loads((run / 'sanity.json').read_text())['passed']:
        raise ValueError('Sanity failed')
    models = {r['model']: r for r in rows if r['run'] != 'bf16_repeat'}
    failures = [r for r in models.values() if r['status'] != 'ok' or r['worker_exit_code'] != 0]
    if failures:
        raise ValueError(f'Failed workers: {failures}')
    report(output, list(models.values()))
    grouped, intervals, pair_rows, sensitivity, stability = [], [], [], [], []
    for name, result in models.items():
        for kind, group, parts in groups(result['documents']):
            grouped.append(dict(model=name, grouping=kind, group=group, documents=len(parts), **aggregate(parts)))
            point, ci = cluster_interval(parts, resamples=resamples)
            for metric, value, lo, hi in zip(METRICS, point, ci.low, ci.high):
                intervals.append(dict(model=name, grouping=kind, group=group, metric=metric,
                                      estimate=float(value), low=float(lo), high=float(hi), confidence=.95,
                                      documents=len(parts), tokens=sum(p['tokens'] for p in parts)))
    fast = models[FAST]['documents']
    family_size = len(peers) * 3
    for peer in peers:
        other = {p['id']: p for p in models[peer]['documents']}
        paired = []
        for a in fast:
            b = other[a['id']]
            assert (a['tokens'], a['document_id'], a['ref_nll']) == (b['tokens'], b['document_id'], b['ref_nll'])
            paired.append(dict(a, ref_nll=b['nll'], kl_sum=a['kl_sum'] - b['kl_sum'], flips=a['flips'] - b['flips']))
        for metric, values in [
            ('delta_nll', [p['nll'] - p['ref_nll'] for p in paired]),
            ('kl_difference', [p['kl_sum'] for p in paired]),
            ('flip_difference', [p['flips'] for p in paired])]:
            values = np.asarray(values, dtype=np.float64) / np.array([p['tokens'] for p in paired])
            stability.append(dict(peer=peer, metric=metric, documents=len(paired),
                fast_lower_documents=int(np.sum(values < 0)), tied_documents=int(np.sum(values == 0)),
                fast_higher_documents=int(np.sum(values > 0)), median=float(np.median(values)),
                p10=float(np.quantile(values, .1)), p90=float(np.quantile(values, .9))))
        for kind, group, parts in groups(paired):
            levels = [.95, 1 - .05 / family_size] if kind == 'all' else [.95]
            for confidence in levels:
                point, ci = cluster_interval(parts, pair=True, resamples=resamples, confidence=confidence)
                for metric, value, lo, hi in zip(PAIR_METRICS, point, ci.low, ci.high):
                    threshold = 1 if metric == 'ppl_ratio' else 0
                    pair_rows.append(dict(model=FAST, peer=peer, grouping=kind, group=group, metric=metric,
                        estimate=float(value), low=float(lo), high=float(hi), confidence=confidence,
                        interval_excludes_equal=bool(lo > threshold or hi < threshold), documents=len(parts)))
        for domain in sorted({p['domain'] for p in paired}):
            parts = [p for p in paired if p['domain'] != domain]
            point, ci = cluster_interval(parts, pair=True, resamples=resamples)
            for metric, value, lo, hi in zip(PAIR_METRICS, point, ci.low, ci.high):
                sensitivity.append(dict(peer=peer, omitted_domain=domain, metric=metric,
                    estimate=float(value), low=float(lo), high=float(hi), confidence=.95))
    export(output / 'grouped', grouped)
    export(output / 'intervals', intervals)
    export(output / 'fast_pairwise', pair_rows)
    export(output / 'leave_one_domain_out', sensitivity)
    export(output / 'document_stability', stability)
    old = {r['model']: r for r in csv.DictReader(first_round.open())}
    comparison = []
    for name, result in models.items():
        if name not in old:
            continue
        for metric in ['ppl', 'ppl_delta', 'kl_bf16_to_candidate', 'top1_flip_rate']:
            comparison.append(dict(model=name, metric=metric, round1=float(old[name][metric]),
                                   expanded=result[metric], round1_tokens=432, expanded_tokens=result['tokens']))
    export(output / 'round1_comparison', comparison)
    import scipy
    write_json(output / 'statistics.json', dict(method='SciPy stratified paired document-cluster percentile bootstrap',
        resamples=resamples, seed=20261016, scipy=scipy.__version__, confidence=.95,
        strata='domain × initial context', cluster='one complete source-document window',
        pair_family=f'{len(peers)} peers × 3 primary metrics (delta NLL, KL difference, flip difference)',
        family_size=family_size, peers=peers,
        primary_pair_metrics=['delta_nll', 'kl_difference', 'flip_difference'],
        derived_pair_summaries=['ppl_ratio', 'ppl_difference; do not use for additional significance claims'],
        family_interval=1 - .05 / family_size, grouped_intervals='exploratory, no multiplicity correction',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        input_sha256=hashlib.sha256((run / 'results.json').read_bytes()).hexdigest()))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--first-round', type=Path, default=Path('build/comparison.csv'))
    p.add_argument('--resamples', type=int, default=19999)
    p.add_argument('--peers', nargs='+', default=PEERS)
    a = p.parse_args()
    analyze(a.run, a.output, a.first_round, a.resamples, a.peers)
