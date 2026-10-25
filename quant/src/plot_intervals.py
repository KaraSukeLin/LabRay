#!/usr/bin/env python3
"""Export a forest plot of Fast minus peer document-bootstrap intervals."""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def plot(analysis):
    rows = list(csv.DictReader((analysis / 'fast_pairwise.csv').open()))
    metrics = [('delta_nll', 'NLL difference (nats/token)'),
               ('kl_difference', 'KL-to-BF16 difference (nats/token)'),
               ('flip_difference', 'Flip-rate difference (percentage points)')]
    peers = list(dict.fromkeys(r['peer'] for r in rows))
    names = {'RadixArk-Qwen3.8-27B-NVFP4-BF16-LMHead': 'RadixArk BF16-LMHead',
             'nvidia-Qwen3.8-27B-NVFP4': 'NVIDIA', 'RedHatAI-Qwen3.8-27B-NVFP4': 'RedHatAI',
             'unsloth-Qwen3.8-27B-NVFP4': 'Unsloth',
             'rdtand-Qwen3.8-27B-PrismaAQUA-5.5bit-vllm': 'PrismaAQUA',
             'ModelOpt-Qwen3.8-27B-AutoQuantize-5.5bit': 'ModelOpt AutoQuantize'}
    labels = [names[p] for p in peers]
    fig, axes = plt.subplots(1, 3, figsize=(12, 2.2 + .3 * len(peers)), layout='constrained')
    for ax, (metric, title) in zip(axes, metrics):
        ax.axvline(0, color='black', linewidth=.8, linestyle='--')
        for j, peer in enumerate(peers):
            selected = [r for r in rows if r['peer'] == peer and r['metric'] == metric
                        and r['grouping'] == 'all']
            for r in selected:
                wide = float(r['confidence']) > .95
                scale = 100 if metric == 'flip_difference' else 1
                lo, hi, point = [float(r[k]) * scale for k in ['low', 'high', 'estimate']]
                ax.plot([lo, hi], [j, j], color='.65' if wide else '#0072B2',
                        linewidth=1.5 if wide else 4, solid_capstyle='butt')
                ax.plot(point, j, 'o', color='#0072B2', markersize=4)
        ax.set_yticks(range(len(peers)), labels if ax == axes[0] else [''] * len(peers))
        ax.invert_yaxis()
        ax.set_xlabel(title)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.grid(axis='x', alpha=.2)
    fig.suptitle('NaviQuant Fast minus peer: paired document bootstrap\nBlue: 95%; gray: family-adjusted intervals; negative favors Fast', fontsize=11)
    fig.savefig(analysis / 'fast_intervals.png', dpi=180)
    fig.savefig(analysis / 'fast_intervals.pdf')
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('analysis', type=Path)
    plot(p.parse_args().analysis)
