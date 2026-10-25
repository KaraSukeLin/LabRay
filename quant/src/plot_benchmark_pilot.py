"""Plot measured pilot lengths/latencies and the declared cost planning ranges."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ecdf


def plot(analysis):
    rows = list(csv.DictReader((analysis/'question_timings.csv').open()))
    costs = list(csv.DictReader((analysis/'full_model_estimates.csv').open()))
    models = list(dict.fromkeys(r['model'] for r in rows))
    names = ['BF16', 'FP8', 'NaviQuant Fast', 'NVIDIA NVFP4', 'RedHatAI NVFP4',
             'Unsloth NVFP4', 'Radix BF16-LMHead', 'PrismaAQUA', 'AutoQuantize']
    tasks = ['ifeval', 'mmmu_standard', 'mmmu_vision']
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout='constrained')
    for j, task in enumerate(tasks):
        for i, model in enumerate(models):
            selected = [r for r in rows if r['model'] == model and r['task'] == task]
            main = [r for r in selected if r['profile'] == 'main']
            fit = ecdf([int(r['output_tokens']) for r in main]).cdf
            color = f'C{i}'
            axes[0, j].step(fit.quantiles, fit.probabilities, where='post', color=color, label=names[i])
            for profile, marker in [('main', 'o'), ('long_probe', 'x')]:
                subset = [r for r in selected if r['profile'] == profile]
                axes[1, j].scatter([int(r['output_tokens']) for r in subset],
                    [float(r['inference_seconds']) for r in subset], color=color, marker=marker, s=10, alpha=.55)
        axes[0, j].axvline(4096 if task == 'ifeval' else 1024, color='.3', linestyle='--', linewidth=.8)
        axes[0, j].set_title(task)
        axes[0, j].set_ylabel('Main response ECDF (pilot, unweighted)')
        axes[1, j].set_ylabel('LLM.generate wall seconds')
        for ax in axes[:, j]:
            ax.set_xlabel('Generated tokens')
            ax.grid(alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, -.055), ncol=5, fontsize=8)
    fig.suptitle('MTP Off pilot: lengths and latency\nCircle: main cap; cross: longer-cap probe', fontsize=11)
    for suffix in ['png', 'pdf']:
        fig.savefig(analysis/f'pilot_lengths_latency.{suffix}', dpi=180, bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.7), layout='constrained')
    profiles = list(dict.fromkeys(r['profile'] for r in costs))
    for j, profile in enumerate(profiles):
        subset = [next(r for r in costs if r['profile'] == profile and r['model'] == model) for model in models]
        point = np.array([float(r['typical_gpu_hours']) for r in subset])
        low = np.array([float(r['optimistic_gpu_hours']) for r in subset])
        high = np.array([float(r['conservative_gpu_hours']) for r in subset])
        ax.errorbar(point, np.arange(len(models))+(j-(len(profiles)-1)/2)*.18, xerr=[point-low, high-point],
                    fmt=['o', 's', '^'][j], capsize=2, markersize=4, label=profile)
    ax.set_yticks(range(len(models)), names)
    ax.invert_yaxis()
    ax.set_xlabel('Projected GPU allocation hours per model (4001 cases)')
    ax.grid(axis='x', alpha=.2)
    ax.legend()
    ax.set_title('MTP Off, H100, TP=1, serial requests\nPlanning range: optimistic to conservative; IFEval cap=4096', fontsize=11)
    for suffix in ['png', 'pdf']:
        fig.savefig(analysis/f'full_cost_ranges.{suffix}', dpi=180)
    plt.close(fig)


def plot_decision(decision):
    rows=list(csv.DictReader((decision/'integrated_comparison.csv').open()))
    names=['BF16','FP8','NaviQuant Fast','NVIDIA NVFP4','RedHatAI','Unsloth',
           'Radix BF16-LMHead','PrismaAQUA','AutoQuantize']
    positions=np.arange(len(rows))
    fig,axes=plt.subplots(1,2,figsize=(12,5.5),layout='constrained')
    axes[0].barh(positions,[float(r['kl_bf16_to_candidate'])*1000 for r in rows],color='C0',alpha=.75)
    axes[0].set_yticks(positions,names)
    axes[0].set_xlabel('Full-vocabulary KL to BF16 (x 1000; lower is better)')
    axes[0].set_title('Verified Fidelity: 55,296 teacher-forced tokens')
    for i,r in enumerate(rows):
        axes[0].text(float(r['kl_bf16_to_candidate'])*1000+.15,i,f'{float(r["weight_GB"]):.2f} GB',va='center',fontsize=8)
    axes[0].set_xlim(0,26)
    for cap,offset,color in [(1024,-.17,'C1'),(4096,.17,'C2')]:
        axes[1].barh(positions+offset,[float(r[f'mmmu_vision_{cap}_correct'])/61*100 for r in rows],
                     height=.32,label=f'{cap}-token cap',color=color,alpha=.8)
    axes[1].set_yticks(positions,names)
    axes[1].set_xlabel('MMMU-Pro Vision correct (%) on 61 shared pilot cases')
    axes[1].set_title('Output cap changes the apparent ordering')
    axes[1].set_xlim(0,80)
    axes[1].legend(loc='lower right')
    for ax in axes:
        ax.invert_yaxis()
        ax.grid(axis='x',alpha=.2)
    fig.suptitle('Fixed checkpoints, MTP Off / thinking Off\nPilot includes stress cases and one sampling seed; this is not a full capability ranking',fontsize=11)
    for suffix in ['png','pdf']:
        fig.savefig(decision/f'fidelity_and_cap_sensitivity.{suffix}',dpi=180,bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('analysis', type=Path)
    p.add_argument('--decision',type=Path)
    args=p.parse_args()
    plot(args.analysis)
    if args.decision:
        plot_decision(args.decision)
