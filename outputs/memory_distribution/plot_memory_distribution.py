"""Plot the supplied PCA and t-SNE coordinates without recomputing embeddings."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

OUT = Path(__file__).resolve().parent
SOURCE = Path('/Users/moka/Downloads')
STAGES = [('first', 'first'), ('33_percent', '33%'), ('67_percent', '67%'), ('final', 'final')]
METHODS = [
    ('baserec', 'BaseRec', '#0072B2', 'o'),
    ('agentcf', 'AgentCF', '#E69F00', 's'),
    ('tiermem', 'TierMem', '#009E73', '^'),
    ('i2agent', 'I2Agent', '#CC79A7', 'D'),
    ('memrec', 'MemRec', '#D55E00', 'v'),
    ('gpt6_one_shot', 'GPT6 · One shot', '#6A51A3', 'P'),
    ('gpt6_sequential_all', 'GPT6 · Sequential all', '#17A7B5', 'X'),
    ('gpt6_sequential_user_provided', 'GPT6 · Sequential user provided', '#59636F', '*'),
]
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 11,
    'axes.labelcolor': '#475569', 'text.color': '#172B4D',
    'xtick.color': '#64748B', 'ytick.color': '#64748B',
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#CBD5E1', 'axes.linewidth': .8,
    'pdf.fonttype': 42, 'svg.fonttype': 'none',
})

for projection, label in [('pca', 'PCA'), ('tsne', 't-SNE')]:
    data = json.loads((SOURCE / f'memory_distribution_{projection}_coordinates.json').read_text())
    coords = np.array([[r[f'{projection}_1'], r[f'{projection}_2']] for r in data])
    assert np.isfinite(coords).all()
    limits = []
    for axis in range(2):
        lo, hi = coords[:, axis].min(), coords[:, axis].max()
        pad = max((hi - lo) * .08, .01)
        limits.append((lo - pad, hi + pad))
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    fig.patch.set_facecolor('white')
    fig.subplots_adjust(left=.09, right=.975, top=.865, bottom=.205, wspace=.16, hspace=.26)
    fig.suptitle(f'Memory distribution · {label}', x=.09, y=.969, ha='left', fontsize=23, fontweight='bold')
    fig.text(.09, .926, 'Each point represents one user within a method. Shared axes across stages.', fontsize=11, color='#64748B')
    for ax, (stage, title), panel in zip(axes.flat, STAGES, 'abcd'):
        ax.set_facecolor('#F8FAFC')
        ax.set_axisbelow(True)
        ax.grid(True, color='#E2E8F0', linewidth=.7)
        subset = [r for r in data if r['stage'] == stage]
        for method, name, color, marker in METHODS:
            rows = [r for r in subset if r['method'] == method]
            if rows:
                ax.scatter([r[f'{projection}_1'] for r in rows], [r[f'{projection}_2'] for r in rows],
                           s=80 if marker != '*' else 125, c=color, marker=marker,
                           alpha=.86, edgecolors='white', linewidths=.65, zorder=3)
        ax.set_title(f'({panel})  {title}', loc='left', fontsize=14, fontweight='bold', pad=12)
        ax.text(1, 1.045, f'n = {len(subset)}', transform=ax.transAxes, ha='right', color='#64748B', fontsize=10)
        ax.set_xlim(limits[0]); ax.set_ylim(limits[1])
        ax.tick_params(length=0, pad=7)
    for ax in axes[-1]:
        ax.set_xlabel('PC 1' if projection == 'pca' else 't-SNE 1', labelpad=9)
    for ax in axes[:, 0]:
        ax.set_ylabel('PC 2' if projection == 'pca' else 't-SNE 2', labelpad=9)
    handles = [Line2D([], [], linestyle='', marker=marker, color=color, markeredgecolor='white',
                      markersize=9, label=name) for _, name, color, marker in METHODS]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.53, .065), ncol=3,
               frameon=False, fontsize=10, columnspacing=2, handletextpad=.55, labelspacing=1.05)
    fig.text(.09, .026, 'Only BaseRec, AgentCF and TierMem have data at every stage; all eight methods appear at final.',
             fontsize=9, color='#64748B')
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(OUT / f'memory_distribution_{projection}.{ext}', dpi=300, facecolor='white')
    plt.close(fig)
    print(f'{label}: plotted {len(data)} points; saved PNG, PDF and SVG to {OUT}')
