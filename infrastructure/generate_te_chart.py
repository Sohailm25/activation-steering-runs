#!/usr/bin/env python3
"""Generate Transfer Efficiency bar chart with CI error bars.

Uses the sohail_research Everforest-inspired palette for white-bg PDF.
Output: figures in both paper (arxiv/) and website (distill) directories.
"""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path

# Everforest-derived palette (white background variant from theme.py)
COLORS = {
    'green':  '#4A7A5B',
    'teal':   '#5A9BA3',
    'coral':  '#C44D5E',
    'text':   '#2d353b',
    'muted':  '#888888',
    'border': '#4a555b',
}

# Data: approved bounded values
conditions = [
    '14B → 32B\n(same-family)',
    '32B → 14B\n(same-family)',
    'Q7B → G9B\n(cross-family)',
    'G9B → Q7B\n(cross-family)',
]
te_values = [1.25, 1.00, 0.17, 0.03]
ci_lower = [1.071, 0.900, 0.036, 0.000]
ci_upper = [1.579, 1.111, 0.321, 0.100]

# Compute error bars (asymmetric)
err_low = [te - lo for te, lo in zip(te_values, ci_lower)]
err_high = [hi - te for te, hi in zip(te_values, ci_upper)]

# Colors: green for same-family, coral for cross-family
bar_colors = [COLORS['green'], COLORS['green'], COLORS['coral'], COLORS['coral']]

# Style
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.linewidth': 0.8,
    'axes.edgecolor': COLORS['border'],
    'axes.labelcolor': COLORS['text'],
    'xtick.color': COLORS['text'],
    'ytick.color': COLORS['text'],
    'text.color': COLORS['text'],
})

fig, ax = plt.subplots(figsize=(7, 4.5))

x = np.arange(len(conditions))
bars = ax.bar(x, te_values, width=0.6, color=bar_colors, edgecolor='white', linewidth=0.5, zorder=3)

# Error bars
ax.errorbar(x, te_values, yerr=[err_low, err_high],
            fmt='none', ecolor=COLORS['text'], elinewidth=1.2, capsize=5, capthick=1.2, zorder=4)

# Reference line at TE=1.0
ax.axhline(y=1.0, color=COLORS['muted'], linestyle='--', linewidth=0.8, alpha=0.7, zorder=2)
ax.text(3.35, 1.02, 'TE = 1.0', color=COLORS['muted'], fontsize=9, va='bottom', ha='right')

# Value labels on bars
for i, (v, lo, hi) in enumerate(zip(te_values, ci_lower, ci_upper)):
    label_y = hi + 0.03
    ax.text(i, label_y, f'{v:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold', color=COLORS['text'])

ax.set_xticks(x)
ax.set_xticklabels(conditions, fontsize=10)
ax.set_ylabel('Transfer Efficiency (TE)', fontsize=11)
ax.set_ylim(0, 1.85)
ax.set_xlim(-0.5, 3.5)

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Grid
ax.yaxis.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
ax.set_axisbelow(True)

# Caption-style title
ax.set_title('Transfer Efficiency of DIM Refusal Directions', fontsize=12, fontweight='bold', pad=12, color=COLORS['text'])

plt.tight_layout()

# Save to multiple locations
repo_root = Path(__file__).parent.parent
out_paths = [
    repo_root / 'arxiv' / 'figures' / 'fig_transfer_efficiency.png',
    repo_root / 'arxiv' / 'figures' / 'fig_transfer_efficiency.pdf',
]

for p in out_paths:
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f'Saved: {p}')

# Also save for website (dark bg version)
# For distill page, save PNG that works on dark background
fig_dark, ax_dark = plt.subplots(figsize=(7, 4.5))
fig_dark.patch.set_facecolor('#2d353b')
ax_dark.set_facecolor('#2d353b')

# Everforest dark palette
EF = {
    'green': '#A7C080', 'aqua': '#83C092', 'red': '#e67e80',
    'text': '#d3c6aa', 'muted': '#859289', 'border': '#4a555b',
    'bg': '#2d353b',
}

bar_colors_dark = [EF['green'], EF['green'], EF['red'], EF['red']]
bars_dark = ax_dark.bar(x, te_values, width=0.6, color=bar_colors_dark, edgecolor=EF['border'], linewidth=0.5, zorder=3)
ax_dark.errorbar(x, te_values, yerr=[err_low, err_high],
                 fmt='none', ecolor=EF['text'], elinewidth=1.2, capsize=5, capthick=1.2, zorder=4)
ax_dark.axhline(y=1.0, color=EF['muted'], linestyle='--', linewidth=0.8, alpha=0.7, zorder=2)
ax_dark.text(3.35, 1.02, 'TE = 1.0', color=EF['muted'], fontsize=9, va='bottom', ha='right')

for i, (v, lo, hi) in enumerate(zip(te_values, ci_lower, ci_upper)):
    label_y = hi + 0.03
    ax_dark.text(i, label_y, f'{v:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold', color=EF['text'])

ax_dark.set_xticks(x)
ax_dark.set_xticklabels(conditions, fontsize=10, color=EF['text'])
ax_dark.set_ylabel('Transfer Efficiency (TE)', fontsize=11, color=EF['text'])
ax_dark.set_ylim(0, 1.85)
ax_dark.set_xlim(-0.5, 3.5)
ax_dark.spines['top'].set_visible(False)
ax_dark.spines['right'].set_visible(False)
for spine in ['bottom', 'left']:
    ax_dark.spines[spine].set_color(EF['border'])
ax_dark.tick_params(colors=EF['text'])
ax_dark.yaxis.grid(True, alpha=0.2, linestyle='-', linewidth=0.5, color=EF['muted'])
ax_dark.set_axisbelow(True)
ax_dark.set_title('Transfer Efficiency of DIM Refusal Directions', fontsize=12, fontweight='bold', pad=12, color=EF['green'])

plt.tight_layout()

website_path = Path.home() / 'Sohailm25.github.io' / 'content' / 'extra' / 'research' / 'activation-steering' / 'figures' / 'fig_transfer_efficiency.png'
fig_dark.savefig(website_path, dpi=300, bbox_inches='tight', facecolor=EF['bg'], edgecolor='none')
print(f'Saved (dark): {website_path}')

plt.close('all')
print('Done.')
