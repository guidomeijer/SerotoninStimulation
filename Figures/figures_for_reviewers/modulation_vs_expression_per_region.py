#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Does the strength (|modulation index|) and latency of light modulation depend on the expression
level of the mouse?

1. Per region: correlate the per-mouse mean |modulation index| (and latency) with the expression
   level of the mouse, over mice.
2. Overall: several ways to ask whether the (signed) modulation index depends on expression,
   without the positive and negative modulations cancelling each other out.

Spearman correlations are used because the number of mice per region is small.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from os.path import join, realpath, dirname, split
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests
from stim_functions import (paths, figure_style, load_subjects, combine_regions,
                            remove_artifact_neurons)

# Settings
MIN_NEURONS_PER_MOUSE = 5   # min. neurons (or modulated neurons for latency) per mouse and region
MIN_MICE = 3                # min. number of mice per region to compute a correlation
ONLY_MODULATED_FOR_MI = True  # use only significantly modulated neurons for |modulation index|
ALPHA = 0.05

# Get paths
f_path, save_path = paths()
fig_path = join(f_path, split(dirname(realpath(__file__)))[-1])

# Load in data
neurons = pd.read_csv(join(save_path, 'light_modulated_neurons.csv'))
expression_df = pd.read_csv(join(save_path, 'expression_levels.csv'))
neurons = remove_artifact_neurons(neurons)
neurons['modulated'] = neurons['modulated'].astype(bool)

# Only sert-cre mice with an expression measure
subjects = load_subjects()
neurons = neurons.merge(subjects[['subject', 'sert-cre']], on='subject')
neurons = neurons[neurons['sert-cre'] == 1]
neurons = neurons.merge(expression_df[['subject', 'rel_fluo']], on='subject')

# Regions
neurons['region'] = combine_regions(neurons['region'], abbreviate=True)
neurons = neurons[~neurons['region'].isin(['root', 'void'])]

# Quantities
neurons['abs_mi'] = neurons['mod_index'].abs()
neurons['latency'] = neurons['latency_peak'] * 1000  # ms
mi_neurons = neurons[neurons['modulated']] if ONLY_MODULATED_FOR_MI else neurons
lat_neurons = neurons[neurons['modulated'] & neurons['latency'].notna()]


def per_region_correlation(df, value):
    """Mean of `value` per mouse and region, then Spearman correlation with expression."""
    per_mouse = (df.groupby(['region', 'subject', 'rel_fluo'])[value]
                 .agg(['mean', 'size']).reset_index())
    per_mouse = per_mouse[per_mouse['size'] >= MIN_NEURONS_PER_MOUSE]
    rows = []
    for region, g in per_mouse.groupby('region'):
        if g.shape[0] < MIN_MICE:
            continue
        rho, p = pearsonr(g['rel_fluo'], g['mean'])
        rows.append(dict(region=region, n_mice=g.shape[0], rho=rho, p=p))
    res = pd.DataFrame(rows)
    res['p_corrected'] = multipletests(res['p'], method='fdr_bh')[1]
    return res.sort_values('rho').reset_index(drop=True), per_mouse


res_mi, mi_per_mouse = per_region_correlation(mi_neurons, 'abs_mi')
res_lat, lat_per_mouse = per_region_correlation(lat_neurons, 'latency')
print('\n|Modulation index| vs expression per region\n', res_mi.round(3).to_string())
print('\nLatency vs expression per region\n', res_lat.round(3).to_string())


# %% Plot per-region correlations
colors, dpi = figure_style()
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(4.5, 2.25), dpi=dpi)
for ax, res, title in [(ax1, res_mi, '|Modulation index|'), (ax2, res_lat, 'Latency')]:
    ax.barh(np.arange(res.shape[0]), res['rho'],
            color=['k' if p < ALPHA else 'lightgrey' for p in res['p_corrected']])
    for i, (p_unc, p_cor) in enumerate(zip(res['p'], res['p_corrected'])):
        if p_cor < ALPHA:
            ax.text(0, i, '**', ha='center', va='center', fontsize=7)
        elif p_unc < ALPHA:
            ax.text(0, i, '*', ha='center', va='center', fontsize=7)
    ax.set(yticks=np.arange(res.shape[0]),
           yticklabels=[f'{r} ({n})' for r, n in zip(res['region'], res['n_mice'])],
           xlabel='Spearman rho with expression', title=title, xlim=[-1, 1])
    ax.axvline(0, color='k', lw=0.5)
sns.despine(trim=True)
plt.tight_layout()
# plt.savefig(join(fig_path, 'modulation_vs_expression_per_region.pdf'))



# %% Scatter plots per region (all regions; one dot per mouse)
SCATTER_MIN_NEURONS = 1  # min. neurons per mouse and region to show a dot
STATS_MIN_MICE = 3       # min. mice to compute and draw a correlation
all_regions = sorted(neurons['region'].unique())
n_cols = 5
n_rows = int(np.ceil(len(all_regions) / n_cols))


def scatter_per_region(groups, ylabel, ylim_zero=False):
    """groups: list of (name, neuron dataframe, value column, color)."""
    pms = []
    for name, df, value, color in groups:
        pm = df.groupby(['region', 'subject', 'rel_fluo'])[value].agg(['mean', 'size']).reset_index()
        pms.append(pm[pm['size'] >= SCATTER_MIN_NEURONS])
    f, axs = plt.subplots(n_rows, n_cols, figsize=(7, 1.6 * n_rows), dpi=dpi,
                          sharex=True, squeeze=False)
    axs = axs.flatten()
    for ax, region in zip(axs, all_regions):
        title = region
        for (name, _, _, color), pm in zip(groups, pms):
            g = pm[pm['region'] == region]
            if g.shape[0] == 0:
                continue
            ax.scatter(g['rel_fluo'], g['mean'], s=5, color=color)
            if g.shape[0] >= STATS_MIN_MICE and g['rel_fluo'].nunique() > 1:
                r, p = pearsonr(g['rel_fluo'], g['mean'])
                sns.regplot(data=g, x='rel_fluo', y='mean', ax=ax, scatter=False, ci=None,
                            line_kws={'color': color, 'lw': 1})
                title += f'\n{name[:3]}: r={r:.2f}, p={p:.2f}, n={g.shape[0]}'
                print(f'{region} {name}: r = {r:.2f}, p = {p:.3f}, n = {g.shape[0]} mice')
            else:
                title += f'\n{name[:3]}: n={g.shape[0]}'
        if ylim_zero:
            ax.axhline(0, color='k', lw=0.5)
        ax.set_title(title, fontsize=6)
        ax.set(xlabel='', xticks=[0, 100, 200, 300], ylabel='')
    for ax in axs[len(all_regions):]:
        ax.axis('off')
    f.supxlabel('Relative expression (%)', fontsize=7, y=0.025)
    f.supylabel(ylabel, fontsize=7)
    sns.despine(trim=False)
    plt.tight_layout()
    return f


# MI of enhanced and suppressed neurons
enh_neurons = neurons[neurons['modulated'] & (neurons['mod_index'] > 0)]
sup_neurons = neurons[neurons['modulated'] & (neurons['mod_index'] < 0)]
f = scatter_per_region([('Enhanced', enh_neurons, 'mod_index', 'tab:red'),
                        ('Suppressed', sup_neurons, 'mod_index', 'tab:blue')],
                       'Modulation index', ylim_zero=True)
plt.savefig(join(fig_path, 'MI_vs_expression.pdf'))
plt.savefig(join(fig_path, 'MI_vs_expression.jpg'), dpi=600)

# Latency
f = scatter_per_region([('Latency', lat_neurons, 'latency', 'k')], 'Latency (ms)')
plt.savefig(join(fig_path, 'latency_vs_expression.pdf'))
plt.savefig(join(fig_path, 'latency_vs_expression.jpg'), dpi=600)


# %% Overall effect of expression on the (signed) modulation index
# The signed MI can't be used directly: enhanced (+) and suppressed (-) neurons cancel out.
# Instead, ask separate questions per mouse:
#   A. mean |MI| of all neurons, corrected for region (neuron's |MI| minus the region mean over
#      all mice, so mice that happened to record from strongly modulated regions don't confound)
#   B. fraction of neurons that is enhanced / suppressed
#   C. mean MI of significantly enhanced and of significantly suppressed neurons separately
neurons['abs_mi_resid'] = neurons['abs_mi'] - neurons.groupby('region')['abs_mi'].transform('mean')
neurons['enhanced'] = neurons['modulated'] & (neurons['mod_index'] > 0)
neurons['suppressed'] = neurons['modulated'] & (neurons['mod_index'] < 0)

per_mouse = neurons.groupby(['subject', 'rel_fluo']).agg(
    abs_mi_resid=('abs_mi_resid', 'mean'),
    perc_enh=('enhanced', lambda x: x.mean() * 100),
    perc_sup=('suppressed', lambda x: x.mean() * 100),
    n_neurons=('abs_mi', 'size')).reset_index()
per_mouse['mi_enh'] = per_mouse['subject'].map(
    neurons[neurons['enhanced']].groupby('subject')['mod_index'].mean())
per_mouse['mi_sup'] = per_mouse['subject'].map(
    neurons[neurons['suppressed']].groupby('subject')['mod_index'].mean())

panels = [('abs_mi_resid', 'Region-corrected\n|modulation index|'),
          ('perc_enh', 'Enhanced neurons (%)'),
          ('perc_sup', 'Suppressed neurons (%)'),
          ('mi_enh', 'MI of enhanced neurons'),
          ('mi_sup', 'MI of suppressed neurons')]
f, axs = plt.subplots(1, len(panels), figsize=(7.5, 1.75), dpi=dpi)
for ax, (col, label) in zip(axs, panels):
    d = per_mouse.dropna(subset=[col])
    rho, p = pearsonr(d['rel_fluo'], d[col])
    print(f'{col}: r = {rho:.2f}, p = {p:.3f}, n = {d.shape[0]} mice')
    sns.regplot(data=d, x='rel_fluo', y=col, ax=ax, ci=None,
                scatter_kws={'s': 5, 'color': 'k'}, line_kws={'color': 'tab:red', 'lw': 1})
    ax.set(xlabel='Relative expression (%)', ylabel=label,
           title=f'rho={rho:.2f}, p={p:.3f}')
    ax.title.set_fontsize(6)
sns.despine(trim=True)
plt.tight_layout()
# plt.savefig(join(fig_path, 'modulation_vs_expression_overall.pdf'))

# Neuron-level mixed model, random intercept per mouse, region as fixed effect
for col in ['abs_mi', 'mod_index']:
    mdl = smf.mixedlm(f'{col} ~ rel_fluo + C(region)', neurons, groups=neurons['subject']).fit()
    print(f'\n{col} ~ expression: beta = {mdl.params["rel_fluo"]:.5f}, '
          f'p = {mdl.pvalues["rel_fluo"]:.3f}')
# Sign-aware test: does expression change how much MI points in the + or - direction?
mdl = smf.mixedlm('mod_index ~ rel_fluo * C(enhanced)', neurons[neurons['modulated']],
                  groups=neurons.loc[neurons['modulated'], 'subject']).fit()
print('\nmodulated neurons, MI ~ expression * enhanced:\n', mdl.summary().tables[1])

plt.show()
