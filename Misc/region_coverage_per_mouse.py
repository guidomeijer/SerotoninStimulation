# -*- coding: utf-8 -*-
"""
Check whether all brain regions are recorded in all mice, or whether some regions are
preferentially recorded in some mice but not others.

Author: Guido Meijer
"""

import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from os.path import join
from stim_functions import paths, figure_style, combine_regions, load_subjects
colors, dpi = figure_style()

# Settings
MIN_NEURONS = 5       # minimum neurons for a region to count as 'recorded' in a mouse
N_PERM = 10000        # permutations for the region x mouse association test
SERT_ONLY = True      # only use sert-cre mice (as in the main figures)
rng = np.random.default_rng(42)

# Get paths
f_path, save_path = paths()

# Load data
light_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons.csv'))
subjects = load_subjects()
light_neurons['sert-cre'] = light_neurons['subject'].map(
    subjects.set_index('subject')['sert-cre'])
light_neurons['full_region'] = combine_regions(light_neurons['region'], abbreviate=True)
light_neurons = light_neurons[~light_neurons['full_region'].str.contains('root')]
if SERT_ONLY:
    light_neurons = light_neurons[light_neurons['sert-cre'] == 1]

# Neuron counts: regions x mice
counts = pd.crosstab(light_neurons['full_region'], light_neurons['subject'])
recorded = counts >= MIN_NEURONS
n_mice = counts.shape[1]
print(f'{n_mice} mice, {counts.shape[0]} regions')

# Per-region coverage
coverage = pd.DataFrame({'n_mice': recorded.sum(axis=1),
                         'frac_mice': recorded.mean(axis=1),
                         'n_neurons': counts.sum(axis=1)})
coverage = coverage.sort_values('n_mice', ascending=False)
print('\nCoverage per region (mice with >= %d neurons):' % MIN_NEURONS)
print(coverage.to_string())
print(f'\nRegions recorded in all mice: {list(coverage.index[coverage["n_mice"] == n_mice])}')

# Per-mouse coverage
mouse_cov = pd.DataFrame({'n_regions': recorded.sum(axis=0), 'n_neurons': counts.sum(axis=0)})
print('\nCoverage per mouse:')
print(mouse_cov.sort_values('n_regions', ascending=False).to_string())

# Test 1: is neuron count distribution over regions the same across mice? (chi-square, with
# Monte Carlo p-value since many cells have small expected counts)
chi2, p_chi, _, _ = stats.chi2_contingency(counts.values)
cramers_v = np.sqrt(chi2 / (counts.values.sum() * (min(counts.shape) - 1)))
region_labels = np.repeat(np.arange(counts.shape[0]), counts.sum(axis=1).values)
mouse_labels = np.concatenate([np.repeat(np.arange(n_mice), counts.values[i])
                               for i in range(counts.shape[0])])
perm_chi = np.empty(N_PERM)
for k in range(N_PERM):
    perm = rng.permutation(mouse_labels)
    perm_counts = np.zeros(counts.shape)
    np.add.at(perm_counts, (region_labels, perm), 1)
    perm_chi[k] = stats.chi2_contingency(perm_counts, correction=False)[0]
p_perm = (np.sum(perm_chi >= chi2) + 1) / (N_PERM + 1)
print(f'\nRegion x mouse association: chi2 = {chi2:.1f}, Cramer\'s V = {cramers_v:.3f}, '
      f'p (asymptotic) = {p_chi:.3g}, p (permutation) = {p_perm:.4g}')

# Test 2: nestedness / non-random presence-absence, compare number of regions per mouse
# to a null where presence is shuffled within region (keeps region prevalence)
obs_var = recorded.sum(axis=0).var()
null_var = np.empty(N_PERM)
rec_vals = recorded.values
for k in range(N_PERM):
    shuffled = np.array([rng.permutation(row) for row in rec_vals])
    null_var[k] = shuffled.sum(axis=0).var()
p_var = (np.sum(null_var >= obs_var) + 1) / (N_PERM + 1)
print(f'Variance in #regions per mouse: observed = {obs_var:.2f}, '
      f'null mean = {null_var.mean():.2f}, p = {p_var:.4g}')

# Test 3: per-region, is it recorded in fewer/more mice than expected? Fisher-style via
# binomial test against overall mean recording probability
p_overall = recorded.values.mean()
binom_p = {r: stats.binomtest(int(recorded.loc[r].sum()), n_mice, p_overall).pvalue
           for r in recorded.index}
coverage['p_binom'] = pd.Series(binom_p)
print(f'\nBinomial test per region vs overall probability ({p_overall:.2f}):')
print(coverage[['n_mice', 'p_binom']].to_string())

# Plot: heatmap of neuron counts and presence
region_order = coverage.index
mouse_order = mouse_cov.sort_values('n_regions', ascending=False).index

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 0.22 * len(region_order) + 1.2), dpi=dpi,
                             sharey=True)
sns.heatmap(np.log10(counts.loc[region_order, mouse_order] + 1), cmap='viridis', ax=ax1,
            cbar_kws={'label': 'log10 (neurons + 1)', 'shrink': 0.5})
ax1.set(xlabel='Mouse', ylabel='', title='Neuron count')
sns.heatmap(recorded.loc[region_order, mouse_order].astype(int), cmap=['white', 'k'], ax=ax2,
            cbar=False, linewidths=0.3, linecolor='lightgrey')
ax2.set(xlabel='Mouse', ylabel='', title=f'Recorded (>= {MIN_NEURONS} neurons)')
plt.tight_layout()
plt.savefig(join(f_path, 'supp_figure_region_coverage', 'region_coverage_per_mouse.pdf'))
plt.show()


# %% Are some regions preferentially recorded in mice with high expression?
expression_df = pd.read_csv(join(save_path, 'expression_levels.csv'))
expr = expression_df.drop_duplicates('subject').set_index('subject')['rel_fluo']
expr = expr.reindex(counts.columns)
if expr.isna().any():
    print(f'\nNo expression data for: {list(expr.index[expr.isna()])}, excluding these mice')
keep = expr.index[expr.notna()]
counts_e, rec_e, expr = counts[keep], recorded[keep], expr[keep]
expr_vals = expr.values
frac_e = counts_e / counts_e.sum(axis=0)   # fraction of each mouse's neurons in each region

# Neuron-weighted mean expression per region vs null (expression shuffled over mice)
def weighted_expr(cnt, ex):
    return (cnt @ ex) / cnt.sum(axis=1)

obs_w = weighted_expr(counts_e.values, expr_vals)
null_w = np.empty((N_PERM, counts_e.shape[0]))
for k in range(N_PERM):
    null_w[k] = weighted_expr(counts_e.values, rng.permutation(expr_vals))
p_w = (np.sum(np.abs(null_w - null_w.mean(0)) >= np.abs(obs_w - null_w.mean(0)), axis=0) + 1) / (N_PERM + 1)

def holm(p):
    p = np.asarray(p)
    order = np.argsort(p)
    m = len(p)
    out = np.empty(m)
    run = 0
    for rank, idx in enumerate(order):
        run = max(run, (m - rank) * p[idx])
        out[idx] = min(run, 1)
    return out

expr_res = pd.DataFrame({'n_neurons': counts_e.sum(axis=1), 'n_mice': rec_e.sum(axis=1),
                         'weighted_expr': obs_w, 'null_expr': null_w.mean(0),
                         'p_perm': p_w}, index=counts_e.index)

# Spearman: expression vs fraction of the mouse's neurons in this region, and recorded vs not
rho, p_rho, p_mw = [], [], []
for r in counts_e.index:
    if frac_e.loc[r].nunique() > 1:
        res = stats.spearmanr(expr_vals, frac_e.loc[r].values)
        rho.append(res[0]); p_rho.append(res[1])
    else:
        rho.append(np.nan); p_rho.append(np.nan)
    a, b = expr_vals[rec_e.loc[r].values], expr_vals[~rec_e.loc[r].values]
    p_mw.append(stats.mannwhitneyu(a, b)[1] if len(a) > 1 and len(b) > 1 else np.nan)
expr_res['rho_frac_vs_expr'] = rho
expr_res['p_spearman'] = p_rho
expr_res['p_rec_vs_not'] = p_mw
expr_res['p_perm_holm'] = holm(expr_res['p_perm'])
valid = expr_res['p_spearman'].notna()
expr_res.loc[valid, 'p_spearman_holm'] = holm(expr_res.loc[valid, 'p_spearman'])
expr_res = expr_res.sort_values('weighted_expr', ascending=False)
print('\nRegion preference for expression level (rel_fluo):')
print(expr_res.round(4).to_string())
print('\nSignificant after Holm (permutation test):',
      list(expr_res.index[expr_res['p_perm_holm'] < 0.05]))

# Global test: do mice with higher expression sample different regions? (per-mouse region
# composition vs expression: PERMANOVA-like, Bray-Curtis distance vs expression distance)
from scipy.spatial.distance import pdist, squareform
d_comp = squareform(pdist(frac_e.T.values, 'braycurtis'))
iu = np.triu_indices(len(expr_vals), 1)
obs_r = stats.spearmanr(d_comp[iu], pdist(expr_vals[:, None]))[0]
null_r = np.empty(N_PERM)
for k in range(N_PERM):
    perm = rng.permutation(len(expr_vals))
    null_r[k] = stats.spearmanr(d_comp[perm][:, perm][iu], pdist(expr_vals[:, None]))[0]
p_mantel = (np.sum(null_r >= obs_r) + 1) / (N_PERM + 1)
print(f'\nMantel test (composition dissimilarity vs expression difference): '
      f'r = {obs_r:.3f}, p = {p_mantel:.4g}')

# %% Plot
plot_df = expr_res.reset_index().rename(columns={'full_region': 'region'})
f, ax1 = plt.subplots(figsize=(2, 3), dpi=dpi)
null_df = pd.DataFrame(null_w, columns=counts_e.index)
null_sorted = [null_df[r].values for r in plot_df['region']]
ypos = np.arange(len(plot_df))
vp = ax1.violinplot(null_sorted, positions=ypos, vert=False, widths=0.9, showextrema=False,
                    side='high' if 'side' in plt.violinplot.__code__.co_varnames else 'both')
for body in vp['bodies']:
    body.set_facecolor('grey')
    body.set_edgecolor('none')
    body.set_alpha(0.5)
    body.set_zorder(2)
ax1.plot(plot_df['weighted_expr'], ypos, '|', color='k', ms=8, mew=1.5, zorder=3)

for i, p in enumerate(plot_df['p_perm']):
    if p < 0.05:
        ax1.text(300, i, '*', va='center', ha='right', fontsize=12)
ax1.set(yticks=ypos, yticklabels=plot_df['region'], xlabel='Neuron-weighted expression (%)',
        ylabel='', xticks=[0, 100, 200, 300])
ax1.invert_yaxis()

sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(f_path, 'supp_figure_region_coverage', 'region_coverage_vs_expression.pdf'))
plt.show()

