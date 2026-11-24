#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 15:19:58 2021
By: Guido Meijer
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from os.path import join, realpath, dirname, split
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import statsmodels.genmod.families.links as sm_links
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from stim_functions import paths, figure_style, load_subjects, remap
from iblatlas.atlas import AllenAtlas
ba = AllenAtlas(res_um=25)
colors, dpi = figure_style()

# Settings
TARGET_VARIABLE = 'latency'
#TARGET_VARIABLE = 'abs_mod_index'
#TARGET_VARIABLE = 'perc_mod'
#TARGET_VARIABLE = 'mod_index'
MIN_MOD_NEURONS = {'perc_mod': 0, 'mod_index': 15, 'abs_mod_index': 15, 'latency': 15}
MIN_NEURONS = {'perc_mod': 5, 'mod_index': 0, 'abs_mod_index': 0, 'latency': 0}
INCL_RECEPTORS = ['5-HT1a', '5-HT1b', '5-HT2a', '5-HT2c', '5-HT3a', '5-HT5a']
#INCL_RECEPTORS = ['5-HT1a', '5-HT1b', '5-HT1d', '5-HT1f', '5-HT2a', '5-HT2b',
#                  '5-HT2c', '5-HT3a', '5-HT3b', '5-HT5a', '5-HT7']
DROP_PROJECTION = False

# Paths
f_path, data_path = paths()
fig_path = join(f_path, split(dirname(realpath(__file__)))[-1])

# Load in expression
expr_df = pd.read_csv(join(data_path, 'receptor_expression.csv'))
expr_df = expr_df[~np.isin(expr_df['acronym'], ['MMme', 'CUL4, 5', 'CUL4, 5gr', 'CUL4, 5mo', 'CP'])]
expr_df['region'] = remap(expr_df['acronym'])
expr_df = expr_df[np.isin(expr_df['receptor'], INCL_RECEPTORS)]
#expr_df['region'] = combine_regions(remap(expr_df['acronym']))
expression_mean = expr_df[['region', 'receptor', 'expression_energy']].groupby(
    ['region', 'receptor']).median().reset_index()

# load structure and expression data set
proj_df = pd.read_csv(join(data_path, 'dr_projection_strength.csv'))
proj_df = proj_df[~np.isin(proj_df['allen_acronym'], ['MMd', 'MMme', 'MMl', 'MMm', 'MMp', 'CUL4, 5', 'CP'])]
proj_df['region'] = remap(proj_df['allen_acronym'])
#proj_df['region'] = combine_regions(remap(proj_df['allen_acronym']))
proj_summary = proj_df[['region', 'projection_density']].groupby(['region']).mean().reset_index()
proj_summary = proj_summary.rename(columns={'projection_density': 'projection'})

# Load in neural data
ephys_data = pd.read_csv(join(data_path, 'light_modulated_neurons.csv'))
ephys_data['abs_mod_index'] = np.abs(ephys_data['mod_index'])
#ephys_data['region'] = combine_regions(ephys_data['region'])
subjects = load_subjects()
for i, nickname in enumerate(np.unique(subjects['subject'])):
    ephys_data.loc[ephys_data['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]

# Calculate percentage modulated neurons
per_mouse_df = ephys_data[ephys_data['sert-cre'] == 1].groupby(['region', 'subject']).sum(numeric_only=True)
per_mouse_df['n_neurons'] = ephys_data[ephys_data['sert-cre'] == 1].groupby(['region', 'subject']).size()
per_mouse_df['perc_mod'] = (per_mouse_df['modulated'] / per_mouse_df['n_neurons'])
per_mouse_df = per_mouse_df.reset_index()
ephys_summary = per_mouse_df[['region', 'perc_mod']].groupby('region').mean()

# Calculate summary per neuron
ephys_summary[['mod_index']] = ephys_data[(ephys_data['sert-cre'] == 1) & (ephys_data['modulated'] == 1)][[
    'region', 'mod_index']].groupby('region').mean()[['mod_index']]
ephys_summary[['abs_mod_index']] = ephys_data[(ephys_data['sert-cre'] == 1) & (ephys_data['modulated'] == 1)][[
    'region', 'abs_mod_index']].groupby('region').mean()[['abs_mod_index']]
ephys_summary[['latenzy']] = ephys_data[(ephys_data['sert-cre'] == 1) & (ephys_data['modulated'] == 1)][[
    'region', 'latenzy']].groupby('region').mean()[['latenzy']]
ephys_summary = ephys_summary.rename(columns={'latenzy': 'latency'})

# Add number of neurons
ephys_summary['n_neurons'] = ephys_data[(ephys_data['sert-cre'] == 1)].groupby(
    'region').size()
ephys_summary['modulated'] = ephys_data[(ephys_data['sert-cre'] == 1) & (ephys_data['modulated'] == 1)].groupby(
    'region').size()

# Drop regions with too few neurons
ephys_summary = ephys_summary[ephys_summary['n_neurons'] >= MIN_NEURONS[TARGET_VARIABLE]]
ephys_summary = ephys_summary[ephys_summary['modulated'] >= MIN_MOD_NEURONS[TARGET_VARIABLE]]

# Drop some regions
ephys_summary = ephys_summary.reset_index()
ephys_summary = ephys_summary[~np.isin(ephys_summary['region'], ['root'])]

# %%
print("--- Starting GLM Analysis ---")

# Reshape receptor expression data to wide format
expression_wide = expression_mean.pivot(
    index='region',
    columns='receptor',
    values='expression_energy'
)

expression_wide.columns.name = None
expression_wide = expression_wide.reset_index()

# Merge all data sources into a single DataFrame

if TARGET_VARIABLE == 'mod_index':
    model_df = ephys_summary[['region', 'mod_index']]
elif TARGET_VARIABLE == 'abs_mod_index':
    model_df = ephys_summary[['region', 'abs_mod_index']]
elif TARGET_VARIABLE == 'perc_mod':
    #model_df = ephys_summary[['region', 'perc_mod']]
    model_df = ephys_summary[['region', 'modulated', 'n_neurons']]
elif TARGET_VARIABLE == 'latency':
    model_df = ephys_summary[['region', 'latency']]
else:
    raise ValueError(f"Unknown TARGET_VARIABLE: '{TARGET_VARIABLE}'")

# Merge projection data
model_df = pd.merge(model_df, proj_summary, on='region', how='inner')

# Merge receptor data
model_df = pd.merge(model_df, expression_wide, on='region', how='inner')

print(f"Final merged DataFrame shape: {model_df.shape}")

# Prepare data for the model
model_df_clean = model_df.dropna()
model_df_clean = model_df_clean.set_index('region') # Use region as index, not a predictor

# Define Target (y) and Predictors (X)
if TARGET_VARIABLE == 'perc_mod':
    # For Binomial GLM, y is [successes, totals]
    y = model_df_clean[['modulated', 'n_neurons']]
    X_full = model_df_clean.drop(columns=['modulated', 'n_neurons'])

    #y = model_df_clean[['perc_mod']]
    #X_full = model_df_clean.drop(columns=['perc_mod'])
else:
    y = model_df_clean[TARGET_VARIABLE]
    X_full = model_df_clean.drop(columns=[TARGET_VARIABLE])

#X_full = X_full.drop(columns=X_full.columns.difference(['projection_density']))

if DROP_PROJECTION:
    X_full = X_full.drop(columns=['projection_density'])

n_obs = X_full.shape[0]
n_pred = X_full.shape[1]
print(f"\nFound {n_obs} observations (regions).")
print(f"Found {n_pred} predictors.")

# Standardize Predictors
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_full)

# Convert back to DataFrame to keep column names
X_scaled = pd.DataFrame(X_scaled, columns=X_full.columns, index=X_full.index)

# Add a constant (intercept) to the model
X_with_const = sm.add_constant(X_scaled)

# Calculate VIF for each predictor
vif_data = pd.DataFrame()
vif_data["feature"] = X_with_const.columns

vif_data["VIF"] = [variance_inflation_factor(X_with_const.values, i)
                          for i in range(len(X_with_const.columns))]

print("\n--- Variance Inflation Factor (VIF) ---")
print(vif_data[vif_data['feature'] != 'const']) # Hide the intercept VIF

# Define GLM families
if (TARGET_VARIABLE == 'abs_mod_index') | (TARGET_VARIABLE == 'mod_index'):
    glm_family = sm.families.Gaussian()
elif TARGET_VARIABLE == 'perc_mod':
    glm_family = sm.families.Binomial()
    #glm_family = sm.families.Gamma(link=sm_links.log())
elif TARGET_VARIABLE == 'latency':
    # Using log link for Gamma is standard to ensure positive predictions
    glm_family = sm.families.Gamma(link=sm_links.log())

# Fit the GLM
glm_model = sm.GLM(y, X_with_const, family=glm_family)
results = glm_model.fit()

# Print the results summary
print("\n--- Full GLM Results Summary ---")
print(results.summary())

# Do leave-one-out cross validation
loo = LeaveOneOut()
loo_coefs = []
y_true = []
y_pred = []
excluded_regions = [] # To store the name of the region left out

# Loop through each 'fold' (leaving one region out each time)
for train_index, test_index in loo.split(X_with_const):
    X_train, X_test = X_with_const.iloc[train_index], X_with_const.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Get the name of the region being tested
    region_name = X_with_const.index[test_index[0]]

    # Fit model on N-1 regions
    try:
        loo_results = sm.GLM(y_train, X_train, family=glm_family).fit()

        # Store coefficients with region name as index
        coefs = loo_results.params.copy()
        coefs['left_out_region'] = region_name
        loo_coefs.append(coefs)

        # Predict
        pred = loo_results.predict(X_test)

        if TARGET_VARIABLE == 'perc_mod':
            y_true.append(y_test.iloc[0, 0] / y_test.iloc[0, 1])
        else:
            y_true.append(y_test.values[0])

        y_pred.append(pred.values[0])
        excluded_regions.append(region_name)

    except Exception as e:
        print(f"Failed for region {region_name}: {e}")
        continue

# Create a DataFrame for the coefficients
loo_coef_df = pd.DataFrame(loo_coefs).set_index('left_out_region')

# Create a DataFrame for the predictions (useful for diagnosing the negative Q2)
loo_pred_df = pd.DataFrame({
    'region': excluded_regions,
    'true_val': y_true,
    'pred_val': y_pred,
    'error': np.array(y_true) - np.array(y_pred)
}).set_index('region')

# %% Plot
# Get params, CIs, and p-values
params = results.params
conf_int = results.conf_int()
pvalues = results.pvalues
reject, pvals_corrected, _, _ = multipletests(pvalues.values, alpha=0.05, method='fdr_bh')

# Combine into a DataFrame
plot_df = pd.DataFrame({'coef': params, 'pvalue': pvalues})
plot_df['ci_low'] = conf_int[0]
plot_df['ci_high'] = conf_int[1]

# Drop the 'const' (intercept) row, we don't plot this
plot_df = plot_df.drop(['const', 'precision'], errors='ignore')

# Sort by coefficient value for a cleaner plot
plot_df = plot_df.sort_values('coef')

# Move projection to bottom
plot_df = pd.concat([plot_df.loc[['projection'], :], plot_df.drop('projection', axis=0)], axis=0)

# Rename
plot_df = plot_df.rename(index={'projection': "Projection"})

# Y-axis positions
y_pos = np.arange(len(plot_df))

f, ax1 = plt.subplots(figsize=(2, 2.1), dpi=dpi)

# Plot error bars (the CIs) and the center points
# Calculate error bar lengths from CI
# xerr = [ (coef - ci_low), (ci_high - coef) ]
x_err = [plot_df['coef'] - plot_df['ci_low'], plot_df['ci_high'] - plot_df['coef']]

ax1.errorbar(x=plot_df['coef'], y=y_pos, xerr=x_err, fmt='o',
             capsize=3, linestyle='None', label='95% Confidence Interval', color='b', zorder=1)
ax1.axvline(x=0, color='grey', linestyle='--', lw=0.75, zorder=0)

if TARGET_VARIABLE == 'mod_index':
    star_x = 0.2
    x_lim = 0.2
    x_loo_lim = 0.1
    ax1.set(xticks=[-x_lim, 0, x_lim], xticklabels=[-x_lim, 0, x_lim], title='Modulation directionality')
elif TARGET_VARIABLE == 'perc_mod':
    x_lim = 0.4
    ax1.set(xticks=[-x_lim, 0, x_lim], xticklabels=[-x_lim, 0, x_lim], title='Modulated neurons')
    star_x = 0.45
elif TARGET_VARIABLE == 'latency':
    x_lim = 0.3
    x_loo_lim = 0.2
    ax1.set(xticks=[-x_lim, 0, x_lim], xticklabels=[-x_lim, 0, x_lim], title='Modulation latency (s)')
    star_x = 0.3
elif TARGET_VARIABLE == 'abs_mod_index':
    x_lim = 0.1
    x_loo_lim = 0.07
    ax1.set(xticks=[-x_lim, 0, x_lim], xticklabels=[-x_lim, 0, x_lim], title='Modulation strength')
    star_x = x_lim + 0.02

# Add significance stars
for i in range(len(plot_df)):
    if plot_df['pvalue'].iloc[i] < 0.05:
        # Place star slightly to the right of the upper CI
        plot_star_x = plot_df['ci_high'].iloc[i] + 0.02
        ax1.text(star_x, y_pos[i] - 0.25, '*',
                 horizontalalignment='center', verticalalignment='center',
                 fontweight='bold', color='k', fontsize=14)

# Formatting
ax1.set_yticks(y_pos, plot_df.index)
ax1.set_xlabel('GLM coefficient')

sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, f'GLM_{TARGET_VARIABLE}.pdf'))

# %%
# Plotting the stability
loo_df = pd.DataFrame(loo_coefs).drop(columns=['const'])
loo_df = loo_df.rename(columns={'projection': "Projection"})
loo_df = loo_df.loc[:, np.flip(plot_df.index)]

f, ax1 = plt.subplots(figsize=(2, 2), dpi=dpi)
#sns.boxplot(data=loo_df, orient='h', fliersize=0, ax=ax1, showmeans=True, zorder=2,
#            meanprops={"marker": "|", "markeredgecolor": "red", "markersize": "8"},
#            boxprops={'facecolor':'none', 'edgecolor':'none'}, medianprops={'color':'none'},
#                     whiskerprops={'color':'none'}, capprops={'color':'none'})
sns.stripplot(data=loo_df, orient='h', color='grey', size=2, ax=ax1, zorder=1)
ax1.axvline(x=0, color='grey', linestyle='--', lw=0.5, zorder=0)
ax1.set(xlabel='GLM coefficient', xticks=[-x_loo_lim, 0, x_loo_lim], xticklabels=[-x_loo_lim, 0, x_loo_lim])

sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, f'GLM_stability_{TARGET_VARIABLE}.pdf'))


# %%
