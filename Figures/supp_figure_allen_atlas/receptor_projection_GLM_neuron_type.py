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
from scipy.stats import pearsonr
import math
import statsmodels.api as sm
import statsmodels.genmod.families.links as sm_links
from sklearn.preprocessing import StandardScaler
from stim_functions import paths, figure_style, load_subjects, combine_regions, remap
from iblatlas.atlas import AllenAtlas
ba = AllenAtlas(res_um=25)
colors, dpi = figure_style()

# Settings
MIN_MOD_NEURONS = 3
MIN_NEURONS = 10
#TARGET_VARIABLE = 'perc_mod'
TARGET_VARIABLE = 'mod_index'
NEURON_TYPE = 'WS'
INCL_RECEPTORS = ['5-HT1a', '5-HT1b', '5-HT2a', '5-HT2c', '5-HT3a', '5-HT5a']

# Paths
f_path, data_path = paths()
fig_path = join(f_path, split(dirname(realpath(__file__)))[-1])

# Load in expression
expr_df = pd.read_csv(join(data_path, 'receptor_expression.csv'))
expr_df = expr_df[~np.isin(expr_df['acronym'], ['MMme', 'CUL4, 5', 'CUL4, 5gr', 'CUL4, 5mo'])]
expr_df['region'] = remap(expr_df['acronym'])
expr_df = expr_df[np.isin(expr_df['receptor'], INCL_RECEPTORS)]
#expr_df['region'] = combine_regions(remap(expr_df['acronym']))
expression_mean = expr_df[['region', 'receptor', 'expression_energy']].groupby(
    ['region', 'receptor']).median().reset_index()

# load structure and expression data set
proj_df = pd.read_csv(join(data_path, 'dr_projection_strength.csv'))
proj_df = proj_df[~np.isin(proj_df['allen_acronym'], ['MMd', 'MMme', 'MMl', 'MMm', 'MMp', 'CUL4, 5'])]
proj_df['region'] = remap(proj_df['allen_acronym'])
#proj_df['region'] = combine_regions(remap(proj_df['allen_acronym']))
proj_summary = proj_df[['region', 'projection_density']].groupby(['region']).mean().reset_index()

# Load in neural data
ephys_data = pd.read_csv(join(data_path, 'light_modulated_neurons.csv'))
#ephys_data['region'] = combine_regions(ephys_data['region'])
subjects = load_subjects()
for i, nickname in enumerate(np.unique(subjects['subject'])):
    ephys_data.loc[ephys_data['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]

# Load in neuron type
type_df = pd.read_csv(join(data_path, 'neuron_type.csv'))
ephys_data = pd.merge(ephys_data, type_df, on=['subject', 'pid', 'eid', 'probe', 'neuron_id'])
ephys_data = ephys_data[ephys_data['type'] == NEURON_TYPE]

# Calculate percentage modulated neurons
per_mouse_df = ephys_data[ephys_data['sert-cre'] == 1].groupby(['region', 'subject']).sum(numeric_only=True)
per_mouse_df['n_neurons'] = ephys_data[ephys_data['sert-cre'] == 1].groupby(['region', 'subject']).size()
per_mouse_df['perc_mod'] = (per_mouse_df['modulated'] / per_mouse_df['n_neurons'])
per_mouse_df = per_mouse_df.reset_index()
ephys_summary = per_mouse_df[['region', 'perc_mod']].groupby('region').mean()

# Calculate summary per neuron
ephys_summary[['mod_index', 'latenzy']] = ephys_data[(ephys_data['sert-cre'] == 1) & (ephys_data['modulated'] == 1)][[
    'region', 'mod_index', 'latenzy']].groupby('region').mean()[['mod_index', 'latenzy']]
ephys_summary = ephys_summary.rename(columns={'latenzy': 'latency'})

# Add number of neurons
ephys_summary['n_neurons'] = ephys_data[(ephys_data['sert-cre'] == 1)].groupby(
    'region').size()
ephys_summary['modulated'] = ephys_data[(ephys_data['sert-cre'] == 1) & (ephys_data['modulated'] == 1)].groupby(
    'region').size()

# Drop regions with too few neurons
ephys_summary = ephys_summary[ephys_summary['n_neurons'] >= MIN_NEURONS]
ephys_summary = ephys_summary[ephys_summary['modulated'] >= MIN_MOD_NEURONS]

# Drop some regions
ephys_summary = ephys_summary.reset_index()
ephys_summary = ephys_summary[~np.isin(ephys_summary['region'], ['root'])]

# %%
print("--- Starting GLM Analysis ---")

# 1. Reshape receptor expression data to wide format
print("Pivoting receptor expression data...")
try:
    expression_wide = expression_mean.pivot(
        index='region',
        columns='receptor',
        values='expression_energy'
    )

    expression_wide.columns.name = None
    expression_wide = expression_wide.reset_index()
except ValueError as e:
    print(f"Error pivoting data: {e}")
    print("Attempting aggregation before pivot...")
    expression_mean_agg = expression_mean.groupby(['region', 'receptor'])['expression_energy'].mean().reset_index()
    expression_wide = expression_mean_agg.pivot(
        index='region',
        columns='receptor',
        values='expression_energy'
    )
    expression_wide.columns.name = None
    expression_wide = expression_wide.reset_index()


# 2. Merge all data sources into a single DataFrame

print("Merging data sources...")

if TARGET_VARIABLE == 'mod_index':
    # Ensure mod_index is present
    if 'mod_index' not in ephys_summary.columns:
        print("Critical Error: 'mod_index' not found in ephys_summary.")
        # Stop execution or raise error
    model_df = ephys_summary[['region', 'mod_index']]
elif TARGET_VARIABLE == 'perc_mod':
    # Ensure count columns are present for Binomial GLM
    if 'modulated' not in ephys_summary.columns or 'n_neurons' not in ephys_summary.columns:
        print("Critical Error: 'modulated' or 'n_neurons' counts not found in ephys_summary.")
        # Stop execution or raise error
    #model_df = ephys_summary[['region', 'perc_mod']]
    model_df = ephys_summary[['region', 'modulated', 'n_neurons']]
elif TARGET_VARIABLE == 'latency':
    # Ensure latency is present
    if 'latency' not in ephys_summary.columns:
        print("Critical Error: 'latency' not found in ephys_summary.")
        # Stop execution or raise error
    model_df = ephys_summary[['region', 'latency']]
else:
    raise ValueError(f"Unknown TARGET_VARIABLE: '{TARGET_VARIABLE}'")

# Merge projection data
model_df = pd.merge(model_df, proj_summary, on='region', how='inner')

# Merge receptor data
model_df = pd.merge(model_df, expression_wide, on='region', how='inner')

print(f"Final merged DataFrame shape: {model_df.shape}")

# 3. Prepare data for the model
model_df_clean = model_df.dropna()
model_df_clean = model_df_clean.set_index('region') # Use region as index, not a predictor
if model_df_clean.empty:
    print("\nCritical Error: No overlapping regions found after merging.")
    print("Cannot build model. Check region names for consistency across files.")

# --- MODIFIED SECTION: Reverted to Full GLM (No PCA) ---
else:
    # 4. Define Target (y) and Predictors (X)

    if TARGET_VARIABLE == 'mod_index':
        y = model_df_clean['mod_index']
        X_full = model_df_clean.drop(columns=['mod_index'])
    elif TARGET_VARIABLE == 'perc_mod':
        # For Binomial GLM, y is [successes, totals]
        y = model_df_clean[['modulated', 'n_neurons']]
        X_full = model_df_clean.drop(columns=['modulated', 'n_neurons'])

        #y = model_df_clean[['perc_mod']]
        #X_full = model_df_clean.drop(columns=['perc_mod'])
    elif TARGET_VARIABLE == 'latency':
        y = model_df_clean['latency']
        X_full = model_df_clean.drop(columns=['latency'])

    #X_full = X_full.drop(columns=X_full.columns.difference(['projection_density']))

    n_obs = X_full.shape[0]
    n_pred = X_full.shape[1]

    print(f"\nFound {n_obs} observations (regions).")
    print(f"Found {n_pred} predictors.")

    # 5. CRITICAL CHECK: N vs P
    # We need N > P + 1 (for the intercept) to fit the model.
    if n_obs <= n_pred + 1:
        print("\n--- CRITICAL WARNING: Not Enough Data ---")
        print(f"You have {n_obs} observations (regions) and {n_pred} predictors.")
        print("To fit the full model, you MUST have more observations than predictors (N > P).")
        print("The model will likely fail or produce unreliable 'Perfect Separation' warnings.")
        print("Please re-run after increasing the number of brain regions in your data.")

    else:
        print("\nSufficient data found (N > P). Proceeding with GLM.")

        # 6. Standardize Predictors (Recommended)
        print("Standardizing predictors (Z-scoring)...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_full)

        # Convert back to DataFrame to keep column names
        X_scaled = pd.DataFrame(X_scaled, columns=X_full.columns, index=X_full.index)

        # 7. Add a constant (intercept) to the model
        X_with_const = sm.add_constant(X_scaled)

        # 8. Fit the GLM

        # --- MODIFIED: Select family based on target ---
        if TARGET_VARIABLE == 'mod_index':
            print("\nFitting Generalized Linear Model (Gaussian family)...")
            # Using log link for Gamma is standard to ensure positive predictions
            glm_family = sm.families.Gaussian()
        elif TARGET_VARIABLE == 'perc_mod':
            print("\nFitting Generalized Linear Model (Binomial family)...")
            print(f"Target variable is {TARGET_VARIABLE} (using [modulated, n_neurons] counts)")
            glm_family = sm.families.Binomial()
            #glm_family = sm.families.Gamma(link=sm_links.log())
        elif TARGET_VARIABLE == 'latency':
            print("\nFitting Generalized Linear Model (Gamma family)...")
            print(f"Target variable is {TARGET_VARIABLE} (using Gamma for positive, continuous duration)")
            # Using log link for Gamma is standard to ensure positive predictions
            glm_family = sm.families.Gamma(link=sm_links.log())

        glm_model = sm.GLM(y, X_with_const, family=glm_family)
        results = glm_model.fit()
        # --- END MODIFIED SECTION ---

        # 9. Print the results summary
        print("\n--- Full GLM Results Summary ---")
        print(results.summary())

        # Get params, CIs, and p-values
        params = results.params
        conf_int = results.conf_int()
        pvalues = results.pvalues

        # Combine into a DataFrame
        plot_df = pd.DataFrame({
            'coef': params,
            'pvalue': pvalues
        })
        plot_df['ci_low'] = conf_int[0]
        plot_df['ci_high'] = conf_int[1]

        # Drop the 'const' (intercept) row, we don't plot this
        plot_df = plot_df.drop('const')

        # Sort by coefficient value for a cleaner plot
        plot_df = plot_df.sort_values('coef')

        # Rename
        plot_df = plot_df.rename(index={'projection_density': "Projection"})

        # Y-axis positions
        y_pos = np.arange(len(plot_df))

        plt.figure(figsize=(2, 2.1), dpi=dpi)

        # Plot error bars (the CIs) and the center points
        # Calculate error bar lengths from CI
        # xerr = [ (coef - ci_low), (ci_high - coef) ]
        x_err = [
            plot_df['coef'] - plot_df['ci_low'],
            plot_df['ci_high'] - plot_df['coef']
        ]

        plt.errorbar(
            x=plot_df['coef'],
            y=y_pos,
            xerr=x_err,
            fmt='o', # 'o' plots the center point
            capsize=3,
            linestyle='None',
            label='95% Confidence Interval',
            color='b',
            zorder=1
        )
        plt.axvline(x=0, color='grey', linestyle='--', lw=0.75, zorder=0)

        if TARGET_VARIABLE == 'mod_index':
            plt.xticks([-0.2, 0, 0.2], [-0.2, 0, 0.2])
            plt.title('Modulation index')
            star_x = 0.2
        elif TARGET_VARIABLE == 'perc_mod':
            plt.xticks([-0.4, 0, 0.4], [-0.4, 0, 0.4])
            plt.title('Modulated neurons')
            star_x = 0.45
        elif TARGET_VARIABLE == 'latency':
            plt.xticks([-0.3, 0, 0.3], [-0.3, 0, 0.3])
            plt.title('Modulation latency (s)')
            star_x = 0.3

        # Add significance stars
        for i in range(len(plot_df)):
            if plot_df['pvalue'].iloc[i] < 0.05:
                # Place star slightly to the right of the upper CI
                plot_star_x = plot_df['ci_high'].iloc[i] + 0.02
                plt.text(star_x, y_pos[i] - 0.5, '*',
                         horizontalalignment='center', verticalalignment='center',
                         fontweight='bold', color='k', fontsize=14)

        # Formatting
        plt.yticks(y_pos, plot_df.index)
        plt.xlabel('GLM coefficient')

        sns.despine(trim=True)
        plt.tight_layout()
        plt.savefig(join(fig_path, f'GLM_{TARGET_VARIABLE}_{NEURON_TYPE}.pdf'))
