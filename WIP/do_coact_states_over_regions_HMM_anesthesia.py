# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 16:11:41 2023

By Guido Meijer
"""

import numpy as np
import pandas as pd
import seaborn as sns
from glob import glob
from os.path import join, split
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from stim_functions import paths, figure_style, load_subjects

# Settings
PRE_TIME = 1
POST_TIME = 4
PLOT = False
ORIG_BIN_SIZE = 0.2  # original bin size
PRE_TIME = 1
POST_TIME = 4

# Plotting
colors, dpi = figure_style()

# Get paths
fig_path, save_path = paths()

# Get all files
all_files = glob(join(save_path, 'HMM', 'Anesthesia', 'prob_mat', '*.npy'))

# Get all recordings
all_rec = np.unique([split(i)[1][:20] for i in all_files])

# Create time axis
time_ax = np.arange(-PRE_TIME + ORIG_BIN_SIZE/2, POST_TIME + ORIG_BIN_SIZE/2, ORIG_BIN_SIZE)

coact_df = pd.DataFrame()
for i, this_rec in enumerate(all_rec):

    # Get all brain regions simultaneously recorded in this recording session
    rec_region_paths = glob(
        join(save_path, 'HMM', 'Anesthesia', 'state_mat', f'{this_rec[:20]}*'))
    rec_region = dict()
    for ii in range(len(rec_region_paths)):
        rec_region[split(rec_region_paths[ii])[1][29:-4]] = np.load(join(rec_region_paths[ii]))

    # Get recording data
    subject = split(rec_region_paths[0])[1][:9]
    date = split(rec_region_paths[0])[1][10:20]

    # Correlate each area with each other area
    for r1, region1 in enumerate(rec_region.keys()):
        for r2, region2 in enumerate(list(rec_region.keys())[r1:]):
            if region1 == region2:
                continue

            # Loop over timebins
            coact_mats = np.empty((2, 2, time_ax.shape[0]))
            coact_mats_null = np.empty((2, 2, time_ax.shape[0]))
            coact = np.empty(time_ax.shape[0])
            coact_null = np.empty(time_ax.shape[0])
            
            for tb, bin_center in enumerate(time_ax):

                # Get coactivation of states
                for state1 in range(2):
                    for state2 in range(2):

                        # Calculate jaccard similarity 
                        reg1_states = rec_region[region1][rec_region[region1].shape[0]//2:, tb]
                        reg2_states = rec_region[region2][rec_region[region2].shape[0]//2:, tb]
                        intersection = np.logical_and(reg1_states == state1,
                                                      reg2_states == state2)
                        union = np.logical_or(reg1_states == state1,
                                              reg2_states == state2)
                        if union.sum() == 0:
                            coact_mats[state1, state2, tb] = 0
                        else:
                            coact_mats[state1, state2, tb] = intersection.sum() / float(union.sum())
                            
                        # No stim condition
                        reg1_states = rec_region[region1][:rec_region[region1].shape[0]//2, tb]
                        reg2_states = rec_region[region2][:rec_region[region2].shape[0]//2, tb]
                        intersection = np.logical_and(reg1_states == state1,
                                                      reg2_states == state2)
                        union = np.logical_or(reg1_states == state1,
                                              reg2_states == state2)
                        if union.sum() == 0:
                            coact_mats_null[state1, state2, tb] = 0
                        else:
                            coact_mats_null[state1, state2, tb] = intersection.sum() / float(union.sum())
                        
                # Get mean over diagonal of coactivation matrix
                coact[tb] = np.max(np.diag(coact_mats[:, :, tb]))
                coact_null[tb] = np.max(np.diag(coact_mats_null[:, :, tb]))

            coact_df = pd.concat((coact_df, pd.DataFrame(data={
                'time': time_ax, 'coact': coact, 
                'region1': region1, 'region2': region2, 'opto': 1,
                'region_pair': f'{np.sort([region1, region2])[0]}-{np.sort([region1, region2])[1]}',
                'subject': subject, 'date': date})))
            coact_df = pd.concat((coact_df, pd.DataFrame(data={
                'time': time_ax, 'coact': coact_null, 
                'region1': region1, 'region2': region2, 'opto': 0,
                'region_pair': f'{np.sort([region1, region2])[0]}-{np.sort([region1, region2])[1]}',
                'subject': subject, 'date': date})))

    # Save output
    coact_df.to_csv(join(save_path, 'state_coactivation_anesthesia.csv'))

# %% Plot

# Add genotype
subjects = load_subjects()
for i, nickname in enumerate(np.unique(subjects['subject'])):
    coact_df.loc[coact_df['subject'] == nickname,
                 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]

f, ax1 = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)
ax1.add_patch(Rectangle((0, 0), 1, 1, color='royalblue', alpha=0.25, lw=0))
sns.lineplot(data=coact_df[coact_df['sert-cre'] == 1], x='time', y='coact',
             errorbar='se', hue='opto', err_kws={'lw': 0},
             hue_order=[0, 1], palette=[colors['no-stim'], colors['stim']])
ax1.set(xlabel='Time from stimulation start (s)', ylim=[0.4, 0.65],
        xticks=[-1, 0, 1, 2, 3, 4], yticks=[0.4, 0.65], yticklabels=[0.4, 0.65])
# ylim=[-0.001, 0.005], yticks=[0, 0.005],
# yticklabels=[0, 0.005])
ax1.set_ylabel('State coactivation', labelpad=-5)
g = ax1.legend(title='', bbox_to_anchor=(0.6, 0.7), prop={'size': 5})
new_labels = ['label 1', 'label 2']
for t, l in zip(g.texts, ['No stim', 'Stim']):
    t.set_text(l)
sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, 'state_coactivation_anesthesia.jpg'), dpi=600)

# %% Plot region pairs seperately
g = sns.FacetGrid(coact_df[coact_df['sert-cre'] == 1], col='region_pair', col_wrap=5, height=2,
                  hue='opto', hue_order=[0, 1],
                  palette=[colors['no-stim'], colors['stim']])
g.map(sns.lineplot, 'time', 'coact', errorbar='se')
#plt.savefig(
#    join(fig_path, 'state_correlation_all_region_pair_task_f{N_STATES_SELECT}.jpg'), dpi=600)