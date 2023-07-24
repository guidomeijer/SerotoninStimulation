# -*- coding: utf-8 -*-
"""
Created on Mon May 30 15:21:26 2022

@author: Guido
"""

import pandas as pd
import numpy as np
from os.path import join
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter
from stim_functions import paths, load_subjects, figure_style

# Settings
ASYM_TIME = 0.05
CCA_TIME = 0.02
BIN_SIZE = 0.01
SMOOTHING = True

# Paths
fig_path, save_path = paths()
fig_path = join(fig_path, 'Extra plots', 'CCA')

# Load in data
jpecc_df = pd.read_pickle(join(save_path, f'jPECC_passive_{BIN_SIZE}.pickle'))

# Add expression
subjects = load_subjects()
for i, nickname in enumerate(np.unique(subjects['subject'])):
    jpecc_df.loc[jpecc_df['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]

# Select sert mice and regions of interest
jpecc_df = jpecc_df[jpecc_df['sert-cre'] == 1]

# Get time axis
time_ax = np.round(jpecc_df['time'].mean(), 3)
time_min = time_ax[0]-BIN_SIZE/2
time_max = time_ax[-1]+BIN_SIZE/2

# Get region pairs with n > 1
region_pairs, counts = np.unique(jpecc_df['region_pair'], return_counts=True)
region_pairs = region_pairs[counts > 1]

# Get 3D array of all jPECC
jPECC, asym, cca_df = dict(), dict(), pd.DataFrame()
for i, rp in enumerate(region_pairs):

    jPECC[rp] = np.dstack(jpecc_df.loc[jpecc_df['region_pair'] == rp, 'r_opto'].to_numpy())

    for jj, subject in zip(range(jPECC[rp].shape[2]), jpecc_df.loc[jpecc_df['region_pair'] == rp, 'subject']):

        # Do some smoothing
        if SMOOTHING:
            jPECC[rp][:, :, jj] = gaussian_filter(jPECC[rp][:, :, jj], 1)

        # OPTO
        this_asym = np.empty(time_ax[int(ASYM_TIME/BIN_SIZE) : -int(ASYM_TIME/BIN_SIZE)].shape[0])
        this_cca = np.empty(time_ax[int(ASYM_TIME/BIN_SIZE) : -int(ASYM_TIME/BIN_SIZE)].shape[0])
        for k, time_bin in enumerate(time_ax[int(ASYM_TIME/BIN_SIZE) : -int(ASYM_TIME/BIN_SIZE)]):
            
            # Assymetry
            this_row = jPECC[rp][k, :, jj]
            this_asym[k] = (np.mean(this_row[(time_ax < time_bin) & (time_ax >= time_bin - ASYM_TIME)])
                            - np.mean(this_row[(time_ax > time_bin) & (time_ax <= time_bin + ASYM_TIME)]))
            this_cca[k] = np.max(this_row[(time_ax >= time_bin - ASYM_TIME)
                                          & (time_ax <= time_bin + ASYM_TIME)])
        # Add to dataframe
        cca_df = pd.concat((cca_df, pd.DataFrame(data={
            'cca': this_cca, 'asym': this_asym,
            'subject': subject, 'region_pair': rp,
            'time': time_ax[int(ASYM_TIME/BIN_SIZE) : -int(ASYM_TIME/BIN_SIZE)]})))



# %% Plot
colors, dpi = figure_style()

for i, region_pair in enumerate(jPECC.keys()):
    f, (ax1, ax_cb) = plt.subplots(1, 2, figsize=(2, 1.75), gridspec_kw={'width_ratios': [0.7, 0.3]},
                                   dpi=dpi)
    VMAX = np.max(jPECC[region_pair])
    ax1.imshow(np.flipud(np.mean(jPECC[region_pair], axis=2)), 
               vmin=-np.max(np.mean(jPECC[region_pair], axis=2)),
               vmax=np.max(np.mean(jPECC[region_pair], axis=2)),
               cmap='icefire', interpolation='nearest', aspect='auto',
               extent=[time_min, time_max, time_min, time_max])
    ax1.plot([time_min, time_max], [time_min, time_max], color='white', ls='--', lw=0.5)
    ax1.plot([time_min, time_max], [0, 0], color='white', ls='--', lw=0.5)
    ax1.plot([0, 0], [time_min, time_max], color='white', ls='--', lw=0.5)
    ax1.set(xlabel='Time from stim. onset (s)', ylabel='Time from stim. onset (s)',
            title=region_pair.replace('-', ' vs '),
            xticks=[time_min, 0, time_max], yticks=[time_min, 0, time_max])
    
    ax_cb.axis('off')
    plt.tight_layout()
    cb_ax = f.add_axes([0.75, 0.3, 0.02, 0.5])
    cbar = f.colorbar(mappable=ax1.images[0], cax=cb_ax)
    cbar.ax.set_ylabel('Population correlation (r)', rotation=270, labelpad=8)
    
    plt.savefig(join(fig_path, f'jPECC_{region_pair}.jpg'), dpi=600)
    plt.close(f)

# %%
colors, dpi = figure_style()
f, axs = plt.subplots(3, 4, figsize=(7, 5), dpi=dpi)
axs = np.concatenate(axs)
for i, region_pair in enumerate(region_pairs):
    #axs[i].plot([-1, 3], [0, 0], ls='--', color='grey')
    axs[i].add_patch(Rectangle((0, 0), 1, 1, color='royalblue', alpha=0.25, lw=0))
    sns.lineplot(x='time', y='cca', data=cca_df[(cca_df['region_pair'] == region_pair)],
                 ax=axs[i], errorbar='se', color='k',
                 legend=None, err_kws={'lw': 0}, zorder=1)
    axs[i].set(ylim=[0, 0.8], xlabel='', ylabel='', title=region_pair)
    
plt.tight_layout()
sns.despine(trim=True)

plt.savefig(join(fig_path, 'jPECC_CCA_passive.jpg'), dpi=600)

# %%
f, axs = plt.subplots(3, 4, figsize=(7, 5), dpi=dpi)
axs = np.concatenate(axs)
for i, region_pair in enumerate(region_pairs):
    #axs[i].plot([-1, 3], [0, 0], ls='--', color='grey')
    axs[i].add_patch(Rectangle((0, -0.4), 1, 1, color='royalblue', alpha=0.25, lw=0))
    sns.lineplot(x='time', y='asym', data=cca_df[(cca_df['region_pair'] == region_pair)],
                 ax=axs[i], errorbar='se', color='k',
                 legend=None, err_kws={'lw': 0}, zorder=1)
    axs[i].set(ylim=[-0.4, 0.4], xlabel='', ylabel='', title=region_pair)
    
plt.tight_layout()
sns.despine(trim=True)

plt.savefig(join(fig_path, 'jPECC_asym_passive.jpg'), dpi=600)

