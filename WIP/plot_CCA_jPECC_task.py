# -*- coding: utf-8 -*-
"""
Created on Mon May 30 15:21:26 2022

@author: Guido
"""

import pandas as pd
import numpy as np
from os.path import join
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter
from stim_functions import paths, load_subjects, figure_style
colors, dpi = figure_style()

# Settings
CENTER_ON = 'stimOn'
#CENTER_ON = 'firstMovement'
ASYM_TIME = 0.05
BIN_SIZE = 0.01
SMOOTHING = False

if CENTER_ON == 'stimOn':
    LABEL = 'Time from trial start (s)'
elif CENTER_ON == 'firstMovement':
    LABEL = 'Time from first movement (s)'

# Paths
fig_path, save_path = paths()
fig_path = join(fig_path, 'Extra plots', 'CCA')

# Load in data
jpecc_df = pd.read_pickle(join(save_path, f'task_jPECC_{CENTER_ON}_{BIN_SIZE}_binsize.pickle'))

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
jPECC_opto, asym_opto, jPECC_no_opto, asym_no_opto, = dict(), dict(), dict(), dict()
cca_df = pd.DataFrame()
for i, rp in enumerate(region_pairs):

    jPECC_opto[rp] = np.dstack(jpecc_df.loc[jpecc_df['region_pair'] == rp, 'r_opto'].to_numpy())
    jPECC_no_opto[rp] = np.dstack(jpecc_df.loc[jpecc_df['region_pair'] == rp, 'r_no_opto'].to_numpy())

    for jj, subject in zip(range(jPECC_opto[rp].shape[2]), jpecc_df.loc[jpecc_df['region_pair'] == rp, 'subject']):

        # Do some smoothing
        if SMOOTHING:
            jPECC_opto[rp][:, :, jj] = gaussian_filter(jPECC_opto[rp][:, :, jj], 1)
            jPECC_no_opto[rp][:, :, jj] = gaussian_filter(jPECC_no_opto[rp][:, :, jj], 1)

        # OPTO
        this_asym_opto = np.empty(time_ax[int(ASYM_TIME/BIN_SIZE) : -int(ASYM_TIME/BIN_SIZE)].shape[0])
        this_cca_opto = np.empty(time_ax[int(ASYM_TIME/BIN_SIZE) : -int(ASYM_TIME/BIN_SIZE)].shape[0])
        for k, time_bin in enumerate(time_ax[int(ASYM_TIME/BIN_SIZE) : -int(ASYM_TIME/BIN_SIZE)]):
            
            # Assymetry
            this_row = jPECC_opto[rp][k, :, jj]
            this_asym_opto[k] = (np.mean(this_row[(time_ax < time_bin) & (time_ax >= time_bin - ASYM_TIME)])
                                 - np.mean(this_row[(time_ax > time_bin) & (time_ax <= time_bin + ASYM_TIME)]))
            this_cca_opto[k] = np.max(this_row[(time_ax >= time_bin - ASYM_TIME)
                                               & (time_ax <= time_bin + ASYM_TIME)])
            
        cca_df = pd.concat((cca_df, pd.DataFrame(data={
            'opto': 1, 'cca': this_cca_opto, 'asym': this_asym_opto,
            'subject': subject, 'region_pair': rp,
            'time': time_ax[int(ASYM_TIME/BIN_SIZE) : -int(ASYM_TIME/BIN_SIZE)]})))
        
        # NO OPTO
        this_asym_no_opto = np.empty(time_ax[int(ASYM_TIME/BIN_SIZE) : -int(ASYM_TIME/BIN_SIZE)].shape[0])
        this_cca_no_opto = np.empty(time_ax[int(ASYM_TIME/BIN_SIZE) : -int(ASYM_TIME/BIN_SIZE)].shape[0])
        for k, time_bin in enumerate(time_ax[int(ASYM_TIME/BIN_SIZE) : -int(ASYM_TIME/BIN_SIZE)]):
            
            # Assymetry
            this_row = jPECC_no_opto[rp][k, :, jj]
            this_asym_no_opto[k] = (np.mean(this_row[(time_ax < time_bin) & (time_ax >= time_bin - ASYM_TIME)])
                                             - np.mean(this_row[(time_ax > time_bin) & (time_ax <= time_bin + ASYM_TIME)]))
            this_cca_no_opto[k] = np.max(this_row[(time_ax >= time_bin - ASYM_TIME)
                                                  & (time_ax <= time_bin + ASYM_TIME)])
            
        cca_df = pd.concat((cca_df, pd.DataFrame(data={
            'opto': 0, 'cca': this_cca_no_opto, 'asym': this_asym_no_opto,
            'subject': subject, 'region_pair': rp,
            'time': time_ax[int(ASYM_TIME/BIN_SIZE) : -int(ASYM_TIME/BIN_SIZE)]})))
     
# Take average per timepoint
cca_df = cca_df.groupby(['time', 'subject', 'region_pair', 'opto']).mean()
cca_long_df = cca_df.melt(ignore_index=False).reset_index()
cca_df = cca_df.reset_index()

# %% Plot

for i, region_pair in enumerate(jPECC_opto.keys()):
    f, (ax1, ax2, ax_cb) = plt.subplots(
        1, 3, figsize=(4, 2), gridspec_kw={'width_ratios': [1, 1, 0.2]}, dpi=dpi)
    ax1.imshow(np.flipud(np.mean(jPECC_opto[region_pair], axis=2)),
               vmin=-np.max(np.mean(jPECC_opto[region_pair], axis=2)),
               vmax=np.max(np.mean(jPECC_opto[region_pair], axis=2)),
               cmap='icefire', interpolation='nearest', aspect='auto',
               extent=[time_min, time_max, time_min, time_max])
    ax1.plot([time_min, time_max], [time_min, time_max], color='white', ls='--', lw=0.5)
    ax1.plot([time_min, time_max], [0, 0], color='white', ls='--', lw=0.5)
    ax1.plot([0, 0], [time_min, time_max], color='white', ls='--', lw=0.5)
    ax1.set(ylabel=LABEL, xlabel=LABEL, title='Opto',
            xticks=[time_min, 0, time_max], yticks=[time_min, 0, time_max])
    
    ax2.imshow(np.flipud(np.mean(jPECC_no_opto[region_pair], axis=2)), vmin=-np.max(jPECC_no_opto[region_pair]),
               vmax=np.max(jPECC_no_opto[region_pair]), cmap='icefire', interpolation='nearest', aspect='auto',
               extent=[time_min, time_max, time_min, time_max])
    ax2.plot([time_min, time_max], [time_min, time_max], color='white', ls='--', lw=0.5)
    ax2.plot([time_min, time_max], [0, 0], color='white', ls='--', lw=0.5)
    ax2.plot([0, 0], [time_min, time_max], color='white', ls='--', lw=0.5)
    ax2.set(ylabel=LABEL, xlabel=LABEL, title='No opto',
            xticks=[time_min, 0, time_max], yticks=[time_min, 0, time_max])
    
    ax_cb.axis('off')
    f.suptitle(region_pair.replace('-', ' vs ') + f' (n = {jPECC_opto[region_pair].shape[2]} mice)')
    plt.tight_layout()
    cb_ax = f.add_axes([0.8, 0.25, 0.02, 0.5])
    cbar = f.colorbar(mappable=ax1.images[0], cax=cb_ax)
    cbar.ax.set_ylabel('Population correlation (r)', rotation=270, labelpad=8)
    
    plt.savefig(join(fig_path, f'jPECC_{region_pair}_{CENTER_ON}.jpg'), dpi=600)
    plt.close(f)

# %%
f, axs = plt.subplots(2, 4, figsize=(7, 3.5), dpi=dpi)
axs = np.concatenate(axs)
for i, region_pair in enumerate(np.unique(cca_long_df['region_pair'])):
    
    sns.lineplot(x='time', y='cca', hue='opto', data=cca_df[(cca_df['region_pair'] == region_pair)],
                 ax=axs[i], errorbar='se', hue_order=[0, 1], palette=[colors['no-stim'], colors['stim']],
                 legend=None, err_kws={'lw': 0}, zorder=1)
    axs[i].set(xlabel=LABEL, ylabel='Canonical correlation (r)', title=region_pair, 
               xticks=[time_min, 0, time_max])
    #axs[i].plot(axs[i].get_xlim(), [0, 0], ls='--', color='grey', zorder=0)
    
plt.tight_layout()
sns.despine(trim=True)

plt.savefig(join(fig_path, f'CCA_{CENTER_ON}.jpg'), dpi=600)

# %%
f, axs = plt.subplots(2, 4, figsize=(7, 3.5), dpi=dpi)
axs = np.concatenate(axs)
for i, region_pair in enumerate(np.unique(cca_long_df['region_pair'])):
    #axs[i].plot([-0.3, 0.1], [0, 0], ls='--', color='grey')
    sns.lineplot(x='time', y='asym', hue='opto', data=cca_df[(cca_df['region_pair'] == region_pair)],
                 ax=axs[i], errorbar='se', hue_order=[0, 1], palette=[colors['no-stim'], colors['stim']],
                 legend=None, err_kws={'lw': 0}, zorder=0)
    axs[i].set(xlabel=LABEL, ylabel='Assymetry',
               ylim=[-0.2, 0.2], title=region_pair)
    axs[i].plot(axs[i].get_xlim(), [0, 0], ls='--', color='grey', zorder=0)
    
plt.tight_layout()
sns.despine(trim=True)

plt.savefig(join(fig_path, f'asym_{CENTER_ON}.jpg'), dpi=600)

