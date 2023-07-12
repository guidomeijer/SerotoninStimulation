# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 16:11:41 2023

By Guido Meijer
"""

import numpy as np
import seaborn as sns
from glob import glob
from os.path import join, split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from stim_functions import paths, figure_style

# Settings
CMAP = 'Set2'
PRE_TIME = 1
POST_TIME = 4

# Plotting
colors, dpi = figure_style()

def get_most_likely_states(state_prob):
    ml_state = np.empty((state_prob.shape[0], state_prob.shape[1]))
    for trial in range(state_prob.shape[0]):
        ml_state[trial, :] = np.argmax(state_prob[trial, : :], axis=1)
    return ml_state

# Get paths
f_path, save_path = paths()
fig_path = join(f_path, 'Extra plots', 'State', 'Regions')

# Get all files
all_files = glob(join(save_path, 'HMM', '*.npy'))

# Get all recordings
all_rec = np.unique([split(i)[1][:28] for i in all_files])

for i, this_rec in enumerate(all_rec):
    
    # Get all brain regions simultaneously recorded in this recording session
    rec_region = glob(join(save_path, 'HMM', f'{this_rec[:20]}*'))
    
    # Plot
    f, axs = plt.subplots(1, len(rec_region), dpi=dpi, figsize=(1.5*len(rec_region), 1.75),
                          sharey=True)
    if len(rec_region) == 1:
        axs = [axs]
    for ii in range(len(rec_region)):
        state_prob = np.load(f'{rec_region[ii]}')
        ml_state = get_most_likely_states(state_prob)
        n_states = state_prob.shape[2]
        axs[ii].imshow(ml_state, aspect='auto', cmap=ListedColormap(sns.color_palette(CMAP, n_states)),
                       vmin=0, vmax=state_prob.shape[2]-1,
                       extent=(-PRE_TIME, POST_TIME, 1, state_prob.shape[1]), interpolation=None)
        axs[ii].plot([0, 0], [1, state_prob.shape[1]], ls='--', color='k', lw=0.75)
        axs[ii].set(yticks=[1, 50], xticks=[-1, 0, 1, 2, 3, 4], title=f'{split(rec_region[ii])[1][29:-4]}')
        if ii == 0:
            axs[ii].set_ylabel('Trials', labelpad=-10)
    f.text(0.5, 0.02, 'Time from stimulation start (s)', ha='center')
    sns.despine(trim=True)
    plt.subplots_adjust(bottom=0.19, left=0.1, right=0.95)
    
    plt.savefig(join(fig_path, f'{this_rec[:20]}.jpg'), dpi=600)
    plt.close(f)