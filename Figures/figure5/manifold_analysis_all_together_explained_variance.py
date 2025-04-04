# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 11:22:23 2023

By Guido Meijer
"""

import numpy as np
from os.path import join, realpath, dirname, split
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from scipy import stats
from matplotlib.patches import Rectangle
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import matplotlib as mpl
from stim_functions import (figure_style, paths, load_subjects, high_level_regions,
                            remap, combine_regions)
from sklearn.decomposition import PCA
colors, dpi = figure_style()

N_DIM = 5
LAST_TIMEPOINTS = 24
DATASET = 'firstMovement_times'
CMAPS = dict({
    'L': 'Reds_r', 'R': 'Purples_r', 'no_opto': 'Oranges_r', 'opto': 'Blues_r',
    'L_opto': 'Reds_r', 'R_opto': 'Purples_r', 'L_no_opto': 'Oranges_r', 'R_no_opto': 'Blues_r'})

# Initialize
colors, dpi = figure_style()

# Get paths
f_path, load_path = paths(save_dir='cache')  # because these data are too large they are not on the repo
fig_path = join(f_path, split(dirname(realpath(__file__)))[-1])

# Load in data
print('Loading in data..')
regions = np.array([])
ses_paths = glob(join(load_path, 'manifold', f'{DATASET}', '*.npy'))
for i, ses_path in enumerate(ses_paths):
    this_dict = np.load(ses_path, allow_pickle=True).flat[0]
    if this_dict['sert-cre'] == 0:
        continue
    n_timepoints = LAST_TIMEPOINTS
    time_ax = this_dict['time'][-LAST_TIMEPOINTS:]
    if i == 0:
        L = np.empty((0, LAST_TIMEPOINTS))
        R = np.empty((0, LAST_TIMEPOINTS))
        opto = np.empty((0, LAST_TIMEPOINTS))
        no_opto = np.empty((0, LAST_TIMEPOINTS))
        L_opto = np.empty((0, LAST_TIMEPOINTS))
        R_opto = np.empty((0, LAST_TIMEPOINTS))
        L_no_opto = np.empty((0, LAST_TIMEPOINTS))
        R_no_opto = np.empty((0, LAST_TIMEPOINTS))

        opto_shuf = np.empty((0, LAST_TIMEPOINTS, this_dict['n_shuffles']))
        no_opto_shuf = np.empty((0, LAST_TIMEPOINTS, this_dict['n_shuffles']))
        L_shuf = np.empty((0, LAST_TIMEPOINTS, this_dict['n_shuffles']))
        R_shuf = np.empty((0, LAST_TIMEPOINTS, this_dict['n_shuffles']))
        L_opto_shuf = np.empty((0, LAST_TIMEPOINTS, this_dict['n_shuffles']))
        R_opto_shuf = np.empty((0, LAST_TIMEPOINTS, this_dict['n_shuffles']))
        L_no_opto_shuf = np.empty((0, LAST_TIMEPOINTS, this_dict['n_shuffles']))
        R_no_opto_shuf = np.empty((0, LAST_TIMEPOINTS, this_dict['n_shuffles']))

    L = np.vstack((L, this_dict['L'][:,-LAST_TIMEPOINTS:]))
    R = np.vstack((R, this_dict['R'][:,-LAST_TIMEPOINTS:]))
    opto = np.vstack((opto, this_dict['opto'][:,-LAST_TIMEPOINTS:]))
    no_opto = np.vstack((no_opto, this_dict['no_opto'][:,-LAST_TIMEPOINTS:]))
    L_opto = np.vstack((L_opto, this_dict['L_opto'][:,-LAST_TIMEPOINTS:]))
    R_opto = np.vstack((R_opto, this_dict['R_opto'][:,-LAST_TIMEPOINTS:]))
    L_no_opto = np.vstack((L_no_opto, this_dict['L_no_opto'][:,-LAST_TIMEPOINTS:]))
    R_no_opto = np.vstack((R_no_opto, this_dict['R_no_opto'][:,-LAST_TIMEPOINTS:]))

    L_shuf = np.vstack((L_shuf, this_dict['shuffle']['L'][:,-LAST_TIMEPOINTS:]))
    R_shuf = np.vstack((R_shuf, this_dict['shuffle']['R'][:,-LAST_TIMEPOINTS:]))
    opto_shuf = np.vstack((opto_shuf, this_dict['shuffle']['opto'][:,-LAST_TIMEPOINTS:]))
    no_opto_shuf = np.vstack((no_opto_shuf, this_dict['shuffle']['no_opto'][:,-LAST_TIMEPOINTS:]))
    L_opto_shuf = np.vstack((L_opto_shuf, this_dict['shuffle']['L_opto'][:,-LAST_TIMEPOINTS:]))
    R_opto_shuf = np.vstack((R_opto_shuf, this_dict['shuffle']['R_opto'][:,-LAST_TIMEPOINTS:]))
    L_no_opto_shuf = np.vstack((L_no_opto_shuf, this_dict['shuffle']['L_no_opto'][:,-LAST_TIMEPOINTS:]))
    R_no_opto_shuf = np.vstack((R_no_opto_shuf, this_dict['shuffle']['R_no_opto'][:,-LAST_TIMEPOINTS:]))

# %%

# Fit PCA on choices without stimulation
all_splits = np.vstack((L_no_opto.T, R_no_opto.T))
pca_choice = PCA(n_components=N_DIM).fit(all_splits)

# Project both opto onto no opto
proj_opto = pca_choice.transform(np.vstack((L_opto.T, R_opto.T)))

# Compute how much variance the stim trials occupy along non-stim PCs
opto_var_no_opto_pcs = np.var(proj_opto, axis=0)

# Normalize by total variance in stim data (so it sums to 1)
opto_var = opto_var_no_opto_pcs / np.sum(np.var(np.vstack((L_opto.T, R_opto.T)), axis=0))

# Shuffles
opto_var_shuffle = np.empty((this_dict['n_shuffles'], N_DIM))
for ii in range(this_dict['n_shuffles']):
    
    this_proj_opto = pca_choice.transform(np.vstack((L_opto_shuf[:, :, ii].T, R_opto_shuf[:, :, ii].T)))
    this_opto_var_no_opto_pcs = np.var(this_proj_opto, axis=0)
    this_opto_var_fraction = this_opto_var_no_opto_pcs / np.sum(np.var(
        np.vstack((L_opto_shuf[:, :, ii].T, R_opto_shuf[:, :, ii].T)), axis=0))
    opto_var_shuffle[ii, :] = this_opto_var_fraction
    
# %%

f, ax = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)

for ii in range(N_DIM):
    ax.plot([ii+1, ii+1], [np.quantile(opto_var_shuffle[:, ii], 0.025), np.quantile(opto_var_shuffle[:, ii], 0.975)],
                lw=5, color='grey', solid_capstyle='butt')
ax.scatter(np.arange(opto_var.shape[0])+1, opto_var, marker='_', color='red', s=30)  
ax.text(1, 0.38, '***', ha='center', va='center', fontsize=8) 
ax.text(2, 0.08, '***', ha='center', va='center', fontsize=8)
ax.set(ylabel='Explained variance ratio', xlabel='Principal components', xticks=[1, 2, 3, 4, 5],
           xlim=[0.5, 5.5], yticks=[0, 0.2, 0.4, 0.6])

sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, 'variance_5HT_dims.pdf'))
