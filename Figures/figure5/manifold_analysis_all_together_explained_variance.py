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

N_DIM = 20
N_SHUFFLES = 500
DATASET = 'firstMovement_times'
CMAPS = dict({
    'L': 'Reds_r', 'R': 'Purples_r', 'no_opto': 'Oranges_r', 'opto': 'Blues_r',
    'L_opto': 'Reds_r', 'R_opto': 'Purples_r', 'L_no_opto': 'Oranges_r', 'R_no_opto': 'Blues_r'})

# Initialize
#pca = PCA(n_components=N_DIM, svd_solver='randomized', random_state=42)
pca = PCA(n_components=N_DIM)
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
    if DATASET == 'timewarped':
        n_timepoints = this_dict['n_bins']
        time_ax = np.arange(n_timepoints)
    else:
        n_timepoints = this_dict['time'].shape[0]
        time_ax = this_dict['time']
    if i == 0:
        L = np.empty((0, n_timepoints))
        R = np.empty((0, n_timepoints))
        opto = np.empty((0, n_timepoints))
        no_opto = np.empty((0, n_timepoints))
        L_opto = np.empty((0, n_timepoints))
        R_opto = np.empty((0, n_timepoints))
        L_no_opto = np.empty((0, n_timepoints))
        R_no_opto = np.empty((0, n_timepoints))

        opto_shuf = np.empty((0, n_timepoints, this_dict['n_shuffles']))
        no_opto_shuf = np.empty((0, n_timepoints, this_dict['n_shuffles']))
        L_shuf = np.empty((0, n_timepoints, this_dict['n_shuffles']))
        R_shuf = np.empty((0, n_timepoints, this_dict['n_shuffles']))
        L_opto_shuf = np.empty((0, n_timepoints, this_dict['n_shuffles']))
        R_opto_shuf = np.empty((0, n_timepoints, this_dict['n_shuffles']))
        L_no_opto_shuf = np.empty((0, n_timepoints, this_dict['n_shuffles']))
        R_no_opto_shuf = np.empty((0, n_timepoints, this_dict['n_shuffles']))

    L = np.vstack((L, this_dict['L']))
    R = np.vstack((R, this_dict['R']))
    opto = np.vstack((opto, this_dict['opto']))
    no_opto = np.vstack((no_opto, this_dict['no_opto']))
    L_opto = np.vstack((L_opto, this_dict['L_opto']))
    R_opto = np.vstack((R_opto, this_dict['R_opto']))
    L_no_opto = np.vstack((L_no_opto, this_dict['L_no_opto']))
    R_no_opto = np.vstack((R_no_opto, this_dict['R_no_opto']))

    L_shuf = np.vstack((L_shuf, this_dict['shuffle']['L']))
    R_shuf = np.vstack((R_shuf, this_dict['shuffle']['R']))
    opto_shuf = np.vstack((opto_shuf, this_dict['shuffle']['opto']))
    no_opto_shuf = np.vstack((no_opto_shuf, this_dict['shuffle']['no_opto']))
    L_opto_shuf = np.vstack((L_opto_shuf, this_dict['shuffle']['L_opto']))
    R_opto_shuf = np.vstack((R_opto_shuf, this_dict['shuffle']['R_opto']))
    L_no_opto_shuf = np.vstack((L_no_opto_shuf, this_dict['shuffle']['L_no_opto']))
    R_no_opto_shuf = np.vstack((R_no_opto_shuf, this_dict['shuffle']['R_no_opto']))

# %%
# Do PCA on L/R choices
lr_splits = np.vstack((L.T, R.T))
pca_fit_lr = pca.fit(lr_splits)
          
# Do PCA on opto / no opto
opto_splits = np.vstack((opto.T, no_opto.T))
pca_fit_opto = pca.fit(opto_splits)

# Fit PCA on choices without stimulation
all_splits = np.vstack((L_no_opto.T, R_no_opto.T))
pca_fit_no_opto = pca.fit(all_splits)

# Project both opto and no opto choices onto no opto
proj_opto = pca.transform(np.vstack((L_opto.T, R_opto.T)))

# Compute how much variance the stim trials occupy along non-stim PCs
opto_var_no_opto_pcs = np.var(proj_opto, axis=0)

# Normalize by total variance in stim data (so it sums to 1)
opto_var = opto_var_no_opto_pcs / np.sum(np.var(np.vstack((L_opto.T, R_opto.T)), axis=0))

# Shuffles
opto_var_shuffle = np.empty((this_dict['n_shuffles'], N_DIM))
for ii in range(this_dict['n_shuffles']):
    
    this_proj_opto = pca.transform(np.vstack((L_opto_shuf[:, :, ii].T, R_opto_shuf[:, :, ii].T)))
    this_opto_var_no_opto_pcs = np.var(this_proj_opto, axis=0)
    this_opto_var_fraction = this_opto_var_no_opto_pcs / np.sum(np.var(
        np.vstack((L_opto_shuf[:, :, ii].T, R_opto_shuf[:, :, ii].T)), axis=0))
    opto_var_shuffle[ii, :] = this_opto_var_fraction
    
# %%

f, axs = plt.subplots(1, 2, figsize=(1.75*2, 1.75), dpi=dpi)

axs[0].plot(np.arange(1, N_DIM+1), np.cumsum(pca_fit_lr.explained_variance_ratio_), marker='o')
axs[0].set(ylabel='Total explained variance', xlabel='Principal components', xticks=[0, 10, 20],
           yticks=[0, 0.2, 0.4, 0.6, 0.8, 1], yticklabels=[0, 0.2, 0.4, 0.6, 0.8, 1])
  
#violin_parts = axs[1].violinplot(opto_var_shuffle, showextrema=False)
#for pc in violin_parts['bodies']:
#    pc.set_facecolor([0.2, 0.2, 0.2])
for ii in range(N_DIM):
    axs[1].plot([ii+1, ii+1], [np.quantile(opto_var_shuffle[:, ii], 0.05), np.quantile(opto_var_shuffle[:, ii], 0.95)],
                lw=5, color='grey', solid_capstyle='butt')
axs[1].scatter(np.arange(opto_var.shape[0])+1, opto_var, marker='_', color='red', s=30)  
axs[1].text(1, 0.38, '***', ha='center', va='center', fontsize=8) 
axs[1].text(2, 0.08, '***', ha='center', va='center', fontsize=8)
axs[1].text(3, 0.08, '***', ha='center', va='center', fontsize=8)
axs[1].set(ylabel='Explained variance ratio', xlabel='Principal components', xticks=[1, 2, 3, 4, 5],
           xlim=[0.5, 5.5], yticks=[0, 0.1, 0.2, 0.3, 0.4])

sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, 'variance_5HT_dims.pdf'))
