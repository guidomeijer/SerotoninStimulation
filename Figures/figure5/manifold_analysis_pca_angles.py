# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 11:22:23 2023

By Guido Meijer
"""

import numpy as np
from os.path import join, realpath, dirname, split
from scipy.linalg import subspace_angles
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
DATASET = 'firstMovement_times_backup'
CMAPS = dict({
    'R': 'Reds_r', 'L': 'Purples_r', 'no_opto': 'Oranges_r', 'opto': 'Blues_r',
    'L_opto': 'Blues_r', 'R_opto': 'Oranges_r', 'L_no_opto': 'Purples_r', 'R_no_opto': 'Reds_r'})

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


"""
# Shuffles
pca_lr_shuffle = np.empty((lr_splits.shape[0], N_DIM, 0))
for ii in range(this_dict['n_shuffles']):
    lr_splits_shuf = np.vstack((L_shuf[:, :, ii].T, R_shuf[:, :, ii].T))
    pca_lr_shuffle = np.dstack((pca_lr_shuffle, pca.fit(lr_splits_shuf)))
"""  
                        

"""
# Shuffles
pca_opto_shuffle = np.empty((opto_splits.shape[0], N_DIM, 0))
for ii in range(this_dict['n_shuffles']):
    opto_splits_shuf = np.vstack((opto_shuf[:, :, ii].T, opto_shuf[:, :, ii].T))
    pca_opto_shuffle = np.dstack((pca_opto_shuffle, pca.fit(opto_splits_shuf)))
"""

# Do PCA on L/R choices without opto
lr_splits = np.vstack((L_no_opto.T, R_no_opto.T))
pca_choice = PCA(n_components=N_DIM).fit(lr_splits)

# Do PCA on choice with opto
opto_splits = np.vstack((L_opto.T, R_opto.T))
pca_opto = PCA(n_components=N_DIM).fit(opto_splits)

# Compute principal angles
angles = subspace_angles(pca_choice.components_.T, pca_opto.components_.T)
pc_angles = np.rad2deg(angles)

# Compute dot product between subspace bases
dot_product_matrix = np.dot(pca_choice.components_, pca_opto.components_.T)
dot_product_norm = np.linalg.norm(dot_product_matrix)

# Compute Singular Value Decomposition (SVD)
_, singular_values, _ = np.linalg.svd(dot_product_matrix)

# Compute singular angles
singular_angles = np.rad2deg(np.arccos(np.clip(singular_values, -1, 1)))

dot_norm_shuf = np.empty(this_dict['n_shuffles'])
sv_shuf, sa_shuf = np.empty((this_dict['n_shuffles'], N_DIM)), np.empty((this_dict['n_shuffles'], N_DIM))
pc_angles_shuf = np.empty((this_dict['n_shuffles'], N_DIM))
for ii in range(this_dict['n_shuffles']):
    
    # Fit PCA
    lr_splits_shuf = np.vstack((L_no_opto_shuf[:, :, ii].T, R_no_opto_shuf[:, :, ii].T))
    pca_choice_shuf = PCA(n_components=N_DIM).fit(lr_splits_shuf)
    opto_splits_shuf = np.vstack((L_opto_shuf[:, :, ii].T, R_opto_shuf[:, :, ii].T))
    pca_opto_shuf = PCA(n_components=N_DIM).fit(opto_splits_shuf)
    
    # Compute principal angles
    angles_shuf = subspace_angles(pca_choice_shuf.components_.T, pca_opto_shuf.components_.T)
    pc_angles_shuf[ii, :] = np.rad2deg(angles_shuf)
    
    # Get dot product
    dpm_shuf = np.dot(pca_choice_shuf.components_, pca_opto_shuf.components_.T)
    dot_norm_shuf[ii] = np.linalg.norm(dpm_shuf)
    
    # Compute Singular Value Decomposition (SVD)
    _, this_sv_shuf, _ = np.linalg.svd(dpm_shuf)
    this_sa_shuf = np.rad2deg(np.arccos(np.clip(this_sv_shuf, -1, 1)))
    
    sv_shuf[ii, :] = this_sv_shuf
    sa_shuf[ii, :] = this_sa_shuf
    
print(f'Dot product: {np.round(dot_product_norm, 2)}, conf interval: [{np.round(np.quantile(dot_norm_shuf, 0.025), 2)}, {np.round(np.quantile(dot_norm_shuf, 0.975), 2)}]') 

# %%

f, axs = plt.subplots(1, 4, figsize=(1.75*4, 1.75), dpi=dpi)

axs[0].scatter(np.arange(N_DIM), pc_angles, marker='_', color='red', s=20)
for i in range(N_DIM):
    axs[0].plot([i, i], [np.quantile(pc_angles_shuf[:, i], 0.025), np.quantile(pc_angles_shuf[:, i], 0.975)],
                color='grey', lw=3, solid_capstyle='butt')
axs[0].set(xlabel='PCs', ylabel='Principal angles (deg)')
    
axs[1].scatter(np.arange(N_DIM), singular_values, marker='_', color='red', s=20)
for i in range(N_DIM):
    axs[1].plot([i, i], [np.quantile(sv_shuf[:, i], 0.025), np.quantile(sv_shuf[:, i], 0.975)],
                color='grey', lw=3, solid_capstyle='butt')
axs[1].set(xlabel='PCs', ylabel='Singular values')
       
axs[2].scatter(np.arange(N_DIM), singular_angles, marker='_', color='red', s=20)
for i in range(N_DIM):
    axs[2].plot([i, i], [np.quantile(sa_shuf[:, i], 0.025), np.quantile(sa_shuf[:, i], 0.975)],
                color='grey', lw=3, solid_capstyle='butt')
axs[2].set(xlabel='PCs', ylabel='Singular angles (deg)')


sns.despine(trim=True)
plt.tight_layout()
