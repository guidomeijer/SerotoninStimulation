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
import pandas as pd
import matplotlib as mpl
from stim_functions import (figure_style, paths, load_subjects, high_level_regions,
                            remap, combine_regions)
from sklearn.decomposition import PCA
colors, dpi = figure_style()

N_DIM = 3
CENTER_ON = 'firstMovement_times'
SPLITS = ['L_opto', 'R_opto', 'L_no_opto', 'R_no_opto']
CMAPS = dict({'L_opto': 'Reds_r', 'R_opto': 'Purples_r', 'L_no_opto': 'Oranges_r', 'R_no_opto': 'Blues_r',
              'L_collapsed': 'Reds_r', 'R_collapsed': 'Purples_r', 'no_opto_collapsed': 'Oranges_r', 'opto_collapsed': 'Blues_r'})

# Initialize
pca = PCA(n_components=N_DIM)
colors, dpi = figure_style()

# Get paths
f_path, load_path = paths(save_dir='cache')  # because these data are too large they are not on the repo
fig_path = join(f_path, split(dirname(realpath(__file__)))[-1])

# Load in data
print('Loading in data..')
ses_paths = glob(join(load_path, 'manifold', f'{CENTER_ON}', '*.npy'))
count = 0
dist_df = pd.DataFrame()
for i, ses_path in enumerate(ses_paths):
    pid = split(ses_path)[-1][:-4]
    mani_dict = np.load(ses_path, allow_pickle=True).flat[0]
    if mani_dict['sert-cre'] == 0:
        continue
    n_timepoints = mani_dict['time'].shape[0]
    time_ax = mani_dict['time']
    print(f'Starting {pid} ({i} of {len(ses_paths)})')
   
    # Get Eucledian distances in neural space between opto and no opto
    dist_opto, dist_opto_shuffle = dict(), dict()
  
    # Get Eucledian distance between opto and no opto per time point for L and R choices seperately 
    opto_dist = np.empty(n_timepoints)
    for t in range(n_timepoints):
        l_dist = np.linalg.norm(mani_dict['L_opto'][:, t] - mani_dict['L_no_opto'][:, t])
        r_dist = np.linalg.norm(mani_dict['R_opto'][:, t] - mani_dict['R_no_opto'][:, t])
        opto_dist[t] = np.max([l_dist, r_dist])
    
    # Do the same for shuffle
    opto_dist_shuf = np.empty((n_timepoints, mani_dict['opto_shuffle']['L_opto'].shape[2]))
    for ii in range(mani_dict['opto_shuffle']['L_opto'].shape[2]):
        for t in range(n_timepoints):
            l_dist = np.linalg.norm(mani_dict['opto_shuffle']['L_opto'][:, t, ii]
                                    - mani_dict['opto_shuffle']['L_no_opto'][:, t, ii])
            r_dist = np.linalg.norm(mani_dict['opto_shuffle']['R_opto'][:, t, ii]
                                    - mani_dict['opto_shuffle']['R_no_opto'][:, t, ii])
            opto_dist_shuf[t, ii] = np.max([l_dist, r_dist])
        
    # Get Eucledian distance between choice L and R 
    choice_dist = np.empty(n_timepoints)
    for t in range(n_timepoints):
        this_opto_dist = np.linalg.norm(mani_dict['L_opto'][:, t] - mani_dict['R_opto'][:, t])
        this_no_opto_dist = np.linalg.norm(mani_dict['L_no_opto'][:, t] - mani_dict['R_no_opto'][:, t])
        choice_dist[t] = np.max([this_opto_dist, this_no_opto_dist])
    
    # Do the same for shuffle
    choice_dist_shuf = np.empty((n_timepoints, mani_dict['choice_shuffle']['L_opto'].shape[2]))
    for ii in range(mani_dict['choice_shuffle']['L_opto'].shape[2]):
        for t in range(n_timepoints):
            this_opto_dist = np.linalg.norm(mani_dict['choice_shuffle']['L_opto'][:, t, ii]
                                            - mani_dict['choice_shuffle']['R_opto'][:, t, ii])
            this_no_opto_dist = np.linalg.norm(mani_dict['choice_shuffle']['L_no_opto'][:, t, ii]
                                               - mani_dict['choice_shuffle']['R_no_opto'][:, t, ii])
            choice_dist_shuf[t, ii] = np.max([this_opto_dist, this_no_opto_dist])
                   
    # Do PCA
    all_splits = np.vstack((mani_dict['L_opto'].T, mani_dict['R_opto'].T,
                            mani_dict['L_no_opto'].T, mani_dict['R_no_opto'].T))
    if all_splits.shape[1] < N_DIM:
        continue
    pca_fit = pca.fit_transform(all_splits) 
           
    # Do PCA for shuffles
    pca_shuffle = np.empty((all_splits.shape[0], N_DIM, 0))
    for ii in range(mani_dict['n_shuffles']):
        all_splits = np.vstack((mani_dict['opto_shuffle']['L_opto'][:, :, ii].T,
                                mani_dict['opto_shuffle']['R_opto'][:, :, ii].T,
                                mani_dict['opto_shuffle']['L_no_opto'][:, :, ii].T,
                                mani_dict['opto_shuffle']['R_no_opto'][:, :, ii].T))
        if np.sum(np.isnan(all_splits)) == 0:
            pca_shuffle = np.dstack((pca_shuffle, pca.fit_transform(all_splits)))
        
    # Get index to which split the PCA belongs to
    split_ids = np.concatenate((['L_opto'] * n_timepoints, ['R_opto'] * n_timepoints,
                                ['L_no_opto'] * n_timepoints, ['R_no_opto'] * n_timepoints))
            
    # Calculate dot product between opto and stim vectors in PCA space
    # Collapse pca onto choice and opto dimensions
    dot_pca, dot_pca_shuffle = dict(), dict()
    angle_pca, angle_pca_shuffle = dict(), dict()
    p_value = dict()
    
    pca_l_col = (pca_fit[split_ids == 'L_opto'] + pca_fit[split_ids == 'L_no_opto']) / 2
    pca_r_col = (pca_fit[split_ids == 'R_opto'] + pca_fit[split_ids == 'R_no_opto']) / 2
    pca_opto_col = (pca_fit[split_ids == 'L_opto'] + pca_fit[split_ids == 'R_opto']) / 2
    pca_no_opto_col = (pca_fit[split_ids == 'L_no_opto'] + pca_fit[split_ids == 'R_no_opto']) / 2

    # Get the dot product and angle between the two vectors
    dot_pca, angle_pca = np.empty(n_timepoints), np.empty(n_timepoints)
    p_value = np.empty(n_timepoints)
    for t in range(n_timepoints):
        choice_vec = pca_l_col[t, :] - pca_r_col[t, :]
        opto_vec = pca_opto_col[t, :] - pca_no_opto_col[t, :]
        dot_prod = np.dot(choice_vec / np.linalg.norm(choice_vec),
                          opto_vec / np.linalg.norm(opto_vec))
        angle_pca[t] = np.degrees(np.arccos(
            dot_prod / (np.linalg.norm(choice_vec) * np.linalg.norm(opto_vec))))
        dot_pca[t] = np.abs(dot_prod)
        
        # Determine whether the dot product is more orthogonal than expected by chance
        z_score = dot_prod / (1 / np.sqrt(choice_vec.shape[0]))
        #z_score = (angle_pca[region][t] - 90) / (1 / np.sqrt(choice_vec.shape[0]))
        p_value[t] = 1 - (stats.norm.sf(abs(z_score)) * 2)
        
    # Do the same for all the shuffles
    dot_pca_shuffle = np.empty((n_timepoints, pca_shuffle.shape[2]))
    angle_pca_shuffle = np.empty((n_timepoints, pca_shuffle.shape[2]))
    for ii in range(pca_shuffle.shape[2]):
        pca_l_col = (pca_shuffle[split_ids == 'L_opto', :, ii]
                     + pca_shuffle[split_ids == 'L_no_opto', :, ii]) / 2
        pca_r_col = (pca_shuffle[split_ids == 'R_opto', :, ii]
                     + pca_shuffle[split_ids == 'R_no_opto', :, ii]) / 2
        pca_opto_col = (pca_shuffle[split_ids == 'L_opto', :, ii]
                        + pca_shuffle[split_ids == 'R_opto', :, ii]) / 2
        pca_no_opto_col = (pca_shuffle[split_ids == 'L_no_opto', :, ii]
                           + pca_shuffle[split_ids == 'R_no_opto', :, ii]) / 2
        
        for t in range(n_timepoints):
            choice_vec = pca_l_col[t, :] - pca_r_col[t, :]
            opto_vec = pca_opto_col[t, :] - pca_no_opto_col[t, :]
            dot_prod = np.dot(choice_vec / np.linalg.norm(choice_vec),
                              opto_vec / np.linalg.norm(opto_vec))
            angle_pca_shuffle[t, ii] = np.degrees(np.arccos(
                dot_prod / (np.linalg.norm(choice_vec) * np.linalg.norm(opto_vec))))
            dot_pca_shuffle[t, ii] = np.abs(dot_prod)
            
    # Get distance in PCA space
    choice_dist_pca = np.empty(n_timepoints)
    for t in range(n_timepoints):
        this_opto_dist = np.linalg.norm(pca_fit[split_ids == 'L_opto'][t, :]
                                        - pca_fit[split_ids == 'R_opto'][t, :])
        this_no_opto_dist = np.linalg.norm(pca_fit[split_ids == 'L_no_opto'][t, :]
                                           - pca_fit[split_ids == 'R_no_opto'][t, :])
        choice_dist_pca[t] = np.max([this_opto_dist, this_no_opto_dist])
        
    opto_dist_pca = np.empty(n_timepoints)
    for t in range(n_timepoints):
        this_l_dist = np.linalg.norm(pca_fit[split_ids == 'L_opto'][t, :]
                                     - pca_fit[split_ids == 'L_no_opto'][t, :])
        this_r_dist = np.linalg.norm(pca_fit[split_ids == 'R_opto'][t, :]
                                     - pca_fit[split_ids == 'R_no_opto'][t, :])
        opto_dist_pca[t] = np.max([this_l_dist, this_r_dist])
        
    # Add to dataframe
    dist_df = pd.concat((dist_df, pd.DataFrame(data={
        'opto_dist': opto_dist, 'choice_dist': choice_dist, 'dot_pca': dot_pca,
        'choice_dist_pca': choice_dist_pca, 'opto_dist_pca': opto_dist_pca,
        'time': time_ax, 'session': pid})))

    
    # Plot PCA trajectories for this session
    fig = plt.figure(figsize=(1.75, 1.75), dpi=dpi)
    ax = fig.add_subplot(projection='3d')
    ax.view_init(elev=-160, azim=220)
    for sp in SPLITS:
        cmap = mpl.colormaps.get_cmap(CMAPS[sp])
        col = [cmap((n_timepoints - p) / n_timepoints) for p in range(n_timepoints)]
        ax.plot(pca_fit[split_ids == sp, 0],
                pca_fit[split_ids == sp, 1],
                pca_fit[split_ids == sp, 2],
                color=col[len(col) // 2], linewidth=1, alpha=0.5, zorder=0)
        ax.scatter(pca_fit[split_ids == sp, 0],
                   pca_fit[split_ids == sp, 1],
                   pca_fit[split_ids == sp, 2],
                   color=col, edgecolors=col, s=10, depthshade=False, zorder=1)
    

# %%

f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(1.75*3, 1.75), dpi=dpi)
sns.lineplot(data=dist_df, x='time', y='opto_dist', ax=ax1, errorbar='se',
             err_kws={'lw': 0})
sns.lineplot(data=dist_df, x='time', y='choice_dist', ax=ax1, errorbar='se',
             err_kws={'lw': 0})


sns.lineplot(data=dist_df, x='time', y='opto_dist_pca', ax=ax2, errorbar='se',
             err_kws={'lw': 0})
sns.lineplot(data=dist_df, x='time', y='choice_dist_pca', ax=ax2, errorbar='se',
             err_kws={'lw': 0})

sns.lineplot(data=dist_df, x='time', y='dot_pca', ax=ax3, 
             err_kws={'lw': 0})

