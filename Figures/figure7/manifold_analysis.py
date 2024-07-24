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
from matplotlib.patches import Rectangle
import pandas as pd
import matplotlib as mpl
from stim_functions import (figure_style, paths, load_subjects, high_level_regions,
                            remap, combine_regions)
from sklearn.decomposition import PCA

N_DIM = 10
SPLIT_ON = 'stim_side'
#SPLIT_ON = 'choice'
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
regions = np.array([])
ses_paths = glob(join(load_path, 'manifold', f'{SPLIT_ON}', '*.npy'))
for i, ses_path in enumerate(ses_paths):
    this_dict = np.load(ses_path, allow_pickle=True).flat[0]
    n_timepoints = this_dict['time'].shape[0]
    time_ax = this_dict['time']
    if i == 0:        
        L_opto = np.empty((0, n_timepoints))
        R_opto = np.empty((0, n_timepoints))
        L_no_opto = np.empty((0, n_timepoints))
        R_no_opto = np.empty((0, n_timepoints))
        
        L_opto_shuf = np.empty((0, n_timepoints, this_dict['n_shuffles']))
        R_opto_shuf = np.empty((0, n_timepoints, this_dict['n_shuffles']))
        L_no_opto_shuf = np.empty((0, n_timepoints, this_dict['n_shuffles']))
        R_no_opto_shuf = np.empty((0, n_timepoints, this_dict['n_shuffles']))
        
        L_opto_choice_shuf = np.empty((0, n_timepoints, this_dict['n_shuffles']))
        R_opto_choice_shuf = np.empty((0, n_timepoints, this_dict['n_shuffles']))
        L_no_opto_choice_shuf = np.empty((0, n_timepoints, this_dict['n_shuffles']))
        R_no_opto_choice_shuf = np.empty((0, n_timepoints, this_dict['n_shuffles']))
        
    L_opto = np.vstack((L_opto, this_dict['L_opto']))
    R_opto = np.vstack((R_opto, this_dict['R_opto']))
    L_no_opto = np.vstack((L_no_opto, this_dict['L_no_opto']))
    R_no_opto = np.vstack((R_no_opto, this_dict['R_no_opto']))
    
    L_opto_shuf = np.vstack((L_opto_shuf, this_dict['opto_shuffle']['L_opto']))
    R_opto_shuf = np.vstack((R_opto_shuf, this_dict['opto_shuffle']['R_opto']))
    L_no_opto_shuf = np.vstack((L_no_opto_shuf, this_dict['opto_shuffle']['L_no_opto']))
    R_no_opto_shuf = np.vstack((R_no_opto_shuf, this_dict['opto_shuffle']['R_no_opto']))
    
    L_opto_choice_shuf = np.vstack((L_opto_choice_shuf, this_dict['choice_shuffle']['L_opto']))
    R_opto_choice_shuf = np.vstack((R_opto_choice_shuf, this_dict['choice_shuffle']['R_opto']))
    L_no_opto_choice_shuf = np.vstack((L_no_opto_choice_shuf, this_dict['choice_shuffle']['L_no_opto']))
    R_no_opto_choice_shuf = np.vstack((R_no_opto_choice_shuf, this_dict['choice_shuffle']['R_no_opto']))
    
    #regions = np.concatenate((regions, high_level_regions(this_dict['region'], input_atlas='Beryl')))
    regions = np.concatenate((regions, combine_regions(remap(this_dict['region']))))

# Get Eucledian distances in neural space between opto and no opto
print('Calculating Eucledian distances..')
dist_opto, dist_opto_shuffle = dict(), dict()
for r, region in enumerate(np.unique(regions)):
    if region == 'root':
        continue
    
    # Get Eucledian distance between opto and no opto per time point for L and R choices seperately 
    this_dist = np.empty(n_timepoints)
    for t in range(n_timepoints):
        l_dist = np.linalg.norm(L_opto[regions == region, t] - L_no_opto[regions == region, t])
        r_dist = np.linalg.norm(R_opto[regions == region, t] - R_no_opto[regions == region, t])
        this_dist[t] = np.mean([l_dist, r_dist])
    dist_opto[region] = this_dist
    
    # Do the same for shuffle
    this_dist = np.empty((n_timepoints, L_opto_shuf.shape[2]))
    for ii in range(L_opto_shuf.shape[2]):
        for t in range(n_timepoints):
            l_dist = np.linalg.norm(L_opto_choice_shuf[regions == region, t, ii]
                                    - L_no_opto_choice_shuf[regions == region, t, ii])
            r_dist = np.linalg.norm(R_opto_choice_shuf[regions == region, t, ii]
                                    - R_no_opto_choice_shuf[regions == region, t, ii])
            this_dist[t, ii] = np.mean([l_dist, r_dist])
    dist_opto_shuffle[region] = this_dist

# Get Eucledian distances in neural space between choice left and right
dist_choice, dist_choice_shuffle = dict(), dict()
for r, region in enumerate(np.unique(regions)):
    if region == 'root':
        continue
    
    # Get Eucledian distance between choice L and R 
    this_dist = np.empty(n_timepoints)
    for t in range(n_timepoints):
        opto_dist = np.linalg.norm(L_opto[regions == region, t] - R_opto[regions == region, t])
        no_opto_dist = np.linalg.norm(L_no_opto[regions == region, t] - R_no_opto[regions == region, t])
        this_dist[t] = np.mean([opto_dist, no_opto_dist])
    dist_choice[region] = this_dist
    
    # Do the same for shuffle
    this_dist = np.empty((n_timepoints, L_opto_shuf.shape[2]))
    for ii in range(L_opto_shuf.shape[2]):
        for t in range(n_timepoints):
            opto_dist = np.linalg.norm(L_opto_choice_shuf[regions == region, t, ii]
                                       - R_opto_choice_shuf[regions == region, t, ii])
            no_opto_dist = np.linalg.norm(L_no_opto_choice_shuf[regions == region, t, ii]
                                          - R_no_opto_choice_shuf[regions == region, t, ii])
            this_dist[t, ii] = np.mean([opto_dist, no_opto_dist])
    dist_choice_shuffle[region] = this_dist
               
# Do PCA
print('Fitting PCA..')
pca_fit, pca_shuffle = dict(), dict()
for r, region in enumerate(np.unique(regions)):
    if region == 'root':
        continue
    print(f'Starting fits for {region}')
   
    # Do PCA on all splits simultaneously to get them in the same PCA space
    all_splits = np.vstack((L_opto[regions == region].T, R_opto[regions == region].T,
                            L_no_opto[regions == region].T, R_no_opto[regions == region].T))
    if all_splits.shape[1] < N_DIM:
        continue
    pca_fit[region] = pca.fit_transform(all_splits) 
           
    # Do PCA for shuffles
    this_pca = np.empty((all_splits.shape[0], N_DIM, 0))
    for ii in range(this_dict['n_shuffles']):
        all_splits = np.vstack((L_opto_shuf[regions == region][:, :, ii].T,
                                R_opto_shuf[regions == region][:, :, ii].T,
                                L_no_opto_shuf[regions == region][:, :, ii].T,
                                R_no_opto_shuf[regions == region][:, :, ii].T))
        if np.sum(np.isnan(all_splits)) == 0:
            this_pca = np.dstack((this_pca, pca.fit_transform(all_splits)))
    pca_shuffle[region] = this_pca
    
# Get index to which split the PCA belongs to
split_ids = np.concatenate((['L_opto'] * n_timepoints, ['R_opto'] * n_timepoints,
                            ['L_no_opto'] * n_timepoints, ['R_no_opto'] * n_timepoints))

"""
# Calculate dot product between opto and stim vectors
# Collapse pca onto choice and opto dimensions
dot_prod, dot_shuffle = dict(), dict()
for r, region in enumerate(pca_fit.keys()):

    # Get the dot product between the two vectors
    dot_prod[region] = np.empty(n_timepoints)
    for t in range(n_timepoints):
        choice_L_vec = (pca_fit[region][split_ids == 'L_no_opto'][t, :]
                        - pca_fit[region][split_ids == 'R_no_opto'][t, :])
        opto_L_vec = (pca_fit[region][split_ids == 'L_no_opto'][t, :]
                      - pca_fit[region][split_ids == 'L_opto'][t, :])
        dot_prod_L = np.abs(np.dot(choice_L_vec, opto_L_vec))
        
        choice_R_vec = (pca_fit[region][split_ids == 'R_no_opto'][t, :]
                        - pca_fit[region][split_ids == 'L_no_opto'][t, :])
        opto_R_vec = (pca_fit[region][split_ids == 'R_no_opto'][t, :]
                      - pca_fit[region][split_ids == 'R_opto'][t, :])
        dot_prod_R = np.abs(np.dot(choice_R_vec, opto_R_vec))
        
        #dot_prod[region][t] = np.mean([dot_prod_L, dot_prod_R])
        dot_prod[region][t] = dot_prod_L
                
    # Do the same for all the shuffles
    dot_shuffle[region] = np.empty((n_timepoints, pca_shuffle[region].shape[2]))
    for ii in range(pca_shuffle[region].shape[2]):
        
        for t in range(n_timepoints):
            choice_L_vec = (pca_shuffle[region][split_ids == 'L_no_opto'][t, :, ii]
                            - pca_shuffle[region][split_ids == 'R_no_opto'][t, :, ii])
            opto_L_vec = (pca_shuffle[region][split_ids == 'L_no_opto'][t, :, ii]
                          - pca_shuffle[region][split_ids == 'L_opto'][t, :, ii])
            dot_prod_L = np.abs(np.dot(choice_L_vec, opto_L_vec))
            
            choice_R_vec = (pca_shuffle[region][split_ids == 'R_no_opto'][t, :, ii]
                            - pca_shuffle[region][split_ids == 'L_no_opto'][t, :, ii])
            opto_R_vec = (pca_shuffle[region][split_ids == 'R_no_opto'][t, :, ii]
                          - pca_shuffle[region][split_ids == 'R_opto'][t, :, ii])
            dot_prod_R = np.abs(np.dot(choice_R_vec, opto_R_vec))
            
            #dot_shuffle[region][t, ii] = np.mean([dot_prod_L, dot_prod_R])
            dot_shuffle[region][t, ii] = dot_prod_L
"""
        
# Calculate dot product between opto and stim vectors
# Collapse pca onto choice and opto dimensions
dot_prod, dot_shuffle = dict(), dict()
for r, region in enumerate(pca_fit.keys()):
    pca_l_col = (pca_fit[region][split_ids == 'L_opto'] + pca_fit[region][split_ids == 'L_no_opto']) / 2
    pca_r_col = (pca_fit[region][split_ids == 'R_opto'] + pca_fit[region][split_ids == 'R_no_opto']) / 2
    pca_opto_col = (pca_fit[region][split_ids == 'L_opto'] + pca_fit[region][split_ids == 'R_opto']) / 2
    pca_no_opto_col = (pca_fit[region][split_ids == 'L_no_opto'] + pca_fit[region][split_ids == 'R_no_opto']) / 2

    # Get the dot product between the two vectors
    dot_prod[region] = np.empty(n_timepoints)
    for t in range(n_timepoints):
        choice_vec = pca_l_col[t, :] - pca_r_col[t, :]
        opto_vec = pca_opto_col[t, :] - pca_no_opto_col[t, :]
        dot_prod[region][t] = np.abs(np.dot(choice_vec, opto_vec))
        
    # Do the same for all the shuffles
    dot_shuffle[region] = np.empty((n_timepoints, pca_shuffle[region].shape[2]))
    for ii in range(pca_shuffle[region].shape[2]):
        pca_l_col = (pca_shuffle[region][split_ids == 'L_opto', :, ii]
                     + pca_shuffle[region][split_ids == 'L_no_opto', :, ii]) / 2
        pca_r_col = (pca_shuffle[region][split_ids == 'R_opto', :, ii]
                     + pca_shuffle[region][split_ids == 'R_no_opto', :, ii]) / 2
        pca_opto_col = (pca_shuffle[region][split_ids == 'L_opto', :, ii]
                        + pca_shuffle[region][split_ids == 'R_opto', :, ii]) / 2
        pca_no_opto_col = (pca_shuffle[region][split_ids == 'L_no_opto', :, ii]
                           + pca_shuffle[region][split_ids == 'R_no_opto', :, ii]) / 2
        
        for t in range(n_timepoints):
            choice_vec = pca_l_col[t, :] - pca_r_col[t, :]
            opto_vec = pca_opto_col[t, :] - pca_no_opto_col[t, :]
            dot_shuffle[region][t, ii] = np.abs(np.dot(choice_vec, opto_vec))
                
                   
# %% Plot Eucledian distance of all regions
f, axs = plt.subplots(1, len(dist_opto.keys()), figsize=(7, 1.75), dpi=dpi)
for r, region in enumerate(dist_opto.keys()):
    axs[r].fill_between(time_ax,
                        np.nanquantile(dist_opto_shuffle[region], 0.05, axis=1),
                        np.nanquantile(dist_opto_shuffle[region], 0.95, axis=1),
                        color='grey', alpha=0.25, lw=0, zorder=1)
    axs[r].plot(time_ax, dist_opto[region], marker='o', zorder=2, ms=1.5)
    axs[r].plot([0, 0], axs[r].get_ylim(), ls='--', color='grey', lw=0.75, zorder=0)
    axs[r].set(title=f'{region}', yticks=[], xlabel='Time to choice (s)')
sns.despine(trim=True, left=True)
plt.tight_layout()
plt.savefig(join(fig_path, 'eu_dist_opto.pdf'))

# %% Eucledian distance choice
f, axs = plt.subplots(1, len(dist_opto.keys()), figsize=(7, 1.75), dpi=dpi)
for r, region in enumerate(dist_opto.keys()):
    axs[r].fill_between(time_ax,
                        np.nanquantile(dist_choice_shuffle[region], 0.05, axis=1),
                        np.nanquantile(dist_choice_shuffle[region], 0.95, axis=1),
                        color='grey', alpha=0.25, lw=0, zorder=1)
    axs[r].plot(time_ax, dist_choice[region], marker='o', zorder=2, ms=1.5)
    axs[r].plot([0, 0], axs[r].get_ylim(), ls='--', color='grey', lw=0.75, zorder=0)
    axs[r].set(title=f'{region}', yticks=[], xlabel='Time to choice (s)')
sns.despine(trim=True, left=True)
plt.tight_layout()
plt.savefig(join(fig_path, 'eu_dist_choice.pdf'))

# %% Plot PCA of all regions
fig = plt.figure(figsize=(len(dist_opto.keys()), 1.75), dpi=dpi)
axs = []
gs = fig.add_gridspec(1, len(dist_opto.keys()))
for r, region in enumerate(pca_fit.keys()):
    axs.append(fig.add_subplot(gs[0, r], projection='3d'))
    axs[r].view_init(elev=-140, azim=200)
    for sp in SPLITS:
        cmap = mpl.colormaps.get_cmap(CMAPS[sp])
        col = [cmap((n_timepoints - p) / n_timepoints) for p in range(n_timepoints)]
        axs[r].plot(pca_fit[region][split_ids == sp, 0],
                    pca_fit[region][split_ids == sp, 1],
                    pca_fit[region][split_ids == sp, 2],
                    color=col[len(col) // 2], linewidth=1, alpha=0.5, zorder=0)
        axs[r].scatter(pca_fit[region][split_ids == sp, 0],
                       pca_fit[region][split_ids == sp, 1],
                       pca_fit[region][split_ids == sp, 2],
                       color=col, edgecolors=col, s=10, depthshade=False, zorder=1)
    axs[r].grid('off')
    axs[r].axis('off')
    axs[r].set_title(f'{region}')
plt.tight_layout()
plt.savefig(join(fig_path, 'pca_all_regions.pdf'))



# %%
f, axs = plt.subplots(1, len(dist_opto.keys()), figsize=(7, 1.75), dpi=dpi, sharey=True)
for r, region in enumerate(dot_prod.keys()):
    axs[r].fill_between(time_ax,
                        np.quantile(dot_shuffle[region], 0.05, axis=1),
                        np.quantile(dot_shuffle[region], 0.95, axis=1),
                        color='grey', alpha=0.25, lw=0)
    axs[r].plot(time_ax, dot_prod[region], marker='o', ms=1.5)
    #axs[r].plot([0, 0], axs[r].get_ylim(), ls='--', color='grey', lw=0.75, zorder=0)
    axs[r].set(title=f'{region}', yticks=[0, 800], ylim=[-20, 800], xlabel='Time to choice (s)')
axs[0].set_ylabel('Dot product', labelpad=-10)
sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, 'dot_prod_all_regions.pdf'))

# %% Plot frontal cortex

fig = plt.figure(figsize=(1.75, 1.75), dpi=dpi)
ax = fig.add_subplot(projection='3d')
ax.view_init(elev=-140, azim=220)
for sp in SPLITS:
    cmap = mpl.colormaps.get_cmap(CMAPS[sp])
    col = [cmap((n_timepoints - p) / n_timepoints) for p in range(n_timepoints)]
    ax.plot(pca_fit['mPFC'][split_ids == sp, 0],
            pca_fit['mPFC'][split_ids == sp, 1],
            pca_fit['mPFC'][split_ids == sp, 2],
            color=col[len(col) // 2], linewidth=1, alpha=0.5, zorder=0)
    ax.scatter(pca_fit['mPFC'][split_ids == sp, 0],
               pca_fit['mPFC'][split_ids == sp, 1],
               pca_fit['mPFC'][split_ids == sp, 2],
               color=col, edgecolors=col, s=10, depthshade=False, zorder=1)
#ax.grid('off')
#ax.axis('off')
ax.set(title='Frontal cortex')
plt.tight_layout()
plt.savefig(join(fig_path, 'pca_frontal-cortex.pdf'))
    
    
    
    