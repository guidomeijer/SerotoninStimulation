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

N_DIM = 3
DATASET = 'firstMovement_times'
#DATASET = 'timewarped'
#CENTER_ON = 'stimOn_times'
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


# Get Eucledian distance between opto and no opto per time point for L and R choices seperately
opto_dist = np.empty(n_timepoints)
for t in range(n_timepoints):
    l_dist = np.linalg.norm(L_opto[:, t] - L_no_opto[:, t])
    r_dist = np.linalg.norm(R_opto[:, t] - R_no_opto[:, t])
    opto_dist[t] = np.max([l_dist, r_dist])

# Do the same for shuffle
opto_dist_shuf = np.empty((n_timepoints, L_opto_shuf.shape[2]))
for ii in range(L_opto_shuf.shape[2]):
    for t in range(n_timepoints):
        l_dist = np.linalg.norm(L_opto_shuf[:, t, ii]
                                - L_no_opto_shuf[:, t, ii])
        r_dist = np.linalg.norm(R_opto_shuf[:, t, ii]
                                - R_no_opto_shuf[:, t, ii])
        opto_dist_shuf[t, ii] = np.max([l_dist, r_dist])

# Get Eucledian distance between choice L and R
choice_dist = np.empty(n_timepoints)
for t in range(n_timepoints):
    this_opto_dist = np.linalg.norm(L_opto[:, t] - R_opto[:, t])
    this_no_opto_dist = np.linalg.norm(L_no_opto[:, t] - R_no_opto[:, t])
    choice_dist[t] = np.max([this_opto_dist, this_no_opto_dist])

# Do the same for shuffle
choice_dist_shuf = np.empty((n_timepoints, L_opto_choice_shuf.shape[2]))
for ii in range(L_opto_choice_shuf.shape[2]):
    for t in range(n_timepoints):
        this_opto_dist = np.linalg.norm(L_opto_choice_shuf[:, t, ii]
                                        - R_opto_choice_shuf[:, t, ii])
        this_no_opto_dist = np.linalg.norm(L_no_opto_choice_shuf[:, t, ii]
                                           - R_no_opto_choice_shuf[:, t, ii])
        choice_dist_shuf[t, ii] = np.max([this_opto_dist, this_no_opto_dist])

# Do PCA
all_splits = np.vstack((L_opto.T, R_opto.T, L_no_opto.T, R_no_opto.T))
pca_fit = pca.fit_transform(all_splits)

# Do PCA for shuffles
pca_shuffle = np.empty((all_splits.shape[0], N_DIM, 0))
for ii in range(this_dict['n_shuffles']):
    all_splits = np.vstack((L_opto_shuf[:, :, ii].T,
                            R_opto_shuf[:, :, ii].T,
                            L_no_opto_shuf[:, :, ii].T,
                            R_no_opto_shuf[:, :, ii].T))
    if np.sum(np.isnan(all_splits)) == 0:
        pca_shuffle = np.dstack((pca_shuffle, pca.fit_transform(all_splits)))
        
pca_choice_shuf = np.empty((all_splits.shape[0], N_DIM, 0))
for ii in range(this_dict['n_shuffles']):
    all_splits = np.vstack((L_opto_choice_shuf[:, :, ii].T,
                            R_opto_choice_shuf[:, :, ii].T,
                            L_no_opto_choice_shuf[:, :, ii].T,
                            R_no_opto_choice_shuf[:, :, ii].T))
    if np.sum(np.isnan(all_splits)) == 0:
        pca_choice_shuf = np.dstack((pca_choice_shuf, pca.fit_transform(all_splits)))

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
    
    #choice_vec = pca_fit[split_ids == 'L_no_opto'][t] - pca_fit[split_ids == 'R_no_opto'][t]
    #opto_vec = pca_fit[split_ids == 'R_opto'][t] - pca_fit[split_ids == 'R_no_opto'][t]
    
    choice_vec = pca_l_col[t, :] - pca_r_col[t, :]
    opto_vec = pca_opto_col[t, :] - pca_no_opto_col[t, :]
    
    dot_prod = np.dot(choice_vec / np.linalg.norm(choice_vec),
                      opto_vec / np.linalg.norm(opto_vec))
    angle_pca[t] = np.degrees(np.arccos(
        dot_prod / (np.linalg.norm(choice_vec) * np.linalg.norm(opto_vec))))
    dot_pca[t] = 1 - np.abs(dot_prod)

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
        
        #choice_vec = pca_shuffle[split_ids == 'L_no_opto', :, ii][t] - pca_shuffle[split_ids == 'R_no_opto', :, ii][t]
        #opto_vec = pca_shuffle[split_ids == 'R_opto', :, ii][t] - pca_shuffle[split_ids == 'R_no_opto', :, ii][t]
        
        choice_vec = pca_l_col[t, :] - pca_r_col[t, :]
        opto_vec = pca_opto_col[t, :] - pca_no_opto_col[t, :]
        
        dot_prod = np.dot(choice_vec / np.linalg.norm(choice_vec),
                          opto_vec / np.linalg.norm(opto_vec))
        angle_pca_shuffle[t, ii] = np.degrees(np.arccos(
            dot_prod / (np.linalg.norm(choice_vec) * np.linalg.norm(opto_vec))))
        dot_pca_shuffle[t, ii] = 1 - np.abs(dot_prod)

# Get distance in PCA space
choice_dist_pca = np.empty(n_timepoints)
for t in range(n_timepoints):
    this_opto_dist = np.linalg.norm(pca_fit[split_ids == 'L_opto'][t, :]
                                    - pca_fit[split_ids == 'R_opto'][t, :])
    this_no_opto_dist = np.linalg.norm(pca_fit[split_ids == 'L_no_opto'][t, :]
                                       - pca_fit[split_ids == 'R_no_opto'][t, :])
    choice_dist_pca[t] = np.max([this_opto_dist, this_no_opto_dist])

choice_dist_pca_shuf = np.empty((n_timepoints, L_opto_choice_shuf.shape[2]))
for ii in range(pca_choice_shuf.shape[2]):
    for t in range(n_timepoints):
        this_opto_dist = np.linalg.norm(pca_choice_shuf[split_ids == 'L_opto', :, ii][t, :]
                                - pca_choice_shuf[split_ids == 'R_opto', :, ii][t, :])
        this_no_opto_dist = np.linalg.norm(pca_choice_shuf[split_ids == 'L_opto', :, ii][t, :]
                                - pca_choice_shuf[split_ids == 'R_opto', :, ii][t, :])
        choice_dist_pca_shuf[t, ii] = np.max([this_opto_dist, this_no_opto_dist])

opto_dist_pca = np.empty(n_timepoints)
for t in range(n_timepoints):
    this_l_dist = np.linalg.norm(pca_fit[split_ids == 'L_opto'][t, :]
                                 - pca_fit[split_ids == 'L_no_opto'][t, :])
    this_r_dist = np.linalg.norm(pca_fit[split_ids == 'R_opto'][t, :]
                                 - pca_fit[split_ids == 'R_no_opto'][t, :])
    opto_dist_pca[t] = np.max([this_l_dist, this_r_dist])

opto_dist_pca_shuf = np.empty((n_timepoints, L_opto_shuf.shape[2]))
for ii in range(pca_shuffle.shape[2]):
    for t in range(n_timepoints):
        l_dist = np.linalg.norm(pca_shuffle[split_ids == 'L_opto', :, ii][t, :]
                                - pca_shuffle[split_ids == 'L_no_opto', :, ii][t, :])
        r_dist = np.linalg.norm(pca_shuffle[split_ids == 'R_opto', :, ii][t, :]
                                - pca_shuffle[split_ids == 'R_no_opto', :, ii][t, :])
        opto_dist_pca_shuf[t, ii] = np.max([l_dist, r_dist])
        
# %% Plot PCA trajectories 
fig = plt.figure(figsize=(1.75, 1.75), dpi=dpi)
ax = fig.add_subplot(projection='3d')
ax.view_init(elev=-140, azim=190)
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
ax.set(xticklabels=[], yticklabels=[], zticklabels=[])
plt.savefig(join(fig_path, f'pca_trajectories_all_together_{DATASET}.pdf'))

"""
def rotate(angle):
    ax.view_init(elev=20, azim=angle)
    
# Create animation
ani = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 360, 2), interval=50)
ani.save(join(fig_path, f'pca_trajectories_all_together_{DATASET}.gif'), writer='pillow', fps=20)
"""

# %% Plot PCA trajectories 
fig = plt.figure(figsize=(1.75, 1.75), dpi=dpi)
ax = fig.add_subplot(projection='3d')
ax.view_init(elev=0, azim=10)
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
ax.set(xticklabels=[], yticklabels=[], zticklabels=[])
plt.savefig(join(fig_path, f'pca_trajectories_all_together_front_{DATASET}.pdf'))


# %%

fig = plt.figure(figsize=(1.75, 1.75), dpi=dpi)
ax = fig.add_subplot(projection='3d')
ax.view_init(elev=-175, azim=190)

cmap = mpl.colormaps.get_cmap(CMAPS['opto_collapsed'])
col = [cmap((n_timepoints - p) / n_timepoints) for p in range(n_timepoints)]
ax.plot(pca_opto_col[:, 0], pca_opto_col[:, 1], pca_opto_col[:, 2],
        color=col[len(col) // 2], linewidth=1, alpha=0.5, zorder=0)
ax.scatter(pca_opto_col[:, 0], pca_opto_col[:, 1], pca_opto_col[:, 2],
           color=col, edgecolors=col, s=10, depthshade=False, zorder=1)

cmap = mpl.colormaps.get_cmap(CMAPS['no_opto_collapsed'])
col = [cmap((n_timepoints - p) / n_timepoints) for p in range(n_timepoints)]
ax.plot(pca_no_opto_col[:, 0], pca_no_opto_col[:, 1], pca_no_opto_col[:, 2],
        color=col[len(col) // 2], linewidth=1, alpha=0.5, zorder=0)
ax.scatter(pca_no_opto_col[:, 0], pca_no_opto_col[:, 1], pca_no_opto_col[:, 2],
           color=col, edgecolors=col, s=10, depthshade=False, zorder=1)

ax.set(xticklabels=[], yticklabels=[], zticklabels=[])
plt.savefig(join(fig_path, f'pca_opto_collapsed_all_together_{DATASET}.pdf'))

# %%

fig = plt.figure(figsize=(1.75, 1.75), dpi=dpi)
ax = fig.add_subplot(projection='3d')
ax.view_init(elev=-175, azim=190)

cmap = mpl.colormaps.get_cmap(CMAPS['L_collapsed'])
col = [cmap((n_timepoints - p) / n_timepoints) for p in range(n_timepoints)]
ax.plot(pca_l_col[:, 0], pca_l_col[:, 1], pca_l_col[:, 2],
        color=col[len(col) // 2], linewidth=1, alpha=0.5, zorder=0)
ax.scatter(pca_l_col[:, 0], pca_l_col[:, 1], pca_l_col[:, 2],
           color=col, edgecolors=col, s=10, depthshade=False, zorder=1)

cmap = mpl.colormaps.get_cmap(CMAPS['R_collapsed'])
col = [cmap((n_timepoints - p) / n_timepoints) for p in range(n_timepoints)]
ax.plot(pca_r_col[:, 0], pca_r_col[:, 1], pca_r_col[:, 2],
        color=col[len(col) // 2], linewidth=1, alpha=0.5, zorder=0)
ax.scatter(pca_r_col[:, 0], pca_r_col[:, 1], pca_r_col[:, 2],
           color=col, edgecolors=col, s=10, depthshade=False, zorder=1)

ax.set(xticklabels=[], yticklabels=[], zticklabels=[])
plt.savefig(join(fig_path, f'pca_choice_collapsed_all_together_{DATASET}.pdf'))


# %%

f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(1.75*3, 1.75), dpi=dpi)

ax1.fill_between(time_ax,
                 np.quantile(choice_dist_shuf, 0.05, axis=1),
                 np.quantile(choice_dist_shuf, 0.95, axis=1),
                 color='lightgrey')
ax1.plot(time_ax, choice_dist, marker='o')
ax1.set(xlabel='Time to choice (s)',ylabel='Choice separation (spks/s)')


ax2.fill_between(time_ax,
                 np.quantile(opto_dist_shuf, 0.05, axis=1),
                 np.quantile(opto_dist_shuf, 0.95, axis=1),
                 color='lightgrey')
ax2.plot(time_ax, opto_dist, marker='o')
ax2.set(xlabel='Time to choice (s)', ylabel='5-HT separation (spks/s)')

ax3.fill_between(time_ax,
                 np.quantile(dot_pca_shuffle, 0.05, axis=1),
                 np.quantile(dot_pca_shuffle, 0.95, axis=1),
                 color='lightgrey')
ax3.plot(time_ax, dot_pca, marker='o')
ax3.set(xlabel='Time to choice (s)',ylabel='Orthogonality\n(1 - norm. dot product)',
        yticks=np.arange(0, 1.1, 0.2))

sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, f'trajectories_all_together_{DATASET}.pdf'))


# %%
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(1.75*3, 1.75), dpi=dpi)

ax1.fill_between(time_ax,
                 np.quantile(choice_dist_pca_shuf, 0.05, axis=1),
                 np.quantile(choice_dist_pca_shuf, 0.95, axis=1),
                 color='lightgrey')
ax1.plot(time_ax, choice_dist_pca, marker='o')
ax1.set(xlabel='Time to choice (s)',ylabel='Choice distance (PC score)', yticks=[0, 100, 200])


ax2.fill_between(time_ax,
                 np.quantile(opto_dist_pca_shuf, 0.05, axis=1),
                 np.quantile(opto_dist_pca_shuf, 0.95, axis=1),
                 color='lightgrey')
ax2.plot(time_ax, opto_dist_pca, marker='o')
ax2.set(xlabel='Time to choice (s)', ylabel='5-HT distance (PC score)', yticks=[0, 50, 100])

ax3.fill_between(time_ax,
                 np.quantile(dot_pca_shuffle, 0.05, axis=1),
                 np.quantile(dot_pca_shuffle, 0.95, axis=1),
                 color='lightgrey')
ax3.plot(time_ax, dot_pca, marker='o')
ax3.set(xlabel='Time to choice (s)',ylabel='Orthogonality\n(1 - norm. dot product)',
        yticks=np.arange(0, 1.1, 0.2))

sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, f'trajectories_all_together_PCA_{DATASET}.pdf'))


