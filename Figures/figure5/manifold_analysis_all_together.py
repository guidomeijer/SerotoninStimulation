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


# Get Eucledian distance between opto and no opto per time point
opto_dist = np.empty(n_timepoints)
for t in range(n_timepoints):
    opto_dist[t] = np.linalg.norm(opto[:, t] - no_opto[:, t])

# Do the same for shuffle
opto_dist_shuf = np.empty((n_timepoints, opto_shuf.shape[2]))
for ii in range(opto_shuf.shape[2]):
    for t in range(n_timepoints):
        opto_dist_shuf[t, ii] = np.linalg.norm(opto_shuf[:, t, ii] - no_opto_shuf[:, t, ii])

# Get Eucledian distance between choice L and R
choice_dist = np.empty(n_timepoints)
for t in range(n_timepoints):
    choice_dist[t] = np.linalg.norm(L[:, t] - R[:, t])

# Do the same for shuffle
choice_dist_shuf = np.empty((n_timepoints, L_shuf.shape[2]))
for ii in range(L_shuf.shape[2]):
    for t in range(n_timepoints):
        choice_dist_shuf[t, ii] = np.linalg.norm(L_shuf[:, t, ii] - R_shuf[:, t, ii])

# Do PCA on L/R choices
lr_splits = np.vstack((L.T, R.T))
pca_fit_lr = pca.fit_transform(lr_splits)
split_ids_lr = np.concatenate((['L'] * n_timepoints, ['R'] * n_timepoints))

# Shuffles
pca_lr_shuffle = np.empty((lr_splits.shape[0], N_DIM, 0))
for ii in range(this_dict['n_shuffles']):
    lr_splits_shuf = np.vstack((L_shuf[:, :, ii].T, R_shuf[:, :, ii].T))
    pca_lr_shuffle = np.dstack((pca_lr_shuffle, pca.fit_transform(lr_splits_shuf)))
                          
# Do PCA on opto / no opto
opto_splits = np.vstack((opto.T, no_opto.T))
pca_fit_opto = pca.fit_transform(opto_splits)
split_ids_opto = np.concatenate((['opto'] * n_timepoints, ['no_opto'] * n_timepoints))

# Shuffles
pca_opto_shuffle = np.empty((opto_splits.shape[0], N_DIM, 0))
for ii in range(this_dict['n_shuffles']):
    opto_splits_shuf = np.vstack((opto_shuf[:, :, ii].T, no_opto_shuf[:, :, ii].T))
    pca_opto_shuffle = np.dstack((pca_opto_shuffle, pca.fit_transform(opto_splits_shuf)))

# Do PCA for data split four ways
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

# Get index to which split the PCA belongs to
split_ids = np.concatenate((['L_opto'] * n_timepoints, ['R_opto'] * n_timepoints,
                            ['L_no_opto'] * n_timepoints, ['R_no_opto'] * n_timepoints))

# Calculate dot product between opto and stim vectors in PCA space
# Collapse pca onto choice and opto dimensions
dot_pca, dot_pca_shuffle = dict(), dict()
angle_pca, angle_pca_shuffle = dict(), dict()
p_value = dict()

# Collapse
pca_l_col = (pca_fit[split_ids == 'L_opto'] + pca_fit[split_ids == 'L_no_opto']) / 2
pca_r_col = (pca_fit[split_ids == 'R_opto'] + pca_fit[split_ids == 'R_no_opto']) / 2
pca_opto_col = (pca_fit[split_ids == 'L_opto'] + pca_fit[split_ids == 'R_opto']) / 2
pca_no_opto_col = (pca_fit[split_ids == 'L_no_opto'] + pca_fit[split_ids == 'R_no_opto']) / 2

# Get the dot product and angle between the two vectors
dot_pca, angle_pca = np.empty(n_timepoints), np.empty(n_timepoints)
p_value, dot_95, dot_5 = np.empty(n_timepoints), np.empty(n_timepoints), np.empty(n_timepoints)
for t in range(n_timepoints):

    #choice_vec = pca_fit[split_ids == 'L_no_opto'][t] - pca_fit[split_ids == 'R_no_opto'][t]
    #opto_vec = pca_fit[split_ids == 'R_opto'][t] - pca_fit[split_ids == 'R_no_opto'][t]
    
    choice_vec = pca_l_col[t, :] - pca_r_col[t, :]
    opto_vec = pca_opto_col[t, :] - pca_no_opto_col[t, :]

    #dot_prod = np.dot(choice_vec, opto_vec)
    dot_prod = np.dot(choice_vec / np.linalg.norm(choice_vec),
                      opto_vec / np.linalg.norm(opto_vec))
    dot_pca[t] = 1 - np.abs(dot_prod)
    #dot_pca[t] = dot_prod
    angle_pca[t] = np.degrees(np.arccos(
        dot_prod / (np.linalg.norm(choice_vec) * np.linalg.norm(opto_vec))))
    
    # Determine whether the dot product is more orthogonal than expected by chance
    z_score = dot_prod / (1 / np.sqrt(N_DIM))
    #z_score = (angle_pca[region][t] - 90) / (1 / np.sqrt(choice_vec.shape[0]))
    p_value[t] = 1 - (stats.norm.sf(abs(z_score)) * 2)
    
    # Calculate the critical dot product values
    dot_95[t] = stats.norm.ppf(1 - 0.05 / 2) * (1 / np.sqrt(N_DIM))

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
        #dot_prod = np.dot(choice_vec, opto_vec)
        angle_pca_shuffle[t, ii] = np.degrees(np.arccos(
            dot_prod / (np.linalg.norm(choice_vec) * np.linalg.norm(opto_vec))))
        dot_pca_shuffle[t, ii] = 1 - np.abs(dot_prod)
        #dot_pca_shuffle[t, ii] = dot_prod
        


# %% Plot PCA left right choices

fig = plt.figure(figsize=(1.75, 1.75), dpi=dpi)
ax = fig.add_subplot(projection='3d')
ax.view_init(elev=-125, azim=-100)
for sp in ['L', 'R']:
    cmap = mpl.colormaps.get_cmap(CMAPS[sp])
    col = [cmap((n_timepoints - p) / n_timepoints) for p in range(n_timepoints)]
    ax.plot(pca_fit_lr[split_ids_lr == sp, 0],
            pca_fit_lr[split_ids_lr == sp, 1],
            pca_fit_lr[split_ids_lr == sp, 2],
            color=col[len(col) // 2], linewidth=1, alpha=0.5, zorder=0)
    ax.scatter(pca_fit_lr[split_ids_lr == sp, 0],
               pca_fit_lr[split_ids_lr == sp, 1],
               pca_fit_lr[split_ids_lr == sp, 2],
               color=col, edgecolors=col, s=10, depthshade=False, zorder=1)

x_pos = -50
y_pos = 50
z_pos = 70
ax.plot([x_pos, x_pos + 50], [y_pos, y_pos], [z_pos, z_pos], color='k')
ax.plot([x_pos, x_pos], [y_pos, y_pos + 40], [z_pos, z_pos], color='k')
ax.plot([x_pos, x_pos], [y_pos, y_pos], [z_pos, z_pos - 40], color='k')
#ax.plot([0, 0], [ax.get_ylim()[0], ax.get_ylim()[0] + 10], [0, 0])
#ax.plot([0, 0], [0, 0], [ax.get_zlim()[0], ax.get_zlim()[0] + 10])
ax.axis('off')

ax.set(xticklabels=[], yticklabels=[], zticklabels=[])
plt.savefig(join(fig_path, f'pca_LR_all_together_{DATASET}.pdf'))

# %% Plot PCA opto

fig = plt.figure(figsize=(1.75, 1.75), dpi=dpi)
ax = fig.add_subplot(projection='3d')
ax.view_init(elev=-140, azim=-120)
for sp in ['opto', 'no_opto']:
    cmap = mpl.colormaps.get_cmap(CMAPS[sp])
    col = [cmap((n_timepoints - p) / n_timepoints) for p in range(n_timepoints)]
    ax.plot(pca_fit_opto[split_ids_opto == sp, 0],
            pca_fit_opto[split_ids_opto == sp, 1],
            pca_fit_opto[split_ids_opto == sp, 2],
            color=col[len(col) // 2], linewidth=1, alpha=0.5, zorder=0)
    ax.scatter(pca_fit_opto[split_ids_opto == sp, 0],
               pca_fit_opto[split_ids_opto == sp, 1],
               pca_fit_opto[split_ids_opto == sp, 2],
               color=col, edgecolors=col, s=10, depthshade=False, zorder=1)

x_pos = -50
y_pos = 50
z_pos = 70
ax.plot([x_pos, x_pos + 50], [y_pos, y_pos], [z_pos, z_pos], color='k')
ax.plot([x_pos, x_pos], [y_pos, y_pos + 40], [z_pos, z_pos], color='k')
ax.plot([x_pos, x_pos], [y_pos, y_pos], [z_pos, z_pos - 40], color='k')
#ax.plot([0, 0], [ax.get_ylim()[0], ax.get_ylim()[0] + 10], [0, 0])
#ax.plot([0, 0], [0, 0], [ax.get_zlim()[0], ax.get_zlim()[0] + 10])
ax.axis('off')

ax.set(xticklabels=[], yticklabels=[], zticklabels=[])
plt.savefig(join(fig_path, f'pca_opto_all_together_{DATASET}.pdf'))


# %% Plot PCA trajectories
fig = plt.figure(figsize=(1.75, 1.75), dpi=dpi)
ax = fig.add_subplot(projection='3d')
ax.view_init(elev=-130, azim=-110)
for sp in ['L_opto', 'L_no_opto', 'R_opto', 'R_no_opto']:
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
x_pos = -50
y_pos = 50
z_pos = 70
ax.plot([x_pos, x_pos + 50], [y_pos, y_pos], [z_pos, z_pos], color='k')
ax.plot([x_pos, x_pos], [y_pos, y_pos + 40], [z_pos, z_pos], color='k')
ax.plot([x_pos, x_pos], [y_pos, y_pos], [z_pos, z_pos - 30], color='k')

ax.axis('off')
plt.savefig(join(fig_path, f'pca_trajectories_all_together_{DATASET}.pdf'))

"""
def rotate(angle):
    ax.view_init(elev=20, azim=angle)

# Create animation
ani = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 360, 2), interval=50)
ani.save(join(fig_path, f'pca_trajectories_all_together_{DATASET}.gif'), writer='pillow', fps=20)
"""



# %%

f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(1.75*3, 1.75), dpi=dpi)

ax1.fill_between(time_ax,
                 np.quantile(choice_dist_shuf, 0.05, axis=1),
                 np.quantile(choice_dist_shuf, 0.95, axis=1),
                 color='lightgrey')
ax1.plot(time_ax, choice_dist, marker='o')
ax1.set(xlabel='Time to choice (s)',ylabel='Choice separation (spks/s)', 
        yticks=[50, 210], xticks=[-0.3, -0.2, -0.1, 0], xticklabels=[-0.3, -0.2, -0.1, 0])


ax2.fill_between(time_ax,
                 np.quantile(opto_dist_shuf, 0.05, axis=1),
                 np.quantile(opto_dist_shuf, 0.95, axis=1),
                 color='lightgrey')
ax2.plot(time_ax, opto_dist, marker='o')
ax2.set(xlabel='Time to choice (s)', ylabel='5-HT separation (spks/s)',
        yticks=[50, 90], xticks=[-0.3, -0.2, -0.1, 0], xticklabels=[-0.3, -0.2, -0.1, 0])

ax3.fill_between(time_ax,
                 np.quantile(dot_pca_shuffle, 0.05, axis=1),
                 np.quantile(dot_pca_shuffle, 0.95, axis=1),
                 color='lightgrey')
ax3.plot(time_ax, dot_pca, marker='o')
ax3.set(xlabel='Time to choice (s)',ylabel='Orthogonality\n(1 - norm. dot product)',
        yticks=np.arange(0, 1.1, 0.2), xticks=[-0.3, -0.2, -0.1, 0], xticklabels=[-0.3, -0.2, -0.1, 0])

sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, f'distance_orthogonality_{DATASET}.pdf'))



