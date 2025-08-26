# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 11:22:23 2023

By Guido Meijer
"""

import numpy as np
import os
from os.path import join, realpath, dirname, split, exists
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import matplotlib as mpl
import requests
import time
import zipfile
from stim_functions import figure_style, paths, add_significance
from sklearn.decomposition import PCA
colors, dpi = figure_style()

N_DIM = 3
CMAPS = dict({
    'R': 'Reds_r', 'L': 'Purples_r', 'no_opto': 'Oranges_r', 'opto': 'Blues_r',
    'L_opto': 'Blues_r', 'R_opto': 'Oranges_r', 'L_no_opto': 'Purples_r', 'R_no_opto': 'Reds_r'})

# Initialize
pca = PCA(n_components=N_DIM)
colors, dpi = figure_style()

# Get paths
f_path, load_path = paths()  
fig_path = join(f_path, split(dirname(realpath(__file__)))[-1])

# Download data from figshare
if not exists(join(load_path, 'firstMovement_times')):
    url = 'https://figshare.com/ndownloader/files/57359308'
    zip_filepath = join(load_path, 'firstMovement_times.zip')
    print('Downloading manifold data (~1.5 GB)')
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))
    bytes_downloaded = 0
    start_time = time.time()
    with open(zip_filepath, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            bytes_downloaded += len(chunk)
            f.write(chunk)
            progress = (bytes_downloaded / total_size) * 100
            if time.time() - start_time > 0:
                speed = bytes_downloaded / (time.time() - start_time)
                print(f"Progress: {progress:.2f}% | Speed: {speed / (1024*1024):.2f} MB/s", end='\r')
    print('Download complete')
    
    # Extract zip file
    print('Extracting files...')
    with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
        zip_ref.extractall(join(load_path))
    print('Extraction complete')

    # Remove zip file
    os.remove(zip_filepath)

# Load in data
print('Loading in data..')
regions = np.array([])
ses_paths = glob(join(load_path, 'firstMovement_times', '*.npy'))
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

# Center data (mean subtraction) before PCA
L -= np.mean(L, axis=0)
R -= np.mean(R, axis=0)
opto -= np.mean(opto, axis=0)
no_opto -= np.mean(no_opto, axis=0)
L_opto -= np.mean(L_opto, axis=0)
R_opto -= np.mean(R_opto, axis=0)
L_no_opto -= np.mean(L_no_opto, axis=0)
R_no_opto -= np.mean(R_no_opto, axis=0)

L_shuf -= np.mean(L_shuf, axis=0)
R_shuf -= np.mean(R_shuf, axis=0)
opto_shuf -= np.mean(opto_shuf, axis=0)
no_opto_shuf -= np.mean(no_opto_shuf, axis=0)
L_opto_shuf -= np.mean(L_opto_shuf, axis=0)
R_opto_shuf -= np.mean(R_opto_shuf, axis=0)
L_no_opto_shuf -= np.mean(L_no_opto_shuf, axis=0)
R_no_opto_shuf -= np.mean(R_no_opto_shuf, axis=0)

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

# Get vectors
choice_vec = ((pca_fit[split_ids == 'R_no_opto'] - pca_fit[split_ids == 'L_no_opto'])
                     + (pca_fit[split_ids == 'R_opto'] - pca_fit[split_ids == 'L_opto'])) / 2
mid_nonstim = (pca_fit[split_ids == 'L_no_opto'] + pca_fit[split_ids == 'R_no_opto']) / 2
mid_stim = (pca_fit[split_ids == 'L_opto'] + pca_fit[split_ids == 'R_opto']) / 2
opto_vec = mid_stim - mid_nonstim

# Get the dot product and angle between the two vectors
dot_pca, angle_pca = np.empty(n_timepoints), np.empty(n_timepoints)
for t in range(n_timepoints):

    dot_prod = np.dot(choice_vec[t, :] / np.linalg.norm(choice_vec[t, :]),
                      opto_vec[t, :] / np.linalg.norm(opto_vec[t, :]))
    dot_pca[t] = 1 - np.abs(dot_prod)
    angle_pca[t] = np.degrees(np.arccos(
        dot_prod / (np.linalg.norm(choice_vec[t, :]) * np.linalg.norm(opto_vec[t, :]))))
    
# Do the same for all the shuffles
dot_pca_shuffle = np.empty((n_timepoints, pca_shuffle.shape[2]))
angle_pca_shuffle = np.empty((n_timepoints, pca_shuffle.shape[2]))
for ii in range(pca_shuffle.shape[2]):
    
    choice_vec = ((pca_shuffle[split_ids == 'R_no_opto', :, ii] - pca_shuffle[split_ids == 'L_no_opto', :, ii])
                  + (pca_shuffle[split_ids == 'R_opto', :, ii] - pca_shuffle[split_ids == 'L_opto', :, ii])) / 2
    mid_nonstim = (pca_shuffle[split_ids == 'L_no_opto', :, ii] + pca_shuffle[split_ids == 'R_no_opto', :, ii]) / 2
    mid_stim = (pca_shuffle[split_ids == 'L_opto', :, ii] + pca_shuffle[split_ids == 'R_opto', :, ii]) / 2
    opto_vec = mid_stim - mid_nonstim

    for t in range(n_timepoints):
        dot_prod = np.dot(choice_vec[t, :] / np.linalg.norm(choice_vec[t, :]),
                          opto_vec[t, :] / np.linalg.norm(opto_vec[t, :]))
        angle_pca_shuffle[t, ii] = np.degrees(np.arccos(
            dot_prod / (np.linalg.norm(choice_vec[t, :]) * np.linalg.norm(opto_vec[t, :]))))
        dot_pca_shuffle[t, ii] = 1 - np.abs(dot_prod)
        


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
plt.savefig(join(fig_path, 'pca_LR_all_together.pdf'))

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
plt.savefig(join(fig_path, 'pca_opto_all_together.pdf'))


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
plt.savefig(join(fig_path, 'pca_trajectories_all_together.pdf'))


# %%

f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(1.75*3, 1.75), dpi=dpi)

ax1.fill_between(time_ax,
                 np.quantile(choice_dist_shuf, 0.025, axis=1),
                 np.quantile(choice_dist_shuf, 0.975, axis=1),
                 color='lightgrey')
ax1.plot(time_ax, choice_dist, marker='o')
add_significance(time_ax, (choice_dist < np.quantile(choice_dist_shuf, 0.975, axis=1)).astype(int), ax1)
ax1.set(xlabel='Time to choice (s)',ylabel='Choice separation (spks/s)', 
        yticks=[50, 250], xticks=[-0.3, -0.2, -0.1, 0], xticklabels=[-0.3, -0.2, -0.1, 0])


ax2.fill_between(time_ax,
                 np.quantile(opto_dist_shuf, 0.025, axis=1),
                 np.quantile(opto_dist_shuf, 0.975, axis=1),
                 color='lightgrey')
ax2.plot(time_ax, opto_dist, marker='o')
add_significance(time_ax, (opto_dist < np.quantile(opto_dist_shuf, 0.975, axis=1)).astype(int), ax2)
ax2.set(xlabel='Time to choice (s)', ylabel='5-HT separation (spks/s)',
        yticks=[50, 110], xticks=[-0.3, -0.2, -0.1, 0], xticklabels=[-0.3, -0.2, -0.1, 0])

ax3.fill_between(time_ax,
                 np.quantile(dot_pca_shuffle, 0.025, axis=1),
                 np.quantile(dot_pca_shuffle, 0.975, axis=1),
                 color='lightgrey')
ax3.plot(time_ax, dot_pca, marker='o')
add_significance(time_ax, (dot_pca < np.quantile(dot_pca_shuffle, 0.975, axis=1)).astype(int), ax3)
ax3.set(xlabel='Time to choice (s)',ylabel='Orthogonality\n(1 - abs. norm. dot product)',
        yticks=np.arange(0, 1.1, 0.2), xticks=[-0.3, -0.2, -0.1, 0], xticklabels=[-0.3, -0.2, -0.1, 0])

sns.despine(trim=True)
plt.tight_layout()

plt.savefig(join(fig_path, 'distance_orthogonality.pdf'))



