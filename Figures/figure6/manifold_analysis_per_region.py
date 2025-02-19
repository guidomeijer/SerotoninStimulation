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

N_DIM = 3
#SPLIT_ON = 'stim_side'
SPLIT_ON = 'firstMovement_times'
CHOICE_5HT_WIN = [-0.05, 0]
ORTH_WIN = [-0.05, 0]
DROP_REGIONS = ['root', 'AI', 'BC', 'ZI', 'RSP']
SPLITS = ['L_opto', 'R_opto', 'L_no_opto', 'R_no_opto']
CMAPS = dict({'L_opto': 'Reds_r', 'R_opto': 'Purples_r', 'L_no_opto': 'Oranges_r', 'R_no_opto': 'Blues_r',
              'L_collapsed': 'Reds_r', 'R_collapsed': 'Purples_r', 'no_opto_collapsed': 'Greys_r', 'opto_collapsed': 'Blues_r'})

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
    if this_dict['sert-cre'] == 0:
        continue
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
    if region in DROP_REGIONS:
        continue

    # Get Eucledian distance between opto and no opto per time point for L and R choices seperately
    this_dist = np.empty(n_timepoints)
    for t in range(n_timepoints):
        l_dist = np.linalg.norm(L_opto[regions == region, t] - L_no_opto[regions == region, t])
        r_dist = np.linalg.norm(R_opto[regions == region, t] - R_no_opto[regions == region, t])
        this_dist[t] = np.max([l_dist, r_dist])
    dist_opto[region] = this_dist

    # Do the same for shuffle
    this_dist = np.empty((n_timepoints, L_opto_shuf.shape[2]))
    for ii in range(L_opto_shuf.shape[2]):
        for t in range(n_timepoints):
            l_dist = np.linalg.norm(L_opto_choice_shuf[regions == region, t, ii]
                                    - L_no_opto_choice_shuf[regions == region, t, ii])
            r_dist = np.linalg.norm(R_opto_choice_shuf[regions == region, t, ii]
                                    - R_no_opto_choice_shuf[regions == region, t, ii])
            this_dist[t, ii] = np.max([l_dist, r_dist])
    dist_opto_shuffle[region] = this_dist

# Get Eucledian distances in neural space between choice left and right
dist_choice, dist_choice_shuffle = dict(), dict()
for r, region in enumerate(np.unique(regions)):
    if region in DROP_REGIONS:
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
            this_dist[t, ii] = np.max([opto_dist, no_opto_dist])
    dist_choice_shuffle[region] = this_dist

# Do PCA
print('Fitting PCA..')
pca_fit, pca_shuffle, pca_choice_shuf = dict(), dict(), dict()
for r, region in enumerate(np.unique(regions)):
    if region in DROP_REGIONS:
        continue
    print(f'Starting fits for {region}')

    # Do PCA on all splits simultaneously to get them in the same PCA space
    all_splits = np.vstack((L_opto[regions == region].T, R_opto[regions == region].T,
                            L_no_opto[regions == region].T, R_no_opto[regions == region].T))
    if all_splits.shape[1] < N_DIM:
        continue
    pca_fit[region] = pca.fit_transform(all_splits)

    # Do PCA for shuffles
    pca_shuffle[region] = np.empty((all_splits.shape[0], N_DIM, 0))
    for ii in range(this_dict['n_shuffles']):
        all_splits = np.vstack((L_opto_shuf[regions == region, :, ii].T,
                                R_opto_shuf[regions == region, :, ii].T,
                                L_no_opto_shuf[regions == region, :, ii].T,
                                R_no_opto_shuf[regions == region, :, ii].T))
        if np.sum(np.isnan(all_splits)) == 0:
            pca_shuffle[region] = np.dstack((pca_shuffle[region], pca.fit_transform(all_splits)))

    pca_choice_shuf[region] = np.empty((all_splits.shape[0], N_DIM, 0))
    for ii in range(this_dict['n_shuffles']):
        all_splits = np.vstack((L_opto_choice_shuf[regions == region, :, ii].T,
                                R_opto_choice_shuf[regions == region, :, ii].T,
                                L_no_opto_choice_shuf[regions == region, :, ii].T,
                                R_no_opto_choice_shuf[regions == region, :, ii].T))
        if np.sum(np.isnan(all_splits)) == 0:
            pca_choice_shuf[region] = np.dstack((pca_choice_shuf[region], pca.fit_transform(all_splits)))

# Get index to which split the PCA belongs to
split_ids = np.concatenate((['L_opto'] * n_timepoints, ['R_opto'] * n_timepoints,
                            ['L_no_opto'] * n_timepoints, ['R_no_opto'] * n_timepoints))

# Calculate dot product between opto and stim vectors in PCA space
# Collapse pca onto choice and opto dimensions
dot_pca, dot_pca_shuffle = dict(), dict()
angle_pca, angle_pca_shuffle = dict(), dict()
p_value = dict()
for r, region in enumerate(pca_fit.keys()):
    pca_l_col = (pca_fit[region][split_ids == 'L_opto'] + pca_fit[region][split_ids == 'L_no_opto']) / 2
    pca_r_col = (pca_fit[region][split_ids == 'R_opto'] + pca_fit[region][split_ids == 'R_no_opto']) / 2
    pca_opto_col = (pca_fit[region][split_ids == 'L_opto'] + pca_fit[region][split_ids == 'R_opto']) / 2
    pca_no_opto_col = (pca_fit[region][split_ids == 'L_no_opto'] + pca_fit[region][split_ids == 'R_no_opto']) / 2

    # Get the dot product and angle between the two vectors
    dot_pca[region], angle_pca[region] = np.empty(n_timepoints), np.empty(n_timepoints)
    p_value[region] = np.empty(n_timepoints)
    for t in range(n_timepoints):
        choice_vec = pca_l_col[t, :] - pca_r_col[t, :]
        opto_vec = pca_opto_col[t, :] - pca_no_opto_col[t, :]
        dot_prod = np.dot(choice_vec / np.linalg.norm(choice_vec),
                          opto_vec / np.linalg.norm(opto_vec))
        angle_pca[region][t] = np.degrees(np.arccos(
            dot_prod / (np.linalg.norm(choice_vec) * np.linalg.norm(opto_vec))))
        dot_pca[region][t] = 1 - np.abs(dot_prod)

        # Determine whether the dot product is more orthogonal than expected by chance
        z_score = dot_prod / (1 / np.sqrt(choice_vec.shape[0]))
        #z_score = (angle_pca[region][t] - 90) / (1 / np.sqrt(choice_vec.shape[0]))
        p_value[region][t] = 1 - (stats.norm.sf(abs(z_score)) * 2)

    # Do the same for all the shuffles
    dot_pca_shuffle[region] = np.empty((n_timepoints, pca_shuffle[region].shape[2]))
    angle_pca_shuffle[region] = np.empty((n_timepoints, pca_shuffle[region].shape[2]))
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
            dot_prod = np.dot(choice_vec / np.linalg.norm(choice_vec),
                              opto_vec / np.linalg.norm(opto_vec))
            angle_pca_shuffle[region][t, ii] = np.degrees(np.arccos(
                dot_prod / (np.linalg.norm(choice_vec) * np.linalg.norm(opto_vec))))
            dot_pca_shuffle[region][t, ii] = 1 - np.abs(dot_prod)


# Get Eucledian distances in PCA space
choice_dist_pca, choice_dist_pca_shuf = dict(), dict()
opto_dist_pca, opto_dist_pca_shuf = dict(), dict()
for r, region in enumerate(np.unique(regions)):
    if region in DROP_REGIONS:
        continue

    choice_dist_pca[region] = np.empty(n_timepoints)
    for t in range(n_timepoints):
        this_opto_dist = np.linalg.norm(pca_fit[region][split_ids == 'L_opto'][t, :]
                                        - pca_fit[region][split_ids == 'R_opto'][t, :])
        this_no_opto_dist = np.linalg.norm(pca_fit[region][split_ids == 'L_no_opto'][t, :]
                                           - pca_fit[region][split_ids == 'R_no_opto'][t, :])
        choice_dist_pca[region][t] = np.max([this_opto_dist, this_no_opto_dist])

    choice_dist_pca_shuf[region] = np.empty((n_timepoints, L_opto_choice_shuf.shape[2]))
    for ii in range(pca_choice_shuf[region].shape[2]):
        for t in range(n_timepoints):
            this_opto_dist = np.linalg.norm(pca_choice_shuf[region][split_ids == 'L_opto', :, ii][t, :]
                                    - pca_choice_shuf[region][split_ids == 'R_opto', :, ii][t, :])
            this_no_opto_dist = np.linalg.norm(pca_choice_shuf[region][split_ids == 'L_opto', :, ii][t, :]
                                    - pca_choice_shuf[region][split_ids == 'R_opto', :, ii][t, :])
            choice_dist_pca_shuf[region][t, ii] = np.max([this_opto_dist, this_no_opto_dist])

    opto_dist_pca[region] = np.empty(n_timepoints)
    for t in range(n_timepoints):
        this_l_dist = np.linalg.norm(pca_fit[region][split_ids == 'L_opto'][t, :]
                                     - pca_fit[region][split_ids == 'L_no_opto'][t, :])
        this_r_dist = np.linalg.norm(pca_fit[region][split_ids == 'R_opto'][t, :]
                                     - pca_fit[region][split_ids == 'R_no_opto'][t, :])
        opto_dist_pca[region][t] = np.max([this_l_dist, this_r_dist])

    opto_dist_pca_shuf[region] = np.empty((n_timepoints, L_opto_shuf.shape[2]))
    for ii in range(pca_shuffle[region].shape[2]):
        for t in range(n_timepoints):
            l_dist = np.linalg.norm(pca_shuffle[region][split_ids == 'L_opto', :, ii][t, :]
                                    - pca_shuffle[region][split_ids == 'L_no_opto', :, ii][t, :])
            r_dist = np.linalg.norm(pca_shuffle[region][split_ids == 'R_opto', :, ii][t, :]
                                    - pca_shuffle[region][split_ids == 'R_no_opto', :, ii][t, :])
            opto_dist_pca_shuf[region][t, ii] = np.max([l_dist, r_dist])


# %% Prepare for plotting

# Get mean over timewindow per region
choice_dist_pca_regions = [np.mean(choice_dist_pca[i][
    (time_ax >= CHOICE_5HT_WIN[0]) & (time_ax <= CHOICE_5HT_WIN[1])])
    for i in choice_dist_pca.keys()]
choice_dist_pca_shuf_regions = [np.mean(choice_dist_pca_shuf[i][
    (time_ax >= CHOICE_5HT_WIN[0]) & (time_ax <= CHOICE_5HT_WIN[1]), :], 0)
    for i in choice_dist_pca_shuf.keys()]
opto_dist_pca_regions = [np.mean(opto_dist_pca[i][
    (time_ax >= CHOICE_5HT_WIN[0]) & (time_ax <= CHOICE_5HT_WIN[1])])
    for i in opto_dist_pca.keys()]
opto_dist_pca_shuf_regions = [np.mean(opto_dist_pca_shuf[i][
    (time_ax >= CHOICE_5HT_WIN[0]) & (time_ax <= CHOICE_5HT_WIN[1]), :], 0)
    for i in opto_dist_pca_shuf.keys()]
dot_pca_regions = [np.mean(dot_pca[i][
    (time_ax >= ORTH_WIN[0]) & (time_ax <= ORTH_WIN[1])])
    for i in dot_pca.keys()]
dot_pca_shuf_regions = [np.mean(dot_pca_shuffle[i][
    (time_ax >= ORTH_WIN[0]) & (time_ax <= ORTH_WIN[1]), :], 0)
    for i in dot_pca_shuffle.keys()]

# Add shuffles to dataframe for plotting
dist_pca_df = pd.DataFrame(data={
    'choice_dist_shuf': [item for sublist in choice_dist_pca_shuf_regions for item in sublist],
    'opto_dist_shuf': [item for sublist in opto_dist_pca_shuf_regions for item in sublist],
    'dot_shuf': [item for sublist in dot_pca_shuf_regions for item in sublist],
    'region': [string for string, sublist in zip(choice_dist_pca_shuf.keys(), choice_dist_pca_shuf_regions) for _ in sublist]})

# %% Choice distance

f, ax1 = plt.subplots(1, 1, figsize=(2.5, 1.75), dpi=dpi)
sns.violinplot(data=dist_pca_df, x='region', y='choice_dist_shuf',
               inner=None, linewidth=0, ax=ax1, color=colors['grey'])
ax1.scatter(np.arange(len(choice_dist_pca_regions)), choice_dist_pca_regions,
            marker='_', color='red', s=20)
for i, this_dist in enumerate(choice_dist_pca_regions):
    if this_dist > np.quantile(choice_dist_pca_shuf_regions[i], 1-(0.001/2)):
        ax1.text(i, this_dist+2, '***', ha='center', va='center', fontsize=7)
    elif this_dist > np.quantile(choice_dist_pca_shuf_regions[i], 1-(0.01/2)):
        ax1.text(i, this_dist+2, '**', ha='center', va='center', fontsize=7)
    elif this_dist > np.quantile(choice_dist_pca_shuf_regions[i], 1-(0.05/2)):
        ax1.text(i, this_dist+2, '*', ha='center', va='center', fontsize=7)
ax1.tick_params(axis='x', labelrotation=90)
ax1.set(xlabel='', ylabel='Choice distance (PCA score)',
        yticks=[0, 40, 80, 120])

sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, 'choice_dist_regions.pdf'))

# %% Opto distance

f, ax1 = plt.subplots(1, 1, figsize=(2.5, 1.75), dpi=dpi)
sns.violinplot(data=dist_pca_df, x='region', y='opto_dist_shuf',
               inner=None, linewidth=0, ax=ax1, color=colors['grey'])
ax1.scatter(np.arange(len(opto_dist_pca_regions)), opto_dist_pca_regions,
            marker='_', color='red', s=20)
for i, this_dist in enumerate(opto_dist_pca_regions):
    if this_dist > np.quantile(opto_dist_pca_shuf_regions[i], 1-(0.001/2)):
        ax1.text(i, this_dist+6, '***', ha='center', va='center', fontsize=7)
    elif this_dist > np.quantile(opto_dist_pca_shuf_regions[i], 1-(0.01/2)):
        ax1.text(i, this_dist+6, '**', ha='center', va='center', fontsize=7)
    elif this_dist > np.quantile(opto_dist_pca_shuf_regions[i], 1-(0.05/2)):
        ax1.text(i, this_dist+6, '*', ha='center', va='center', fontsize=7)
ax1.tick_params(axis='x', labelrotation=90)
ax1.set(xlabel='', ylabel='5-HT distance (PCA score)',
        yticks=[0, 40, 80])

sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, 'opto_dist_regions.pdf'))

# %% Dot product

f, ax1 = plt.subplots(1, 1, figsize=(2.5, 1.75), dpi=dpi)
sns.violinplot(data=dist_pca_df, x='region', y='dot_shuf',
               inner=None, linewidth=0, ax=ax1, color=colors['grey'])
ax1.scatter(np.arange(len(dot_pca_regions)), dot_pca_regions,
            marker='_', color='red', s=20)
for i, this_dist in enumerate(dot_pca_regions):
    if this_dist > np.quantile(dot_pca_shuf_regions[i], 1-(0.001/2)):
        ax1.text(i, 1.15, '***', ha='center', va='center', fontsize=7)
    elif this_dist > np.quantile(dot_pca_shuf_regions[i], 1-(0.01/2)):
        ax1.text(i, 1.15, '**', ha='center', va='center', fontsize=7)
    elif this_dist > np.quantile(dot_pca_shuf_regions[i], 1-(0.05/2)):
        ax1.text(i, 1.15, '*', ha='center', va='center', fontsize=7)
ax1.tick_params(axis='x', labelrotation=90)
ax1.set(xlabel='', ylabel='Orthogonality\n(1 - norm. dot prod.)',
        yticks=[0, 1])

sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, 'dot_prod_regions.pdf'))

# %%



"""

# %% Plot choice and opto distance

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(3, 1.75), dpi=dpi, sharey=True)

for i, region in enumerate(dist_choice.keys()):
    if dist_choice[region][-1] > np.nanquantile(dist_choice_shuffle[region], 0.99, axis=1)[-1]:
        ax1.plot(time_ax, dist_choice[region], color=colors[region])
        ax1.text(time_ax[-1] + 0.005, dist_choice[region][-1], region, color=colors[region],
                 va='center')
ax1.set(ylabel='Separation (spks/s)', title='Choice', yticks=[0, 50, 100, 150],
        xticks=[-0.15, -0.1, -0.05, 0], xticklabels=[-150, -100, -50, 0])

for i, region in enumerate(dist_opto.keys()):
    if dist_choice[region][-1] > np.nanquantile(dist_choice_shuffle[region], 0.99, axis=1)[-1]:
        ax2.plot(time_ax, dist_opto[region], color=colors[region])
        ax2.text(time_ax[-1] + 0.005, dist_opto[region][-1], region, color=colors[region],
                 va='center')
ax2.set(title='5-HT', xticks=[-0.15, -0.1, -0.05, 0], xticklabels=[-150, -100, -50, 0],
        yticks=[0, 50, 100, 150])

f.text(0.5, 0.04, 'Time to choice (ms)', ha='center')

sns.despine(trim=True)
plt.subplots_adjust(bottom=0.21, wspace=0.3, left=0.15)
plt.savefig(join(fig_path, 'choice_and_opto.pdf'))

# %% Plot choice and opto distance PCA

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(3, 1.75), dpi=dpi, sharey=True)

for i, region in enumerate(dist_choice.keys()):
    ax1.plot(time_ax, choice_dist_pca[region], color=colors[region])
    ax1.text(time_ax[-1] + 0.005, choice_dist_pca[region][-1], region, color=colors[region],
             va='center')
ax1.set(ylabel='PCA distance', title='Choice', yticks=[0, 50, 100, 150],
        xticks=[-0.15, -0.1, -0.05, 0], xticklabels=[-150, -100, -50, 0])

for i, region in enumerate(dist_opto.keys()):
    ax2.plot(time_ax, choice_dist_pca[region], color=colors[region])
    ax2.text(time_ax[-1] + 0.005, choice_dist_pca[region][-1], region, color=colors[region],
             va='center')
ax2.set(title='5-HT', xticks=[-0.15, -0.1, -0.05, 0], xticklabels=[-150, -100, -50, 0],
        yticks=[0, 50, 100, 150])

f.text(0.5, 0.04, 'Time to choice (ms)', ha='center')

sns.despine(trim=True)
plt.subplots_adjust(bottom=0.21, wspace=0.3, left=0.15)
plt.savefig(join(fig_path, 'choice_and_opto_pca.pdf'))


# %% Plot dot product

f, ax1 = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi, sharey=True)

for i, region in enumerate(dist_choice.keys()):
    if dist_choice[region][-1] > np.nanquantile(dist_choice_shuffle[region], 0.99, axis=1)[-1]:
        ax1.plot(time_ax, dot_pca[region], color=colors[region])
        if p_value[region][-1] < 0.001:
            ax1.text(time_ax[-1] + 0.005, dot_pca[region][-1], '***', color=colors[region],
                     va='top', fontsize=12)
        elif p_value[region][-1] < 0.01:
            ax1.text(time_ax[-1] + 0.005, dot_pca[region][-1], '**', color=colors[region],
                     va='top', fontsize=12)
        elif p_value[region][-1] < 0.05:
            ax1.text(time_ax[-1] + 0.005, dot_pca[region][-1] + 0.025, '*', color=colors[region],
                     va='top', fontsize=12)
ax1.set(ylabel='Normalized dot product', yticks=[0, 0.2, 0.4, 0.6, 0.8, 1],
        xticks=[-0.15, -0.1, -0.05, 0], xticklabels=[-150, -100, -50, 0],
        xlabel='Time to choice (ms)')

sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, 'dot_product.pdf'))

# %%
f, ax1 = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi, sharey=True)

ax1.plot(time_ax, dot_pca['mPFC'], color=colors['mPFC'])
ax1.text(time_ax[-1] + 0.005, dot_pca['mPFC'][-1] + 0.025, '*', color=colors['mPFC'],
         va='top', fontsize=12)
ax1.set(ylabel='Normalized dot product', yticks=[0, 0.2, 0.4],
        xticks=[-0.15, -0.1, -0.05, 0], xticklabels=[-150, -100, -50, 0],
        xlabel='Time to choice (ms)')

sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, 'dot_product_mpfc.pdf'))

# %% Plot angle

f, ax1 = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi, sharey=True)

for i, region in enumerate(dist_choice.keys()):
    if dist_choice[region][-1] > np.nanquantile(dist_choice_shuffle[region], 0.95, axis=1)[-1]:
        ax1.plot(time_ax, angle_pca[region], color=colors[region])
        #ax1.text(time_ax[-1] + 0.005, angle_pca[region][-1], region, color=colors[region],
        #         va='center')
        ax1.set(ylabel='Angle (deg)', title='Choice', yticks=[89.8, 90, 90.2],
                xticks=[-0.15, -0.1, -0.05, 0], xticklabels=[-0.15, -0.1, -0.05, 0])

sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, 'angle.pdf'))


# %% Plot example region

EXAMPLE_REGION = 'mPFC'

fig = plt.figure(figsize=(1.75, 1.75), dpi=dpi)
ax = fig.add_subplot(projection='3d')
ax.view_init(elev=-160, azim=220)
for sp in SPLITS:
    cmap = mpl.colormaps.get_cmap(CMAPS[sp])
    col = [cmap((n_timepoints - p) / n_timepoints) for p in range(n_timepoints)]
    ax.plot(pca_fit[EXAMPLE_REGION][split_ids == sp, 0],
            pca_fit[EXAMPLE_REGION][split_ids == sp, 1],
            pca_fit[EXAMPLE_REGION][split_ids == sp, 2],
            color=col[len(col) // 2], linewidth=1, alpha=0.5, zorder=0)
    ax.scatter(pca_fit[EXAMPLE_REGION][split_ids == sp, 0],
               pca_fit[EXAMPLE_REGION][split_ids == sp, 1],
               pca_fit[EXAMPLE_REGION][split_ids == sp, 2],
               color=col, edgecolors=col, s=10, depthshade=False, zorder=1)
#ax.grid('off')
#ax.axis('off')
ax.set(xticklabels=[], yticklabels=[], zticklabels=[])
plt.savefig(join(fig_path, 'pca_frontal-cortex.pdf'))

# %% Plot choice collapse

pca_l_col = (pca_fit[EXAMPLE_REGION][split_ids == 'L_opto'] + pca_fit['mPFC'][split_ids == 'L_no_opto']) / 2
pca_r_col = (pca_fit[EXAMPLE_REGION][split_ids == 'R_opto'] + pca_fit['mPFC'][split_ids == 'R_no_opto']) / 2

fig = plt.figure(figsize=(1.75, 1.75), dpi=dpi)
ax = fig.add_subplot(projection='3d')
ax.view_init(elev=-179, azim=225)

cmap_l = mpl.colormaps.get_cmap(CMAPS['L_collapsed'])
col_l = [cmap_l((n_timepoints - p) / n_timepoints) for p in range(n_timepoints)]
ax.plot(pca_l_col[:, 0], pca_l_col[:, 1], pca_l_col[:, 2],
        color=col_l[len(col) // 2], linewidth=1, alpha=0.5, zorder=0)
ax.scatter(pca_l_col[:, 0], pca_l_col[:, 1], pca_l_col[:, 2],
           color=col_l, edgecolors=col_l, s=10, depthshade=False, zorder=1)

cmap_r = mpl.colormaps.get_cmap(CMAPS['R_collapsed'])
col_r = [cmap_r((n_timepoints - p) / n_timepoints) for p in range(n_timepoints)]
ax.plot(pca_r_col[:, 0], pca_r_col[:, 1], pca_r_col[:, 2],
        color=col_r[len(col) // 2], linewidth=1, alpha=0.5, zorder=0)
ax.scatter(pca_r_col[:, 0], pca_r_col[:, 1], pca_r_col[:, 2],
           color=col_r, edgecolors=col_r, s=10, depthshade=False, zorder=1)
#ax.grid('off')
#ax.axis('off')
ax.set(xticklabels=[], yticklabels=[], zticklabels=[])
plt.savefig(join(fig_path, 'pca_frontal-cortex_choice.pdf'))

# %% Plot opto collapse

pca_opto_col = (pca_fit[EXAMPLE_REGION][split_ids == 'L_opto'] + pca_fit['mPFC'][split_ids == 'R_opto']) / 2
pca_no_opto_col = (pca_fit[EXAMPLE_REGION][split_ids == 'L_no_opto'] + pca_fit['mPFC'][split_ids == 'R_no_opto']) / 2

fig = plt.figure(figsize=(1.75, 1.75), dpi=dpi)
ax = fig.add_subplot(projection='3d')
ax.view_init(elev=-179, azim=225)

cmap_opto = mpl.colormaps.get_cmap(CMAPS['opto_collapsed'])
col_opto = [cmap_opto((n_timepoints - p) / n_timepoints) for p in range(n_timepoints)]
ax.plot(pca_opto_col[:, 0], pca_opto_col[:, 1], pca_opto_col[:, 2],
        color=col_opto[len(col) // 2], linewidth=1, alpha=0.5, zorder=0)
ax.scatter(pca_opto_col[:, 0], pca_opto_col[:, 1], pca_opto_col[:, 2],
           color=col_opto, edgecolors=col_opto, s=10, depthshade=False, zorder=1)

cmap_nopto = mpl.colormaps.get_cmap(CMAPS['no_opto_collapsed'])
col_nopto = [cmap_nopto((n_timepoints - p) / n_timepoints) for p in range(n_timepoints)]
ax.plot(pca_no_opto_col[:, 0], pca_no_opto_col[:, 1], pca_no_opto_col[:, 2],
        color=col_nopto[len(col) // 2], linewidth=1, alpha=0.5, zorder=0)
ax.scatter(pca_no_opto_col[:, 0], pca_no_opto_col[:, 1], pca_no_opto_col[:, 2],
           color=col_nopto, edgecolors=col_nopto, s=10, depthshade=False, zorder=1)
#ax.grid('off')
#ax.axis('off')
ax.set(xticklabels=[], yticklabels=[], zticklabels=[])
plt.savefig(join(fig_path, 'pca_frontal-cortex_opto.pdf'))

# %% Plot both

fig = plt.figure(figsize=(1.75, 1.75), dpi=dpi)
ax = fig.add_subplot(projection='3d')
ax.view_init(elev=-179, azim=225)

ax.plot(pca_opto_col[:, 0], pca_opto_col[:, 1], pca_opto_col[:, 2],
        color=col_opto[len(col) // 2], linewidth=1, alpha=0.5, zorder=0)
ax.scatter(pca_opto_col[:, 0], pca_opto_col[:, 1], pca_opto_col[:, 2],
           color=col_opto, edgecolors=col_opto, s=10, depthshade=False, zorder=1)

ax.plot(pca_no_opto_col[:, 0], pca_no_opto_col[:, 1], pca_no_opto_col[:, 2],
        color=col_nopto[len(col) // 2], linewidth=1, alpha=0.5, zorder=0)
ax.scatter(pca_no_opto_col[:, 0], pca_no_opto_col[:, 1], pca_no_opto_col[:, 2],
           color=col_nopto, edgecolors=col_nopto, s=10, depthshade=False, zorder=1)

ax.plot(pca_l_col[:, 0], pca_l_col[:, 1], pca_l_col[:, 2],
        color=col_l[len(col) // 2], linewidth=1, alpha=0.5, zorder=0)
ax.scatter(pca_l_col[:, 0], pca_l_col[:, 1], pca_l_col[:, 2],
           color=col_l, edgecolors=col_l, s=10, depthshade=False, zorder=1)

ax.plot(pca_r_col[:, 0], pca_r_col[:, 1], pca_r_col[:, 2],
        color=col_r[len(col) // 2], linewidth=1, alpha=0.5, zorder=0)
ax.scatter(pca_r_col[:, 0], pca_r_col[:, 1], pca_r_col[:, 2],
           color=col_r, edgecolors=col_r, s=10, depthshade=False, zorder=1)
#ax.grid('off')
#ax.axis('off')
ax.set(xticklabels=[], yticklabels=[], zticklabels=[])
plt.savefig(join(fig_path, 'pca_frontal-cortex_both.pdf'))



# %%

f, axs = plt.subplots(1, len(dist_opto.keys()), figsize=(7, 1.75), dpi=dpi, sharey=True)
for r, region in enumerate(dot_pca.keys()):
    axs[r].fill_between(time_ax,
                        np.quantile(dot_pca_shuffle[region], 0.05, axis=1),
                        np.quantile(dot_pca_shuffle[region], 0.95, axis=1),
                        color='grey', alpha=0.25, lw=0)
    axs[r].plot(time_ax, dot_pca[region], marker='o', ms=1.5)
    #axs[r].plot([0, 0], axs[r].get_ylim(), ls='--', color='grey', lw=0.75, zorder=0)
    #axs[r].set(title=f'{region}', yticks=[0, 800], ylim=[-20, 800], xlabel='Time to choice (s)')
    axs[r].set(title=f'{region}')

axs[0].set_ylabel('Dot product', labelpad=-10)
sns.despine(trim=True)

plt.tight_layout()

# %%

f, axs = plt.subplots(1, len(dist_opto.keys()), figsize=(7, 1.75), dpi=dpi, sharey=True)
for r, region in enumerate(angle_pca.keys()):
    axs[r].fill_between(time_ax,
                        np.quantile(angle_pca_shuffle[region], 0.05, axis=1),
                        np.quantile(angle_pca_shuffle[region], 0.95, axis=1),
                        color='grey', alpha=0.25, lw=0)
    axs[r].plot(time_ax, angle_pca[region], marker='o', ms=1.5)
    axs[r].plot(axs[r].get_xlim(), [90, 90], ls='--', color='grey')
    #axs[r].plot([0, 0], axs[r].get_ylim(), ls='--', color='grey', lw=0.75, zorder=0)
    #axs[r].set(title=f'{region}', yticks=[0, 800], ylim=[-20, 800], xlabel='Time to choice (s)')
    axs[r].set(title=f'{region}')

axs[0].set_ylabel('Angle', labelpad=-10)
sns.despine(trim=True)
plt.tight_layout()


# %%

f, axs = plt.subplots(1, len(dist_opto.keys()), figsize=(7, 1.75), dpi=dpi, sharey=True)
for r, region in enumerate(p_value.keys()):

    axs[r].plot(time_ax, p_value[region], marker='o', ms=1.5)
    #axs[r].plot(axs[r].get_xlim(), [90, 90], ls='--', color='grey')
    #axs[r].plot([0, 0], axs[r].get_ylim(), ls='--', color='grey', lw=0.75, zorder=0)
    #axs[r].set(title=f'{region}', yticks=[0, 800], ylim=[-20, 800], xlabel='Time to choice (s)')
    axs[r].set(title=f'{region}')

axs[0].set_ylabel('P', labelpad=-10)
sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, 'p_value.pdf'))


# %%

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
for r, region in enumerate(dot_pca.keys()):
    axs[r].fill_between(time_ax,
                        np.quantile(dot_pca_shuffle[region], 0.05, axis=1),
                        np.quantile(dot_pca_shuffle[region], 0.95, axis=1),
                        color='grey', alpha=0.25, lw=0)
    axs[r].plot(time_ax, dot_pca[region], marker='o', ms=1.5)
    #axs[r].plot([0, 0], axs[r].get_ylim(), ls='--', color='grey', lw=0.75, zorder=0)
    axs[r].set(title=f'{region}', yticks=[0, 800], ylim=[-20, 800], xlabel='Time to choice (s)')
axs[0].set_ylabel('Dot product', labelpad=-10)
sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, 'dot_prod_pca_all_regions.pdf'))
"""

