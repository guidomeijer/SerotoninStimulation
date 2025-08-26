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
import pandas as pd
import requests
import time
import zipfile
from stim_functions import figure_style, paths, remap, combine_regions
from sklearn.decomposition import PCA

N_DIM = 3
CHOICE_WIN = [-0.01, 0]
OPTO_WIN = [-0.01, 0]
ORTH_WIN = [-0.05, 0]
DROP_REGIONS = ['root', 'AI', 'BC', 'ZI', 'RSP']
CMAPS = dict({
    'L': 'Reds_r', 'R': 'Purples_r', 'no_opto': 'Oranges_r', 'opto': 'Blues_r',
    'L_opto': 'Reds_r', 'R_opto': 'Purples_r', 'L_no_opto': 'Oranges_r', 'R_no_opto': 'Blues_r'})

# Initialize
pca = PCA(n_components=N_DIM)
colors, dpi = figure_style()

# Get paths
f_path, load_path = paths()  # because these data are too large they are not on the repo
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
    
    regions = np.concatenate((regions, combine_regions(remap(this_dict['region']))))

# %%
# Get Eucledian distances in neural space between opto and no opto
dist_opto, dist_opto_shuffle = dict(), dict()
dist_choice, dist_choice_shuffle = dict(), dict()
dot_pca, dot_pca_shuffle = dict(), dict()   
for r, region in enumerate(np.unique(regions)):
    if region in DROP_REGIONS:
        continue
    print(f'Processing {region}')
    
    # Get Eucledian distance between opto and no opto per time point for L and R choices seperately
    dist_opto[region] = np.empty(n_timepoints)
    for t in range(n_timepoints):
        dist_opto[region][t] = np.linalg.norm(opto[regions == region, t] - no_opto[regions == region, t])
    
    # Do the same for shuffle
    dist_opto_shuffle[region] = np.empty((n_timepoints, opto_shuf.shape[2]))
    for ii in range(opto_shuf.shape[2]):
        for t in range(n_timepoints):
            dist_opto_shuffle[region][t, ii] = np.linalg.norm(opto_shuf[regions == region, t, ii]
                                                              - no_opto_shuf[regions == region, t, ii])
        
    # Get Eucledian distance between opto and no opto per time point for L and R choices seperately
    dist_choice[region] = np.empty(n_timepoints)
    for t in range(n_timepoints):
        dist_choice[region][t] = np.linalg.norm(L[regions == region, t] - R[regions == region, t])
    
    # Do the same for shuffle
    dist_choice_shuffle[region] = np.empty((n_timepoints, L_shuf.shape[2]))
    for ii in range(L_shuf.shape[2]):
        for t in range(n_timepoints):
            dist_choice_shuffle[region][t, ii] = np.linalg.norm(L_shuf[regions == region, t, ii]
                                                                - R_shuf[regions == region, t, ii])

    # Get dot product
    # Do PCA for data split four ways
    all_splits = np.vstack((L_opto[regions == region].T, R_opto[regions == region].T,
                            L_no_opto[regions == region].T, R_no_opto[regions == region].T))
    pca_fit = pca.fit_transform(all_splits)
    
    # Do PCA for shuffles
    pca_shuffle = np.empty((all_splits.shape[0], N_DIM, 0))
    for ii in range(this_dict['n_shuffles']):
        all_splits = np.vstack((L_opto_shuf[regions == region, :, ii].T,
                                R_opto_shuf[regions == region, :, ii].T,
                                L_no_opto_shuf[regions == region, :, ii].T,
                                R_no_opto_shuf[regions == region, :, ii].T))
        if np.sum(np.isnan(all_splits)) == 0:
            pca_shuffle = np.dstack((pca_shuffle, pca.fit_transform(all_splits)))
    
    # Get index to which split the PCA belongs to
    split_ids = np.concatenate((['L_opto'] * n_timepoints, ['R_opto'] * n_timepoints,
                                ['L_no_opto'] * n_timepoints, ['R_no_opto'] * n_timepoints))
    

    # Calculate dot product between opto and stim vectors in PCA space
    # Collapse pca onto choice and opto dimensions 
    pca_l_col = (pca_fit[split_ids == 'L_opto'] + pca_fit[split_ids == 'L_no_opto']) / 2
    pca_r_col = (pca_fit[split_ids == 'R_opto'] + pca_fit[split_ids == 'R_no_opto']) / 2
    pca_opto_col = (pca_fit[split_ids == 'L_opto'] + pca_fit[split_ids == 'R_opto']) / 2
    pca_no_opto_col = (pca_fit[split_ids == 'L_no_opto'] + pca_fit[split_ids == 'R_no_opto']) / 2
    
    # Get the dot product and angle between the two vectors
    dot_pca[region] = np.empty(n_timepoints)
    for t in range(n_timepoints):
    
        # Calculate normalized dot product
        choice_vec = pca_l_col[t, :] - pca_r_col[t, :]
        opto_vec = pca_opto_col[t, :] - pca_no_opto_col[t, :]
        dot_prod = np.dot(choice_vec / np.linalg.norm(choice_vec),
                          opto_vec / np.linalg.norm(opto_vec))
        dot_pca[region][t] = 1 - np.abs(dot_prod)
    
    # Do the same for all the shuffles
    dot_pca_shuffle[region] = np.empty((n_timepoints, pca_shuffle.shape[2]))
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
            
            # Calculate normalized dot product
            choice_vec = pca_l_col[t, :] - pca_r_col[t, :]
            opto_vec = pca_opto_col[t, :] - pca_no_opto_col[t, :]
    
            dot_prod = np.dot(choice_vec / np.linalg.norm(choice_vec),
                              opto_vec / np.linalg.norm(opto_vec))
            dot_pca_shuffle[region][t, ii] = 1 - np.abs(dot_prod)

    

# %% Prepare for plotting

# Get mean over timewindow per region
choice_dist_pca_regions = [np.mean(dist_choice[i][
    (time_ax >= CHOICE_WIN[0]) & (time_ax <= CHOICE_WIN[1])])
    for i in dist_choice.keys()]
choice_dist_pca_shuf_regions = [np.mean(dist_choice_shuffle[i][
    (time_ax >= CHOICE_WIN[0]) & (time_ax <= CHOICE_WIN[1]), :], 0)
    for i in dist_choice_shuffle.keys()]
opto_dist_pca_regions = [np.mean(dist_opto[i][
    (time_ax >= OPTO_WIN[0]) & (time_ax <= OPTO_WIN[1])])
    for i in dist_opto.keys()]
opto_dist_pca_shuf_regions = [np.mean(dist_opto_shuffle[i][
    (time_ax >= OPTO_WIN[0]) & (time_ax <= OPTO_WIN[1]), :], 0)
    for i in dist_opto_shuffle.keys()]
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
    'region': [string for string, sublist in zip(dist_choice_shuffle.keys(), choice_dist_pca_shuf_regions) for _ in sublist]})

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
ax1.set(xlabel='', ylabel='Choice distance (spks/s)',
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
ax1.set(xlabel='', ylabel='5-HT distance (spks/s)', yticks=[0, 50], ylim=[0, 50])

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



