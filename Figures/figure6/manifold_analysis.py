# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 11:22:23 2023

By Guido Meijer
"""

import numpy as np
from os.path import join, realpath, dirname, split
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import pandas as pd
import matplotlib as mpl
from stim_functions import figure_style, paths
from sklearn.decomposition import PCA

#SPLITS = ['L_no_opto', 'R_no_opto']
SPLITS = ['L_opto', 'R_opto', 'L_no_opto', 'R_no_opto']
CMAPS = dict({'L_opto': 'Reds_r', 'R_opto': 'Purples_r', 'L_no_opto': 'Oranges_r', 'R_no_opto': 'Blues_r',
              'L_collapsed': 'Reds_r', 'R_collapsed': 'Purples_r', 'no_opto_collapsed': 'Oranges_r', 'opto_collapsed': 'Blues_r'})

# Initialize
pca = PCA(n_components=3)

# Get paths
f_path, save_path = paths()
fig_path = join(f_path, split(dirname(realpath(__file__)))[-1])

# Load in data
peths_df = pd.read_pickle(join(save_path, 'psth_task.pickle'))
peths_df = peths_df[(peths_df['high_level_region'] != 'root') & (peths_df['sert-cre'] == 1)].reset_index()

colors, dpi = figure_style()
fig = plt.figure(figsize=(7, 1.75))
axs = []
gs = fig.add_gridspec(1, 7) 

for r, region in enumerate(np.unique(peths_df['high_level_region'])):
    
    # Get array of all PETHs and select time limits
    time_ax = peths_df['time'][0]
       
    # Create subplot
    axs.append(fig.add_subplot(gs[0, r], projection='3d'))
    
    # Get stacked psth data
    for i, sp in enumerate(SPLITS):
        psth_data = np.array(peths_df.loc[(peths_df['high_level_region'] == region)
                                       & (peths_df['split'] == sp), 'peth'].tolist())
        n_obs = psth_data.shape[1]
        if i == 0:
            all_psth_data = psth_data.T
        else:
            all_psth_data = np.vstack((all_psth_data, psth_data.T))
                
    # Dimensionality reduction by PCA
    pca_fit = pca.fit_transform(all_psth_data)
    
    # Plot
    for i, sp in enumerate(SPLITS):
        this_pca_fit = pca_fit[n_obs * i : n_obs * (i + 1)]
        
        cmap = mpl.colormaps.get_cmap(CMAPS[sp])
        col = [cmap((n_obs - p) / n_obs) for p in range(n_obs)]

    
        axs[r].plot(this_pca_fit[:, 0], this_pca_fit[:, 1], this_pca_fit[:, 2],
                    color=col[len(col) // 2], linewidth=5, alpha=0.5)
    
        axs[r].scatter(this_pca_fit[:, 0], this_pca_fit[:, 1], this_pca_fit[:, 2],
                       color=col, edgecolors=col, s=20, depthshade=False)
        
        axs[r].grid('off')
        axs[r].axis('off')
        axs[r].set_title(f'{region}')
        
plt.tight_layout()

# %% Frontal cortex
   


# Create subplot
fig = plt.figure(figsize=(1.75, 1.75), dpi=dpi)
ax = fig.add_subplot(projection='3d')
ax.view_init(elev=10, azim=0)

# Get stacked psth data
for i, sp in enumerate(SPLITS):
    psth_data = np.array(peths_df.loc[(peths_df['high_level_region'] == 'Frontal cortex')
                                   & (peths_df['split'] == sp), 'peth'].tolist())
    n_obs = psth_data.shape[1]
    if i == 0:
        all_psth_data = psth_data.T
    else:
        all_psth_data = np.vstack((all_psth_data, psth_data.T))
         
# Dimensionality reduction by PCA on stacked PSTH
all_pca_fit = pca.fit_transform(all_psth_data)

# Unstack pca projection
pca_fit = dict()
for i, sp in enumerate(SPLITS):
    pca_fit[sp] = all_pca_fit[n_obs * i : n_obs * (i + 1)]
    
# Get choice and opto vector 
pca_fit['L_collapsed'] = (pca_fit['L_opto'] + pca_fit['L_no_opto']) / 2
pca_fit['R_collapsed'] = (pca_fit['R_opto'] + pca_fit['R_no_opto']) / 2
pca_fit['opto_collapsed'] = (pca_fit['L_opto'] + pca_fit['R_opto']) / 2
pca_fit['no_opto_collapsed'] = (pca_fit['L_no_opto'] + pca_fit['R_no_opto']) / 2

# Get dot product
dot_prod = np.empty(pca_fit['L_opto'].shape[0])
for t in range(pca_fit['L_opto'].shape[0]):
    choice_vec = pca_fit['L_collapsed'][t, :] - pca_fit['R_collapsed'][t, :]
    opto_vec = pca_fit['opto_collapsed'][t, :] - pca_fit['no_opto_collapsed'][t, :]
    dot_prod[t] = np.abs(np.dot(choice_vec, opto_vec))

# Plot
for i, sp in enumerate(SPLITS):
    cmap = mpl.colormaps.get_cmap(CMAPS[sp])
    col = [cmap((n_obs - p) / n_obs) for p in range(n_obs)]

    ax.plot(pca_fit[sp][:, 0], pca_fit[sp][:, 1], pca_fit[sp][:, 2],
                color=col[len(col) // 2], linewidth=5, alpha=0.5)
    ax.scatter(pca_fit[sp][:, 0], pca_fit[sp][:, 1], pca_fit[sp][:, 2],
                   color=col, edgecolors=col, s=20, depthshade=False)

ax.set(yticklabels=[], xticklabels=[], zticklabels=[])
#ax.grid('off')
#ax.axis('off')
    
    
    
    