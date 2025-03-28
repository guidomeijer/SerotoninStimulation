# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 12:09:32 2023

@author: Guido
"""

import numpy as np
import pandas as pd
import seaborn as sns
from glob import glob
from itertools import permutations
from scipy.stats import pearsonr
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from os.path import join, split
import matplotlib.pyplot as plt
from stim_functions import paths, figure_style, init_one, load_trials
one = init_one()

# Settings
PRE_TIME = 1
POST_TIME = 4
BIN_SIZE = 0.1
PLOT = False
N_STATES_SELECT = 'global'

# Plotting
colors, dpi = figure_style()

# Get paths
f_path, save_path = paths()
fig_path = join(f_path, 'Extra plots', 'State', 'Opto Task', f'{N_STATES_SELECT}')

# %% State matrix

# Get all files
all_files = glob(join(save_path, 'HMM', 'Task', f'{N_STATES_SELECT}', 'state_mat', '*.npy'))

for i, this_file in enumerate(all_files):

    # Get recording data
    subject = split(this_file)[1][:9]
    date = split(this_file)[1][10:20]
    probe = split(this_file)[1][21:28]
    region = split(this_file)[1][29:-4]

    # Load in brain states
    state_mat = np.load(this_file)

    # Load in trials
    eid = one.search(subject=subject, date=date)[0]
    trials = load_trials(eid, laser_stimulation=True, one=one)

    # Get number of states
    n_states = np.unique(state_mat).shape[0]

    # Plot
    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(1.75*4, 1.75), dpi=dpi)
    ax1.imshow(state_mat, aspect='auto',
               cmap=ListedColormap(colors['states_light']),
               vmin=0, vmax=n_states-1,
               extent=(-PRE_TIME, POST_TIME, 1, state_mat.shape[0]), interpolation=None)
    ax1.set(yticks=[1, state_mat.shape[0]], xticks=[-1, 0, 1, 2, 3, 4], title=f'{region}',
            ylabel='Trials')

    probe_opto_trials = (trials['probe_trial'] == 1) & (trials['laser_stimulation'] == 1)
    ax2.imshow(state_mat[probe_opto_trials], aspect='auto',
               cmap=ListedColormap(colors['states_light']),
               vmin=0, vmax=n_states-1,
               extent=(-PRE_TIME, POST_TIME, 1, np.sum(probe_opto_trials)), interpolation=None)
    ax2.set(yticks=[1, np.sum(probe_opto_trials)],
            xticks=[-1, 0, 1, 2, 3, 4], title='Probe trials')

    ax3.imshow(state_mat[trials['laser_stimulation'] == 1], aspect='auto',
               cmap=ListedColormap(colors['states_light']),
               vmin=0, vmax=n_states-1,
               extent=(-PRE_TIME, POST_TIME, 1, trials['laser_stimulation'].sum()), interpolation=None)
    ax3.set(yticks=[1, trials['laser_stimulation'].sum()], xticks=[-1, 0, 1, 2, 3, 4],
            title='Opto trials')

    ax4.imshow(state_mat[trials['laser_stimulation'] == 0], aspect='auto',
               cmap=ListedColormap(colors['states_light']),
               vmin=0, vmax=n_states-1,
               extent=(-PRE_TIME, POST_TIME, 1, (trials['laser_stimulation'] == 0).sum()),
               interpolation=None)
    ax4.set(yticks=[1,  (trials['laser_stimulation'] == 0).sum()], xticks=[-1, 0, 1, 2, 3, 4],
            title='No opto trials')

    sns.despine(trim=True)
    plt.tight_layout()
    plt.savefig(join(fig_path, f'{subject}_{date}_{probe}_{region}.jpg'), dpi=600)
    plt.savefig(join(fig_path, f'{subject}_{date}_{probe}_{region}.pdf'))
    plt.close(f)

    # %% Probability state plots

    # Contstruct time ax
    time_ax = np.arange(-PRE_TIME + BIN_SIZE/2, (POST_TIME+BIN_SIZE) - BIN_SIZE/2, BIN_SIZE)

    # Get all files
    all_files = glob(join(save_path, 'HMM', 'Task', f'{N_STATES_SELECT}', 'prob_mat', '*.npy'))

    state_colors = sns.color_palette(colors['states'], n_states)

    for i, this_file in enumerate(all_files):

        # Get recording data
        subject = split(this_file)[1][:9]
        date = split(this_file)[1][10:20]
        probe = split(this_file)[1][21:28]
        region = split(this_file)[1][29:-4]

        # Load in brain states
        prob_mat = np.load(this_file)

        # Load in trials
        eid = one.search(subject=subject, date=date)[0]
        trials = load_trials(eid, laser_stimulation=True, one=one)

        # Get number of states
        n_states = prob_mat.shape[2]

        # Plot
        f, axs = plt.subplots(2, 6, figsize=(7, 2.5), dpi=dpi)
        axs = np.concatenate(axs)
        for j in range(n_states):
            no_opto_mean = np.mean(prob_mat[trials['laser_stimulation'] == 0, :, j], axis=0)
            no_opto_sem = (np.std(prob_mat[trials['laser_stimulation'] == 0, :, j], axis=0)
                           / np.sqrt(trials['laser_stimulation'].sum()))
            axs[j].fill_between(time_ax, no_opto_mean - no_opto_sem, no_opto_mean + no_opto_sem,
                                alpha=0.25, color=colors['no-stim'], lw=0)
            axs[j].plot(time_ax, no_opto_mean, color=colors['no-stim'])
            opto_mean = np.mean(prob_mat[trials['laser_stimulation'] == 1, :, j], axis=0)
            opto_sem = (np.std(prob_mat[trials['laser_stimulation'] == 1, :, j], axis=0)
                        / np.sqrt(trials['laser_stimulation'].sum()))
            axs[j].fill_between(time_ax, opto_mean - opto_sem, opto_mean + opto_sem, alpha=0.25,
                                color=colors['states'][j], lw=0)
            axs[j].plot(time_ax, opto_mean, color=colors['states'][j])
            axs[j].plot([0, 0], [0, axs[j].get_ylim()[1]], ls='--', color='grey')
            axs[j].plot([-PRE_TIME-0.1, -PRE_TIME-0.1], [0, 0.1], color='k')
            axs[j].plot([-PRE_TIME-0.1, -0.1], [0, 0], color='k')
            axs[j].axis('off')
            #axs[j].set_title(f'State {j+1}', color=colors['states'][j])
        axs[0].text(-1.4, 0.05, '10%', ha='center', va='center', rotation=90)
        axs[0].text(-0.55, -0.015, '1s', ha='center', va='center')
        axs[-1].axis('off')
        sns.despine(trim=True)
        plt.tight_layout()
        
        plt.savefig(join(fig_path, f'{subject}_{date}_{probe}_{region}_stateprobs.jpg'), dpi=600)
        plt.savefig(join(fig_path, f'{subject}_{date}_{probe}_{region}_stateprobs.pdf'))

        plt.close(f)
