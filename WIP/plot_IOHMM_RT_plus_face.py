# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 11:13:53 2025

By Guido Meijer
"""

import numpy as np
from os import path
from glob import glob
import pandas as pd
from scipy.stats import zscore
import seaborn as sns
import matplotlib.pyplot as plt
from stim_functions import paths, figure_style
colors, dpi = figure_style()

# Settings
PLOT_SES = False
WIN_STARTS = np.arange(-20, 70) 
WIN_SIZE = 15
trial_win_labels = WIN_STARTS + (WIN_SIZE/2)

# Get processed sessions
fig_path, save_path = paths()
ses = glob(path.join(save_path, 'IOHMM', '*.csv'))

# Load in all sessions into one big dataframe
all_trials, block_df = pd.DataFrame(), pd.DataFrame()
for i, ses_path in enumerate(ses):
    
    # Load in this session
    trials_df = pd.read_csv(ses_path)
    trials_df['subject'] = path.split(ses_path)[1][:9]
    trials_df['date'] = path.split(ses_path)[1][10:20]
    
    # Z-score metrics for plotting
    trials_df['pupil_zscore'] = zscore(trials_df['pupil'])
    trials_df['whisking_zscore'] = zscore(trials_df['whisking'])
    trials_df['sniffing_zscore'] = zscore(trials_df['sniffing'])
    trials_df['rt_zscore'] = zscore(np.log10(trials_df['reaction_times']))
    
    # Remove probe trials
    trials_df.loc[(trials_df['laser_probability'] == 0.25)
                  & (trials_df['laser_stimulation'] == 1), 'laser_stimulation'] = 0
    trials_df.loc[(trials_df['laser_probability'] == 0.75)
                  & (trials_df['laser_stimulation'] == 0), 'laser_stimulation'] = 1

    # Get states centered at opto block switches
    all_blocks = 0
    trials_df['opto_block_switch'] = np.concatenate((
        [False], np.diff(trials_df['laser_stimulation']) != 0))
    opto_block_switch_ind = trials_df[trials_df['opto_block_switch']].index
    for b, trial_ind in enumerate(opto_block_switch_ind):
        all_blocks += 1
          
        # Binned trials
        this_p_eng, this_p_expl, this_p_diseng, these_trial_bins = [], [], [], []
        for tt, this_start in enumerate(WIN_STARTS-1):
            trial_win = trials_df[trial_ind+this_start:trial_ind+this_start+WIN_SIZE]
            if trial_win.shape[0] == WIN_SIZE:
                this_p_eng.append(trial_win['p_engaged'].mean())
                this_p_expl.append(trial_win['p_exploratory'].mean())
                this_p_diseng.append(trial_win['p_disengaged'].mean())
                these_trial_bins.append(trial_win_labels[tt])
        this_p_eng = np.array(this_p_eng)
        this_p_expl = np.array(this_p_expl)
        this_p_diseng = np.array(this_p_diseng)   
        these_trial_bins = np.array(these_trial_bins)
                                
        block_df = pd.concat((block_df, pd.DataFrame(data={
            'p_engaged': this_p_eng, 'p_exploratory': this_p_expl,  'p_disengaged': this_p_diseng,
            'p_eng_bl': (this_p_eng - np.mean(this_p_eng[these_trial_bins < -5])) * 100,
            'p_expl_bl': (this_p_expl - np.mean(this_p_expl[these_trial_bins < -5])) * 100,
            'p_diseng_bl': (this_p_diseng - np.mean(this_p_diseng[these_trial_bins < -5])) * 100,
            'trial_bin': these_trial_bins,
            'opto_switch': all_blocks, 'subject': path.split(ses_path)[1][:9],
            'opto': trials_df.loc[trial_ind, 'laser_stimulation']})), ignore_index=True)
    
    # Add to df
    all_trials = pd.concat((all_trials, trials_df))
        
    # Plot session
    if PLOT_SES:
        f, ax1 = plt.subplots(1, 1, figsize=(6, 2), dpi=dpi)
        
        ax1.plot(np.arange(trials_df.shape[0]), trials_df['p_engaged'], label='Engaged', zorder=0)
        ax1.plot(np.arange(trials_df.shape[0]), trials_df['p_exploratory'], label='Exploratory', zorder=0)
        ax1.plot(np.arange(trials_df.shape[0]), trials_df['p_disengaged'], label='Disengaged', zorder=0)
        
        norm_rt = ((trials_df['rt_log'] - np.min(trials_df['rt_log']))
                   / (np.max(trials_df['rt_log']) - np.min(trials_df['rt_log'])))
        ax1.scatter(np.arange(trials_df.shape[0]), norm_rt, color='magenta', label='RT')
        norm_whisk = ((trials_df['whisking'] - np.min(trials_df['whisking']))
                      / (np.max(trials_df['whisking']) - np.min(trials_df['whisking'])))        
        ax1.scatter(np.arange(trials_df.shape[0]), norm_whisk, color='cyan', label='Whisking')
        norm_snif = ((trials_df['sniffing'] - np.min(trials_df['sniffing']))
                     / (np.max(trials_df['sniffing']) - np.min(trials_df['sniffing'])))        
        ax1.scatter(np.arange(trials_df.shape[0]), norm_snif, color='gold', label='Sniffing')
                
        ax1.set(xlim=[250, 300], xlabel='Trials')
        ax1.legend(bbox_to_anchor=(1.2, 0.4))
        
        plt.tight_layout()
        sns.despine(trim=True)
        plt.savefig(path.join(fig_path, 'Extra plots', 'IOHMM', f'{path.split(ses_path)[1][:-4]}.jpg'))
        plt.close(f)
  
# Calculate means per subject for opto and no opto
per_sub_opto = all_trials.groupby(['subject', 'laser_stimulation']).mean(numeric_only=True)    
per_sub_diff = (per_sub_opto.xs(key=1, level='laser_stimulation')
                - per_sub_opto.xs(key=0, level='laser_stimulation')) * 100

# Calculate means per subject for the three states
per_sub_state = all_trials.groupby(['subject', 'state']).mean(numeric_only=True)    
per_sub_diff = (per_sub_opto.xs(key=1, level='laser_stimulation')
                - per_sub_opto.xs(key=0, level='laser_stimulation')) * 100

# Get mean per subject for block switches
per_sub_block = block_df.groupby(['subject', 'opto', 'trial_bin']).mean(numeric_only=True).reset_index()


# %% Plot

f, ax1 = plt.subplots(1, 1, figsize=(2, 1.75), dpi=dpi)
per_state = all_trials.groupby('state').mean(numeric_only=True)
im = ax1.imshow(per_state[['rt_zscore', 'pupil_zscore', 'whisking_zscore', 'sniffing_zscore']].to_numpy().T,
                cmap='coolwarm', vmin=-0.3, vmax=0.3)
ax1.set(yticks=[0, 1, 2, 3], yticklabels=['RT', 'Pupil', 'Whisking', 'Sniffing'], 
        xticks=[0, 1, 2], xticklabels=[1, 2, 3], xlabel='State')
f.colorbar(im)
plt.tight_layout()
plt.savefig(path.join(fig_path, 'Extra plots', 'RT_plus_face_IOHMM_states.jpg'), dpi=600)

# %%

long_df = per_sub_diff.melt(value_vars=['p_engaged', 'p_exploratory', 'p_disengaged'])
f, ax1 = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)
sns.swarmplot(data=long_df, x='variable', y='value')
ax1.plot([-0.5, 2.5], [0, 0], ls='--', lw=0.75, color='grey')
ax1.set(ylabel='5-HT induced change (%)', xlabel='', xticks=[0, 1, 2],
        xticklabels=['Eng.', 'Expl.', 'Diseng.'], yticks=[-5, 0, 5])


plt.tight_layout()
sns.despine(trim=True)
plt.savefig(path.join(fig_path, 'Extra plots', 'RT_plus_face_IOHMM_change.jpg'), dpi=600)

# %% Plot
    
f, ax1 = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)
long_df = pd.melt(per_sub_block[per_sub_block['opto'] == 1], id_vars=['trial_bin'],
                  value_vars=['p_eng_bl', 'p_expl_bl', 'p_diseng_bl'])
sns.lineplot(data=long_df, x='trial_bin', y='value', hue='variable', ax=ax1,
             errorbar='se', err_kws={'lw': 0})
ax1.plot(ax1.get_xlim(), [0, 0], ls='--', color='grey', lw=0.5)

ax1.set(ylabel='State probability (%)', yticks=[-10, -5, 0, 5],
        xticks=[-20, 0, 20, 40, 60, 80],
        xlabel='Trials since 5-HT start')

leg_handles, _ = ax1.get_legend_handles_labels()
leg_labels = ['Engaged', 'Exploratory', 'Disengaged']
ax1.legend(leg_handles, leg_labels, prop={'size': 5}, bbox_to_anchor=[0.6, 0.4], frameon=False)

sns.despine(trim=True)
plt.tight_layout()
plt.savefig(path.join(fig_path, 'Extra plots', 'RT_plus_face_IOHMM_baseline.jpg'), dpi=600)

# %%

f, ax1 = plt.subplots(1, 1, figsize=(1.75, 2), dpi=dpi)
long_df = pd.melt(per_sub_block[per_sub_block['opto'] == 1], id_vars=['trial_bin'],
                  value_vars=['p_engaged', 'p_exploratory', 'p_disengaged'])
sns.lineplot(data=long_df, x='trial_bin', y='value', hue='variable', ax=ax1,
             errorbar='se', err_kws={'lw': 0})
ax1.plot(ax1.get_xlim(), [0, 0], ls='--', color='grey', lw=0.5)

ax1.set(ylabel='Prob. of engaged state (%)',
        xticks=[-20, 0, 20, 40, 60, 80],
       xlabel='Trials since 5-HT start')

leg_handles, _ = ax1.get_legend_handles_labels()
leg_labels = ['Engaged', 'Exploratory', 'Disengaged']
ax1.legend(leg_handles, leg_labels, prop={'size': 5}, bbox_to_anchor=[0.52, 1.1], frameon=False)

sns.despine(trim=True)
plt.tight_layout()
plt.savefig(path.join(fig_path, 'Extra plots', 'RT_plus_face_IOHMM.jpg'), dpi=600)
