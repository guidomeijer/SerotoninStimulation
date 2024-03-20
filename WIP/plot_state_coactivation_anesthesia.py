# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 10:56:02 2023

@author: Guido
"""

import numpy as np
import pandas as pd
import seaborn as sns
from glob import glob
from matplotlib.patches import Rectangle
from scipy.stats import pearsonr
from os.path import join, split
import matplotlib.pyplot as plt
from stim_functions import paths, figure_style, load_subjects

N_STATES_SELECT = 'global'
RANDOM_TIMES = 'jitter'

# Plotting
colors, dpi = figure_style()

# Get paths
f_path, save_path = paths()
fig_path = join(f_path, 'Extra plots', 'State')

# Load in data
coact_df = pd.read_csv(join(save_path, 'state_coactivation_anesthesia.csv'))

# Add genotype
subjects = load_subjects()
for i, nickname in enumerate(np.unique(subjects['subject'])):
    coact_df.loc[coact_df['subject'] == nickname,
                 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]
   




# %% Passive over time
f, ax1 = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)
#ax1.add_patch(Rectangle((0, 0), 1, 0.6, color='royalblue', alpha=0.25, lw=0))
sns.lineplot(data=coact_df[coact_df['sert-cre'] == 1], x='time', y='coact',
             errorbar='se', hue='opto', err_kws={'lw': 0},
             hue_order=[0, 1], palette=[colors['no-stim'], colors['stim']])
#ax1.set(xlabel='Time from stimulation start (s)', ylim=[0.3, 0.5],
#        xticks=[-1, 0, 1, 2, 3], yticks=[0.3, 0.5], yticklabels=[0.3, 0.5])
# ylim=[-0.001, 0.005], yticks=[0, 0.005],
# yticklabels=[0, 0.005])
ax1.set_ylabel('State coactivation', labelpad=-5)
g = ax1.legend(title='', bbox_to_anchor=(0.6, 0.3), prop={'size': 5})
new_labels = ['label 1', 'label 2']
for t, l in zip(g.texts, ['No stim', 'Stim']):
    t.set_text(l)
sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, 'state_coactivation_anesthesia.jpg'), dpi=600)

