# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 13:58:31 2023

@author: Guido Meijer
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from os.path import join, realpath, dirname, split
from stim_functions import paths, figure_style, load_subjects, high_level_regions

# Get paths
f_path, save_path = paths()
fig_path = join(f_path, split(dirname(realpath(__file__)))[-1])

# Load in results
task_neurons = pd.read_csv(join(save_path, 'task_modulated_neurons.csv'))

# Add expression
subjects = load_subjects()
for i, nickname in enumerate(np.unique(subjects['subject'])):
    task_neurons.loc[task_neurons['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]

# Select significant neurons from sert-cre animals
task_neurons = task_neurons[task_neurons['task_responsive'] & task_neurons['sert-cre']]

# %%
colors, dpi = figure_style()
f, ax1 = plt.subplots(figsize=(1.75, 1.75), dpi=dpi)
sns.scatterplot(data=task_neurons, x='task_opto_roc', y='task_no_opto_roc', ax=ax1)