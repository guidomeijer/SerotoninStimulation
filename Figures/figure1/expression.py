# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 09:18:27 2025 by Guido Meijer
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from os.path import join, realpath, dirname, split
from stim_functions import paths, figure_style
colors, dpi = figure_style()

# Get paths
f_path, save_path = paths()
fig_path = join(f_path, split(dirname(realpath(__file__)))[-1])

# Load in data
expr_df = pd.read_csv(join(save_path, 'expression_levels.csv'))

# Plot overview plot
f, ax1 = plt.subplots(1, 1, figsize=(1.3, 1.6), dpi=dpi)
sns.swarmplot(x='sert-cre', y='rel_fluo', data=expr_df, order=[1, 0], color='k', ax=ax1,
              legend=None, size=2.5)
ax1.text(0, 410, f'n={np.sum(expr_df["sert-cre"])}', ha='center', va='center')
ax1.text(1, 410, f'n={np.sum(expr_df["sert-cre"] == 0)}', ha='center', va='center')
ax1.set(xticks=[0, 1], xticklabels=['SERT', 'WT'], ylabel='Relative fluoresence (%)', xlabel='Mice',
        yticks=[0, 200, 400])

plt.tight_layout()
sns.despine(trim=True)
plt.savefig(join(fig_path, 'expression_levels_SERT_WT.pdf'))

