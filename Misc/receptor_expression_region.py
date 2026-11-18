# -*- coding: utf-8 -*-
"""
Author: Guido Meijer
Date: 06/05/2026
"""
# %%

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from os.path import join
from stim_functions import paths, figure_style, remap
colors, dpi = figure_style()

# Load in expression
_, data_path = paths()
expr_df = pd.read_csv(join(data_path, 'receptor_expression.csv'))
expr_df = expr_df[~np.isin(expr_df['acronym'], ['MMme', 'CUL4, 5', 'CUL4, 5gr', 'CUL4, 5mo'])]
expr_df['region'] = remap(expr_df['acronym'])

# Plot
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(2.5, 1.75), dpi=dpi)
ax1.bar(expr_df)