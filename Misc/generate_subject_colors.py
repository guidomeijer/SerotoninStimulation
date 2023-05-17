# -*- coding: utf-8 -*-
"""
Created on Wed May 17 10:13:16 2023

By Guido Meijer
"""

import numpy as np
import json
from os.path import join
import seaborn as sns
from matplotlib import colors
from stim_functions import load_subjects, paths

_, save_path = paths()
subjects = load_subjects()

use_colors = np.concatenate((sns.color_palette('tab20'), [colors.to_rgb('maroon'), np.array([0, 0, 0])]))

subject_colors = dict()
for i, subject in enumerate(subjects['subject'].values):
        subject_colors[subject] = list(use_colors[i])
        
# Save
with open(join(save_path, 'subject_colors.json'), 'w') as outfile:
    json.dump(subject_colors, outfile)
    