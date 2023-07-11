# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 16:11:41 2023

By Guido Meijer
"""

import numpy as np
from glob import glob
from os.path import join, split
from stim_functions import paths

# Get paths
f_path, save_path = paths()
fig_path = join(f_path, 'Extra plots', 'State', 'Regions')

# Get all files
all_files = glob(join(save_path, 'HMM', '*.npy'))

# Get all recordings
all_rec = np.unique([split(i)[1][:28] for i in all_files])

for i, this_rec in enumerate(all_rec):
    glob(join(save_path, ))
    