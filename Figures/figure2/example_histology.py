# -*- coding: utf-8 -*-
"""
Created on Wed May 24 11:52:04 2023

By Guido Meijer
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import join, realpath, dirname, split, isfile
from matplotlib.patches import Rectangle
from stim_functions import paths, query_ephys_sessions, load_subjects, figure_style
from atlaselectrophysiology.load_histology import download_histology_data
from pathlib import Path
from iblatlas.atlas import AllenAtlas
ba = AllenAtlas()

# Settings
#RAW_DATA_PATH = 'E:\\Flatiron\\mainenlab\\Subjects'
#RAW_DATA_PATH = 'D:\\Flatiron\\mainenlab\\Subjects'
RAW_DATA_PATH = 'C:\\Users\\guido\\Data\\Flatiron\\mainenlab\\Subjects'
SUBJECTS = ['ZFM-03330', 'ZFM-02600']
AP_EXT = [-4400, -4600]
EXP_WIN_XY = [190, 130]  # top left point
CTRL_WIN_XY = [190, 85]
WIN_WIDTH = 80
WIN_HEIGHT = 30
f_path, save_path = paths()
fig_path = join(f_path, split(dirname(realpath(__file__)))[-1])

colors, dpi = figure_style()
f, axs = plt.subplots(2, 1, figsize=(1, 1.75), dpi=dpi)

for i, subject in enumerate(SUBJECTS):
    
    # Get paths to green and red channel of the histology data
    gr_path = Path(join(RAW_DATA_PATH, subject, 'histology', f'STD_ds_{subject}_GR.nrrd'))

    # Download histology if not already on disk
    if ~isfile(gr_path):
        _ = download_histology_data(subject, 'mainenlab')

    # Initialize Allen atlas objects
    gr_hist = AllenAtlas(hist_path=gr_path)

    all_rel_fluo = np.empty(np.arange(AP_EXT[0], AP_EXT[1]-25, -25).shape[0])
    for j, ap in enumerate(np.arange(AP_EXT[0], AP_EXT[1]-25, -25)):

        slice_im = np.moveaxis(gr_hist.slice(ap/1e6, axis=1), 0, 1)

        test_slice = slice_im[EXP_WIN_XY[1]:EXP_WIN_XY[1]+WIN_HEIGHT, EXP_WIN_XY[0]:WIN_WIDTH+EXP_WIN_XY[0]]
        control_slice = slice_im[CTRL_WIN_XY[1]:CTRL_WIN_XY[1]+WIN_HEIGHT, CTRL_WIN_XY[0]:WIN_WIDTH+EXP_WIN_XY[0]]

        all_rel_fluo[j] = (np.sum(test_slice[test_slice > np.percentile(test_slice, 99)])
                           / np.sum(control_slice[control_slice > np.percentile(control_slice, 99)]))

    rel_fluo = (np.max(all_rel_fluo) * 100) - 100

    # Plot figures
    slice_im = np.moveaxis(gr_hist.slice(
        np.arange(AP_EXT[0], AP_EXT[1]-25, -25)[np.argmax(all_rel_fluo)]/1e6, axis=1), 0, 1)
    axs[i].imshow(slice_im, cmap='bone', vmin=0,
                  vmax=np.mean(slice_im)+(np.std(slice_im)*4))
    axs[i].axis('off')
    axs[i].set(ylim=[200, 40], xlim=[150, 300])
    
plt.tight_layout()
plt.savefig(join(fig_path, 'example_histology.pdf'))