#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 11:42:01 2021

@author: guido
"""

from os.path import join, realpath, dirname, split
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from glob import glob
from matplotlib.patches import Rectangle
from stim_functions import init_one, figure_style, paths, get_dlc_XYs

# Init
one = init_one()
f_path, save_path = paths()
fig_path = join(f_path, split(dirname(realpath(__file__)))[-1])

# Settings
EID = '299bf322-e039-4444-8f04-67fde247ae5e'

# Load data
one.load_dataset(EID, dataset='_iblrig_leftCamera.raw.mp4', download_only=True)
roi_win = pos = one.load_dataset(EID, dataset='leftROIMotionEnergy.position.npy')
video_times, XYs = get_dlc_XYs(one, EID)

# Plot one frame
ses_path = one.eid2path(EID)
cap = cv2.VideoCapture(glob(join(ses_path, 'raw_video_data', '*.mp4'))[0])
cap.set(1, 1000)
ret, image = cap.retrieve()

# %%
colors, dpi = figure_style()
f, ax1 = plt.subplots(1, 1, figsize=(3, 3), dpi=dpi)
ax1.imshow(image)
rect = Rectangle((roi_win[2], roi_win[3]), roi_win[0], roi_win[1], linewidth=1,
                 edgecolor=sns.color_palette('Dark2')[1], facecolor='none')
ax1.plot(XYs['nose_tip'][1000, 0], XYs['nose_tip'][1000, 1], marker='x',
         color=sns.color_palette('Dark2')[2], markersize=5)
ax1.add_patch(rect)
ax1.axis('off')

plt.savefig(join(fig_path, 'example_video_frame.jpg'), dpi=600)

