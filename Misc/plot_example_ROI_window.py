#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 11:42:01 2021

@author: guido
"""

from os.path import join
import matplotlib.pyplot as plt
import cv2
from glob import glob
from matplotlib.patches import Rectangle
from stim_functions import init_one, figure_style, paths

# Init
one = init_one()
fig_path, save_path = paths()
fig_path = join(fig_path, 'Extra plots', 'Facial movement')

# Settings
EID = '299bf322-e039-4444-8f04-67fde247ae5e'

# Load data
one.load_dataset(EID, dataset='_iblrig_leftCamera.raw.mp4', download_only=True)
roi_win = pos = one.load_dataset(EID, dataset='leftROIMotionEnergy.position.npy')

# Plot one frame
ses_path = one.eid2path(EID)
cap = cv2.VideoCapture(glob(join(ses_path, 'raw_video_data', '*.mp4'))[0])
cap.set(1, 1000)
ret, image = cap.retrieve()

colors, dpi = figure_style()
f, ax1 = plt.subplots(1, 1, figsize=(3, 3), dpi=dpi)
ax1.imshow(image)
rect = Rectangle((roi_win[2], roi_win[3]), roi_win[0], roi_win[1], linewidth=1, edgecolor='red', facecolor='none')
ax1.add_patch(rect)
ax1.axis('off')
plt.savefig(join(fig_path, 'example_ROI.jpg'), dpi=600)
