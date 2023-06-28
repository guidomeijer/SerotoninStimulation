#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 14:40:01 2020

@author: guido
"""

import pathlib
import pandas as pd
import numpy as np
import seaborn as sns
from mayavi import mlab
from os.path import join, realpath, dirname, split
import matplotlib.pyplot as plt
import ibllib.atlas as atlas
from matplotlib import colors
from atlaselectrophysiology import rendering
from stim_functions import query_ephys_sessions, paths, figure_style, load_subjects
from one.api import ONE
ba = atlas.AllenAtlas(25)
one = ONE()

# Set colormap
SUBJECT = 'ZFM-03330'
DATE = '2022-02-16'

# Paths
f_path, save_path = paths()
fig_path = join(f_path, split(dirname(realpath(__file__)))[-1])

# Plot insertions from example recording
rec = query_ephys_sessions(anesthesia='all')
rec = rec[(rec['subject'] == SUBJECT) & (rec['date'] == DATE)]

fig = rendering.figure(grid=False, size=(1024, 768))

for i, pid in enumerate(rec['pid']):
    ins_q = one.alyx.rest('trajectories', 'list', provenance='Ephys aligned histology track',
                          probe_insertion=pid)
    ins = atlas.Insertion(x=ins_q[0]['x'] / 1000000, y=ins_q[0]['y'] / 1000000,
                          z=ins_q[0]['z'] / 1000000, phi=ins_q[0]['phi'],
                          theta=ins_q[0]['theta'], depth=ins_q[0]['depth'] / 1000000)
    mlapdv = ba.xyz2ccf(ins.xyz)
    mlab.plot3d(mlapdv[:, 1], mlapdv[:, 2], mlapdv[:, 0],
                line_width=1, tube_radius=75,
                color=colors.to_rgb('tab:red'))

# Add fiber to plot
fiber = atlas.Insertion(x=0, y=-0.00664, z=-0.0005, phi=270, theta=32, depth=0.004)
mlapdv = ba.xyz2ccf(fiber.xyz)
mlab.plot3d(mlapdv[:, 1], mlapdv[:, 2], mlapdv[:, 0],
            line_width=1, tube_radius=150, color=(.6, .6, .6))
mlab.show()

