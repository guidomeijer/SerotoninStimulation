#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 14:40:01 2020

@author: guido
"""

import numpy as np
from mayavi import mlab
from os.path import join, realpath, dirname, split
from iblatlas import atlas
from matplotlib.transforms import Affine2D, offset_copy
import matplotlib.pyplot as plt
from atlaselectrophysiology import rendering
from stim_functions import query_ephys_sessions, paths, figure_style, load_subjects
from one.api import ONE
ba = atlas.AllenAtlas(25)
one = ONE()

RENDER_VIDEO = False

# Paths
f_path, save_path = paths()
fig_path = join(f_path, split(dirname(realpath(__file__)))[-1])

# Load in recordings and subjects
rec = query_ephys_sessions(anesthesia='all')
subjects = load_subjects()

plot_colors, dpi = figure_style()
fig = rendering.figure(grid=False, size=(1024, 768))
for i, pid in enumerate(rec['pid']):
    ins_q = one.alyx.rest('trajectories', 'list', provenance='Ephys aligned histology track',
                          probe_insertion=pid)
    ins = atlas.Insertion(x=ins_q[0]['x'] / 1000000, y=ins_q[0]['y'] / 1000000,
                          z=ins_q[0]['z'] / 1000000, phi=ins_q[0]['phi'],
                          theta=ins_q[0]['theta'], depth=ins_q[0]['depth'] / 1000000)
    mlapdv = ba.xyz2ccf(ins.xyz)
    sub_nr = subjects.loc[subjects['subject'] == rec.loc[rec['pid'] == pid, 'subject'].values[0], 'subject_nr'].values[0]
    mlab.plot3d(mlapdv[:, 1], mlapdv[:, 2], mlapdv[:, 0],
                line_width=1, tube_radius=40,
                color=tuple(plot_colors['subject_palette'][sub_nr]))
    
# Add fiber to plot
fiber = atlas.Insertion(x=0, y=-0.00664, z=-0.0005, phi=270, theta=32, depth=0.004)
mlapdv = ba.xyz2ccf(fiber.xyz)
mlab.plot3d(mlapdv[:, 1], mlapdv[:, 2], mlapdv[:, 0],
            line_width=1, tube_radius=150, color=(.6, .6, .6))

if RENDER_VIDEO:
    rendering.rotating_video(join(fig_path, 'rotation_brain_insertions.avi'), fig, fps=30, secs=12)

mlab.show()

# %% Plot subject number in corresponding color


def rainbow_text(x, y, strings, colors, orientation='horizontal',
                 ax=None, **kwargs):
    """
    Take a list of *strings* and *colors* and place them next to each
    other, with text strings[i] being shown in colors[i].

    Parameters
    ----------
    x, y : float
        Text position in data coordinates.
    strings : list of str
        The strings to draw.
    colors : list of color
        The colors to use.
    orientation : {'horizontal', 'vertical'}
    ax : Axes, optional
        The Axes to draw into. If None, the current axes will be used.
    **kwargs
        All other keyword arguments are passed to plt.text(), so you can
        set the font size, family, etc.
    """
    if ax is None:
        ax = plt.gca()
    t = ax.transData
    fig = ax.figure
    canvas = fig.canvas

    assert orientation in ['horizontal', 'vertical']
    if orientation == 'vertical':
        kwargs.update(rotation=90, verticalalignment='bottom')

    for s, c in zip(strings, colors):
        text = ax.text(x, y, s + " ", color=c, transform=t, **kwargs)

        # Need to draw to update the text position.
        text.draw(canvas.get_renderer())
        ex = text.get_window_extent()
        # Convert window extent from pixels to inches
        # to avoid issues displaying at different dpi
        ex = fig.dpi_scale_trans.inverted().transform_bbox(ex)

        if orientation == 'horizontal':
            t = text.get_transform() + \
                offset_copy(Affine2D(), fig=fig, x=ex.width, y=0)
        else:
            t = text.get_transform() + \
                offset_copy(Affine2D(), fig=fig, x=0, y=ex.height)


f, ax = plt.subplots(1, 1, figsize=(2.48, 0.11), dpi=dpi)
sub_str = str(np.arange(len(subjects))+1)[1:-1].split()
rainbow_text(0, 0.1, sub_str, plot_colors['subject_palette'], ax=ax, size=7)
ax.axis('off')

plt.subplots_adjust(left=0)
plt.savefig(join(fig_path, 'mouse_colors.pdf'))



