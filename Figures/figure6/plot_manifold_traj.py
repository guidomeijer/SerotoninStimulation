# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 16:04:33 2023

@author: Guido
"""

from manifold.state_space import plot_grand_traj, plot_traj_and_dist, plot_all

SPLIT = 'choice'
plot_grand_traj(SPLIT)

plot_traj_and_dist('choice', 'mPFC')

#plot_all(['stim', 'fback', 'choice'])
#plot_all(['choice', 'stim', 'fback', 'opto'])