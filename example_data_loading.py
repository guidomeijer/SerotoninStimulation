# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 15:27:43 2025

By Guido Meijer
"""

from brainbox.io.one import SpikeSortingLoader
from stim_functions import (init_one, paths, query_ephys_sessions, load_passive_opto_times,
                            load_trials)
from iblatlas.atlas import AllenAtlas
ba = AllenAtlas()

# Initalize connection to ONE
one = init_one()

# Set up the paths on your computer, if this is the first time you run this script you will be 
# prompted to input the path to the path where you want to save the figures and the path where 
# you want to save any processed data. It also automatically saves the path to the cloned repository
# which you need to load the times of the optogenetic stimulation.
fig_path, save_path = paths()

# Query ephys recordings done in the hippocampus
rec = query_ephys_sessions(acronym='CA1', one=one)

# For the first recording load in the trials. The column laser_stimulation indicates which trials
# had 5-HT stimulation 
trials_df = load_trials(rec.loc[0, 'eid'], laser_stimulation=True, one=one)

# Now load in the times of the passive serotonin stimulation after the task
passive_opto_times = load_passive_opto_times(rec.loc[0, 'eid'], one=one)

# This is how you load in the neural data 
sl = SpikeSortingLoader(pid=rec.loc[0, 'pid'], one=one, atlas=ba)
spikes, clusters, channels = sl.load_spike_sorting()
clusters = sl.merge_clusters(spikes, clusters, channels)

"""
There are two kinds of identifiers: eid and pid. The eid is the identifier of the session whereas
the pid is the identifier of the probe insertion. A session typically has two probes so for one
eid there can be two different pids. Trials for example are session-level so you use eid to
load them in. Neural data is probe insertion specific so you use the pid.
"""