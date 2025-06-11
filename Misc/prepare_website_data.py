# -*- coding: utf-8 -*-
"""
Created on Mon May  5 17:01:49 2025 by Guido Meijer
"""


import numpy as np
import pandas as pd
from os.path import join
from stim_functions import paths
from iblbrainviewer.api import FeatureUploader
up = FeatureUploader()

# Get paths
_, save_path = paths()

# Load in results
light_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons.csv'))
mod_idx_df = pd.read_pickle(join(save_path, 'mod_over_time.pickle'))

# Get values per region
grouped_df = light_neurons.groupby('allen_acronym').size().to_frame()
grouped_df = grouped_df.rename(columns={0: 'n_neurons'})
grouped_df['n_mod_neurons'] = light_neurons[light_neurons['modulated']].groupby('allen_acronym').size()
grouped_df['n_mod_neurons'] = grouped_df['n_mod_neurons'].fillna(0).astype(int)
grouped_df['perc_mod'] = (grouped_df['n_mod_neurons'] / grouped_df['n_neurons']) * 100
grouped_df['mod_index'] = light_neurons[['allen_acronym', 'mod_index']].groupby('allen_acronym').mean()['mod_index']

# Exclude root, void, and fiber tracts
grouped_df = grouped_df.reset_index()
grouped_df = grouped_df[(grouped_df['allen_acronym'] != 'root') & (grouped_df['allen_acronym'] != 'void')]
grouped_df = grouped_df[~grouped_df['allen_acronym'].str.islower()]

# Put into numpy arrays
acronyms = grouped_df['allen_acronym'].values.astype(str)
fname1 = 'Number of recorded neurons'
values1 = grouped_df['n_neurons'].values
fname2 = 'Percentage of 5-HT modulated neurons'
values2 = grouped_df['perc_mod'].values
fname3 = 'Modulation index'
values3 = grouped_df['mod_index'].values

# Upload to local bucket
up.local_features(fname1, acronyms, values1, hemisphere='left', output_dir=save_path)
up.local_features(fname2, acronyms, values2, hemisphere='left', output_dir=save_path)
up.local_features(fname3, acronyms, values3, hemisphere='left', output_dir=save_path)

# Create bucket
up = FeatureUploader('meijer_serotonin')

# Upload the features.
up.patch_features(fname1, acronyms, values1, hemisphere='left')
up.patch_features(fname2, acronyms, values2, hemisphere='left')
up.patch_features(fname3, acronyms, values3, hemisphere='left')

# Generate url
url = up.get_buckets_url(['meijer_serotonin'])
print(url)



     

