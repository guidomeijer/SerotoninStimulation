# -*- coding: utf-8 -*-
"""
Created on Mon May  5 17:01:49 2025 by Guido Meijer
"""


import numpy as np
import pandas as pd
from os.path import join
from stim_functions import paths, remap
from iblatlas.regions import BrainRegions
from iblbrainviewer.api import FeatureUploader
up = FeatureUploader()
br = BrainRegions()

# Get paths
_, save_path = paths()

# Load in results
light_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons.csv'))
light_neurons['beryl_acronym'] = remap(light_neurons['allen_acronym'])
mod_idx_df = pd.read_pickle(join(save_path, 'mod_over_time.pickle'))

# Get min/max over time per region
for r, region in enumerate(np.unique(mod_idx_df['region'])):
    reg_mod = np.vstack(mod_idx_df.loc[mod_idx_df['region'] == region, 'mod_idx'].to_numpy())
    mean_mod = np.mean(reg_mod, axis=0)
    this_mod = mean_mod[np.argmax(np.abs(mean_mod))]
    light_neurons.loc[light_neurons['beryl_acronym'] == region, 'min_max_mod'] = this_mod

# Get values per region
grouped_df = light_neurons.groupby('beryl_acronym').size().to_frame()
grouped_df = grouped_df.rename(columns={0: 'n_neurons'})
grouped_df['n_mod_neurons'] = light_neurons[light_neurons['modulated']].groupby('beryl_acronym').size()
grouped_df['n_mod_neurons'] = grouped_df['n_mod_neurons'].fillna(0).astype(int)
grouped_df['perc_mod'] = (grouped_df['n_mod_neurons'] / grouped_df['n_neurons']) * 100
grouped_df['mod_index'] = light_neurons[['beryl_acronym', 'min_max_mod']].groupby('beryl_acronym').mean()['min_max_mod']
#grouped_df['mod_index'] = light_neurons[['beryl_acronym', 'mod_index']].groupby('beryl_acronym').mean()['mod_index']
grouped_df['latency'] = light_neurons[['beryl_acronym', 'latency_peak_onset']].groupby('beryl_acronym').median()['latency_peak_onset']
grouped_df.loc[np.isnan(grouped_df['mod_index']), 'mod_index'] = 0

# Exclude root, void, and fiber tracts
grouped_df = grouped_df.reset_index()
grouped_df = grouped_df[(grouped_df['beryl_acronym'] != 'root') & (grouped_df['beryl_acronym'] != 'void')]
grouped_df = grouped_df[~grouped_df['beryl_acronym'].str.islower()]

# Put into numpy arrays
acronyms = grouped_df['beryl_acronym'].values.astype(str)
fname1 = 'number_of_recorded_neurons'
values1 = grouped_df['n_neurons'].values.astype(int)
fname2 = 'percentage_of_5HT_modulated_neurons'
values2 = grouped_df['perc_mod'].values
fname3 = 'modulation_index'
values3 = grouped_df['mod_index'].values
fname4 = 'latency'
values4 = grouped_df['latency'].values

# Upload to local bucket
up.local_features(fname1, acronyms, values1, hemisphere='left', output_dir=save_path)
up.local_features(fname2, acronyms, values2, hemisphere='left', output_dir=save_path)
up.local_features(fname3, acronyms, values3, hemisphere='left', output_dir=save_path)

# Load bucket
up = FeatureUploader('meijer_serotonin')

# Descriptions
short_desc = 'Serotonin modulation across the brain'
long_desc = 'A longer more comprehensive description about the bucket, e.g abstract of associated publication'
up.patch_bucket(short_desc=short_desc, long_desc=long_desc)

# Upload the features
if up.features_exist(fname1):
    up.patch_features(fname1, acronyms, values1, hemisphere='left')
else:
    up.create_features(fname1, acronyms, values1, hemisphere='left')
if up.features_exist(fname2):
    up.patch_features(fname2, acronyms, values2, hemisphere='left')
else:
    up.create_features(fname2, acronyms, values2, hemisphere='left')
if up.features_exist(fname3):
    up.patch_features(fname3, acronyms, values3, hemisphere='left')
else:
    up.create_features(fname3, acronyms, values3, hemisphere='left')
if up.features_exist(fname4):
    up.patch_features(fname4, acronyms, values4, hemisphere='left')
else:
    up.create_features(fname4, acronyms, values4, hemisphere='left')

# Create and upload tree
tree = {'Number of recorded neurons': fname1,
        '5-HT modulated neurons (%)': fname2,
        'Modulation index': fname3,
        'Modulation latency (s)': fname4}
up.patch_bucket(tree=tree)

# Generate url
url = up.get_buckets_url(['meijer_serotonin'])
print(url)

# Get token
print(up.token)



     

