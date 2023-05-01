# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 15:48:06 2023

@author: Guido
"""

import numpy as np
import pandas as pd
from os.path import join
from stim_functions import paths

# Get paths
_, save_path = paths()

neuron_type = pd.read_csv(join(save_path, 'neuron_type.csv'))
neuron_type = neuron_type[neuron_type['type'] != 'Und.']