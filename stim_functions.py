# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 16:22:01 2020

By: Guido Meijer
"""

import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import tkinter as tk
from scipy.stats import binned_statistic
from scipy.signal import gaussian, convolve
import pathlib
from brainbox import singlecell
from os.path import join, realpath, dirname, isfile
from glob import glob
from matplotlib import colors as matplotlib_colors
from scipy.interpolate import interp1d
import json
from pathlib import Path
from brainbox.io.spikeglx import spikeglx
from brainbox.metrics.single_units import spike_sorting_metrics
from brainbox.io.one import SpikeSortingLoader
from iblutil.numerical import ismember
from iblatlas.regions import BrainRegions
from iblatlas.atlas import AllenAtlas
from one.api import ONE

# Whether to use released open-source data
OPEN_ONE = False

# Number of states of HMM
N_STATES = 7
N_STATES_TASK = 11
N_STATES_REGIONS = {'Frontal cortex': 8, 'Amygdala': 6, 'Hippocampus': 7, 'Midbrain': 7,
                    'Sensory cortex': 8, 'Striatum': 7, 'Thalamus': 5}
N_STATES_TASK_REGIONS = {'Frontal cortex': 9, 'Amygdala': 12, 'Hippocampus': 13, 'Midbrain': 14,
                         'Sensory cortex': 10, 'Striatum': 9, 'Thalamus': 7}
N_STATES_CONT = 7


def init_one(local=False, open_one=OPEN_ONE):
    if local:
        mode='local'
    else:
        mode='auto'
    if open_one:
        one = ONE(mode=mode, base_url='https://openalyx.internationalbrainlab.org',
                  password='international', silent=True)
    else:
        one = ONE(mode=mode)
    return one


def load_subjects(anesthesia='all', behavior=None):
    assert anesthesia in ['no', 'yes', 'both', 'all', 'no&both', 'yes&both']
    subjects = pd.read_csv(join(pathlib.Path(__file__).parent.resolve(), 'subjects.csv'),
                           delimiter=';|,', engine='python')
    subjects = subjects[subjects['include'] == 1]
    if anesthesia == 'yes':
        subjects = subjects[subjects['anesthesia'] == 2]
    elif anesthesia == 'no':
        subjects = subjects[subjects['anesthesia'] == 0]
    elif anesthesia == 'both':
        subjects = subjects[subjects['anesthesia'] == 1]
    elif anesthesia == 'no&both':
        subjects = subjects[(subjects['anesthesia'] == 1) | (subjects['anesthesia'] == 0)]
    elif anesthesia == 'yes&both':
        subjects = subjects[(subjects['anesthesia'] == 1) | (subjects['anesthesia'] == 2)]
    subjects['subject_nr'] = subjects['subject_nr'].astype(int)
    subjects = subjects.reset_index(drop=True)
    return subjects


def paths(save_dir='repo'):
    """
    Load in figure path from paths.json, if this file does not exist it will be generated from
    user input

    Save directory can be either the repository (for small files) or the one cache directory
    (for large files)

    Input
    ------------------------
    save_dir : str
        'repo' or 'cache' for saving in the repository or one cache, respectively

    Output
    ------------------------
    fig_path : str
        Path to where to save the figures

    save_path : str
        Path to where to save the output data
    """
    if not isfile(join(dirname(realpath(__file__)), 'paths.json')):
        path_dict = dict()
        path_dict['fig_path'] = input('Path folder to save figures: ')
        path_dict['save_path'] = join(dirname(realpath(__file__)), 'Data')
        path_file = open(join(dirname(realpath(__file__)), 'paths.json'), 'w')
        json.dump(path_dict, path_file)
        path_file.close()
    with open(join(dirname(realpath(__file__)), 'paths.json')) as json_file:
        path_dict = json.load(json_file)
    if save_dir == 'cache':
        one = ONE(mode='local')
        save_dir = Path(one.cache_dir, 'serotonin')
        save_dir.mkdir(exist_ok=True)
    elif save_dir == 'repo':
        save_dir = path_dict['save_path']
    else:
        print('save_dir must be either repo or cache')
    return path_dict['fig_path'], save_dir


def figure_style():
    """
    Set style for plotting figures
    """
    sns.set(style="ticks", context="paper",
            font="Arial",
            rc={"font.size": 7,
                "figure.titlesize": 7,
                "axes.titlesize": 7,
                "axes.labelsize": 7,
                "axes.linewidth": 0.5,
                "lines.linewidth": 1,
                "lines.markersize": 3,
                "xtick.labelsize": 7,
                "ytick.labelsize": 7,
                "savefig.transparent": True,
                "xtick.major.size": 2.5,
                "ytick.major.size": 2.5,
                "xtick.major.width": 0.5,
                "ytick.major.width": 0.5,
                "xtick.minor.size": 2,
                "ytick.minor.size": 2,
                "xtick.minor.width": 0.5,
                "ytick.minor.width": 0.5,
                'legend.fontsize': 7,
                'legend.title_fontsize': 7,
                'legend.frameon': False,
                })
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    subject_pal = sns.color_palette(
        np.concatenate((sns.color_palette('tab20'),
                        [matplotlib_colors.to_rgb('maroon'), np.array([0, 0, 0])])))
    state_dark_pal = sns.color_palette('Dark2')
    state_light_pal = sns.color_palette('Set2')
   
    """
    state_dark_pal = sns.color_palette(
        np.concatenate((sns.color_palette('Dark2')[:5],
                        [sns.color_palette('Dark2')[6],
                         sns.color_palette('Dark2')[7],
                         sns.color_palette('tab10')[0],
                        sns.color_palette('tab10')[3],
                        sns.color_palette('tab10')[9],
                        sns.color_palette('tab10')[6]])))
    state_light_pal = sns.color_palette(
        np.concatenate((sns.color_palette('Set2')[:5],
                        [sns.color_palette('Set2')[6],
                         sns.color_palette('Set2')[7],
                         sns.color_palette('tab20')[1],
                        sns.color_palette('tab20')[7],
                        sns.color_palette('tab20')[19],
                        sns.color_palette('tab20')[13]])))
    """
    frontal = sns.color_palette('Dark2')[1]
    sensory = sns.color_palette('Dark2')[5]
    hipp = sns.color_palette('Dark2')[4]
    amygdala = sns.color_palette('Dark2')[2]
    thalamus = sns.color_palette('Dark2')[0]
    striatum = sns.color_palette('Set1')[7]
    midbrain = sns.color_palette('Dark2')[3]

    colors = {'subject_palette': subject_pal,
              'grey': [0.7, 0.7, 0.7],
              'sert': sns.color_palette('Dark2')[0],
              'wt': [0.7, 0.7, 0.7],
              'awake': sns.color_palette('Dark2')[2],
              'anesthesia': sns.color_palette('Dark2')[3],
              'enhanced': sns.color_palette('colorblind')[3],
              'suppressed': sns.color_palette('colorblind')[0],
              'down-state': sns.color_palette('colorblind')[3],
              'up-state': [1, 1, 1],
              'stim': [0, 0, 0],
              'no-stim': [0.7, 0.7, 0.7],
              'NS': sns.color_palette('Set2')[0],
              'WS': sns.color_palette('Set2')[1],
              'WS1': sns.color_palette('Set2')[1],
              'WS2': sns.color_palette('Set2')[2],
              'states': state_dark_pal,
              'states_light': state_light_pal,
              'main_states': sns.diverging_palette(20, 210, l=55, center='dark'),
              'Frontal cortex': frontal,
              'Sensory cortex': sensory,
              'Midbrain': midbrain,
              'Amygadala': amygdala,
              'Thalamus': thalamus,
              'Hippocampus': hipp,
              'Striatum': striatum,
              'OFC': sns.color_palette('Set1')[7],
              'mPFC': sns.color_palette('Dark2')[1],
              'M2': sns.color_palette('Dark2')[2],
              'Amyg.': sns.color_palette('Dark2')[6],
              'HPC': sns.color_palette('Dark2')[3],
              'VIS': sns.color_palette('Dark2')[5],
              'Pir.': sns.color_palette('Dark2')[4],
              'SC': sns.color_palette('Dark2')[7],
              'Thal.': sns.color_palette('tab10')[9],
              'PAG': sns.color_palette('Dark2')[0],
              'BC': sns.color_palette('Accent')[0],
              'Str.': sns.color_palette('Accent')[1],
              'MRN': sns.color_palette('Accent')[2],
              'OLF': sns.color_palette('tab10')[8],
              'Orbitofrontal cortex': sns.color_palette('Dark2')[0],
              'Medial prefrontal cortex': sns.color_palette('Dark2')[1],
              'Secondary motor cortex': sns.color_palette('Dark2')[2],
              'Amygdala': sns.color_palette('Dark2')[3],
              'Visual cortex': sns.color_palette('Dark2')[5],
              'Piriform': sns.color_palette('Dark2')[6],
              'Superior colliculus': sns.color_palette('Dark2')[7],
              'Periaqueductal gray': sns.color_palette('Set1')[7],
              'Barrel cortex': sns.color_palette('Set2')[0],
              'Tail of the striatum': sns.color_palette('Set2')[1],
              'Midbrain reticular nucleus': sns.color_palette('Accent')[2],
              'Olfactory areas': sns.color_palette('tab10')[8],
              'Substantia nigra': [0.75, 0.75, 0.75],
              'Retrosplenial cortex': 'r',
              'RSP': 'r',
              'SNr': [0.75, 0.75, 0.75],
              'left-stim': sns.color_palette('Paired')[7],
              'left-no-stim': sns.color_palette('Paired')[6],
              'right-stim': sns.color_palette('Paired')[9],
              'right-no-stim': sns.color_palette('Paired')[8]}
    screen_width = tk.Tk().winfo_screenwidth()
    dpi = screen_width / 10
    return colors, dpi


"""
 'OFC': sns.color_palette('Dark2')[0],
 'mPFC': sns.color_palette('Dark2')[1],
 'M2': sns.color_palette('Dark2')[2],
 'Amyg': sns.color_palette('Dark2')[3],
 'Hipp': sns.color_palette('Dark2')[4],
 'VIS': sns.color_palette('Dark2')[5],
 'Pir': sns.color_palette('Dark2')[6],
 'SC': sns.color_palette('Dark2')[7],
 'Thal': sns.color_palette('tab10')[9],
 'PAG': sns.color_palette('Set1')[7],
 'BC': sns.color_palette('Accent')[0],
 'Str': sns.color_palette('Accent')[1],
 'MRN': sns.color_palette('Accent')[2],
 'OLF': sns.color_palette('tab10')[8],
 'Orbitofrontal cortex': sns.color_palette('Dark2')[0],
 'Medial prefrontal cortex': sns.color_palette('Dark2')[1],
 'Secondary motor cortex': sns.color_palette('Dark2')[2],
 'Amygdala': sns.color_palette('Dark2')[3],
 'Hippocampus': sns.color_palette('Dark2')[4],
 'Visual cortex': sns.color_palette('Dark2')[5],
 'Piriform': sns.color_palette('Dark2')[6],
 'Superior colliculus': sns.color_palette('Dark2')[7],
 'Periaqueductal gray': sns.color_palette('Set1')[7],
 'Barrel cortex': sns.color_palette('Set2')[0],
 'Thalamus': sns.color_palette('tab10')[9],
 'Tail of the striatum': sns.color_palette('Set2')[1],
 'Midbrain reticular nucleus': sns.color_palette('Accent')[2],
 'Olfactory areas': sns.color_palette('tab10')[8],
 'Substantia nigra': [0.75, 0.75, 0.75],
 'Striatum': sns.color_palette('Accent')[1],
 'Retrosplenial cortex': 'r',
 'RSP': 'r',
 'SNr': [0.75, 0.75, 0.75],
 """


def get_artifact_neurons():
    artifact_neurons = pd.read_csv(
        join(pathlib.Path(__file__).parent.resolve(), 'artifact_neurons.csv'))
    return artifact_neurons


def remove_artifact_neurons(df):
    artifact_neurons = pd.read_csv(
        join(pathlib.Path(__file__).parent.resolve(), 'artifact_neurons.csv'))
    for i, column in enumerate(df.columns):
        if df[column].dtype == bool:
            df[column] = df[column].astype('boolean')
    if 'pid' in df.columns:
        df = pd.merge(df, artifact_neurons, indicator=True, how='outer',
                      on=['pid', 'neuron_id', 'subject', 'probe', 'date']).query('_merge=="left_only"').drop('_merge', axis=1)
    else:
        df = pd.merge(df, artifact_neurons, indicator=True, how='outer',
                      on=['subject', 'probe', 'date', 'neuron_id']).query('_merge=="left_only"').drop('_merge', axis=1)
    return df


def query_ephys_sessions(aligned=True, behavior_crit=False, n_trials=0, anesthesia='no',
                         acronym=None, one=None):
    assert anesthesia in ['no', 'both', 'yes', 'all', 'no&both', 'yes&both']
    if one is None:
        one = ONE()

    # Construct django query string
    DJANGO_STR = ('session__projects__name__icontains,serotonin_inference,'
                  'session__qc__lt,50')
    if aligned:
        # Query all ephys-histology aligned sessions
        DJANGO_STR += ',json__extended_qc__alignment_count__gt,0'

    if behavior_crit:
        # Query sessions with an alignment and that meet behavior criterion
        DJANGO_STR += ',session__extended_qc__behavior,1'

    # Query sessions with at least this many trials
    if n_trials > 0:
        DJANGO_STR += f',session__n_trials__gte,{n_trials}'

    # Query sessions
    if acronym is None:
        ins = one.alyx.rest('insertions', 'list', django=DJANGO_STR)
    elif type(acronym) is str:
        ins = one.alyx.rest('insertions', 'list', django=DJANGO_STR, atlas_acronym=acronym)
    else:
        ins = []
        for i, ac in enumerate(acronym):
            ins = ins + one.alyx.rest('insertions', 'list', django=DJANGO_STR, atlas_acronym=ac)

    # Only include subjects from subjects.csv
    incl_subjects = load_subjects(anesthesia=anesthesia)
    ins = [i for i in ins if i['session_info']['subject'] in incl_subjects['subject'].values]

    # Get list of eids and probes
    rec = pd.DataFrame()
    rec['pid'] = np.array([i['id'] for i in ins])
    rec['eid'] = np.array([i['session'] for i in ins])
    rec['probe'] = np.array([i['name'] for i in ins])
    rec['subject'] = np.array([i['session_info']['subject'] for i in ins])
    rec['date'] = np.array([i['session_info']['start_time'][:10] for i in ins])
    rec = rec.drop_duplicates('pid', ignore_index=True)
    return rec


def remap(acronyms, source='Allen', dest='Beryl', combine=False, split_thalamus=False,
          abbreviate=True, brainregions=None):
    br = brainregions or BrainRegions()
    _, inds = ismember(br.acronym2id(acronyms), br.id[br.mappings[source]])
    remapped_acronyms = br.get(br.id[br.mappings[dest][inds]])['acronym']
    if combine:
        return combine_regions(remapped_acronyms, split_thalamus=split_thalamus, abbreviate=abbreviate)
    else:
        return remapped_acronyms


def combine_regions(acronyms, split_thalamus=False, abbreviate=True):
    """
    Combines regions into groups, input Beryl atlas acronyms: use remap function first
    """
    regions = np.array(['root'] * len(acronyms), dtype=object)
    if abbreviate:
        regions[np.in1d(acronyms, ['ILA', 'PL', 'ACAd', 'ACAv'])] = 'mPFC'
        regions[np.in1d(acronyms, ['MOs'])] = 'M2'
        regions[np.in1d(acronyms, ['ORBl', 'ORBm'])] = 'OFC'
        if split_thalamus:
            regions[np.in1d(acronyms, ['PO'])] = 'PO'
            regions[np.in1d(acronyms, ['LP'])] = 'LP'
            regions[np.in1d(acronyms, ['LD'])] = 'LD'
            regions[np.in1d(acronyms, ['RT'])] = 'RT'
            regions[np.in1d(acronyms, ['VAL'])] = 'VAL'
        else:
            regions[np.in1d(acronyms, ['PO', 'LP', 'LD', 'RT', 'VAL'])] = 'Thal.'
        regions[np.in1d(acronyms, ['SCm', 'SCs', 'SCig', 'SCsg', 'SCdg'])] = 'SC'
        regions[np.in1d(acronyms, ['RSPv', 'RSPd'])] = 'RSP'
        regions[np.in1d(acronyms, ['ZI'])] = 'ZI'
        regions[np.in1d(acronyms, ['PAG'])] = 'PAG'
        regions[np.in1d(acronyms, ['SSp-bfd'])] = 'BC'
        # regions[np.in1d(acronyms, ['LGv', 'LGd'])] = 'LG'
        regions[np.in1d(acronyms, ['PIR'])] = 'Pir.'
        # regions[np.in1d(acronyms, ['SNr', 'SNc', 'SNl'])] = 'SN'
        regions[np.in1d(acronyms, ['VISa', 'VISam', 'VISp', 'VISpm'])] = 'VIS'
        regions[np.in1d(acronyms, ['AId', 'AIv', 'AIp'])] = 'AI'
        regions[np.in1d(acronyms, ['MEA', 'CEA', 'BLA', 'COAa'])] = 'Amyg.'
        regions[np.in1d(acronyms, ['AON', 'TTd', 'DP'])] = 'OLF'
        regions[np.in1d(acronyms, ['CP', 'STR', 'STRd', 'STRv'])] = 'Str.'
        regions[np.in1d(acronyms, ['CA1', 'CA3', 'DG'])] = 'HPC'
    else:
        regions[np.in1d(acronyms, ['ILA', 'PL', 'ACAd', 'ACAv'])] = 'Medial prefrontal cortex'
        regions[np.in1d(acronyms, ['MOs'])] = 'Secondary motor cortex'
        regions[np.in1d(acronyms, ['ORBl', 'ORBm'])] = 'Orbitofrontal cortex'
        if split_thalamus:
            regions[np.in1d(acronyms, ['PO'])] = 'Thalamus (PO)'
            regions[np.in1d(acronyms, ['LP'])] = 'Thalamus (LP)'
            regions[np.in1d(acronyms, ['LD'])] = 'Thalamus (LD)'
            regions[np.in1d(acronyms, ['RT'])] = 'Thalamus (RT)'
            regions[np.in1d(acronyms, ['VAL'])] = 'Thalamus (VAL)'
        else:
            regions[np.in1d(acronyms, ['PO', 'LP', 'LD', 'RT', 'VAL'])] = 'Thalamus'
        regions[np.in1d(acronyms, ['SCm', 'SCs', 'SCig', 'SCsg', 'SCdg'])] = 'Superior colliculus'
        regions[np.in1d(acronyms, ['RSPv', 'RSPd'])] = 'Retrosplenial cortex'
        regions[np.in1d(acronyms, ['AON', 'TTd', 'DP'])] = 'Olfactory areas'
        regions[np.in1d(acronyms, ['ZI'])] = 'Zona incerta'
        regions[np.in1d(acronyms, ['PAG'])] = 'Periaqueductal gray'
        regions[np.in1d(acronyms, ['AId', 'AIv', 'AIp'])] = 'Insular cortex'
        regions[np.in1d(acronyms, ['SSp-bfd'])] = 'Barrel cortex'
        # regions[np.in1d(acronyms, ['LGv', 'LGd'])] = 'Lateral geniculate'
        regions[np.in1d(acronyms, ['PIR'])] = 'Piriform'
        # regions[np.in1d(acronyms, ['SNr', 'SNc', 'SNl'])] = 'Substantia nigra'
        regions[np.in1d(acronyms, ['VISa', 'VISam', 'VISp', 'VISpm'])] = 'Visual cortex'
        regions[np.in1d(acronyms, ['MEA', 'CEA', 'BLA', 'COAa'])] = 'Amygdala'
        regions[np.in1d(acronyms, ['CP', 'STR', 'STRd', 'STRv'])] = 'Striatum'
        regions[np.in1d(acronyms, ['CA1', 'CA3', 'DG'])] = 'Hippocampus'
    return regions


def high_level_regions(acronyms, merge_cortex=False, only_vis=False, input_atlas='Allen'):
    """
    """
    if input_atlas == 'Allen':
        acronyms = remap(acronyms)
    first_level_regions = combine_regions(acronyms, abbreviate=True)
    cosmos_regions = remap(acronyms, dest='Cosmos')
    regions = np.array(['root'] * len(first_level_regions), dtype=object)
    if merge_cortex:
        # regions[cosmos_regions == 'Isocortex'] = 'Cortex'
        # regions[first_level_regions == 'Pir'] = 'Cortex'
        regions[np.in1d(first_level_regions, ['mPFC', 'OFC', 'M2', 'Pir', 'BC', 'VIS'])] = 'Cortex'
    else:
        regions[np.in1d(first_level_regions, ['mPFC', 'OFC', 'M2'])] = 'Frontal cortex'
        if only_vis:
            regions[np.in1d(first_level_regions, ['VIS'])] = 'Visual cortex'
        else:
            regions[np.in1d(first_level_regions, ['Pir', 'BC', 'VIS'])] = 'Sensory cortex'
    regions[cosmos_regions == 'MB'] = 'Midbrain'
    regions[cosmos_regions == 'HPF'] = 'Hippocampus'
    regions[cosmos_regions == 'TH'] = 'Thalamus'
    regions[np.in1d(first_level_regions, ['Amyg'])] = 'Amygdala'
    regions[np.in1d(acronyms, ['CP', 'ACB', 'FS'])] = 'Striatum'
    return regions


def get_full_region_name(acronyms):
    brainregions = BrainRegions()
    full_region_names = []
    for i, acronym in enumerate(acronyms):
        try:
            regname = brainregions.name[np.argwhere(brainregions.acronym == acronym).flatten()][0]
            full_region_names.append(regname)
        except IndexError:
            full_region_names.append(acronym)
    if len(full_region_names) == 1:
        return full_region_names[0]
    else:
        return full_region_names


def load_passive_opto_times(eid, one=None, force_rerun=False, anesthesia=False, freq=25):
    """
    Load in the time stamps of the optogenetic stimulation at the end of the recording, after the
    taks and the spontaneous activity. Or when it's a long stimulation session with different
    frequencies, only return those stimulation bouts of the specified Hz (default is 25 Hz).

    Returns
    opto_train_times : 1D array
        Timestamps of the start of each pulse train
    opto_pulse_times : 1D array
        Timestamps of all individual pulses
    """

    if one is None:
        one = ONE()

    # See if this is an anesthesia session
    anesthesia_sub = load_subjects(anesthesia='both')
    ses_details = one.get_details(eid)
    subject = ses_details['subject']
    date = ses_details['date']
    if subject in anesthesia_sub['subject'].values:
        anesthesia_ses = True
    else:
        anesthesia_ses = False

    # Load in pulses from disk if already extracted
    _, save_path = paths()
    save_path = join(save_path, 'OptoTimes')
    if isfile(join(save_path, f'{subject}_{date}_pulse_trains.npy')) & ~force_rerun & ~anesthesia:
        opto_train_times = np.load(join(save_path, f'{subject}_{date}_pulse_trains_{freq}hz.npy'))
        opto_on_times = np.load(join(save_path, f'{subject}_{date}_ind_pulses_{freq}hz.npy'))
        return opto_train_times, opto_on_times
    elif isfile(join(save_path, f'{subject}_{date}_pulse_trains_anesthesia.npy')) & ~force_rerun & anesthesia:
        opto_train_times = np.load(
            join(save_path, f'{subject}_{date}_pulse_trains_anesthesia_{freq}hz.npy'))
        opto_on_times = np.load(
            join(save_path, f'{subject}_{date}_ind_pulses_anesthesia_{freq}hz.npy'))
        return opto_train_times, opto_on_times
    else:
        # Load in laser pulses
        try:
            one.load_datasets(eid, datasets=[
                '_spikeglx_ephysData_g0_t0.nidq.cbin', '_spikeglx_ephysData_g0_t0.nidq.meta',
                '_spikeglx_ephysData_g0_t0.nidq.ch'], download_only=True)
        except:
            one.load_datasets(eid, datasets=[
                '_spikeglx_ephysData_g1_t0.nidq.cbin', '_spikeglx_ephysData_g1_t0.nidq.meta',
                '_spikeglx_ephysData_g1_t0.nidq.ch'], download_only=True)
        session_path = one.eid2path(eid)
        nidq_file = glob(str(session_path.joinpath(
            'raw_ephys_data/_spikeglx_ephysData_g*_t0.nidq.cbin')))[-1]
        sr = spikeglx.Reader(nidq_file)
        if anesthesia_ses & ~anesthesia:
            offset = int(300 * sr.fs)
            end = int(1020 * sr.fs)
            opto_trace = sr.read_sync_analog(slice(offset, end))[:, 1]
            opto_times = np.arange(offset, end) / sr.fs
        else:
            offset = int((sr.shape[0] / sr.fs - 720) * sr.fs)
            opto_trace = sr.read_sync_analog(slice(offset, sr.shape[0]))[:, 1]
            opto_times = np.arange(offset, sr.shape[0]) / sr.fs

        if np.sum(np.diff(opto_trace) > 1) == 0:
            print(f'No pulses found for {eid}')
            return [], []

        # Get start times of pulse trains
        opto_on_times = opto_times[np.concatenate((np.diff(opto_trace), [0])) > 1]

        # Get the times of the onset of each pulse train
        opto_train_times = opto_on_times[np.concatenate(([True], np.diff(opto_on_times) > 1))]

        # Get the stimulation frequencies
        opto_freqs = np.empty(opto_train_times.shape)
        for i, t_time in enumerate(opto_train_times):
            # opto_freqs[i] = 1/np.mean(np.diff(opto_on_times[(opto_on_times >= t_time)
            #                                                & (opto_on_times <= t_time + 1)]))
            opto_freqs[i] = opto_on_times[(opto_on_times >= t_time)
                                          & (opto_on_times <= t_time + 1)].shape[0]
        opto_freqs = opto_freqs - opto_freqs % 5  # round to 5
        opto_freqs[opto_freqs == 0] = 1

        # If there are different stimulation frequencies than 25 Hz it's a long stim session
        if np.any(np.isin([1, 5, 10], opto_freqs)) & (np.sum(opto_freqs == 40) < 5):
            print('Long opto stim session detected')

            # Load in the trace in chunks and only extract the 25Hz trains
            opto_train_times = []
            opto_on_times = []
            chunk_edges = np.arange(0, sr.shape[0], 500 * sr.fs).astype(int)
            for j in range(len(chunk_edges[:-1])):

                # Load in chunk of trace
                trace_chunk = sr.read_sync_analog(slice(chunk_edges[j], chunk_edges[j+1]))[:, 1]
                times_chunk = np.arange(chunk_edges[j], chunk_edges[j+1]) / sr.fs

                # Get start times of pulse trains
                these_on_times = times_chunk[np.concatenate((np.diff(trace_chunk), [0])) > 1]

                # Get the times of the onset of each pulse train
                these_train_times = these_on_times[np.concatenate(
                    ([True], np.diff(these_on_times) > 1))]

                # Get the stimulation frequencies
                these_freqs = np.empty(these_train_times.shape)
                for ii, t_time in enumerate(these_train_times):
                    these_freqs[ii] = these_on_times[(these_on_times >= t_time) & (
                        these_on_times <= t_time + 1)].shape[0]
                these_freqs = these_freqs - these_freqs % 5  # round to 5
                these_freqs[these_freqs == 0] = 1

                # Get stimulation amplitudes
                if sr.read_sync_analog(slice(chunk_edges[j], chunk_edges[j+1])).shape[1] > 3:
                    analog_chunk = sr.read_sync_analog(
                        slice(chunk_edges[j], chunk_edges[j+1]))[:, 3]
                    these_amps = np.empty(these_train_times.shape)
                    for ii, t_time in enumerate(these_train_times):
                        this_amp = np.max(
                            analog_chunk[(times_chunk >= t_time) & (times_chunk <= t_time + 1)])
                        if this_amp > 0.6:
                            these_amps[ii] = 1
                        else:
                            these_amps[ii] = 0.5
                else:
                    these_amps = np.ones(these_train_times.shape)

                # Add the pulse trains of 25 Hz full power to the array
                opto_train_times.append(
                    these_train_times[(these_freqs == freq) & (these_amps == 1)])
                for kk, this_train_time in enumerate(these_train_times[(these_freqs == freq) & (these_amps == 1)]):
                    opto_on_times.append(these_on_times[(these_on_times >= this_train_time)
                                                        & (these_on_times <= this_train_time + 1)])

            # Convert to arrays
            opto_train_times = np.concatenate(opto_train_times)
            opto_on_times = np.concatenate(opto_on_times)

            # Save extracted pulses to disk
            if anesthesia:
                np.save(
                    join(save_path, f'{subject}_{date}_pulse_trains_anesthesia_{freq}hz.npy'), opto_train_times)
                np.save(
                    join(save_path, f'{subject}_{date}_ind_pulses_anesthesia_{freq}hz.npy'), opto_on_times)
            else:
                np.save(
                    join(save_path, f'{subject}_{date}_pulse_trains_{freq}hz.npy'), opto_train_times)
                np.save(
                    join(save_path, f'{subject}_{date}_ind_pulses_{freq}hz.npy'), opto_on_times)

            return opto_train_times, opto_on_times

        # Find the opto pulses after the spontaneous activity (after a long break, here 100s)
        if np.sum(np.diff(opto_train_times) > 100) > 0:
            first_pulse = np.where(np.diff(opto_train_times) > 100)[0][0]+1
        elif opto_train_times[0] - opto_times[0] > 50:
            first_pulse = 0
        else:
            print('Could not find passive laser pulses')
            return [], []
        opto_train_times = opto_train_times[first_pulse:]
        opto_on_times = opto_on_times[first_pulse:]

        # Save extracted pulses to disk
        if anesthesia:
            np.save(
                join(save_path, f'{subject}_{date}_pulse_trains_anesthesia_{freq}hz.npy'), opto_train_times)
            np.save(
                join(save_path, f'{subject}_{date}_ind_pulses_anesthesia_{freq}hz.npy'), opto_on_times)
        else:
            np.save(
                join(save_path, f'{subject}_{date}_pulse_trains_{freq}hz.npy'), opto_train_times)
            np.save(join(save_path, f'{subject}_{date}_ind_pulses_{freq}hz.npy'), opto_on_times)

        return opto_train_times, opto_on_times


def load_trials(eid, laser_stimulation=False, invert_choice=False, invert_stimside=False, one=None):
    one = one or ONE()

    data = one.load_object(eid, 'trials')
    data = {your_key: data[your_key] for your_key in [
        'stimOn_times', 'feedback_times', 'goCue_times', 'probabilityLeft', 'contrastLeft',
        'contrastRight', 'feedbackType', 'choice', 'firstMovement_times']}
    trials = pd.DataFrame(data=data)
    if trials.shape[0] == 0:
        return
    trials['signed_contrast'] = trials['contrastRight']
    trials.loc[trials['signed_contrast'].isnull(), 'signed_contrast'] = -trials['contrastLeft']
    if laser_stimulation:
        trials['laser_stimulation'] = one.load_dataset(
            eid, dataset='_ibl_trials.laserStimulation.npy')
        try:
            trials['laser_probability'] = one.load_dataset(
                eid, dataset='_ibl_trials.laserProbability.npy')
            trials['probe_trial'] = ((trials['laser_stimulation'] == 0) & (trials['laser_probability'] == 0.75)
                                     | (trials['laser_stimulation'] == 1) & (trials['laser_probability'] == 0.25)).astype(int)
        except:
            trials['laser_probability'] = trials['laser_stimulation'].copy()
            trials.loc[(trials['signed_contrast'] == 0)
                       & (trials['laser_stimulation'] == 0), 'laser_probability'] = 0.25
            trials.loc[(trials['signed_contrast'] == 0)
                       & (trials['laser_stimulation'] == 1), 'laser_probability'] = 0.75

    trials['correct'] = trials['feedbackType']
    trials.loc[trials['correct'] == -1, 'correct'] = 0
    trials['right_choice'] = -trials['choice']
    trials.loc[trials['right_choice'] == -1, 'right_choice'] = 0
    trials['stim_side'] = (trials['signed_contrast'] > 0).astype(int)
    trials.loc[trials['stim_side'] == 0, 'stim_side'] = -1
    trials.loc[(trials['signed_contrast'] == 0) & (trials['contrastLeft'].isnull()),
               'stim_side'] = 1
    trials.loc[(trials['signed_contrast'] == 0) & (trials['contrastRight'].isnull()),
               'stim_side'] = -1
    if 'firstMovement_times' in trials.columns.values:
        trials['reaction_times'] = trials['firstMovement_times'] - trials['goCue_times']
    if invert_choice:
        trials['choice'] = -trials['choice']
    if invert_stimside:
        trials['stim_side'] = -trials['stim_side']
        trials['signed_contrast'] = -trials['signed_contrast']
    return trials


def get_neuron_qc(pid, one=None, ba=None, force_rerun=False):
    one = one or ONE()

    # Check if QC is already computed
    eid, probe = one.pid2eid(pid)
    session_path = one.eid2path(eid)
    if isfile(join(session_path, 'alf', probe, 'neuron_qc_metrics.csv')) & ~force_rerun:
        print('Neuron QC metrics loaded from disk')
        qc_metrics = pd.read_csv(join(session_path, 'alf', probe, 'neuron_qc_metrics.csv'))
        return qc_metrics

    # Load in spikes
    if ba is None:
        sl = SpikeSortingLoader(pid=pid, one=one)
    else:
        sl = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
    spikes, clusters, channels = sl.load_spike_sorting()
    clusters = sl.merge_clusters(spikes, clusters, channels)

    # Calculate QC metrics
    print('Calculating neuron QC metrics')
    qc_metrics, _ = spike_sorting_metrics(spikes.times, spikes.clusters,
                                          spikes.amps, spikes.depths,
                                          cluster_ids=np.arange(clusters.channels.size))
    qc_metrics.to_csv(join(session_path, 'alf', probe, 'neuron_qc_metrics.csv'))
    return qc_metrics


def load_lfp(eid, probe, time_start, time_end, relative_to='begin', destriped=False, one=None):
    one = one or ONE()
    destriped_lfp_path = join(paths()[1], 'LFP')

    # Download LFP data
    if destriped:
        ses_details = one.get_details(eid)
        subject = ses_details['subject']
        date = ses_details['start_time'][:10]
        lfp_path = join(destriped_lfp_path, f'{subject}_{date}_{probe}_destriped_lfp.cbin')
    else:
        lfp_paths, _ = one.load_datasets(eid, download_only=True, datasets=[
            '_spikeglx_ephysData_g*_t0.imec*.lf.cbin', '_spikeglx_ephysData_g*_t0.imec*.lf.meta',
            '_spikeglx_ephysData_g*_t0.imec*.lf.ch'], collections=[f'raw_ephys_data/{probe}'] * 3)
        lfp_path = lfp_paths[0]
    sr = spikeglx.Reader(lfp_path)

    # Convert time to samples
    if relative_to == 'begin':
        samples_start = int(time_start * sr.fs)
        samples_end = int(time_end * sr.fs)
    elif relative_to == 'end':
        samples_start = sr.shape[0] - int(time_start * sr.fs)
        samples_end = sr.shape[0] - int(time_end * sr.fs)

    # Load in lfp slice
    signal = sr.read(nsel=slice(samples_start, samples_end, None), csel=slice(None, None, None))[0]
    signal = signal.T
    time = np.arange(samples_start, samples_end) / sr.fs

    return signal, time


def plot_scalar_on_slice(
        regions, values, coord=-1000, slice='coronal', mapping='Beryl', hemisphere='left',
        cmap='viridis', background='boundary', clevels=None, brain_atlas=None, colorbar=False, ax=None):
    """
    Function to plot scalar value per allen region on histology slice
    :param regions: array of acronyms of Allen regions
    :param values: array of scalar value per acronym. If hemisphere is 'both' and different values want to be shown on each
    hemispheres, values should contain 2 columns, 1st column for LH values, 2nd column for RH values
    :param coord: coordinate of slice in um (not needed when slice='top')
    :param slice: orientation of slice, options are 'coronal', 'sagittal', 'horizontal', 'top' (top view of brain)
    :param mapping: atlas mapping to use, options are 'Allen', 'Beryl' or 'Cosmos'
    :param hemisphere: hemisphere to display, options are 'left', 'right', 'both'
    :param background: background slice to overlay onto, options are 'image' or 'boundary'
    :param cmap: colormap to use
    :param clevels: min max color levels [cim, cmax]
    :param brain_atlas: AllenAtlas object
    :param colorbar: whether to plot a colorbar
    :param ax: optional axis object to plot on
    :return:
    """

    if clevels is None:
        clevels = (np.min(values), np.max(values))

    ba = brain_atlas or AllenAtlas()
    br = ba.regions

    # Find the mapping to use
    map_ext = '-lr'
    map = mapping + map_ext

    region_values = np.zeros_like(br.id) * np.nan

    if len(values.shape) == 2:
        for r, vL, vR in zip(regions, values[:, 0], values[:, 1]):
            region_values[np.where(br.acronym[br.mappings[map]] == r)[0][0]] = vR
            region_values[np.where(br.acronym[br.mappings[map]] == r)[0][1]] = vL
    else:
        for r, v in zip(regions, values):
            region_values[np.where(br.acronym[br.mappings[map]] == r)[0]] = v

        lr_divide = int((br.id.shape[0] - 1) / 2)
        if hemisphere == 'left':
            region_values[0:lr_divide] = np.nan
        elif hemisphere == 'right':
            region_values[lr_divide:] = np.nan
            region_values[0] = np.nan

    if ax:
        fig = ax.get_figure()
    else:
        fig, ax = plt.subplots()

    if background == 'boundary':
        cmap_bound = matplotlib.cm.get_cmap("bone_r").copy()
        cmap_bound.set_under([1, 1, 1], 0)

    if slice == 'coronal':

        if background == 'image':
            ba.plot_cslice(coord / 1e6, volume='image', mapping=map, ax=ax)
            ba.plot_cslice(
                coord / 1e6, volume='value', region_values=region_values, mapping=map, cmap=cmap,
                vmin=clevels[0], vmax=clevels[1], ax=ax)
        else:
            ba.plot_cslice(
                coord / 1e6, volume='value', region_values=region_values, mapping=map, cmap=cmap,
                vmin=clevels[0], vmax=clevels[1], ax=ax)
            ba.plot_cslice(
                coord / 1e6, volume='boundary', mapping=map, ax=ax, cmap=cmap_bound, vmin=0.01,
                vmax=0.8)

    elif slice == 'sagittal':
        if background == 'image':
            ba.plot_sslice(coord / 1e6, volume='image', mapping=map, ax=ax)
            ba.plot_sslice(
                coord / 1e6, volume='value', region_values=region_values, mapping=map, cmap=cmap,
                vmin=clevels[0], vmax=clevels[1], ax=ax)
        else:
            ba.plot_sslice(
                coord / 1e6, volume='value', region_values=region_values, mapping=map, cmap=cmap,
                vmin=clevels[0], vmax=clevels[1], ax=ax)
            ba.plot_sslice(
                coord / 1e6, volume='boundary', mapping=map, ax=ax, cmap=cmap_bound, vmin=0.01,
                vmax=0.8)

    elif slice == 'horizontal':
        if background == 'image':
            ba.plot_hslice(coord / 1e6, volume='image', mapping=map, ax=ax)
            ba.plot_hslice(
                coord / 1e6, volume='value', region_values=region_values, mapping=map, cmap=cmap,
                vmin=clevels[0], vmax=clevels[1], ax=ax)
        else:
            ba.plot_hslice(
                coord / 1e6, volume='value', region_values=region_values, mapping=map, cmap=cmap,
                vmin=clevels[0], vmax=clevels[1], ax=ax)
            ba.plot_hslice(
                coord / 1e6, volume='boundary', mapping=map, ax=ax, cmap=cmap_bound, vmin=0.01,
                vmax=0.8)

    elif slice == 'top':
        if background == 'image':
            ba.plot_top(volume='image', mapping=map, ax=ax)
            ba.plot_top(
                volume='value', region_values=region_values, mapping=map, cmap=cmap,
                vmin=clevels[0], vmax=clevels[1], ax=ax)
        else:
            ba.plot_top(
                volume='value', region_values=region_values, mapping=map, cmap=cmap,
                vmin=clevels[0], vmax=clevels[1], ax=ax)
            ba.plot_top(
                volume='boundary', mapping=map, ax=ax, cmap=cmap_bound, vmin=0.01, vmax=0.8)

    return fig, ax


def make_bins(signal, timestamps, start_times, stop_times, binsize):

    # Loop over start times
    binned_signal = []
    for (start, end) in np.vstack((start_times, stop_times)).T:
        binned_signal.append(binned_statistic(timestamps, signal, bins=int((end-start)*(1/binsize)),
                                              range=(start, end), statistic=np.nanmean)[0])
    return binned_signal


def calculate_peths(
        spike_times, spike_clusters, cluster_ids, align_times, pre_time=0.2,
        post_time=0.5, bin_size=0.025, smoothing=0.025, return_fr=True):
    """
    Calcluate peri-event time histograms; return means and standard deviations
    for each time point across specified clusters

    :param spike_times: spike times (in seconds)
    :type spike_times: array-like
    :param spike_clusters: cluster ids corresponding to each event in `spikes`
    :type spike_clusters: array-like
    :param cluster_ids: subset of cluster ids for calculating peths
    :type cluster_ids: array-like
    :param align_times: times (in seconds) to align peths to
    :type align_times: array-like
    :param pre_time: time (in seconds) to precede align times in peth
    :type pre_time: float
    :param post_time: time (in seconds) to follow align times in peth
    :type post_time: float
    :param bin_size: width of time windows (in seconds) to bin spikes
    :type bin_size: float
    :param smoothing: standard deviation (in seconds) of Gaussian kernel for
        smoothing peths; use `smoothing=0` to skip smoothing
    :type smoothing: float
    :param return_fr: `True` to return (estimated) firing rate, `False` to return spike counts
    :type return_fr: bool
    :return: peths, binned_spikes
    :rtype: peths: Bunch({'mean': peth_means, 'std': peth_stds, 'tscale': ts, 'cscale': ids})
    :rtype: binned_spikes: np.array (n_align_times, n_clusters, n_bins)
    """

    # initialize containers
    n_offset = 5 * int(np.ceil(smoothing / bin_size))  # get rid of boundary effects for smoothing
    n_bins_pre = int(np.ceil(pre_time / bin_size)) + n_offset
    n_bins_post = int(np.ceil(post_time / bin_size)) + n_offset
    n_bins = n_bins_pre + n_bins_post
    binned_spikes = np.zeros(shape=(len(align_times), len(cluster_ids), n_bins))

    # build gaussian kernel if requested
    if smoothing > 0:
        w = n_bins - 1 if n_bins % 2 == 0 else n_bins
        window = gaussian(w, std=smoothing / bin_size)
        # half (causal) gaussian filter
        # window[int(np.ceil(w/2)):] = 0
        window /= np.sum(window)
        binned_spikes_conv = np.copy(binned_spikes)

    ids = np.unique(cluster_ids)

    # filter spikes outside of the loop
    idxs = np.bitwise_and(spike_times >= np.min(align_times) - (n_bins_pre + 1) * bin_size,
                          spike_times <= np.max(align_times) + (n_bins_post + 1) * bin_size)
    idxs = np.bitwise_and(idxs, np.isin(spike_clusters, cluster_ids))
    spike_times = spike_times[idxs]
    spike_clusters = spike_clusters[idxs]

    # compute floating tscale
    tscale = np.arange(-n_bins_pre, n_bins_post + 1) * bin_size
    # bin spikes
    for i, t_0 in enumerate(align_times):
        # define bin edges
        ts = tscale + t_0
        # filter spikes
        idxs = np.bitwise_and(spike_times >= ts[0], spike_times <= ts[-1])
        i_spikes = spike_times[idxs]
        i_clusters = spike_clusters[idxs]

        # bin spikes similar to bincount2D: x = spike times, y = spike clusters
        xscale = ts
        xind = (np.floor((i_spikes - np.min(ts)) / bin_size)).astype(np.int64)
        yscale, yind = np.unique(i_clusters, return_inverse=True)
        nx, ny = [xscale.size, yscale.size]
        ind2d = np.ravel_multi_index(np.c_[yind, xind].transpose(), dims=(ny, nx))
        r = np.bincount(ind2d, minlength=nx * ny, weights=None).reshape(ny, nx)

        # store (ts represent bin edges, so there are one fewer bins)
        bs_idxs = np.isin(ids, yscale)
        binned_spikes[i, bs_idxs, :] = r[:, :-1]

        # smooth
        if smoothing > 0:
            idxs = np.where(bs_idxs)[0]
            for j in range(r.shape[0]):
                binned_spikes_conv[i, idxs[j], :] = convolve(
                    r[j, :], window, mode='same', method='auto')[:-1]

    # average
    if smoothing > 0:
        binned_spikes_ = np.copy(binned_spikes_conv)
    else:
        binned_spikes_ = np.copy(binned_spikes)
    if return_fr:
        binned_spikes_ /= bin_size

    peth_means = np.mean(binned_spikes_, axis=0)
    peth_stds = np.std(binned_spikes_, axis=0)

    if smoothing > 0:
        peth_means = peth_means[:, n_offset:-n_offset]
        peth_stds = peth_stds[:, n_offset:-n_offset]
        binned_spikes = binned_spikes_[:, :, n_offset:-n_offset]
        tscale = tscale[n_offset:-n_offset]

    # package output
    tscale = (tscale[:-1] + tscale[1:]) / 2
    peths = dict({'means': peth_means, 'stds': peth_stds, 'tscale': tscale, 'cscale': ids})
    return peths, binned_spikes


def peri_multiple_events_time_histogram(
        spike_times, spike_clusters, events, event_ids, cluster_id,
        t_before=0.2, t_after=0.5, bin_size=0.025, smoothing=0.025, as_rate=True,
        include_raster=False, error_bars='sem', ax=None,
        pethline_kwargs=[{'color': 'blue', 'lw': 2}, {'color': 'red', 'lw': 2}],
        errbar_kwargs=[{'color': 'blue', 'alpha': 0.5}, {'color': 'red', 'alpha': 0.5}],
        raster_kwargs=[{'color': 'blue', 'lw': 0.5}, {'color': 'red', 'lw': 0.5}],
        eventline_kwargs={'color': 'black', 'alpha': 0.5}, **kwargs):
    """
    Plot peri-event time histograms, with the meaning firing rate of units centered on a given
    series of events. Can optionally add a raster underneath the PETH plot of individual spike
    trains about the events.

    Parameters
    ----------
    spike_times : array_like
        Spike times (in seconds)
    spike_clusters : array-like
        Cluster identities for each element of spikes
    events : array-like
        Times to align the histogram(s) to
    event_ids : array-like
        Identities of events
    cluster_id : int
        Identity of the cluster for which to plot a PETH

    t_before : float, optional
        Time before event to plot (default: 0.2s)
    t_after : float, optional
        Time after event to plot (default: 0.5s)
    bin_size :float, optional
        Width of bin for histograms (default: 0.025s)
    smoothing : float, optional
        Sigma of gaussian smoothing to use in histograms. (default: 0.025s)
    as_rate : bool, optional
        Whether to use spike counts or rates in the plot (default: `True`, uses rates)
    include_raster : bool, optional
        Whether to put a raster below the PETH of individual spike trains (default: `False`)
    error_bars : {'std', 'sem', 'none'}, optional
        Defines which type of error bars to plot. Options are:
        -- `'std'` for 1 standard deviation
        -- `'sem'` for standard error of the mean
        -- `'none'` for only plotting the mean value
        (default: `'std'`)
    ax : matplotlib axes, optional
        If passed, the function will plot on the passed axes. Note: current
        behavior causes whatever was on the axes to be cleared before plotting!
        (default: `None`)
    pethline_kwargs : dict, optional
        Dict containing line properties to define PETH plot line. Default
        is a blue line with weight of 2. Needs to have color. See matplotlib plot documentation
        for more options.
        (default: `{'color': 'blue', 'lw': 2}`)
    errbar_kwargs : dict, optional
        Dict containing fill-between properties to define PETH error bars.
        Default is a blue fill with 50 percent opacity.. Needs to have color. See matplotlib
        fill_between documentation for more options.
        (default: `{'color': 'blue', 'alpha': 0.5}`)
    eventline_kwargs : dict, optional
        Dict containing fill-between properties to define line at event.
        Default is a black line with 50 percent opacity.. Needs to have color. See matplotlib
        vlines documentation for more options.
        (default: `{'color': 'black', 'alpha': 0.5}`)
    raster_kwargs : dict, optional
        Dict containing properties defining lines in the raster plot.
        Default is black lines with line width of 0.5. See matplotlib vlines for more options.
        (default: `{'color': 'black', 'lw': 0.5}`)

    Returns
    -------
        ax : matplotlib axes
            Axes with all of the plots requested.
    """

    # Check to make sure if we fail, we fail in an informative way
    if not len(spike_times) == len(spike_clusters):
        raise ValueError('Spike times and clusters are not of the same shape')
    if len(events) == 1:
        raise ValueError('Cannot make a PETH with only one event.')
    if error_bars not in ('std', 'sem', 'none'):
        raise ValueError('Invalid error bar type was passed.')
    if not all(np.isfinite(events)):
        raise ValueError('There are NaN or inf values in the list of events passed. '
                         ' Please remove non-finite data points and try again.')

    # Construct an axis object if none passed
    if ax is None:
        plt.figure()
        ax = plt.gca()
    # Plot the curves and add error bars
    mean_max, bars_max = [], []
    for i, event_id in enumerate(np.unique(event_ids)):
        # Compute peths
        peths, binned_spikes = singlecell.calculate_peths(spike_times, spike_clusters, [cluster_id],
                                                          events[event_ids == event_id], t_before,
                                                          t_after, bin_size, smoothing, as_rate)
        mean = peths.means[0, :]
        ax.plot(peths.tscale, mean, **pethline_kwargs[i])
        if error_bars == 'std':
            bars = peths.stds[0, :]
        elif error_bars == 'sem':
            bars = peths.stds[0, :] / np.sqrt(np.sum(event_ids == event_id))
        else:
            bars = np.zeros_like(mean)
        if error_bars != 'none':
            ax.fill_between(peths.tscale, mean - bars, mean + bars, **errbar_kwargs[i])
        mean_max.append(mean.max())
        bars_max.append(bars[mean.argmax()])

    # Plot the event marker line. Extends to 5% higher than max value of means plus any error bar.
    plot_edge = (np.max(mean_max) + bars_max[np.argmax(mean_max)]) * 1.05
    ax.vlines(0., 0., plot_edge, **eventline_kwargs)
    # Set the limits on the axes to t_before and t_after. Either set the ylim to the 0 and max
    # values of the PETH, or if we want to plot a spike raster below, create an equal amount of
    # blank space below the zero where the raster will go.
    ax.set_xlim([-t_before, t_after])
    ax.set_ylim([-plot_edge if include_raster else 0., plot_edge])
    # Put y ticks only at min, max, and zero
    if mean.min() != 0:
        ax.set_yticks([0, mean.min(), mean.max()])
    else:
        ax.set_yticks([0., mean.max()])
    # Move the x axis line from the bottom of the plotting space to zero if including a raster,
    # Then plot the raster
    if include_raster:
        ax.axhline(0., color='black', lw=0.5)
        tickheight = plot_edge / len(events)  # How much space per trace
        tickedges = np.arange(0., -plot_edge - 1e-5, -tickheight)
        clu_spks = spike_times[spike_clusters == cluster_id]
        ii = 0
        for k, event_id in enumerate(np.unique(event_ids)):
            for i, t in enumerate(events[event_ids == event_id]):
                idx = np.bitwise_and(clu_spks >= t - t_before, clu_spks <= t + t_after)
                event_spks = clu_spks[idx]
                ax.vlines(event_spks - t, tickedges[i + ii + 1], tickedges[i + ii],
                          **raster_kwargs[k])
            ii += np.sum(event_ids == event_id)
        ax.set_ylabel('Firing Rate' if as_rate else 'Number of spikes', y=0.75)
    else:
        ax.set_ylabel('Firing Rate' if as_rate else 'Number of spikes')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('Time (s) after event')
    return ax


def calculate_mi(spike_counts_A, spike_counts_B):
       
    # Calculate joint probability distribution
    joint_hist, _, _ = np.histogram2d(spike_counts_A, spike_counts_B, bins=spike_counts_A.shape[0])
    joint_prob = joint_hist / np.sum(joint_hist)
    
    # Calculate marginal probabilities
    marg_prob_A = np.sum(joint_prob, axis=1)
    marg_prob_B = np.sum(joint_prob, axis=0)
    
    # Calculate mutual information
    non_zero_indices = joint_prob > 0
    mutual_info = np.sum(joint_prob[non_zero_indices] * \
                         np.log2(joint_prob[non_zero_indices] / \
                                 (np.outer(marg_prob_A, marg_prob_B))[non_zero_indices]))
    
    return mutual_info


def get_dlc_XYs(one, eid, view='left', likelihood_thresh=0.9):
    ses_details = one.get_details(eid)
    subject = ses_details['subject']
    date = ses_details['date']
    _, save_path = paths()
    if f'alf/_ibl_{view}Camera.times.npy' in one.list_datasets(eid):
        times = one.load_dataset(eid, f'_ibl_{view}Camera.times.npy')
    elif isfile(join(save_path, 'CameraTimestamps', f'{subject}_{date}_{view}Camera.npy')):
        times = np.load(join(save_path, 'CameraTimestamps', f'{subject}_{date}_{view}Camera.npy'))
    else:
        print('could not load camera timestamps')
        return None, None
    try:
        cam = one.load_dataset(eid, '_ibl_%sCamera.dlc.pqt' % view)
    except KeyError:
        print('not all dlc data available')
        return None, None
    points = np.unique(['_'.join(x.split('_')[:-1]) for x in cam.keys()])
    # Set values to nan if likelyhood is too low # for pqt: .to_numpy()
    XYs = {}
    for point in points:
        x = np.ma.masked_where(cam[point + '_likelihood'] < likelihood_thresh, cam[point + '_x'])
        x = x.filled(np.nan)
        y = np.ma.masked_where(cam[point + '_likelihood'] < likelihood_thresh, cam[point + '_y'])
        y = y.filled(np.nan)
        XYs[point] = np.array([x, y]).T
    return times, XYs


def smooth_interpolate_signal_sg(signal, window=31, order=3, interp_kind='cubic'):
    """Run savitzy-golay filter on signal, interpolate through nan points.

    Parameters
    ----------
    signal : np.ndarray
        original noisy signal of shape (t,), may contain nans
    window : int
        window of polynomial fit for savitzy-golay filter
    order : int
        order of polynomial for savitzy-golay filter
    interp_kind : str
        type of interpolation for nans, e.g. 'linear', 'quadratic', 'cubic'
    Returns
    -------
    np.array
        smoothed, interpolated signal for each time point, shape (t,)

    """

    signal_noisy_w_nans = np.copy(signal)
    timestamps = np.arange(signal_noisy_w_nans.shape[0])
    good_idxs = np.where(~np.isnan(signal_noisy_w_nans))[0]
    # perform savitzky-golay filtering on non-nan points
    signal_smooth_nonans = non_uniform_savgol(
        timestamps[good_idxs], signal_noisy_w_nans[good_idxs], window=window, polynom=order)
    signal_smooth_w_nans = np.copy(signal_noisy_w_nans)
    signal_smooth_w_nans[good_idxs] = signal_smooth_nonans
    # interpolate nan points
    interpolater = interp1d(
        timestamps[good_idxs], signal_smooth_nonans, kind=interp_kind, fill_value='extrapolate')

    signal = interpolater(timestamps)

    return signal


def non_uniform_savgol(x, y, window, polynom):
    """Applies a Savitzky-Golay filter to y with non-uniform spacing as defined in x.
    This is based on
    https://dsp.stackexchange.com/questions/1676/savitzky-golay-smoothing-filter-for-not-equally-spaced-data
    The borders are interpolated like scipy.signal.savgol_filter would do
    https://dsp.stackexchange.com/a/64313
    Parameters
    ----------
    x : array_like
        List of floats representing the x values of the data
    y : array_like
        List of floats representing the y values. Must have same length as x
    window : int (odd)
        Window length of datapoints. Must be odd and smaller than x
    polynom : int
        The order of polynom used. Must be smaller than the window size
    Returns
    -------
    np.array
        The smoothed y values
    """

    if len(x) != len(y):
        raise ValueError('"x" and "y" must be of the same size')

    if len(x) < window:
        raise ValueError('The data size must be larger than the window size')

    if type(window) is not int:
        raise TypeError('"window" must be an integer')

    if window % 2 == 0:
        raise ValueError('The "window" must be an odd integer')

    if type(polynom) is not int:
        raise TypeError('"polynom" must be an integer')

    if polynom >= window:
        raise ValueError('"polynom" must be less than "window"')

    half_window = window // 2
    polynom += 1

    # Initialize variables
    A = np.empty((window, polynom))  # Matrix
    tA = np.empty((polynom, window))  # Transposed matrix
    t = np.empty(window)  # Local x variables
    y_smoothed = np.full(len(y), np.nan)

    # Start smoothing
    for i in range(half_window, len(x) - half_window, 1):
        # Center a window of x values on x[i]
        for j in range(0, window, 1):
            t[j] = x[i + j - half_window] - x[i]

        # Create the initial matrix A and its transposed form tA
        for j in range(0, window, 1):
            r = 1.0
            for k in range(0, polynom, 1):
                A[j, k] = r
                tA[k, j] = r
                r *= t[j]

        # Multiply the two matrices
        tAA = np.matmul(tA, A)

        # Invert the product of the matrices
        tAA = np.linalg.inv(tAA)

        # Calculate the pseudoinverse of the design matrix
        coeffs = np.matmul(tAA, tA)

        # Calculate c0 which is also the y value for y[i]
        y_smoothed[i] = 0
        for j in range(0, window, 1):
            y_smoothed[i] += coeffs[0, j] * y[i + j - half_window]

        # If at the end or beginning, store all coefficients for the polynom
        if i == half_window:
            first_coeffs = np.zeros(polynom)
            for j in range(0, window, 1):
                for k in range(polynom):
                    first_coeffs[k] += coeffs[k, j] * y[j]
        elif i == len(x) - half_window - 1:
            last_coeffs = np.zeros(polynom)
            for j in range(0, window, 1):
                for k in range(polynom):
                    last_coeffs[k] += coeffs[k, j] * y[len(y) - window + j]

    # Interpolate the result at the left border
    for i in range(0, half_window, 1):
        y_smoothed[i] = 0
        x_i = 1
        for j in range(0, polynom, 1):
            y_smoothed[i] += first_coeffs[j] * x_i
            x_i *= x[i] - x[half_window]

    # Interpolate the result at the right border
    for i in range(len(x) - half_window, len(x), 1):
        y_smoothed[i] = 0
        x_i = 1
        for j in range(0, polynom, 1):
            y_smoothed[i] += last_coeffs[j] * x_i
            x_i *= x[i] - x[-half_window - 1]

    return y_smoothed


def get_pupil_diameter(XYs):
    """Estimate pupil diameter by taking median of different computations.

    In the two most obvious ways:
    d1 = top - bottom, d2 = left - right

    In addition, assume the pupil is a circle and estimate diameter from other pairs of
    points

    Author: Michael Schartner

    Parameters
    ----------
    XYs : dict
        keys should include `pupil_top_r`, `pupil_bottom_r`,
        `pupil_left_r`, `pupil_right_r`
    Returns
    -------
    np.array
        pupil diameter estimate for each time point, shape (n_frames,)

    """

    # direct diameters
    t = XYs['pupil_top_r'][:, :2]
    b = XYs['pupil_bottom_r'][:, :2]
    l = XYs['pupil_left_r'][:, :2]
    r = XYs['pupil_right_r'][:, :2]

    def distance(p1, p2):
        return ((p1[:, 0] - p2[:, 0]) ** 2 + (p1[:, 1] - p2[:, 1]) ** 2) ** 0.5

    # get diameter via top-bottom and left-right
    ds = []
    ds.append(distance(t, b))
    ds.append(distance(l, r))

    def dia_via_circle(p1, p2):
        # only valid for non-crossing edges
        u = distance(p1, p2)
        return u * (2 ** 0.5)

    # estimate diameter via circle assumption
    for side in [[t, l], [t, r], [b, l], [b, r]]:
        ds.append(dia_via_circle(side[0], side[1]))
    diam = np.nanmedian(ds, axis=0)

    return diam


def get_raw_smooth_pupil_diameter(XYs):

    # threshold (in standard deviations) beyond which a point is labeled as an outlier
    std_thresh = 5

    # threshold (in seconds) above which we will not interpolate nans, but keep them
    # (for long stretches interpolation may not be appropriate)
    nan_thresh = 1

    # compute framerate of camera
    fr = 60  # set by hardware
    window = 61  # works well empirically

    # compute diameter using raw values of 4 markers (will be noisy and have missing data)
    diam0 = get_pupil_diameter(XYs)

    # run savitzy-golay filter on non-nan timepoints to denoise
    diam_sm0 = smooth_interpolate_signal_sg(
        diam0, window=window, order=3, interp_kind='linear')

    # find outliers, set to nan
    errors = diam0 - diam_sm0
    std = np.nanstd(errors)
    diam1 = np.copy(diam0)
    diam1[(errors < (-std_thresh * std)) | (errors > (std_thresh * std))] = np.nan
    # run savitzy-golay filter again on (possibly reduced) non-nan timepoints to denoise
    diam_sm1 = smooth_interpolate_signal_sg(
        diam1, window=window, order=3, interp_kind='linear')

    # don't interpolate long strings of nans
    t = np.diff(1 * np.isnan(diam1))
    begs = np.where(t == 1)[0]
    ends = np.where(t == -1)[0]
    if begs.shape[0] > ends.shape[0]:
        begs = begs[:ends.shape[0]]
    for b, e in zip(begs, ends):
        if (e - b) > (fr * nan_thresh):
            diam_sm1[(b + 1):(e + 1)] = np.nan  # offset by 1 due to earlier diff

    # diam_sm1 is the final smoothed pupil diameter estimate
    return diam0, diam_sm1


def SNR(diam0, diam_sm1):

    # compute signal to noise ratio between raw and smooth dia
    good_idxs = np.where(~np.isnan(diam_sm1) & ~np.isnan(diam0))[0]
    snr = (np.var(diam_sm1[good_idxs]) /
           np.var(diam_sm1[good_idxs] - diam0[good_idxs]))

    return snr


def query_opto_sessions(subject, include_ephys=False, one=None):
    one = one or ONE()
    if include_ephys:
        sessions = one.alyx.rest('sessions', 'list', subject=subject,
                                 task_protocol='_iblrig_tasks_opto_')
    else:
        sessions = one.alyx.rest('sessions', 'list', subject=subject,
                                 task_protocol='_iblrig_tasks_opto_biasedChoiceWorld')
    return [sess['url'][-36:] for sess in sessions]


def behavioral_criterion(eids, min_perf=0.75, min_trials=100, return_excluded=False,
                         verbose=True, one=None):
    if one is None:
        one = ONE()
    use_eids, excl_eids = [], []
    for j, eid in enumerate(eids):
        try:
            trials = load_trials(eid, one=one)
            perf = (np.sum(trials.loc[np.abs(trials['signed_contrast']) == 1, 'feedbackType'] == 1)
                    / trials[np.abs(trials['signed_contrast']) == 1].shape[0])
            details = one.get_details(eid)
            if (perf > min_perf) & (trials.shape[0] > min_trials):
                use_eids.append(eid)
            else:
                if verbose:
                    print('%s %s excluded (perf: %.2f, n_trials: %d)'
                          % (details['subject'], details['start_time'][:10], perf, trials.shape[0]))
                excl_eids.append(eid)
        except Exception:
            if verbose:
                print('Could not load session %s' % eid)
    if return_excluded:
        return use_eids, excl_eids
    else:
        return use_eids


def fit_psychfunc(stim_levels, n_trials, proportion):
    # Fit a psychometric function with two lapse rates
    #
    # Returns vector pars with [bias, threshold, lapselow, lapsehigh]
    import psychofit as psy
    assert (stim_levels.shape == n_trials.shape == proportion.shape)
    if stim_levels.max() <= 1:
        stim_levels = stim_levels * 100

    pars, _ = psy.mle_fit_psycho(np.vstack((stim_levels, n_trials, proportion)),
                                 P_model='erf_psycho_2gammas',
                                 parstart=np.array([0, 20, 0.05, 0.05]),
                                 parmin=np.array([-100, 5, 0, 0]),
                                 parmax=np.array([100, 100, 1, 1]))
    return pars


def plot_psychometric(trials, ax, color='b', linestyle='solid', fraction=True):
    import psychofit as psy
    if trials['signed_contrast'].max() <= 1:
        trials['signed_contrast'] = trials['signed_contrast'] * 100

    stim_levels = np.sort(trials['signed_contrast'].unique())
    pars = fit_psychfunc(stim_levels, trials.groupby('signed_contrast').size(),
                         trials.groupby('signed_contrast').mean()['right_choice'])

    # plot psychfunc
    sns.lineplot(x=np.arange(-27, 27), y=psy.erf_psycho_2gammas(pars, np.arange(-27, 27)),
                 ax=ax, color=color, linestyle=linestyle)

    # plot psychfunc: -100, +100
    sns.lineplot(x=np.arange(-36, -31), y=psy.erf_psycho_2gammas(pars, np.arange(-103, -98)),
                 ax=ax, color=color, linestyle=linestyle)
    sns.lineplot(x=np.arange(31, 36), y=psy.erf_psycho_2gammas(pars, np.arange(98, 103)),
                 ax=ax, color=color, linestyle=linestyle)

    # now break the x-axis
    trials['signed_contrast'].replace(-100, -35)
    trials['signed_contrast'].replace(100, 35)

    # plot datapoints with errorbars on top
    sns.lineplot(x=trials['signed_contrast'], y=trials['right_choice'], ax=ax,
                 **{**{'err_style': "bars",
                       'linewidth': 0, 'linestyle': 'None', 'mew': 0.5,
                       'marker': 'o', 'errorbar': 'se'}}, color=color)

    ax.set(xticks=[-35, -25, -12.5, 0, 12.5, 25, 35], xlim=[-40, 40], ylim=[0, 1.02],
           xlabel='Contrast (%)')
    ax.set_xticklabels(['-100', '-25', '-12.5', '0', '12.5', '25', '100'])
    if fraction:
        ax.set(yticks=[0, 0.25, 0.5, 0.75, 1], yticklabels=['0', '.25', '.5', '.75', '1'])
        ax.set_ylabel('Fraction of right choices', labelpad=1)
    else:
        ax.set(yticks=[0, 0.25, 0.5, 0.75, 1], yticklabels=['0', '25', '50', '75', '100'],
               ylabel='Right choices (%)')
    # break_xaxis()


def break_xaxis(y=-0.004, **kwargs):

    # axisgate: show axis discontinuities with a quick hack
    # https://twitter.com/StevenDakin/status/1313744930246811653?s=19
    # first, white square for discontinuous axis
    plt.text(-30, y, '-', fontsize=14, fontweight='bold',
             horizontalalignment='center', verticalalignment='center',
             color='w')
    plt.text(30, y, '-', fontsize=14, fontweight='bold',
             horizontalalignment='center', verticalalignment='center',
             color='w')

    # put little dashes to cut axes
    plt.text(-30, y, '/ /', horizontalalignment='center',
             verticalalignment='center', fontsize=12, fontweight='bold')
    plt.text(30, y, '/ /', horizontalalignment='center',
             verticalalignment='center', fontsize=12, fontweight='bold')


def get_bias(trials):
    import psychofit as psy
    """
    Calculate bias by fitting psychometric curves to the 80/20 and 20/80 blocks, finding the
    point on the y-axis when contrast = 0% and getting the difference.
    """
    if len(trials) == 0:
        return np.nan

    # 20/80 blocks
    these_trials = trials[trials['probabilityLeft'] == 0.2]
    stim_levels = np.sort(these_trials['signed_contrast'].unique())
    pars_right = fit_psychfunc(stim_levels, these_trials.groupby('signed_contrast').size(),
                               these_trials.groupby('signed_contrast').mean()['right_choice'])
    bias_right = psy.erf_psycho_2gammas(pars_right, 0)

    # 80/20 blocks
    these_trials = trials[trials['probabilityLeft'] == 0.8]
    stim_levels = np.sort(these_trials['signed_contrast'].unique())
    pars_left = fit_psychfunc(stim_levels, these_trials.groupby('signed_contrast').size(),
                              these_trials.groupby('signed_contrast').mean()['right_choice'])
    bias_left = psy.erf_psycho_2gammas(pars_left, 0)

    return bias_right - bias_left
