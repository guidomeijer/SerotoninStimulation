from dateutil import parser
import numpy as np
import pandas as pd
from pathlib import Path

from iblutil.numerical import ismember
from brainbox.io.one import SpikeSortingLoader, SessionLoader
from ibllib.atlas.regions import BrainRegions
from one.remote import aws
from stim_functions import get_neuron_qc, get_artifact_neurons
from ibllib.atlas import AllenAtlas
ba = AllenAtlas()


def load_good_units(one, pid, compute_metrics=False, **kwargs):
    """
    Function to load the cluster information and spike trains for clusters that pass all quality metrics.

    Parameters
    ----------
    one: one.api.ONE
        Instance to be used to connect to local or remote database
    pid: str
        A probe insertion UUID
    compute_metrics: bool
        If True, force SpikeSortingLoader.merge_clusters to recompute the cluster metrics. Default is False
    kwargs:
        Keyword arguments passed to SpikeSortingLoader upon initiation. Specifically, if one instance offline,
        you need to pass 'eid' and 'pname' here as they cannot be inferred from pid in offline mode.

    Returns
    -------
    good_spikes: dict
        Spike trains associated with good clusters. Dictionary with keys ['depths', 'times', 'clusters', 'amps']
    good_clusters: pandas.DataFrame
        Information of clusters for this pid that pass all quality metrics
    """
    eid = kwargs.pop('eid', '')
    pname = kwargs.pop('pname', '')
    spike_loader = SpikeSortingLoader(pid=pid, one=one, eid=eid, pname=pname, atlas=ba)
    spikes, clusters, channels = spike_loader.load_spike_sorting()
    clusters = spike_loader.merge_clusters(spikes, clusters, channels)
    
    # Select IBL good neurons and exclude artifact neurons
    clusters_labeled = get_neuron_qc(pid, one=one, ba=ba)
    clusters_labeled['atlas_id'] = clusters['atlas_id']
    iok = clusters_labeled['label'] == 1
    good_clusters = clusters_labeled[iok]
    artifact_neurons = get_artifact_neurons()
    good_clusters = good_clusters[~good_clusters['cluster_id'].isin(
        artifact_neurons.loc[artifact_neurons['pid'] == pid, 'neuron_id'])]

    spike_idx, ib = ismember(spikes['clusters'], good_clusters.index)
    good_clusters.reset_index(drop=True, inplace=True)
    # Filter spike trains for only good clusters
    good_spikes = {k: v[spike_idx] for k, v in spikes.items()}
    good_spikes['clusters'] = good_clusters.index[ib].astype(np.int32)

    return good_spikes, good_clusters


def merge_probes(spikes_list, clusters_list):
    """
    Merge spikes and clusters information from several probes as if they were recorded from the same probe.
    This can be used to account for the fact that data from the probes recorded in the same session are not
    statistically independent as they have the same underlying behaviour.

    NOTE: The clusters dataframe will be re-indexed to avoid duplicated indices. Accordingly, spikes['clusters']
    will be updated. To unambiguously identify clusters use the column 'uuids'

    Parameters
    ----------
    spikes_list: list of dicts
        List of spike dictionaries as loaded by SpikeSortingLoader or brainwidemap.load_good_units
    clusters_list: list of pandas.DataFrames
        List of cluster dataframes as loaded by SpikeSortingLoader.merge_clusters or brainwidemap.load_good_units

    Returns
    -------
    merged_spikes: dict
        Merged and time-sorted spikes in single dictionary, where 'clusters' is adjusted to index into merged_clusters
    merged_clusters: pandas.DataFrame
        Merged clusters in single dataframe, re-indexed to avoid duplicate indices.
        To unambiguously identify clusters use the column 'uuids'
    """

    assert (len(clusters_list) == len(spikes_list)), 'clusters_list and spikes_list must have the same length'
    assert all([isinstance(s, dict) for s in spikes_list]), 'spikes_list must contain only dictionaries'
    assert all([isinstance(c, pd.DataFrame) for c in clusters_list]), 'clusters_list must contain only pd.DataFrames'

    merged_spikes = []
    merged_clusters = []
    cluster_max = 0
    for clusters, spikes in zip(clusters_list, spikes_list):
        spikes['clusters'] += cluster_max
        cluster_max = clusters.index.max() + 1
        merged_spikes.append(spikes)
        merged_clusters.append(clusters)
    merged_clusters = pd.concat(merged_clusters, ignore_index=True)
    merged_spikes = {k: np.concatenate([s[k] for s in merged_spikes]) for k in merged_spikes[0].keys()}
    # Sort spikes by spike time
    sort_idx = np.argsort(merged_spikes['times'], kind='stable')
    merged_spikes = {k: v[sort_idx] for k, v in merged_spikes.items()}

    return merged_spikes, merged_clusters


def load_trials_and_mask(
        one, eid, min_rt=0.08, max_rt=2., nan_exclude='default', min_trial_len=None,
        max_trial_len=None, exclude_unbiased=False, exclude_nochoice=False, sess_loader=None):
    """
    Function to load all trials for a given session and create a mask to exclude all trials that have a reaction time
    shorter than min_rt or longer than max_rt or that have NaN for one of the specified events.

    Parameters
    ----------
    one: one.api.ONE
        Instance to be used to connect to local or remote database
    eid: str
        A session UUID
    min_rt: float or None
        Minimum admissible reaction time in seconds for a trial to be included. Default is 0.08. If None, don't apply.
    max_rt: float or None
        Maximum admissible reaction time in seconds for a trial to be included. Default is 2. If None, don't apply.
    nan_exclude: list or 'default'
        List of trial events that cannot be NaN for a trial to be included. If set to 'default' the list is
        ['stimOn_times','choice','feedback_times','probabilityLeft','firstMovement_times','feedbackType']
    min_trial_len: float or None
        Minimum admissible trial length measured by goCue_time (start) and feedback_time (end).
        Default is None.
    max_trial_len: float or Nona
        Maximum admissible trial length measured by goCue_time (start) and feedback_time (end).
        Default is None.
    exclude_unbiased: bool
        True to exclude trials that fall within the unbiased block at the beginning of session.
        Default is False.
    exclude_nochoice: bool
        True to exclude trials where the animal does not respond. Default is False.
    sess_loader: brainbox.io.one.SessionLoader or NoneType
        Optional SessionLoader object; if None, this object will be created internally

    Returns
    -------
    trials: pandas.DataFrame
        Trials table containing all trials for this session. If complete with columns:
        ['stimOff_times','goCueTrigger_times','feedbackType','contrastLeft','contrastRight','rewardVolume',
        'goCue_times','choice','feedback_times','stimOn_times','response_times','firstMovement_times',
        'probabilityLeft', 'intervals_0', 'intervals_1']
    mask: pandas.Series
        Boolean Series to mask trials table for trials that pass specified criteria. True for all trials that should be
        included, False for all trials that should be excluded.
    """

    if nan_exclude == 'default':
        nan_exclude = [
            'stimOn_times',
            'choice',
            'feedback_times',
            'probabilityLeft',
            'firstMovement_times',
            'feedbackType'
        ]

    if sess_loader is None:
        sess_loader = SessionLoader(one, eid)

    if sess_loader.trials.empty:
        sess_loader.load_trials()

    # Create a mask for trials to exclude
    # Remove trials that are outside the allowed reaction time range
    if min_rt is not None:
        query = f'(firstMovement_times - stimOn_times < {min_rt})'
    else:
        query = ''
    if max_rt is not None:
        query += f' | (firstMovement_times - stimOn_times > {max_rt})'
    # Remove trials that are outside the allowed trial duration range
    if min_trial_len is not None:
        query += f' | (feedback_times - goCue_times < {min_trial_len})'
    if max_trial_len is not None:
        query += f' | (feedback_times - goCue_times > {max_trial_len})'
    # Remove trials with nan in specified events
    for event in nan_exclude:
        query += f' | {event}.isnull()'
    # Remove trials in unbiased block at beginning
    if exclude_unbiased:
        query += ' | (probabilityLeft == 0.5)'
    # Remove trials where animal does not respond
    if exclude_nochoice:
        query += ' | (choice == 0)'
    # If min_rt was None we have to clean up the string
    if min_rt is None:
        query = query[3:]

    # Create mask
    mask = ~sess_loader.trials.eval(query)
    
    # Add opto stimulated trials
    trials = sess_loader.trials
    trials['opto'] = one.load_dataset(eid, dataset='_ibl_trials.laserStimulation.npy').astype(int)

    return trials, mask


def download_aggregate_tables(one, target_path=None, type='clusters', tag='2022_Q4_IBL_et_al_BWM', overwrite=False):
    """
    Function to download the aggregated clusters information associated with the given data release tag from AWS.

    Parameters
    ----------
    one: one.api.ONE
        Instance to be used to connect to database.
    target_path: str or pathlib.Path
        Directory to which clusters.pqt should be downloaded. If None, downloads to one.cache_dir/bwm_tables
    type: {'clusters', 'trials'}
        Which type of aggregate table to load, clusters or trials table.
    tag: str
        Tag for which to download the clusters table. Default is '2022_Q4_IBL_et_al_BWM'.
    overwrite : bool
        If True, will re-download files even if file exists locally and file sizes match.

    Returns
    -------
    agg_path: pathlib.Path
        Path to the downloaded aggregate
    """

    if target_path is None:
        target_path = Path(one.cache_dir).joinpath('bwm_tables')
        target_path.mkdir(exist_ok=True)
    else:
        assert target_path.exists(), 'The target_path you passed does not exist.'

    agg_path = target_path.joinpath(f'{type}.pqt')
    s3, bucket_name = aws.get_s3_from_alyx(alyx=one.alyx)
    aws.s3_download_file(f"aggregates/{tag}/{type}.pqt", agg_path, s3=s3,
                         bucket_name=bucket_name, overwrite=overwrite)

    if not agg_path.exists():
        print(f'Downloading of {type} table failed.')
        return
    return agg_path


def filter_units_region(eids, clusters_table=None, one=None, mapping='Beryl', min_qc=1., min_units_sessions=(10, 2)):
    """
    Filter to retain only units that satisfy certain region based criteria.

    Parameters
    ----------
    eids: list or pandas.Series
        List of session UUIDs to include at start.
    clusters_table: str or pathlib.Path
        Absolute path to clusters table to be used for filtering. If None, requires to provide one.api.ONE instance
        to download the latest version.
    mapping: str
        Mapping from atlas id to brain region acronym to be applied. Default is 'Beryl'.
    one: one.api.ONE
        Instance to be used to connect to download clusters_table if this is not explicitly provided.
    min_qc: float or None
        Minimum QC label for a spike sorted unit to be retained.
        Default is 1. If None, criterion is not applied.
    min_units_sessions: tuple or None
        If tuple, the first entry is the minimum of units per session per region for a session to be retained, the
        second entry is the minimum number of those sessions per region for a region to be retained.
        Default is (10, 2). If None, criterion is not applied

    Returns
    -------
    regions_df: pandas.DataFrame
        Dataframe of units that survive region based criteria.
    """

    if not any([min_qc, min_units_sessions]):
        print('No criteria selected. Aborting.')
        return

    if clusters_table is None:
        if one is None:
            print(f'You either need to provide a path to clusters_table or an instance of one.api.ONE to '
                  f'download clusters_table.')
            return
        else:
            clusters_table = download_aggregate_tables(one, type='clusters')
    clus_df = pd.read_parquet(clusters_table)

    # Only consider given pids
    clus_df = clus_df.loc[clus_df['eid'].isin(eids)]
    diff = set(eids).difference(set(clus_df['eid']))
    if len(diff) != 0:
        print('WARNING: Not all eids in bwm_df are found in cluster table.')

    # Only consider units that pass min_qc
    if min_qc:
        clus_df = clus_df.loc[clus_df['label'] >= min_qc]

    # Add region acronyms column and remove root and void regions
    br = BrainRegions()
    clus_df[f'{mapping}'] = br.id2acronym(clus_df['atlas_id'], mapping=f'{mapping}')
    clus_df = clus_df.loc[~clus_df[f'{mapping}'].isin(['void', 'root'])]

    # Group by regions and filter for sessions per region
    if min_units_sessions:
        units_count = clus_df.groupby([f'{mapping}', 'eid']).aggregate(
            n_units=pd.NamedAgg(column='cluster_id', aggfunc='count'),
        )
        # Only keep sessions with at least min_units_sessions[0] units
        units_count = units_count[units_count['n_units'] >= min_units_sessions[0]]
        # Only keep regions with at least min_units_sessions[1] sessions left
        units_count = units_count.reset_index(level=['eid'])
        region_df = units_count.groupby([f'{mapping}']).aggregate(
            n_sessions=pd.NamedAgg(column='eid', aggfunc='count'),
        )
        region_df = region_df[region_df['n_sessions'] >= min_units_sessions[1]]
        # Merge back to get the eids and clusters
        region_session_df = pd.merge(region_df, units_count, on=f'{mapping}', how='left')
        region_session_df = region_session_df.reset_index(level=[f'{mapping}'])
        region_session_df.drop(labels=['n_sessions', 'n_units'], axis=1, inplace=True)
        clus_df = pd.merge(region_session_df, clus_df, on=['eid', f'{mapping}'], how='left')

    # Reset index
    clus_df.reset_index(inplace=True, drop=True)

    return clus_df


def filter_sessions(eids, trials_table, bwm_include=True, min_errors=3, min_trials=None):
    """
    Filters eids for sessions that pass certain criteria.
    The function first loads an aggregate of all trials for the brain wide map dataset
     that contains already pre-computed acceptance critera


    Parameters
    ----------
    eids: list or pandas.Series
        Session ids to map to regions. Typically, the 'eid' column of the bwm_df returned by bwm_query.
        Note that these eids must be represented in trials_table to be considered for the filter.
    trials_table: str or pathlib.Path
        Absolute path to trials table to be used for filtering.
    bwm_include: bool
        Whether to filter for BWM inclusion criteria (see defaults of function load_trials_and_mask()). Default is True.
    min_errors: int or None
        Minimum number of error trials after other criteria are applied. Default is 3.
    min_trials: int or None
        Minimum number of trials that pass default criteria (see load_trials_and_mask()) for a session to be retained.
        Default is None, i.e. not applied

    Returns
    -------
    eids: pandas.Series
        Session ids that pass the criteria
    """

    # Load trials table
    trials_df = pd.read_parquet(trials_table)

    # Keep only eids
    trials_df = trials_df.loc[trials_df['eid'].isin(eids)]

    # Aggregate and filter
    if bwm_include:
        trials_df = trials_df[trials_df['bwm_include']]

    trials_agg = trials_df.groupby('eid').aggregate(
        n_trials=pd.NamedAgg(column='eid', aggfunc='count'),
        n_error=pd.NamedAgg(column='feedbackType', aggfunc=lambda x: (x == -1).sum()),
    )
    if min_trials:
        trials_agg = trials_agg.loc[trials_agg['n_trials'] >= min_trials]
    if min_errors:
        trials_agg = trials_agg.loc[trials_agg['n_error'] >= min_errors]

    return trials_agg.index.to_list()
