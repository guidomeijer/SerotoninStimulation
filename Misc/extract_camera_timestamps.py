#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 13:04:25 2022
By: Guido Meijer
"""

import numpy as np
from os.path import join, isfile
from serotonin_functions import paths, query_ephys_sessions
from ibllib.io.extractors.camera import CameraTimestampsFPGA
from ibllib.io.extractors.ephys_fpga import get_sync_and_chn_map, get_main_probe_sync, load_channel_map
from one.api import ONE
one = ONE()

# Get path to save
_, save_path = paths()

# Query ephys sessions
rec = query_ephys_sessions()

for i, eid in enumerate(np.unique(rec['eid'])):

    # Get session details
    print(f'Processing {eid} [{i+1} of {len(np.unique(rec["eid"]))}]')

    try:
        cam_times = one.load_dataset(eid, dataset='_ibl_leftCamera.times.npy', collection='alf')
    except:
        print('\nNo extracted timestamps found, starting manual extraction..\n')

        # Get subject and date of recording
        ses_details = one.get_details(eid)
        subject = ses_details['subject']
        date = ses_details['date']

        if not isfile(join(save_path, 'CameraTimestamps', f'{subject}_{date}_leftCamera.npy')):

            # Download sync
            one.load_datasets(eid, download_only=True, datasets=['_spikeglx_sync.channels.npy',
                                                                 '_spikeglx_sync.polarities.npy',
                                                                 '_spikeglx_sync.times.npy'])

            # Download video
            one.load_dataset(eid, download_only=True, dataset='_iblrig_leftCamera.raw.mp4',
                             collection='raw_video_data')

            # Extract timestamps
            try:
                session_path = one.eid2path(eid)
                sync, chmap = get_sync_and_chn_map(session_path, 'raw_ephys_data')
                extractor = CameraTimestampsFPGA('left', session_path=session_path)
                left_times = extractor.extract(sync=sync, chmap=chmap)[0]
                np.save(join(save_path, 'CameraTimestamps', f'{subject}_{date}_leftCamera.npy'),
                        left_times)
                print('Extracted timestamps saved')
            except Exception as err:
                print(err)
                continue

