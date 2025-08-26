# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 11:32:45 2025

By Guido Meijer
"""

from stim_functions import load_subjects, init_one, query_opto_sessions

one = init_one(open_one=False)
subjects = load_subjects()

eids = []
for i, nickname in enumerate(subjects['subject']):
        
    these_eids = one.alyx.rest('sessions', 'list', subject=nickname,
                               task_protocol='_iblrig_tasks_opto_biasedChoiceWorld')
    eids.append(these_eids)