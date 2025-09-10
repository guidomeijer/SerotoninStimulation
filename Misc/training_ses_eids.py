# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 11:32:45 2025

By Guido Meijer
"""


from one.api import ONE
#one = ONE()

one = ONE(mode='remote', base_url='https://openalyx.internationalbrainlab.org',
          password='international', silent=True)
subjects = ['ZFM-02181',
            'ZFM-02600',
            'ZFM-02601',
            'ZFM-03321',
            'ZFM-03323',
            'ZFM-03324',
            'ZFM-03329',
            'ZFM-03330',
            'ZFM-03331',
            'ZFM-03332',
            'ZFM-04080',
            'ZFM-04083',
            'ZFM-04122',
            'ZFM-04300',
            'ZFM-04811',
            'ZFM-05170',
            'ZFM-04820']
eids = []
for i, nickname in enumerate(subjects):
    these_eids = one.alyx.rest('sessions', 'list', subject=nickname,
                               task_protocol='_iblrig_tasks_opto_biasedChoiceWorld')
    eids.append(these_eids)
print(eids)