"""
Register on Alyx the set of tracked traces (after histology) for a given mouse.
All your tracks should be in a single folder, and the files names should follow the nomenclature
{yyyy-mm-dd}_{SubjectName}_{SessionNumber}_{ProbeLabel}_pts.csv
Edit the variable 'path_tracks'(set it either to your local directory [example given here],
either to the Google folder if synched to your machine).
To check if the registration worked, go to the admin interface
> experiments > trajectory estimates > search for the subject
If you want to test first, use  ALYX_URL = "https://dev.alyx.internationalbrainlab.org"
And check the data appears on:
https://dev.alyx.internationalbrainlab.org/admin/experiments/trajectoryestimate/?
When you feel confident you can upload without error,
set EXAMPLE_OVERWRITE = False ,
change to the   ALYX_URL = "https://alyx.internationalbrainlab.org"
and re-run.
With EXAMPLE_OVERWRITE = True, the script downloads an example dataset and runs
the registration (used for automatic testing of the example).
"""
# Author: Olivier, Gaelle
from pathlib import Path

from one.api import ONE

from ibllib.pipes import histology

# ======== EDIT FOR USERS ====

# Edit so as to reflect the directory containing your electrode tracks
#path_tracks = "/home/guido/Histology/ZFM-05182/tracks/"
#path_tracks = "E:\\Histology\\ZFM-05495\\tracks"
path_tracks = 'D:\\Flatiron\\mainenlab\\Subjects\\ZFM-05170\\histology\\tracks'

# ======== DO NOT EDIT BELOW ====
ALYX_URL = "https://alyx.internationalbrainlab.org"
one = ONE(base_url=ALYX_URL)
histology.register_track_files(path_tracks=path_tracks, one=one, overwrite=True)
histology.detect_missing_histology_tracks(path_tracks=path_tracks, one=one)