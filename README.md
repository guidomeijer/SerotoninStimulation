# SerotoninStimulation

To access the data you need to create an environment and setup the ONE API by following these instructions: https://int-brain-lab.github.io/iblenv/notebooks_external/data_download.html. 

You will also need to install ssm (https://github.com/lindermanlab/ssm), for further requirements see requirements.txt.

The data will be loaded in using ONE. However, the onset times of optogenetic stimulation are not extracted on the database, instead they can be found in ./Data/OptoTimes. The function `load_passive_opto_times` in `serotonin_functions` can load in the already extraxted NPY files. 




