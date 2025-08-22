# Serotonin drives choice-independent reconfiguration of distributed neural activity
<img src="https://github.com/user-attachments/assets/af61346d-2d8a-4df5-b764-7f6932d9ad01" width="40%" align="right"/>

📄 [Link to the preprint](https://doi.org/10.1101/2025.08.01.668048) 📄

This repository contains all the code to reproduce the figures in the publication. The data is hosted by the International Brain Laboratory and accessable through the [Open Neurophysiology Environment (ONE)](https://int-brain-lab.github.io/ONE/one_reference.html) interface. The dataset contains 86 Neuropixel recordings from 17 mice with ~7500 good single neurons in total. The recordings include optogenetic stimulation of serotonergic neurons in the dorsal raphe nucleus during performance of the IBL steering wheel task and during quiet wakefulness. 

### Installation
1. Create an Anaconda or Miniforge (recommended) environment `conda create -n serotonin python=3.10 git` (or `mamba` instead of `conda` if you use Miniforge)
2. Clone this repository `git clone https://github.com/guidomeijer/SerotoninStimulation`
3. Install the required packages `pip install -r requirements.txt`

### Instructions

The first time you run the code it will ask you to input some paths to folders on your computer, for example where to save the figures. The code to generate the figures in the publication can be found in `Figures`. The processed data necessary for plotting (e.g. which neurons are significantly 5-HT modulated) is provided in `Data`. The processed manifold data was too large to include in the repository and will have to be generated using the `Preprocess\prepare_manifold_psth_data.py` script. You can also rerun any processing done to obtain the processed data by running the corresponding script in `Preprocessing`. 

An example code on how to load the data can be found in `example_data_loading.py`. Documentation is available on how to [query](https://int-brain-lab.github.io/ONE/notebooks/one_search/one_search.html), [explore](https://int-brain-lab.github.io/ONE/notebooks/one_list/one_list.html), and [load](https://int-brain-lab.github.io/ONE/notebooks/one_load/one_load.html) the data. Furthermore, extensive documentation as to what all the dataset types (e.g. `spikes.times`) entail can be found [here](https://docs.google.com/document/d/1OqIqqakPakHXRAwceYLwFY9gOrm8_P62XIfCTnHwstg).







