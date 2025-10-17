# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 12:22:34 2025

By Gemini Pro 
"""

# Import necessary libraries from the AllenSDK and others
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
import os

# Specify the directory for caching the Allen Brain Atlas data.
# This saves time on subsequent runs.
cache_dir = r"D:\AllenAtlas"
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

# Initialize the MouseConnectivityCache.
# The manifest file helps manage the data downloads.
mcc = MouseConnectivityCache(manifest_file=os.path.join(cache_dir, 'mouse_connectivity_manifest.json'),
                             cache=True)

# --- Step 1: Get the structure tree and find the Dorsal Raphe Nucleus ---

# The structure tree contains information about all brain regions,
# including their names, acronyms, and IDs.
structure_tree = mcc.get_structure_tree()

# We can find the Dorsal Raphe Nucleus by its name.
# This returns a list of matching structures; we'll take the first one.
dr_structures = structure_tree.get_structures_by_name(['Dorsal nucleus raphe'])
if not dr_structures:
    raise ValueError("Could not find 'Dorsal raphe nucleus' in the structure tree.")
dr_structure = dr_structures[0]
dr_id = dr_structure['id']

print(f"Found Dorsal Raphe Nucleus (DR) with ID: {dr_id}\n")


# --- Step 2: Find experiments with injections in the Dorsal Raphe Nucleus ---

# We search for all experiments where the injection was primarily in the DR.
experiments = mcc.get_experiments(injection_structure_ids=[dr_id])
experiment_ids = [exp['id'] for exp in experiments]

print(f"Found {len(experiment_ids)} experiments with injections in the DR.")
print(f"Experiment IDs: {experiment_ids}\n")

if not experiment_ids:
    print("No experiments found for the Dorsal Raphe Nucleus. Exiting.")
    exit()

# --- Step 3: Get projection data for these experiments ---

# 'get_structure_unionizes' fetches the projection data for a list of experiments.
# This method calculates the volume and density of projections to various target structures.
# This can take some time as it may need to download data for each experiment.
print("Fetching projection data... (This may take a while)")
unionizes_df = mcc.get_structure_unionizes(experiment_ids=experiment_ids)
print("Projection data fetched successfully.\n")


# --- Step 4: Process and aggregate the projection data ---

# We are interested in the average projection density across all DR experiments.
# We group the data by the target structure_id and calculate the mean.
# We will focus on 'projection_density' as the measure of projection strength.
aggregated_projections = unionizes_df.groupby('structure_id')['projection_density'].mean().reset_index()

# --- Step 5: Map structure IDs to names for readability ---

# The structure tree allows us to map the structure IDs back to their names.
# We'll create a mapping from ID to name.
structure_names = structure_tree.get_structures_by_id(aggregated_projections['structure_id'])
id_to_name_map = {st['id']: st['name'] for st in structure_names}
id_to_acronym_map = {st['id']: st['acronym'] for st in structure_names}

# Add the structure names and acronyms to our aggregated data.
aggregated_projections['region_name'] = aggregated_projections['structure_id'].map(id_to_name_map)
aggregated_projections['allen_acronym'] = aggregated_projections['structure_id'].map(id_to_acronym_map)

# --- Step 6: Finalize and save the results ---

# Sort the data to see the strongest projections first.
sorted_projections = aggregated_projections.sort_values(by='projection_density', ascending=False)

# Reorder columns for a cleaner output file.
final_df = sorted_projections[['structure_id', 'allen_acronym', 'region_name', 'projection_density']]

# Save the results to a CSV file.
output_filename = r'C:\Users\Guido1\Repositories\SerotoninStimulation\Data\dr_projection_strength.csv'
final_df.to_csv(output_filename, index=False)

print(f"Analysis complete. Results saved to '{output_filename}'")
print("\n--- Top 10 Projection Targets from Dorsal Raphe ---")
print(final_df.head(10).to_string(index=False))
print("\n-------------------------------------------------")
