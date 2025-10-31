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


# --- Step 3: Get projection data for these experiments ---

# 'get_structure_unionizes' fetches the projection data for a list of experiments.
# This method calculates the volume and density of projections to various target structures.
# This can take some time as it may need to download data for each experiment.
print("Fetching projection data... (This may take a while)")
unionizes_df = mcc.get_structure_unionizes(experiment_ids=[480074702, 114155190, 128055110])
print("Projection data fetched successfully.\n")

# Exclude injection site itself
unionizes_df = unionizes_df[~unionizes_df['is_injection']]

# --- Step 4: Process and aggregate the projection data ---

# We are interested in the average projection density across all DR experiments.
# We group the data by the target structure_id and calculate the mean.
# We will focus on 'projection_density' as the measure of projection strength.
aggregated_projections = unionizes_df.groupby('structure_id')['projection_density'].mean().reset_index()

# --- Step 5: Map structure IDs to names for readability ---

# The structure tree allows us to map the structure IDs back to their names.
# We'll create a mapping from ID to name.
structure_tree = mcc.get_structure_tree()
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
output_filename = r'C:\Users\guido\Repositories\SerotoninStimulation\Data\dr_projection_strength.csv'
final_df.to_csv(output_filename, index=False)

print(f"Analysis complete. Results saved to '{output_filename}'")
print("\n--- Top 10 Projection Targets from Dorsal Raphe ---")
print(final_df.head(10).to_string(index=False))
print("\n-------------------------------------------------")
