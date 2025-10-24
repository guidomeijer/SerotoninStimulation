import pandas as pd
from allensdk.api.queries.rma_api import RmaApi
from allensdk.core.structure_tree import StructureTree

HTR_GENES = [
    'Htr1a', 'Htr1b', 'Htr1d', 'Htr1f',
    'Htr2a', 'Htr2b', 'Htr2c',
    'Htr3a', 'Htr3b',
    'Htr4',
    'Htr5a', 'Htr5b',
    'Htr6',
    'Htr7'
]


def get_expression_by_region(gene_acronym='Htr1a'):
    """
    Fetches gene expression data per brain region for a given gene acronym
    from the Allen Mouse Brain Atlas.

    Args:
        gene_acronym (str): The official acronym for the gene (e.g., 'Htr1a').

    Returns:
        pandas.DataFrame: A DataFrame containing expression metrics for each
        brain region, merged with structure information (name, acronym).
        Returns None if the gene or experiment is not found.
    """
    api = RmaApi()

    print(f"Searching for gene: {gene_acronym}")

    # 1. Find the Gene ID
    # We query the 'Gene' model where the acronym matches our gene
    # and the organism is mouse ('Mus musculus').
    try:
        gene_data = api.model_query(
            'Gene',
            criteria=f"[acronym$eq'{gene_acronym}'],organism[name$eq'Mus musculus']"
        )
    except Exception as e:
        print(f"Error querying for gene: {e}")
        return None

    if not gene_data:
        print(f"Error: Gene '{gene_acronym}' not found for 'Mus musculus'.")
        return None

    gene_id = gene_data[0]['id']
    print(f"Found Gene ID: {gene_id}")

    # 2. Find a relevant SectionDataSet (ISH experiment)
    # We look for a coronal ISH experiment (product_id=1) for our gene.
    # We also filter for 'failed$eqfalse' to get valid experiments.
    try:
        # Product ID 1 is the Mouse Brain Atlas ISH dataset
        experiment_data = api.model_query(
            'SectionDataSet',
            criteria=f"[failed$eqfalse],products[id$eq1],genes[id$eq{gene_id}]"
        )
    except Exception as e:
        print(f"Error querying for experiment: {e}")
        return None

    if not experiment_data:
        print(f"Error: No valid ISH experiment found for gene ID {gene_id}.")
        return None

    # We'll just use the first experiment found
    # You could add logic here to select a specific experiment (e.g., sagittal)
    experiment_id = experiment_data[0]['id']
    print(f"Found Experiment (SectionDataSet ID): {experiment_id}")

    # 3. Get StructureTree to map structure IDs to names/acronyms
    # We get all structures from the Allen Mouse Brain Reference Atlas (graph_id 1)
    try:
        # Query for all structures in the mouse brain graph (graph_id=1)
        # We MUST specify num_rows='all' to ensure we get all structures
        # and not just the first paginated results.
        structures = api.model_query(
            'Structure',
            criteria="[graph_id$eq1]", # ID 1 is the Mouse Brain Atlas
            num_rows='all'
        )

        if not structures:
            print("Error: Could not fetch structures for graph_id 1.")
            return None

        # The API returns 'structure_id_path' as a string (e.g., '/997/1009/').
        # StructureTree's constructor expects a list of integers.
        # 'clean_structures' is the helper that performs this conversion.
        processed_structures = StructureTree.clean_structures(structures)

        # Use StructureTree to easily map IDs
        structure_tree = StructureTree(processed_structures)

        # get_id_acronym_map() returns {acronym: id}
        structure_map = structure_tree.get_id_acronym_map()
        # get_name_map() returns {id: name}
        name_map = structure_tree.get_name_map()

        # Combine into a DataFrame for easy merging
        # --- FIX: Correctly assign keys() to acronym and values() to structure_id ---
        structure_df = pd.DataFrame(
            {'acronym': list(structure_map.keys()),
             'structure_id': list(structure_map.values())}
        )

        # Map names using the structure_id
        structure_df['name'] = structure_df['structure_id'].map(name_map)

        # --- FIX: Ensure structure_id is int64 ---
        structure_df['structure_id'] = structure_df['structure_id'].astype('int64')

    except Exception as e:
        print(f"Error fetching structure tree: {e}")
        return None

    # 4. Get Expression Data per Region (StructureUnionize)
    # This query fetches the summarized expression data for our chosen
    # experiment, aggregated by brain structure.
    print(f"Fetching expression data for experiment {experiment_id}...")
    try:
        unionize_data = api.model_query(
            'StructureUnionize',
            criteria=f"[section_data_set_id$eq{experiment_id}]",
            num_rows='all' # Also apply here for safety
        )
    except Exception as e:
        print(f"Error querying for StructureUnionize data: {e}")
        return None

    if not unionize_data:
        print(f"Error: No unionize data found for experiment {experiment_id}.")
        return None

    print(f"Successfully fetched {len(unionize_data)} structure entries.")

    # 5. Format the data into a pandas DataFrame
    expression_df = pd.DataFrame(unionize_data)

    # Select and rename key columns for clarity
    expression_df = expression_df[[
        'structure_id',
        'expression_energy',
        'expression_density',
        'sum_expressing_pixel_intensity',
        'sum_expressing_pixels',
        'sum_pixels'
    ]]

    # --- FIX: Ensure structure_id is int64 ---
    expression_df['structure_id'] = expression_df['structure_id'].astype('int64')

    # 6. Merge expression data with structure information
    final_df = pd.merge(
        structure_df,
        expression_df,
        on='structure_id',
        how='inner' # Only keep structures present in the expression data
    )

    return final_df

# --- Main execution ---
if __name__ == "__main__":

    # Loop over genes
    expression_data = pd.DataFrame()
    for gene_acronym in HTR_GENES:

        # Fetch data for this gene
        this_expression_data = get_expression_by_region(gene_acronym=gene_acronym)
        if this_expression_data is None:
            continue
        this_expression_data['receptor'] = '5-HT' + gene_acronym[3:]
        expression_data = pd.concat((expression_data, this_expression_data))

    # Save to disk
    expression_data.to_csv('receptor_expression.csv', index=False)


