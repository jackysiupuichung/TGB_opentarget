import pandas as pd
import sys

def main():
    base_path = "tgb/datasets/thgl_opentargets"
    
    edgelist_path = f"{base_path}/thgl-opentargets_edgelist.csv"
    ds_map_path = f"{base_path}/thgl-opentargets_datasourcemapping.csv"
    rel_map_path = f"{base_path}/thgl-opentargets_edgemapping.csv"

    print(f"Reading {edgelist_path}...")
    try:
        df_edges = pd.read_csv(edgelist_path)
    except FileNotFoundError:
        print(f"Error: Could not find {edgelist_path}")
        sys.exit(1)

    print(f"Reading {ds_map_path}...")
    try:
        df_ds_map = pd.read_csv(ds_map_path)
    except FileNotFoundError:
        print(f"Error: Could not find {ds_map_path}")
        sys.exit(1)

    print(f"Reading {rel_map_path}...")
    try:
        df_rel_map = pd.read_csv(rel_map_path)
    except FileNotFoundError:
        print(f"Error: Could not find {rel_map_path}")
        sys.exit(1)

    # Get unique combinations of relation and datasource IDs and their counts
    unique_pairs = df_edges.groupby(['relation', 'datasource']).size().reset_index(name='count')
    
    # Rename columns for merging
    # Edgelist has 'relation' (id), 'datasource' (id)
    # Maps have 'relation_id', 'relation_name' and 'datasource_id', 'datasource_name'
    
    # Merge with Relation Map
    merged = unique_pairs.merge(
        df_rel_map, 
        left_on='relation', 
        right_on='relation_id', 
        how='left'
    )
    
    # Merge with Datasource Map
    merged = merged.merge(
        df_ds_map, 
        left_on='datasource', 
        right_on='datasource_id', 
        how='left'
    )
    
    # Select and order columns
    result = merged[['relation', 'relation_name', 'datasource', 'datasource_name', 'count']].copy()
    
    # Sort by relation and datasource for deterministic ordering
    result = result.sort_values(['relation', 'datasource']).reset_index(drop=True)

    # Add a new unique ID for the combination
    result.insert(0, 'rel_ds_id', result.index)

    print("\nUnique Combinations of Relation and Datasource:")
    print("-" * 110)
    print(f"{'ID':<4} {'RelID':<6} {'Relation Name':<30} {'DsID':<6} {'Datasource Name':<20} {'Count':<10}")
    print("-" * 110)
    
    for _, row in result.iterrows():
        print(f"{row['rel_ds_id']:<4} {row['relation']:<6} {row['relation_name']:<30} {row['datasource']:<6} {row['datasource_name']:<20} {row['count']:<10}")

    # Save to mapping file
    out_path = f"{base_path}/thgl-opentargets_reldsmapping.csv"
    print(f"\nSaving mapping to {out_path}...")
    result.to_csv(out_path, index=False)
    print("Done.")

if __name__ == "__main__":
    main()
