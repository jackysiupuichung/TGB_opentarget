#!/usr/bin/env python3
"""
Construct a Temporal Heterogeneous Graph (THGL) from Open Targets evidence.

Time is represented as discrete yearly snapshots.

This script supports dynamic edges plus optional static edges (static edges are
duplicated across all observed dynamic years).

Output files (prefix: thgl-opentargets):
- *_edgelist.csv: timestamp, head, tail, relation, datasource, score
- *_nodetype.csv: node_id, type
- *_edgemapping.csv: relation_name, relation_id
- *_datasourcemapping.csv: datasource_name, datasource_id
- *_nodemapping.csv: node_type_name, node_type_id
"""

import argparse
import csv
import os
from collections import defaultdict

import pandas as pd


# ===========================================================
# GRAPH CONSTRUCTION
# ===========================================================

def build_combined_edgelist(dynamic_edges: pd.DataFrame, static_edges: pd.DataFrame):
    """Build temporal edge dict and id mappings.

    Returns:
        node_dict           {node_name: node_id}
        node_type_dict      {node_id: node_type_id}
        edge_dict           {year: {(u, v, r, ds, s): 1}}
        rel_type_dict       {relation_name: rel_id}
        ds_type_dict        {datasource_name: ds_id}
        node_type_mapping   {node_type_name: node_type_id}
    """

    node_dict = {}
    node_type_dict = {}
    node_type_mapping = {}
    rel_type_dict = {}
    ds_type_dict = {}
    edge_dict = defaultdict(dict)

    def get_node_id(name, ntype):
        if ntype not in node_type_mapping:
            node_type_mapping[ntype] = len(node_type_mapping)
        if name not in node_dict:
            node_dict[name] = len(node_dict)
            node_type_dict[node_dict[name]] = node_type_mapping[ntype]
        return node_dict[name]

    def get_rel_id(rel):
        if rel not in rel_type_dict:
            rel_type_dict[rel] = len(rel_type_dict)
        return rel_type_dict[rel]

    def get_ds_id(ds):
        if ds not in ds_type_dict:
            ds_type_dict[ds] = len(ds_type_dict)
        return ds_type_dict[ds]

    # 1) Dynamic edges
    print("Processing dynamic edges...")
    for _, row in dynamic_edges.iterrows():
        ts = int(row["year"])
        u = get_node_id(row["sourceId"], row["source_type"])
        v = get_node_id(row["targetId"], row["target_type"])
        r = get_rel_id(row["relation"])
        ds = get_ds_id(row["datasourceId"])
        s = float(row["score"])
        edge_dict[ts][(u, v, r, ds, s)] = 1

    # 2) Static edges duplicated across dynamic years
    print("Processing static edges (duplicating across all dynamic years)...")
    unique_years = sorted(edge_dict.keys())

    for _, row in static_edges.iterrows():
        u = get_node_id(row["sourceId"], row["source_type"])
        v = get_node_id(row["targetId"], row["target_type"])
        r = get_rel_id(row["relation"])
        ds = get_ds_id(row["datasourceId"])
        s = float(row["score"])

        for yr in unique_years:
            edge_dict[yr][(u, v, r, ds, s)] = 1

    num_nodes = len(node_dict)
    num_years = len(unique_years)
    total_edges = sum(len(edges) for edges in edge_dict.values())

    print("Graph summary:")
    if unique_years:
        print(f"  Nodes: {num_nodes:,}")
        print(f"  Years: {num_years} ({unique_years[0]} to {unique_years[-1]})")
    else:
        print(f"  Nodes: {num_nodes:,}")
        print("  Years: 0")
    print(f"  Total temporal edges (after duplication): {total_edges:,}")

    return node_dict, node_type_dict, edge_dict, rel_type_dict, ds_type_dict, node_type_mapping


# ===========================================================
# WRITERS
# ===========================================================

def write_edgelist(edge_dict, outname):
    num_lines = 0
    with open(outname, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "head", "tail", "relation", "datasource", "score"])

        for ts in sorted(edge_dict):
            for h, t, r, ds, s in edge_dict[ts]:
                writer.writerow([ts, h, t, r, ds, s])
                num_lines += 1

    print(f"Wrote {num_lines:,} edges → {outname}")


def write_node_types(node_type_dict, outname):
    with open(outname, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["node_id", "type"])
        for nid, ntype in node_type_dict.items():
            writer.writerow([nid, ntype])


def write_relation_mapping(rel_type_dict, outname):
    with open(outname, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["relation_name", "relation_id"])
        for r, rid in rel_type_dict.items():
            writer.writerow([r, rid])


def write_datasource_mapping(ds_type_dict, outname):
    with open(outname, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["datasource_name", "datasource_id"])
        for ds, dsid in ds_type_dict.items():
            writer.writerow([ds, dsid])


def write_node_type_mapping(node_type_mapping, outname):
    with open(outname, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["node_type_name", "node_type_id"])
        for k, v in node_type_mapping.items():
            writer.writerow([k, v])


# ===========================================================
# MAIN
# ===========================================================

def main(dynamic_path, static_path, out_dir):
    out_prefix = os.path.join(out_dir, "thgl-opentargets")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading dynamic edges from {dynamic_path}...")
    dynamic_edges = pd.read_parquet(dynamic_path)

    print(f"Loading static edges from {static_path}...")
    static_edges = pd.read_parquet(static_path)

    (
        node_dict,
        node_type_dict,
        edge_dict,
        rel_type_dict,
        ds_type_dict,
        node_type_mapping,
    ) = build_combined_edgelist(dynamic_edges, static_edges)

    write_edgelist(edge_dict, f"{out_prefix}_edgelist.csv")
    write_node_types(node_type_dict, f"{out_prefix}_nodetype.csv")
    write_relation_mapping(rel_type_dict, f"{out_prefix}_edgemapping.csv")
    write_datasource_mapping(ds_type_dict, f"{out_prefix}_datasourcemapping.csv")
    write_node_type_mapping(node_type_mapping, f"{out_prefix}_nodemapping.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Construct a temporal heterogeneous graph from Open Targets.")
    parser.add_argument("--dynamic_path", type=str, required=True, help="Path to dynamic edges parquet")
    parser.add_argument("--static_path", type=str, required=True, help="Path to static edges parquet")
    parser.add_argument("--out_dir", type=str, default="datasets/thgl_opentargets", help="Output directory")
    args = parser.parse_args()

    main(args.dynamic_path, args.static_path, args.out_dir)
