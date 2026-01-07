#!/usr/bin/env python3
"""
Construct a Temporal Heterogeneous Graph (THGL) from Open Targets evidence
(dynamic edges only).

Time is represented as discrete yearly snapshots.
Static edges are ignored in this version.
"""

import argparse
import csv
import os
import pandas as pd
from collections import defaultdict


# ===========================================================
# IO
# ===========================================================
def load_opentargets_parquet(fname):
    df = pd.read_parquet(fname)
    print(f"Loaded {len(df):,} rows from {fname}")
    return df


def extract_dynamic_edges(df):
    dynamic = df[df["year"].notna()].reset_index(drop=True)
    print(f"Dynamic edges: {len(dynamic):,}")
    return dynamic

def extract_static_edges(df):
    static = df[df["year"].isna()].reset_index(drop=True)
    print(f"Static edges: {len(static):,}")
    return static


# ===========================================================
# GRAPH CONSTRUCTION (GitHub-style temporal snapshots)
# ===========================================================
def load_edgelist(dynamic_edges):
    """
    Returns:
        node_dict           {node_name: node_id}
        node_type_dict      {node_id: node_type_id}
        edge_dict           {year: {(h, t, r): 1}}
        rel_type_dict       {relation_name: rel_id}
        node_type_mapping   {node_type_name: node_type_id}
    """
    node_dict = {}
    node_type_dict = {}
    node_type_mapping = {}
    rel_type_dict = {}
    edge_dict = defaultdict(dict)

    num_edges = 0

    for _, row in dynamic_edges.iterrows():
        ts = int(row["year"])
        head = row["sourceId"]
        tail = row["targetId"]
        # combination of data source and relation type
        rel = row["relation_key"]

        head_type = row["source_type"]
        tail_type = row["target_type"]

        # node types
        for t in (head_type, tail_type):
            if t not in node_type_mapping:
                node_type_mapping[t] = len(node_type_mapping)

        # nodes
        if head not in node_dict:
            node_dict[head] = len(node_dict)
            node_type_dict[node_dict[head]] = node_type_mapping[head_type]

        if tail not in node_dict:
            node_dict[tail] = len(node_dict)
            node_type_dict[node_dict[tail]] = node_type_mapping[tail_type]

        # relations
        if rel not in rel_type_dict:
            rel_type_dict[rel] = len(rel_type_dict)

        edge = (
            node_dict[head],
            node_dict[tail],
            rel_type_dict[rel]
        )

        edge_dict[ts][edge] = 1
        num_edges += 1

    print(f"There are {len(node_dict):,} nodes")
    print(f"There are {num_edges:,} temporal edges")
    print(f"There are {len(edge_dict):,} timesteps")

    return node_dict, node_type_dict, edge_dict, rel_type_dict, node_type_mapping

def load_static_edgelist(static_edges, node_dict, node_type_dict, rel_type_dict, node_type_mapping):
    """
    Update the existing dictionaries with static edges.
    Returns:
        node_dict           {node_name: node_id}
        node_type_dict      {node_id: node_type_id}
        edge_dict           {year: {(h, t, r): 1}}
        rel_type_dict       {relation_name: rel_id}
        node_type_mapping   {node_type_name: node_type_id}
    """
    static_edge_dict = defaultdict(dict)
    num_static_edges = 0

    for _, row in static_edges.iterrows():
        head = row["sourceId"]
        tail = row["targetId"]
        rel = row["relation_key"]

        head_type = row["source_type"]
        tail_type = row["target_type"]

        # node types
        for t in (head_type, tail_type):
            if t not in node_type_mapping:
                node_type_mapping[t] = len(node_type_mapping)

        # nodes
        if head not in node_dict:
            node_dict[head] = len(node_dict)
            node_type_dict[node_dict[head]] = node_type_mapping[head_type]

        if tail not in node_dict:
            node_dict[tail] = len(node_dict)
            node_type_dict[node_dict[tail]] = node_type_mapping[tail_type]

        # relations
        if rel not in rel_type_dict:
            rel_type_dict[rel] = len(rel_type_dict)

        edge = (
            node_dict[head],
            node_dict[tail],
            rel_type_dict[rel]
        )

        # Static edges can be assigned to a special timestamp, e.g., 0
        static_edge_dict[0][edge] = 1
        num_static_edges += 1

    print(f"After adding static edges:")
    print(f"There are {len(node_dict):,} nodes")
    print(f"There are {num_static_edges:,} static edges added")
    print(f"There are {len(static_edge_dict):,} timesteps")

    return node_dict, node_type_dict, static_edge_dict, rel_type_dict, node_type_mapping

# ===========================================================
# WRITERS
# ===========================================================
def write_edgelist(edge_dict, outname):
    num_lines = 0
    with open(outname, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "head", "tail", "relation_type"])

        for ts in sorted(edge_dict):
            for h, t, r in edge_dict[ts]:
                writer.writerow([ts, h, t, r])
                num_lines += 1

    print(f"Wrote {num_lines:,} edges → {outname}")

def write_static_edgelist(static_edge_dict, outname):
    num_lines = 0
    with open(outname, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "head", "tail", "relation_type"])

        for ts in sorted(static_edge_dict):
            for h, t, r in static_edge_dict[ts]:
                writer.writerow([ts, h, t, r])
                num_lines += 1

    print(f"Wrote {num_lines:,} static edges → {outname}")

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


def write_node_type_mapping(node_type_mapping, outname):
    with open(outname, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["node_type_name", "node_type_id"])
        for k, v in node_type_mapping.items():
            writer.writerow([k, v])


# ===========================================================
# MAIN
# ===========================================================
def main(data_dir):
    parquet_file = os.path.join(
        data_dir,
        "progression_graph",
        "datasource_harmonic.parquet"
    )

    out_prefix = os.path.join(data_dir, "thgl", "thgl-opentargets")

    df = load_opentargets_parquet(parquet_file)
    dynamic_edges = extract_dynamic_edges(df)
    static_edges = extract_static_edges(df)

    node_dict, node_type_dict, edge_dict, rel_type_dict, node_type_mapping = load_edgelist(dynamic_edges)
    node_dict, node_type_dict, static_edge_dict, rel_type_dict, node_type_mapping = load_static_edgelist(
        static_edges, node_dict, node_type_dict, rel_type_dict, node_type_mapping
    )

    # placeholder for finding node features from node_dict

    write_edgelist(edge_dict, f"{out_prefix}_edgelist.csv")
    write_static_edgelist(static_edge_dict, f"{out_prefix}_static_edgelist.csv")
    write_node_types(node_type_dict, f"{out_prefix}_nodetype.csv")
    write_relation_mapping(rel_type_dict, f"{out_prefix}_edgemapping.csv")
    write_node_type_mapping(node_type_mapping, f"{out_prefix}_nodemapping.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Construct a temporal heterogeneous graph from Open Targets."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Base directory containing data"
    )
    args = parser.parse_args()

    main(args.data_dir)
