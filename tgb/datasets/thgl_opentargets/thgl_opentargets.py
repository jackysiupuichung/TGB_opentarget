#!/usr/bin/env python3
"""
Construct a temporal heterogeneous graph (THGL) from the Open Targets Platform edges.

Input schema (per parquet):
    source, target, source_type, target_type, relation, datasourceId, score, year, id

Output:
    - thgl-opentargets_edgelist.csv
    - thgl-opentargets_nodeIDmapping.csv
    - thgl-opentargets_nodetype.csv
    - thgl-opentargets_relation_mapping.csv
    - thgl-opentargets_skipped_edges.csv

Usage:
    python build_thgl_opentargets.py \
        --data_dir /path/to/opentarget_het_graph \
        --relation_mode relation        # (or "source_type")
"""

import pandas as pd
import glob
import csv
import os
import argparse
from collections import defaultdict
from tqdm import tqdm
from tgb.utils.info import NODE_TYPE_MAP, RELATION_TYPE_MAP, CLINICAL_STAGE_MAP, SOURCEID_TYPE_MAP


# ===========================================================
# FUNCTIONS
# ===========================================================
def get_or_add_node(node_name, node_type, node_dict, node_type_dict):
    """Return existing node_id or create a new one"""
    if node_name not in node_dict:
        node_id = len(node_dict)
        node_dict[node_name] = node_id
        curr_node_type = NODE_TYPE_MAP.get(node_type, 99)
        if curr_node_type == 99:
            print(f"⚠️  WARNING: Unknown node type '{node_type}' for node '{node_name}'")
        else:
            node_type_dict[node_id] = curr_node_type
    return node_dict[node_name]


def get_or_add_relation(rel_name, relation_dict, RELATION_TYPE_MAP):
    """Return numeric relation ID based on RELATION_TYPE_MAP or add new"""
    if rel_name in RELATION_TYPE_MAP:
        rel_id = RELATION_TYPE_MAP[rel_name]
    else:
        # Assign dynamically if not predefined
        rel_id = len(RELATION_TYPE_MAP)
        print(f"⚠️  WARNING: Unknown relation '{rel_name}' — assigning new ID {rel_id}")
        RELATION_TYPE_MAP[rel_name] = rel_id

    # Record relation in relation_dict (for reverse lookup or reference)
    relation_dict[rel_name] = rel_id
    return rel_id


def write_mapping_csv(mapping, outname, headers):
    """Generic CSV writer for mapping dicts"""
    os.makedirs(os.path.dirname(outname), exist_ok=True)
    with open(outname, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for k, v in mapping.items():
            writer.writerow([k, v])


def write_edges(out_dict, outname):
    os.makedirs(os.path.dirname(outname), exist_ok=True)
    with open(outname, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["year", "src", "dst", "relation_type", "score"])
        for year in sorted(out_dict.keys()):
            for edge, vals in out_dict[year].items():
                src, dst, rel = edge
                score = vals[0]
                writer.writerow([year, src, dst, rel, score])


def write_skipped_edges(skipped_list, outname):
    """Log edges skipped due to missing or unknown node types"""
    os.makedirs(os.path.dirname(outname), exist_ok=True)
    if skipped_list:
        df = pd.DataFrame(skipped_list)
        df.to_csv(outname, index=False)
        print(f"⚠️  Logged {len(df)} skipped edges → {outname}")
    else:
        print("✅ No skipped edges.")


# ===========================================================
# MAIN
# ===========================================================
def main(data_dir, relation_mode):
    edge_dir = os.path.join(data_dir, "data/kg_output/edges")
    script_dir = os.path.dirname(os.path.abspath(__file__))

    out_edge_csv = os.path.join(script_dir, "thgl-opentargets_edgelist.csv")
    out_nodemap_csv = os.path.join(script_dir, "thgl-opentargets_nodeIDmapping.csv")
    out_nodetype_csv = os.path.join(script_dir, "thgl-opentargets_nodetype.csv")
    out_nodetypemapping_csv = os.path.join(script_dir, "thgl-opentargets_nodetype_mapping.csv")
    out_relmap_csv = os.path.join(script_dir, "thgl-opentargets_relation_mapping.csv")
    out_skipped_csv = os.path.join(script_dir, "thgl-opentargets_skipped_edges.csv")

    print(f"🔹 Relation mode: {relation_mode}")
    print("🔹 Loading all edge parquet files...")
    edge_files = glob.glob(f"{edge_dir}/*.parquet")
    if not edge_files:
        raise FileNotFoundError(f"No parquet files found in {edge_dir}")
    edges = pd.concat([pd.read_parquet(f) for f in edge_files], ignore_index=True)
    print(f"✅ Loaded {len(edges):,} edges from {len(edge_files)} files")

    # Initialize structures
    node_dict = {}
    node_type_dict = {}
    relation_dict = {}
    out_dict = defaultdict(dict)
    skipped_edges = []

    # Select correct mapping for relations
    if relation_mode == "relation":
        relation_map = RELATION_TYPE_MAP
        relation_field = "relation"
    elif relation_mode == "datasourceId":
        relation_map = SOURCEID_TYPE_MAP
        relation_field = "datasourceId"
    else:
        raise ValueError("relation_mode must be either 'relation' or 'datasourceId'")

    print(f"🔹 Using relation field: '{relation_field}' with predefined map ({len(relation_map)} entries)")

    print("🔹 Building temporal heterogeneous graph data ...")

    for _, row in tqdm(edges.iterrows(), total=len(edges), desc="Processing edges"):
        src = row["sourceId"]
        dst = row["targetId"]
        src_type = row["source_type"]
        dst_type = row["target_type"]
        rel_label = row[relation_field]
        score = float(row["score"]) if pd.notna(row["score"]) else 0.0
        year = int(row["year"]) if not pd.isna(row["year"]) else 0

        # ✅ SAFETY CHECK
        if pd.isna(src_type) or pd.isna(dst_type) or \
           src_type not in NODE_TYPE_MAP or dst_type not in NODE_TYPE_MAP:
            skipped_edges.append({
                "year": year,
                "source": src,
                "target": dst,
                "source_type": src_type,
                "target_type": dst_type,
                "relation": rel_label,
                "score": score
            })
            continue

        # Nodes
        src_id = get_or_add_node(src, src_type, node_dict, node_type_dict)
        dst_id = get_or_add_node(dst, dst_type, node_dict, node_type_dict)

        # Relation ID (uses correct mapping)
        rel_id = get_or_add_relation(rel_label, relation_dict, relation_map)

        # Temporal edge record
        out_dict[year][(src_id, dst_id, rel_id)] = (score,)

    # Write outputs
    print(f"✅ Constructed temporal edges for {len(out_dict)} years")
    print(f"✅ Unique nodes: {len(node_dict):,}, relations: {len(relation_dict):,}")
    print(f"⚠️  Skipped {len(skipped_edges):,} edges due to missing or unknown node types")

    write_mapping_csv(node_dict, out_nodemap_csv, ["node_name", "node_id"])
    write_mapping_csv(node_type_dict, out_nodetype_csv, ["node_id", "node_type"])
    write_mapping_csv(NODE_TYPE_MAP, out_nodetypemapping_csv, ["node_type_name", "node_type"])
    write_mapping_csv(relation_dict, out_relmap_csv, [f"relation_name", f"relation"])
    write_edges(out_dict, out_edge_csv)
    write_skipped_edges(skipped_edges, out_skipped_csv)

    print("\n✅ Graph export complete:")
    print(f"   • Edges → {out_edge_csv}")
    print(f"   • Node mappings → {out_nodemap_csv}")
    print(f"   • Node types → {out_nodetype_csv}")
    print(f"   • Node type mapping → {out_nodetypemapping_csv}")
    print(f"   • Relation mapping → {out_relmap_csv}")
    print(f"   • Skipped edges log → {out_skipped_csv}")


# ===========================================================
# ENTRY POINT
# ===========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Construct a Temporal Heterogeneous Graph (THGL) from Open Targets.")
    parser.add_argument("--data_dir", type=str, required=True, help="Base directory containing data/kg_output/edges/")
    parser.add_argument(
        "--relation_mode",
        type=str,
        choices=["relation", "datasourceId"],
        default="relation",
        help="Choose how to define the relation type: 'relation' or 'datasourceId'."
    )
    args = parser.parse_args()

    main(args.data_dir, args.relation_mode)
