#!/usr/bin/env python3
"""
Generate negative edges for the validation and test sets
restricted to therapeutic (OpenTargets) edges only.
"""

import time
import os
from tgb.linkproppred.thg_negative_generator import OpenTargetsNegativeEdgeGenerator
from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset

def main():
    print("*** Negative Sample Generation for OpenTargets THGL ***")

    # ===========================================================
    # CONFIG
    # ===========================================================
    name = "thgl-opentargets"
    root = "datasets"
    num_neg_e_per_pos = 10
    neg_sample_strategy = "node-type-filtered"
    rnd_seed = 42

    # ===========================================================
    # LOAD DATASET
    # ===========================================================
    dataset = PyGLinkPropPredDataset(name=name, root=root)
    data = dataset.get_TemporalData()

    train_mask = dataset.train_mask
    val_mask = dataset.val_mask
    test_mask = dataset.test_mask

    data_splits = {
        "train": data[train_mask],
        "val": data[val_mask],
        "test": data[test_mask],
    }

    # ===========================================================
    # INITIALIZE NEGATIVE EDGE GENERATOR
    # ===========================================================

    neg_sampler = OpenTargetsNegativeEdgeGenerator(
        dataset_name=name,
        node_type=dataset.node_type,
        edge_data=data,
        num_neg_e=num_neg_e_per_pos,
        rnd_seed=rnd_seed,
    )

    # ===========================================================
    # GENERATE NEGATIVE SAMPLES (VAL + TEST)
    # ===========================================================
    for split_mode in ["val", "test"]:
        start_time = time.time()
        print(
            f"INFO: Start generating negative samples: {split_mode} --- {neg_sample_strategy}"
        )
        neg_sampler.generate_negative_samples(
            pos_edges=data_splits[split_mode],
            split_mode=split_mode,
            partial_path=os.path.dirname(os.path.abspath(__file__))
        )
        print(
            f"INFO: End of negative samples generation for {split_mode}. "
            f"Elapsed Time (s): {time.time() - start_time: .4f}"
        )


if __name__ == "__main__":
    main()
