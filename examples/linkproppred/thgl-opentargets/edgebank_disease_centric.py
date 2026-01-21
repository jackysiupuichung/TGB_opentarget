"""
EdgeBank for Disease-Centric Novel Target Prioritization
Adapted from TGB EdgeBank implementation for recommendation-style evaluation.
"""

import timeit
import numpy as np
import math
from tqdm import tqdm
import os
import os.path as osp
import sys
import argparse

# Internal imports
tgb_modules_path = osp.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(tgb_modules_path)

from tgb.linkproppred.dataset import LinkPropPredDataset
from tgb.linkproppred.evaluate_prioritisation import DiseaseCentricEvaluator
from tgb.linkproppred.evaluate import Evaluator
from modules.edgebank_predictor import EdgeBankPredictor
from tgb.utils.utils import set_random_seed, save_results
from collections import defaultdict


def test(data, test_mask, neg_sampler, split_mode):
    r"""
    Evaluated the dynamic link prediction
    Evaluation happens as 'one vs. many', meaning that each positive edge is evaluated against many negative edges

    Parameters:
        data: a dataset object
        test_mask: required masks to load the test set edges
        neg_sampler: an object that gives the negative edges corresponding to each positive edge
        split_mode: specifies whether it is the 'validation' or 'test' set to correctly load the negatives
    Returns:
        perf_metric: the result of the performance evaluation
    """
    num_batches = math.ceil(len(data['sources'][test_mask]) / BATCH_SIZE)
    perf_list = []
    hits_list = []
    for batch_idx in tqdm(range(num_batches)):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, len(data['sources'][test_mask]))
        pos_src, pos_dst, pos_t, pos_edge = (
            data['sources'][test_mask][start_idx: end_idx],
            data['destinations'][test_mask][start_idx: end_idx],
            data['timestamps'][test_mask][start_idx: end_idx],
            data['edge_type'][test_mask][start_idx: end_idx],
        )
        neg_batch_list = neg_sampler.query_batch(pos_src, pos_dst, pos_t, pos_edge, split_mode=split_mode)
        
        for idx, neg_batch in enumerate(neg_batch_list):
            query_src = np.array([int(pos_src[idx]) for _ in range(len(neg_batch) + 1)])
            query_dst = np.concatenate([np.array([int(pos_dst[idx])]), neg_batch])

            y_pred = edgebank.predict_link(query_src, query_dst)
            # compute MRR
            input_dict = {
                "y_pred_pos": np.array([y_pred[0]]),
                "y_pred_neg": np.array(y_pred[1:]),
                "eval_metric": [metric],
            }
            results = evaluator.eval(input_dict)
            perf_list.append(results[metric])
            hits_list.append(results['hits@10'])
            
        # update edgebank memory after each positive batch
        edgebank.update_memory(pos_src, pos_dst, pos_t)

    perf_metrics = float(np.mean(perf_list))
    perf_hits = float(np.mean(hits_list))

    return perf_metrics, perf_hits


def test_rec(evaluator, edgebank, test_diseases):
    """
    Evaluate EdgeBank on disease-centric novel target recommendation.
    Only evaluates on the test set diseases.
    
    Parameters:
        evaluator: DiseaseCentricEvaluator instance with test set configuration
        edgebank: EdgeBankPredictor instance initialized with training data
        test_diseases: List of disease IDs to evaluate
    
    Returns:
        dict: Average metrics across all test diseases
    """
    if len(test_diseases) == 0:
        print("No test diseases found. Exiting.")
        return None
    
    print(f"Evaluating {len(test_diseases)} test diseases...")
    
    # Accumulators for metrics
    metric_sums = defaultdict(float)
    disease_count = 0
    
    # Get max node ID for score array sizing
    max_node_id = max(
        evaluator.full_data['sources'].max(),
        evaluator.full_data['destinations'].max()
    )
    num_nodes = max_node_id + 1
    
    # Iterate through each test disease
    for disease_id in tqdm(test_diseases, desc="Evaluating test diseases"):
        # Create query for all possible targets
        query_src = np.full(num_nodes, disease_id, dtype=np.int64)
        query_dst = np.arange(num_nodes, dtype=np.int64)
        
        # Get predictions from EdgeBank
        # Returns 1.0 for edges in memory (historical), 0.0 for novel
        scores = edgebank.predict_link(query_src, query_dst).astype(np.float32)
        
        # Evaluate this disease using the evaluator
        metrics = evaluator.eval_disease(disease_id, scores)
        
        if metrics is None:
            continue
        
        # Accumulate metrics
        for k, v in metrics.items():
            metric_sums[k] += v
        disease_count += 1
    
    if disease_count == 0:
        print("No valid diseases evaluated.")
        return None
    
    # Compute averages
    results = {}
    for k, v in metric_sums.items():
        results[k] = v / disease_count
    
    return results


def get_args():
    parser = argparse.ArgumentParser('*** TGB EdgeBank: Disease-Centric Recommendation ***')
    parser.add_argument('-d', '--data', type=str, help='Dataset name', default='thgl-opentargets')
    parser.add_argument('--k_value', type=int, help='k_value for computing ranking metrics', default=100)
    parser.add_argument('--seed', type=int, help='Random seed', default=1)
    parser.add_argument('--mem_mode', type=str, help='Memory mode', default='unlimited', 
                       choices=['unlimited', 'fixed_time_window'])
    parser.add_argument('--time_window_ratio', type=float, help='Time window ratio', default=0.15)
    parser.add_argument('--root', type=str, help='Dataset root directory', default='datasets')

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    return args, sys.argv


# ==================
# Main Execution
# ==================

if __name__ == "__main__":
    start_overall = timeit.default_timer()
    
    # Get arguments
    args, _ = get_args()
    
    SEED = args.seed
    set_random_seed(SEED)
    MEMORY_MODE = args.mem_mode
    K_VALUE = args.k_value
    K_VALUE_REC = 200
    TIME_WINDOW_RATIO = args.time_window_ratio
    DATA = args.data
    ROOT = args.root
    BATCH_SIZE = 200
    MODEL_NAME = 'EdgeBank'
    
    print("="*60)
    print(f"EdgeBank Evaluation: MRR + Disease-Centric Recommendation")
    print(f"Dataset: {DATA}")
    print(f"Memory Mode: {MEMORY_MODE}")
    print(f"K Value (for MRR): {K_VALUE}")
    print(f"K Value (for recommendation): {K_VALUE_REC}")
    print("="*60)
    
    # Load dataset
    dataset = LinkPropPredDataset(name=DATA, root=ROOT, preprocess=True)
    data = dataset.full_data
    metric = dataset.eval_metric
    
    # Get masks
    train_mask = dataset.train_mask
    val_mask = dataset.val_mask
    test_mask = dataset.test_mask
    
    # Prepare EdgeBank memory with training data only
    print("\nInitializing EdgeBank with training data...")
    hist_src = data['sources'][train_mask]
    hist_dst = data['destinations'][train_mask]
    hist_ts = data['timestamps'][train_mask]
    
    edgebank = EdgeBankPredictor(
        src=hist_src,
        dst=hist_dst,
        ts=hist_ts,
        memory_mode=MEMORY_MODE,
        time_window_ratio=TIME_WINDOW_RATIO,
        pos_prob=1.0
    )
    
    print(f"EdgeBank initialized with {len(edgebank.memory)} edges in memory.")
    
    # ==================== PART 1: Standard MRR Evaluation ====================
    print("\n" + "="*60)
    print("PART 1: Standard Link Prediction (MRR-based)")
    print("="*60)
    
    evaluator = Evaluator(name=DATA)
    neg_sampler = dataset.negative_sampler
    
    # Validation
    print("\n--- Validation Set ---")
    dataset.load_val_ns()
    start_val = timeit.default_timer()
    perf_metric_val, perf_hits_val = test(data, val_mask, neg_sampler, 'val')
    val_time = timeit.default_timer() - start_val
    
    print(f"\nValidation Results:")
    print(f"  {metric}: {perf_metric_val:.4f}")
    print(f"  Hits@10: {perf_hits_val:.4f}")
    print(f"  Time (s): {val_time:.4f}")
    
    # Test
    print("\n--- Test Set ---")
    dataset.load_test_ns()
    start_test_mrr = timeit.default_timer()
    perf_metric_test, perf_hits_test = test(data, test_mask, neg_sampler, 'test')
    test_mrr_time = timeit.default_timer() - start_test_mrr
    
    print(f"\nTest Results:")
    print(f"  {metric}: {perf_metric_test:.4f}")
    print(f"  Hits@10: {perf_hits_test:.4f}")
    print(f"  Time (s): {test_mrr_time:.4f}")
    
    # ==================== PART 2: Disease-Centric Recommendation ====================
    print("\n" + "="*60)
    print("PART 2: Disease-Centric Novel Target Prioritization")
    print("="*60)
    
    # Re-initialize EdgeBank with training data only (reset after MRR evaluation)
    print("\nRe-initializing EdgeBank with training data...")
    edgebank_rec = EdgeBankPredictor(
        src=hist_src,
        dst=hist_dst,
        ts=hist_ts,
        memory_mode=MEMORY_MODE,
        time_window_ratio=TIME_WINDOW_RATIO,
        pos_prob=1.0
    )
    
    # Initialize DiseaseCentricEvaluator for test set
    print("Initializing Disease-Centric Evaluator...")
    dc_evaluator = DiseaseCentricEvaluator(
        dataset_name=DATA,
        root=ROOT,
        eval_split='test',
        k_value=K_VALUE_REC
    )
    
    test_diseases = dc_evaluator.get_test_diseases()
    
    print("\nNOTE: EdgeBank only remembers historical edges from training.")
    print("Novel targets will receive score 0.0 and rank poorly.")
    print("This is expected behavior for this baseline.\n")
    
    # Run disease-centric evaluation
    start_test_rec = timeit.default_timer()
    rec_results = test_rec(dc_evaluator, edgebank_rec, test_diseases)
    test_rec_time = timeit.default_timer() - start_test_rec
    
    # Print recommendation results
    if rec_results:
        print("\n" + "="*60)
        print("DISEASE-CENTRIC RECOMMENDATION RESULTS")
        print(f"Evaluated on {len(test_diseases)} test diseases")
        print("-" * 60)
        for metric_name, value in sorted(rec_results.items()):
            print(f"{metric_name}: {value:.6f}")
        print(f"\nTime (s): {test_rec_time:.4f}")
        print("="*60)
    
    # ==================== Save All Results ====================
    results_path = f'{osp.dirname(osp.abspath(__file__))}/saved_results'
    os.makedirs(results_path, exist_ok=True)
    results_filename = f'{results_path}/{MODEL_NAME}_{MEMORY_MODE}_{DATA}_combined_results.json'
    
    combined_results = {
        'model': MODEL_NAME,
        'memory_mode': MEMORY_MODE,
        'data': DATA,
        'seed': SEED,
        'k_value': K_VALUE,
        # Standard MRR results
        'val_mrr': perf_metric_val,
        'val_hits10': perf_hits_val,
        'val_time': val_time,
        'test_mrr': perf_metric_test,
        'test_hits10': perf_hits_test,
        'test_mrr_time': test_mrr_time,
        # Disease-centric recommendation results
        'test_rec_time': test_rec_time,
    }
    
    if rec_results:
        for k, v in rec_results.items():
            combined_results[f'test_rec_{k}'] = v
    
    save_results(combined_results, results_filename)
    
    print(f"\n\nAll results saved to: {results_filename}")
    
    total_time = timeit.default_timer() - start_overall
    print(f"Total Elapsed Time (s): {total_time:.4f}")

