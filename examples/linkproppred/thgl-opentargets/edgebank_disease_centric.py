"""
EdgeBank for Disease-Centric Novel Target Prioritization
Adapted from TGB EdgeBank implementation for recommendation-style evaluation.
"""

import timeit
import numpy as np
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
from modules.edgebank_predictor import EdgeBankPredictor
from tgb.utils.utils import set_random_seed, save_results
from collections import defaultdict


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
    TIME_WINDOW_RATIO = args.time_window_ratio
    DATA = args.data
    ROOT = args.root
    MODEL_NAME = 'EdgeBank'
    
    print("="*60)
    print(f"EdgeBank Disease-Centric Novel Target Prioritization")
    print(f"Dataset: {DATA}")
    print(f"Memory Mode: {MEMORY_MODE}")
    print(f"K Value: {K_VALUE}")
    print("="*60)
    
    # Load dataset
    dataset = LinkPropPredDataset(name=DATA, root=ROOT, preprocess=True)
    data = dataset.full_data
    
    # Get masks
    train_mask = dataset.train_mask
    val_mask = dataset.val_mask
    test_mask = dataset.test_mask
    
    # Initialize DiseaseCentricEvaluator for test set
    print("\nInitializing Disease-Centric Evaluator...")
    evaluator = DiseaseCentricEvaluator(
        dataset_name=DATA,
        root=ROOT,
        eval_split='test',  # Evaluate on test set
        k_value=K_VALUE
    )
    
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
    
    # Get test diseases
    test_diseases = evaluator.get_test_diseases()
    
    print("\n" + "="*60)
    print("Running Test Evaluation...")
    print("="*60)
    print("NOTE: EdgeBank only remembers historical edges from training.")
    print("Novel targets will receive score 0.0 and rank poorly.")
    print("This is expected behavior for this baseline.")
    print("="*60 + "\n")
    
    # Run test evaluation
    start_test = timeit.default_timer()
    results = test_rec(evaluator, edgebank, test_diseases)
    test_time = timeit.default_timer() - start_test
    
    # Print results
    if results:
        print("\n" + "="*60)
        print("DISEASE-CENTRIC NOVEL TARGET PRIORITIZATION RESULTS")
        print(f"Dataset: {DATA}")
        print(f"Model: {MODEL_NAME} ({MEMORY_MODE})")
        print(f"Evaluated on {len(test_diseases)} test diseases")
        print("-" * 60)
        for metric, value in sorted(results.items()):
            print(f"{metric}: {value:.6f}")
        print(f"\nTest Time (s): {test_time:.4f}")
        print("="*60)
        
        # Save results
        results_path = f'{osp.dirname(osp.abspath(__file__))}/saved_results'
        os.makedirs(results_path, exist_ok=True)
        results_filename = f'{results_path}/{MODEL_NAME}_{MEMORY_MODE}_{DATA}_disease_centric_results.json'
        
        save_results({
            'model': MODEL_NAME,
            'memory_mode': MEMORY_MODE,
            'data': DATA,
            'eval_type': 'disease_centric_recommendation',
            'k_value': K_VALUE,
            'seed': SEED,
            'test_time': test_time,
            **results
        }, results_filename)
        
        print(f"\nResults saved to: {results_filename}")
    
    total_time = timeit.default_timer() - start_overall
    print(f"\nTotal Elapsed Time (s): {total_time:.4f}")
