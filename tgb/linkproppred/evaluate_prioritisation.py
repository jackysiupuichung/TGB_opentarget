
import argparse
import numpy as np
import pandas as pd
import os
import os.path as osp
import torch
from tqdm import tqdm
from collections import defaultdict
import pickle
from tgb.linkproppred.dataset import LinkPropPredDataset

def load_therapeutic_ids(dataset_root, dataset_name, target_rel="clinical_trial__chembl"):
    """
    Identifies therapeutic edge type IDs by checking the edge mapping file.
    """
    mapping_file = osp.join(dataset_root, f"{dataset_name}_edgemapping.csv")
    if not osp.exists(mapping_file):
        raise FileNotFoundError(f"Mapping file not found: {mapping_file}")
    
    df = pd.read_csv(mapping_file)
    row = df[df['relation_name'] == target_rel]
    if row.empty:
        print(f"Warning: No relation named '{target_rel}' found in mapping.")
        return set()
        
    return set(row['relation_id'].values)


def load_target_ids(dataset_root, dataset_name, dataset_obj=None, target_type="target"):
    """
    Identifies target node IDs.
    Requires dataset_obj to access node_type array.
    """
    mapping_file = osp.join(dataset_root, f"{dataset_name}_nodemapping.csv")
    if not osp.exists(mapping_file):
        print(f"Warning: Node mapping file not found: {mapping_file}")
        return None
        
    df = pd.read_csv(mapping_file)
    row = df[df['node_type_name'].str.lower() == target_type.lower()]
    
    if row.empty:
        print(f"Warning: Node type '{target_type}' not found in mapping.")
        return None
        
    target_type_id = row['node_type_id'].values[0]
    
    if dataset_obj is None or dataset_obj.node_type is None:
        print("Warning: Dataset object or node_type array not provided.")
        return None
        
    return np.where(dataset_obj.node_type == target_type_id)[0]


class DiseaseCentricEvaluator:
    """
    Disease-centric novel target prioritization evaluator.
    Follows recommendation system logic where:
    - Users = Diseases
    - Items = Targets
    - Evaluation filters historical interactions and ranks novel targets
    """
    def __init__(self, dataset_name, root="datasets", eval_split='test', k_value=100):
        self.dataset_name = dataset_name
        self.root = root
        self.eval_split = eval_split
        self.k_value = k_value
        
        print(f"Loading dataset {dataset_name} from {self.root}...")
        self.dataset = LinkPropPredDataset(name=dataset_name, root=self.root, preprocess=True)
        self.full_data = self.dataset.full_data
        
        # Identify therapeutic edge type IDs
        self.therapeutic_ids = load_therapeutic_ids(self.dataset.root, self.dataset_name)
        print(f"Therapeutic Edge Type IDs: {self.therapeutic_ids}")

        # Identify target node IDs (items in recommendation terminology)
        self.target_ids = load_target_ids(self.dataset.root, self.dataset_name, self.dataset)
        if self.target_ids is not None:
             print(f"Restricting candidates to {len(self.target_ids)} target nodes.")
             # Pre-compute target mask for fast filtering
             self.is_target_node_mask = np.zeros(self.dataset.num_nodes, dtype=bool)
             self.is_target_node_mask[self.target_ids] = True
        else:
             raise ValueError("Target node IDs not found. Cannot proceed.")
        
        # Data structures
        self.history_therapeutic = defaultdict(set)  # Disease -> set of historical targets
        self.disease_positives = defaultdict(set)    # Disease -> set of novel test targets
        self.test_diseases = []
        
        # Build indices immediately
        self._build_indices(eval_split=self.eval_split)
        
    def _build_indices(self, eval_split='test'):
        """
        Builds history and test positive sets based on temporal cutoff.
        Similar to SampleGenerator logic for train/test splitting.
        """
        train_mask = self.dataset.train_mask
        val_mask = self.dataset.val_mask
        test_mask = self.dataset.test_mask
        
        if eval_split == 'test':
            history_mask = np.logical_or(train_mask, val_mask)
            eval_mask = test_mask
            print(f"Evaluation Split: TEST. History includes Train + Val edges.")
        elif eval_split == 'val':
            history_mask = train_mask
            eval_mask = val_mask
            print(f"Evaluation Split: VAL. History includes Train edges.")
        else:
            raise ValueError(f"Invalid split: {eval_split}")
            
        # Build historical therapeutic interactions (known positives)
        print("Building historical therapeutic index...")
        hist_src = self.full_data['sources'][history_mask]
        hist_dst = self.full_data['destinations'][history_mask]
        hist_et = self.full_data['edge_type'][history_mask]
        
        self.history_therapeutic = defaultdict(set)
        
        if len(self.therapeutic_ids) > 0:
            is_ther_hist = np.isin(hist_et, list(self.therapeutic_ids))
            for s, d in zip(hist_src[is_ther_hist], hist_dst[is_ther_hist]):
                self.history_therapeutic[s].add(d)
        
        # Identify novel positives in evaluation set
        print("Identifying novel positives...")
        eval_src = self.full_data['sources'][eval_mask]
        eval_dst = self.full_data['destinations'][eval_mask]
        eval_et = self.full_data['edge_type'][eval_mask]
        
        eval_ther_mask = np.isin(eval_et, list(self.therapeutic_ids))
        
        self.disease_positives = defaultdict(set)
        
        num_novel = 0
        for s, d in zip(eval_src[eval_ther_mask], eval_dst[eval_ther_mask]):
            # Novelty check: not in historical interactions
            if d not in self.history_therapeutic[s]:
                self.disease_positives[s].add(d)
                num_novel += 1
                
        self.test_diseases = sorted(list(self.disease_positives.keys()))
        print(f"Found {len(self.test_diseases)} diseases with {num_novel} novel targets in {eval_split} set.")

    def get_test_diseases(self):
        """Returns list of disease IDs to evaluate."""
        return self.test_diseases

    def eval_disease(self, disease_id, all_node_scores):
        """
        Evaluates one disease given scores for ALL nodes.
        Follows the same logic as SampleGenerator.evaluate_data_full:
        - Filters out historical interactions (negative_evaluation_samples logic)
        - Ranks novel targets against all valid candidates
        - Computes ranking metrics
        
        Args:
            disease_id: The disease (user) ID
            all_node_scores: Score array where index = node_id
        
        Returns:
            dict: Metrics (MRR, Recall@K, Precision@K, NDCG@K)
        """
        
        if disease_id not in self.disease_positives:
            return None
            
        positives = self.disease_positives[disease_id]
        if not positives:
            return None
            
        # Copy scores to avoid side effects
        scores = all_node_scores.copy()
        
        # Filter candidate pool (similar to negative_evaluation_samples filtering):
        # 1. Mask non-target nodes
        if self.is_target_node_mask is not None:
            scores[~self.is_target_node_mask] = -np.inf

        # 2. Mask historical therapeutic targets (known positives from training)
        hist = self.history_therapeutic.get(disease_id, set())
        if hist:
            valid_hist_indices = [h for h in hist if h < len(scores)]
            scores[valid_hist_indices] = -np.inf
            
        # 3. Mask self (disease node)
        if disease_id < len(scores):
            scores[disease_id] = -np.inf
        
        # Calculate ranks for each positive target
        ranks = []
        for p in positives:
            if p >= len(scores):
                continue
            
            p_score = scores[p]
            
            # Skip if positive was masked (shouldn't happen for valid data)
            if p_score == -np.inf:
                continue
            
            # Rank = 1 + number of candidates with strictly higher scores
            rank = np.sum(scores > p_score) + 1
            ranks.append(rank)
            
        if not ranks:
            return None
            
        ranks = np.array(ranks)
        
        # Compute metrics
        metrics = {}
        
        # MRR
        metrics['mrr'] = float(np.mean(1.0 / ranks))
        
        # Precision, Recall, NDCG @ K
        hits = (ranks <= self.k_value).sum()
        
        # Recall@K = hits / total_positives
        metrics[f'recall@{self.k_value}'] = float(hits / len(positives))
        
        # Precision@K = hits / k
        metrics[f'precision@{self.k_value}'] = float(hits / self.k_value)
        
        # NDCG@K
        relevant_ranks = ranks[ranks <= self.k_value]
        dcg = np.sum(1.0 / np.log2(relevant_ranks + 1))
        
        num_ideal = min(len(positives), self.k_value)
        ideal_ranks = np.arange(1, num_ideal + 1)
        idcg = np.sum(1.0 / np.log2(ideal_ranks + 1))
        
        metrics[f'ndcg@{self.k_value}'] = float(dcg / idcg) if idcg > 0 else 0.0
            
        return metrics

    def eval(self, predictor):
        """
        Performs full evaluation across all test diseases.
        Similar to the test() function in edgebank.py but for disease-centric ranking.
        
        Args:
            predictor: Object with predict_all_nodes(disease_id) method
        """
        if len(self.test_diseases) == 0:
            print("No novel therapeutic targets found. Exiting eval.")
            return None

        print("Starting ranking evaluation...")
        
        metric_sums = defaultdict(float)
        disease_count = 0
        
        for disease_id in tqdm(self.test_diseases):
            scores = predictor.predict_all_nodes(disease_id)
            metrics = self.eval_disease(disease_id, scores)
            
            if metrics is None:
                continue
                
            for k, v in metrics.items():
                metric_sums[k] += v
            disease_count += 1

        if disease_count == 0:
            print("No valid diseases evaluated.")
            return None
            
        print("="*50)
        print(f"DISEASE-CENTRIC NOVEL TARGET PRIORITIZATION RESULTS")
        print(f"Dataset: {self.dataset_name}")
        print(f"Evaluated on {disease_count} diseases")
        print("-" * 30)
        
        results = {}
        for k, v in metric_sums.items():
            avg = v / disease_count
            results[k] = avg
            print(f"{k}: {avg:.6f}")
            
        print("="*50)
        
        return results


class DotProductPredictor:
    """Predictor using pre-trained embeddings and cosine similarity."""
    def __init__(self, embeddings_path):
        print(f"Loading embeddings from {embeddings_path}...")
        if embeddings_path.endswith('.pkl'):
            with open(embeddings_path, 'rb') as f:
                self.emb_dict = pickle.load(f)
        elif embeddings_path.endswith('.pt') or embeddings_path.endswith('.pth'):
             loaded = torch.load(embeddings_path, map_location='cpu')
             if isinstance(loaded, dict) and 'emb' in loaded:
                 self.emb_dict = loaded['emb']
             else:
                 self.emb_dict = loaded
        else:
            raise ValueError("Unsupported format.")
            
        self.sorted_node_ids = np.array(sorted(list(self.emb_dict.keys())))
        self.node_id_to_idx = {nid: i for i, nid in enumerate(self.sorted_node_ids)}
        
        dim = len(self.emb_dict[self.sorted_node_ids[0]])
        self.matrix = np.zeros((len(self.sorted_node_ids), dim), dtype=np.float32)
        
        for i, nid in enumerate(self.sorted_node_ids):
            self.matrix[i] = self.emb_dict[nid]
            
        # Normalize for cosine similarity
        norms = np.linalg.norm(self.matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        self.matrix = self.matrix / norms
        self.id_map_max = self.sorted_node_ids[-1]

    def predict_all_nodes(self, src_id):
        """
        Computes cosine similarity between src_id and ALL nodes.
        Returns array where array[i] = score for node i.
        """
        out = np.full(self.id_map_max + 1, -np.inf, dtype=np.float32)
        
        if src_id not in self.node_id_to_idx:
            return out
            
        src_idx = self.node_id_to_idx[src_id]
        src_vec = self.matrix[src_idx]
        
        scores = np.dot(self.matrix, src_vec)
        out[self.sorted_node_ids] = scores
        
        return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='thgl-opentargets')
    parser.add_argument('--emb', type=str, required=True, help='Path to embeddings (.pkl or .pt)')
    parser.add_argument('--root', type=str, default="datasets")
    parser.add_argument('--k', type=int, default=100)
    args = parser.parse_args()
    
    evaluator = DiseaseCentricEvaluator(args.dataset, args.root, k_value=args.k)
    predictor = DotProductPredictor(args.emb)
    evaluator.eval(predictor)
