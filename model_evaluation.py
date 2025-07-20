import os
import json
import mrcfile
import zarr
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import precision_recall_curve, average_precision_score
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import seaborn as sns

from config import particle_names, particle_to_index, particle_radius
from yolo_predictor import run_yolo_inference
from unet_model import run_unet_inference
from blend import blend_predictions

class CryoETEvaluator:
    def __init__(self, dataset_path: str, distance_threshold: float = 50.0):
        self.dataset_path = Path(dataset_path)
        self.distance_threshold = distance_threshold
        self.protein_classes = {
            'apo-ferritin': 1,
            'beta-amylase': 2, 
            'beta-galactosidase': 3,
            'ribosome-80s': 4,
            'thyroglobulin': 5,
            'vlp': 6
        }
        self.class_names = {v: k for k, v in self.protein_classes.items()}
        
    def load_ground_truth_annotations(self, run_name: str, voxel_spacing: str = "10.000") -> Dict[str, np.ndarray]:
        """Load ground truth annotation data"""
        annotations = {}
        
        run_path = self.dataset_path / run_name
        annotation_dir = run_path / "Tomograms" / f"VoxelSpacing{voxel_spacing}" / "CanonicalTomogram" / "Annotations"
        
        if not annotation_dir.exists():
            print(f"Warning: Annotation directory not found for {run_name}")
            return annotations
        
        for protein_dir in annotation_dir.iterdir():
            if protein_dir.is_dir():
                protein_name = protein_dir.name
                
                # Map protein names
                if protein_name == 'ribosome-80s':
                    mapped_name = 'ribosome'
                elif protein_name == 'vlp':
                    mapped_name = 'virus-like-particle'
                else:
                    mapped_name = protein_name
                
                zarr_path = protein_dir / "point_annotations.zarr"
                if zarr_path.exists():
                    try:
                        zarr_data = zarr.open(zarr_path, mode='r')
                        points = zarr_data[:]
                        
                        # Convert coordinates (Z,Y,X) -> (X,Y,Z) and multiply by voxel spacing
                        if len(points) > 0:
                            points_xyz = points[:, [2, 1, 0]] * float(voxel_spacing)
                            annotations[mapped_name] = points_xyz
                            print(f"Loaded {len(points_xyz)} annotations for {mapped_name}")
                    except Exception as e:
                        print(f"Error loading annotations for {protein_name}: {e}")
        
        return annotations
    
    def calculate_matching_metrics(self, pred_points: np.ndarray, gt_points: np.ndarray, 
                                 distance_threshold: float = None) -> Dict[str, float]:
        """Calculate matching metrics"""
        if distance_threshold is None:
            distance_threshold = self.distance_threshold
            
        if len(pred_points) == 0 and len(gt_points) == 0:
            return {
                'true_positives': 0,
                'false_positives': 0,
                'false_negatives': 0,
                'precision': 1.0,
                'recall': 1.0,
                'f1_score': 1.0,
                'localization_errors': []
            }
        elif len(pred_points) == 0:
            return {
                'true_positives': 0,
                'false_positives': 0,
                'false_negatives': len(gt_points),
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'localization_errors': []
            }
        elif len(gt_points) == 0:
            return {
                'true_positives': 0,
                'false_positives': len(pred_points),
                'false_negatives': 0,
                'precision': 0.0,
                'recall': 0.0 if len(gt_points) > 0 else 1.0,
                'f1_score': 0.0,
                'localization_errors': []
            }
        
        # Calculate distance matrix
        distances = cdist(pred_points, gt_points)
        
        # Use Hungarian algorithm for optimal matching
        pred_indices, gt_indices = linear_sum_assignment(distances)
        
        # Determine valid matches
        valid_matches = distances[pred_indices, gt_indices] <= distance_threshold
        valid_pred_indices = pred_indices[valid_matches]
        valid_gt_indices = gt_indices[valid_matches]
        
        true_positives = len(valid_pred_indices)
        false_positives = len(pred_points) - true_positives
        false_negatives = len(gt_points) - true_positives
        
        precision = true_positives / len(pred_points) if len(pred_points) > 0 else 0.0
        recall = true_positives / len(gt_points) if len(gt_points) > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Calculate localization errors
        localization_errors = []
        if len(valid_pred_indices) > 0:
            localization_errors = distances[valid_pred_indices, valid_gt_indices].tolist()
        
        return {
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'localization_errors': localization_errors
        }
    
    def calculate_f4_score(self, precision: float, recall: float) -> float:
        """Calculate F4 score (emphasizes recall)"""
        beta = 4
        if precision + recall == 0:
            return 0.0
        return (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
    
    def calculate_iou_3d(self, pred_points: np.ndarray, gt_points: np.ndarray, 
                        radius: float) -> float:
        """Calculate 3D IoU (using sphere approximation)"""
        if len(pred_points) == 0 and len(gt_points) == 0:
            return 1.0
        elif len(pred_points) == 0 or len(gt_points) == 0:
            return 0.0
        
        # Simplified IoU calculation: count overlapping points within the given radius
        distances = cdist(pred_points, gt_points)
        overlapping_pred = np.any(distances <= radius, axis=1)
        overlapping_gt = np.any(distances <= radius, axis=0)
        
        intersection = np.sum(overlapping_pred)
        union = len(pred_points) + len(gt_points) - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_ap_score(self, pred_points: np.ndarray, gt_points: np.ndarray,
                          confidences: Optional[np.ndarray] = None) -> float:
        """Calculate Average Precision"""
        if len(pred_points) == 0 or len(gt_points) == 0:
            return 0.0
        
        if confidences is None:
            confidences = np.ones(len(pred_points))
        
        # Determine whether each predicted point is a true positive
        distances = cdist(pred_points, gt_points)
        min_distances = np.min(distances, axis=1)
        y_true = (min_distances <= self.distance_threshold).astype(int)
        
        # Calculate AP
        try:
            ap = average_precision_score(y_true, confidences)
        except:
            ap = 0.0
        
        return ap
    
    def evaluate_single_run(self, run_name: str, predictions: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Evaluate a single run"""
        print(f"\nEvaluating run: {run_name}")
        
        gt_annotations = self.load_ground_truth_annotations(run_name)
        
        # Filter predictions for this run
        run_predictions = predictions[predictions['experiment'] == run_name]
        
        results = {}
        
        for protein_name in particle_names:
            print(f"  Evaluating {protein_name}...")
            
            # Get ground truth annotations
            gt_points = gt_annotations.get(protein_name, np.array([]).reshape(0, 3))
            
            # Get predictions
            pred_subset = run_predictions[run_predictions['particle_type'] == protein_name]
            
            if len(pred_subset) > 0:
                pred_points = pred_subset[['x', 'y', 'z']].values
                confidences = pred_subset.get('confidence', np.ones(len(pred_subset))).values
            else:
                pred_points = np.array([]).reshape(0, 3)
                confidences = np.array([])
            
            # Calculate basic matching metrics
            matching_metrics = self.calculate_matching_metrics(pred_points, gt_points)
            
            # Calculate additional metrics
            f4_score = self.calculate_f4_score(matching_metrics['precision'], matching_metrics['recall'])
            
            protein_radius = particle_radius.get(protein_name, 100.0)
            iou = self.calculate_iou_3d(pred_points, gt_points, protein_radius)
            
            ap_score = self.calculate_ap_score(pred_points, gt_points, confidences)
            
            # Calculate localization error statistics
            loc_errors = matching_metrics['localization_errors']
            mean_loc_error = np.mean(loc_errors) if loc_errors else 0.0
            std_loc_error = np.std(loc_errors) if loc_errors else 0.0
            
            # Calculate rate metrics
            false_discovery_rate = matching_metrics['false_positives'] / len(pred_points) if len(pred_points) > 0 else 0.0
            miss_rate = matching_metrics['false_negatives'] / len(gt_points) if len(gt_points) > 0 else 0.0
            
            results[protein_name] = {
                'true_positives': matching_metrics['true_positives'],
                'false_positives': matching_metrics['false_positives'],
                'false_negatives': matching_metrics['false_negatives'],
                'precision': matching_metrics['precision'],
                'recall': matching_metrics['recall'],
                'f1_score': matching_metrics['f1_score'],
                'f4_score': f4_score,
                'iou': iou,
                'ap_score': ap_score,
                'mean_localization_error': mean_loc_error,
                'std_localization_error': std_loc_error,
                'false_discovery_rate': false_discovery_rate,
                'miss_rate': miss_rate,
                'num_predictions': len(pred_points),
                'num_ground_truth': len(gt_points)
            }
            
            print(f"    Recall: {matching_metrics['recall']:.3f}, "
                  f"F4: {f4_score:.3f}, "
                  f"IoU: {iou:.3f}, "
                  f"AP: {ap_score:.3f}")
        
        return results
    
    def evaluate_all_runs(self, predictions: pd.DataFrame, max_runs: int = None) -> Dict[str, Dict[str, float]]:
        """Evaluate all runs"""
        print("Starting comprehensive evaluation...")
        
        # Get all runs
        runs = [d.name for d in self.dataset_path.iterdir() 
                if d.is_dir() and d.name.startswith('TS_')]
        runs.sort()
        
        if max_runs:
            runs = runs[:max_runs]
        
        print(f"Evaluating {len(runs)} runs")
        
        all_results = {}
        
        for run_name in runs:
            try:
                run_results = self.evaluate_single_run(run_name, predictions)
                all_results[run_name] = run_results
            except Exception as e:
                print(f"Error evaluating {run_name}: {e}")
                continue
        
        return all_results
    
    def aggregate_results(self, all_results: Dict[str, Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
        """Aggregate results from all runs"""
        aggregated = {}
        
        for protein_name in particle_names:
            metrics = {
                'total_tp': 0, 'total_fp': 0, 'total_fn': 0,
                'total_predictions': 0, 'total_ground_truth': 0,
                'recalls': [], 'precisions': [], 'f1_scores': [], 'f4_scores': [],
                'ious': [], 'ap_scores': [], 'localization_errors': [],
                'false_discovery_rates': [], 'miss_rates': []
            }
            
            for run_name, run_results in all_results.items():
                if protein_name in run_results:
                    result = run_results[protein_name]
                    
                    metrics['total_tp'] += result['true_positives']
                    metrics['total_fp'] += result['false_positives']
                    metrics['total_fn'] += result['false_negatives']
                    metrics['total_predictions'] += result['num_predictions']
                    metrics['total_ground_truth'] += result['num_ground_truth']
                    
                    if result['num_ground_truth'] > 0:  # Only include runs with ground truth annotations
                        metrics['recalls'].append(result['recall'])
                        metrics['f1_scores'].append(result['f1_score'])
                        metrics['f4_scores'].append(result['f4_score'])
                        metrics['ious'].append(result['iou'])
                        metrics['ap_scores'].append(result['ap_score'])
                        metrics['miss_rates'].append(result['miss_rate'])
                    
                    if result['num_predictions'] > 0:
                        metrics['precisions'].append(result['precision'])
                        metrics['false_discovery_rates'].append(result['false_discovery_rate'])
                    
                    if result['mean_localization_error'] > 0:
                        metrics['localization_errors'].append(result['mean_localization_error'])
            
            # Calculate aggregated metrics
            overall_precision = metrics['total_tp'] / (metrics['total_tp'] + metrics['total_fp']) if (metrics['total_tp'] + metrics['total_fp']) > 0 else 0.0
            overall_recall = metrics['total_tp'] / (metrics['total_tp'] + metrics['total_fn']) if (metrics['total_tp'] + metrics['total_fn']) > 0 else 0.0
            overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
            overall_f4 = self.calculate_f4_score(overall_precision, overall_recall)
            
            aggregated[protein_name] = {
                # Basic counts
                'total_true_positives': metrics['total_tp'],
                'total_false_positives': metrics['total_fp'],
                'total_false_negatives': metrics['total_fn'],
                'total_predictions': metrics['total_predictions'],
                'total_ground_truth': metrics['total_ground_truth'],
                
                # Overall metrics
                'overall_precision': overall_precision,
                'overall_recall': overall_recall,
                'overall_f1_score': overall_f1,
                'overall_f4_score': overall_f4,
                
                # Average metrics
                'mean_recall': np.mean(metrics['recalls']) if metrics['recalls'] else 0.0,
                'std_recall': np.std(metrics['recalls']) if metrics['recalls'] else 0.0,
                'mean_precision': np.mean(metrics['precisions']) if metrics['precisions'] else 0.0,
                'std_precision': np.std(metrics['precisions']) if metrics['precisions'] else 0.0,
                'mean_f1_score': np.mean(metrics['f1_scores']) if metrics['f1_scores'] else 0.0,
                'mean_f4_score': np.mean(metrics['f4_scores']) if metrics['f4_scores'] else 0.0,
                'mean_iou': np.mean(metrics['ious']) if metrics['ious'] else 0.0,
                'mean_ap_score': np.mean(metrics['ap_scores']) if metrics['ap_scores'] else 0.0,
                'mean_localization_error': np.mean(metrics['localization_errors']) if metrics['localization_errors'] else 0.0,
                'std_localization_error': np.std(metrics['localization_errors']) if metrics['localization_errors'] else 0.0,
                'mean_false_discovery_rate': np.mean(metrics['false_discovery_rates']) if metrics['false_discovery_rates'] else 0.0,
                'mean_miss_rate': np.mean(metrics['miss_rates']) if metrics['miss_rates'] else 0.0,
            }
        
        return aggregated
    
    def print_evaluation_report(self, aggregated_results: Dict[str, Dict[str, float]]):
        print("\n" + "="*80)
        print("COMPREHENSIVE EVALUATION REPORT")
        print("="*80)
        
        # Print detailed results by particle type
        for protein_name in particle_names:
            if protein_name in aggregated_results:
                result = aggregated_results[protein_name]
                
                print(f"\n{protein_name.upper()}")
                print("-" * 60)
                print(f"Ground Truth Objects: {result['total_ground_truth']}")
                print(f"Predicted Objects: {result['total_predictions']}")
                print(f"True Positives: {result['total_true_positives']}")
                print(f"False Positives: {result['total_false_positives']}")
                print(f"False Negatives: {result['total_false_negatives']}")
                print()
                print(f"Overall Recall: {result['overall_recall']:.4f}")
                print(f"Overall Precision: {result['overall_precision']:.4f}")
                print(f"Overall F1 Score: {result['overall_f1_score']:.4f}")
                print(f"Overall F4 Score: {result['overall_f4_score']:.4f}")
                print(f"Mean IoU: {result['mean_iou']:.4f}")
                print(f"Mean AP Score: {result['mean_ap_score']:.4f}")
                print(f"Mean Localization Error: {result['mean_localization_error']:.2f} ± {result['std_localization_error']:.2f}")
                print(f"False Discovery Rate: {result['mean_false_discovery_rate']:.4f}")
                print(f"Miss Rate: {result['mean_miss_rate']:.4f}")
        
        # Calculate overall averages
        print(f"\n{'OVERALL SUMMARY':^60}")
        print("-" * 60)
        
        overall_metrics = {}
        for metric in ['overall_recall', 'overall_f4_score', 'mean_iou', 'mean_ap_score', 
                      'mean_localization_error', 'mean_false_discovery_rate', 'mean_miss_rate']:
            values = [result[metric] for result in aggregated_results.values() 
                     if result['total_ground_truth'] > 0]
            overall_metrics[metric] = np.mean(values) if values else 0.0
        
        print(f"Average Recall: {overall_metrics['overall_recall']:.4f}")
        print(f"Average F4 Score: {overall_metrics['overall_f4_score']:.4f}")
        print(f"Average IoU: {overall_metrics['mean_iou']:.4f}")
        print(f"Average AP: {overall_metrics['mean_ap_score']:.4f}")
        print(f"Average Localization Error: {overall_metrics['mean_localization_error']:.2f}")
        print(f"Average False Discovery Rate: {overall_metrics['mean_false_discovery_rate']:.4f}")
        print(f"Average Miss Rate: {overall_metrics['mean_miss_rate']:.4f}")
        
        return overall_metrics
    
    def save_results_to_csv(self, all_results: Dict, aggregated_results: Dict, output_path: str = "./evaluation_results.csv"):
        """Save results to CSV file"""
        rows = []
        
        for run_name, run_results in all_results.items():
            for protein_name, metrics in run_results.items():
                row = {
                    'run': run_name,
                    'protein': protein_name,
                    **metrics
                }
                rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        print(f"\nDetailed results saved to: {output_path}")
        
        # Save aggregated results
        agg_output_path = output_path.replace('.csv', '_aggregated.csv')
        agg_rows = []
        for protein_name, metrics in aggregated_results.items():
            row = {'protein': protein_name, **metrics}
            agg_rows.append(row)
        
        agg_df = pd.DataFrame(agg_rows)
        agg_df.to_csv(agg_output_path, index=False)
        print(f"Aggregated results saved to: {agg_output_path}")

def run_comprehensive_evaluation(dataset_path: str = "./data/10445", 
                               max_runs: int = 10,
                               distance_threshold: float = 50.0):
    """Run the complete evaluation process"""
    print("Starting comprehensive model evaluation...")
    
    # 1. Run model inference to obtain predictions
    print("Step 1: Running model inference...")
    try:
        yolo_results = run_yolo_inference()
        unet_results = run_unet_inference()
        predictions = blend_predictions(yolo_results, unet_results)
        print(f"Generated {len(predictions)} predictions")
    except Exception as e:
        print(f"Error during inference: {e}")
        print("Loading dummy predictions for testing...")
        predictions = generate_dummy_predictions(dataset_path, max_runs)
    
    # 2. Initialize evaluator
    evaluator = CryoETEvaluator(dataset_path, distance_threshold)
    
    # 3. Evaluate all runs
    print("Step 2: Evaluating predictions...")
    all_results = evaluator.evaluate_all_runs(predictions, max_runs)
    
    # 4. Aggregate results
    print("Step 3: Aggregating results...")
    aggregated_results = evaluator.aggregate_results(all_results)
    
    # 5. Print report
    print("Step 4: Generating report...")
    overall_metrics = evaluator.print_evaluation_report(aggregated_results)
    
    # 6. Save results
    print("Step 5: Saving results...")
    evaluator.save_results_to_csv(all_results, aggregated_results)
    
    return overall_metrics, aggregated_results, all_results

def generate_dummy_predictions(dataset_path: str, max_runs: int) -> pd.DataFrame:
    print("Generating dummy predictions for testing...")
    
    dataset_path = Path(dataset_path)
    runs = [d.name for d in dataset_path.iterdir() 
            if d.is_dir() and d.name.startswith('TS_')][:max_runs]
    
    predictions = []
    
    for run_name in runs:
        for protein_name in particle_names:
            num_preds = np.random.randint(5, 25)
            for i in range(num_preds):
                predictions.append({
                    'id': len(predictions),
                    'experiment': run_name,
                    'particle_type': protein_name,
                    'x': np.random.uniform(0, 2000),
                    'y': np.random.uniform(0, 2000),
                    'z': np.random.uniform(0, 1000),
                    'confidence': np.random.uniform(0.3, 0.9)
                })
    
    return pd.DataFrame(predictions)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate CryoET model performance')
    parser.add_argument('--dataset_path', type=str, default='./data/10445',
                       help='Path to CryoET dataset')
    parser.add_argument('--max_runs', type=int, default=10,
                       help='Maximum number of runs to evaluate')
    parser.add_argument('--distance_threshold', type=float, default=50.0,
                       help='Distance threshold for matching predictions to ground truth')
    
    args = parser.parse_args()
    
    overall_metrics, aggregated_results, all_results = run_comprehensive_evaluation(
        args.dataset_path, args.max_runs, args.distance_threshold
    )