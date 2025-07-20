import os
import argparse
import sys
import warnings
import numpy as np
import pandas as pd
import torch

from yolo_predictor import run_yolo_inference
from unet_model import run_unet_inference
from blend import blend_predictions

def install_dependencies():
    deps_path = './data/dependencies'
    if os.path.exists(f'{deps_path}/requirements.txt'):
        os.system(f'pip install -q --no-index --find-links {deps_path} --requirement {deps_path}/requirements.txt')
    
    try:
        import zarr
    except: 
        wheel_path = './data/wheel_files'
        if os.path.exists(wheel_path):
            os.system(f"cp -r '{wheel_path}' './working/'")
            os.system("pip install ./working/wheel_files/asciitree-0.3.3/asciitree-0.3.3")
            os.system("pip install --no-index --find-links=./working/wheel_files zarr")
            os.system("pip install --no-index --find-links=./working/wheel_files connected-components-3d")

def run_inference():
    """inference for kaggle competition"""
    print("Starting model inference pipeline...")
    
    install_dependencies()
    
    print("Step 1: YOLO inference...")
    yolo_results = run_yolo_inference()
    print(f"YOLO results: {len(yolo_results)} predictions")
    
    print("Step 2: UNet inference...")
    unet_results = run_unet_inference()
    print(f"UNet results: {len(unet_results)} predictions")
    
    print("Step 3: Blending predictions...")
    final_results = blend_predictions(yolo_results, unet_results)
    print(f"Final results: {len(final_results)} predictions")
    
    final_results.to_csv('submission.csv', index=False)
    print("Submission saved to submission.csv")
    
    return final_results

def run_evaluation(dataset_path: str = "./data/10445", max_runs: int = 10):
    """eva"""
    print("Starting model evaluation...")
    
    try:
        from model_evaluation import run_comprehensive_evaluation
        from evaluation_visualizer import visualize_evaluation_results
        
        overall_metrics, aggregated_results, all_results = run_comprehensive_evaluation(
            dataset_path=dataset_path,
            max_runs=max_runs
        )
        
        visualize_evaluation_results(overall_metrics, aggregated_results)
        
        print(f"\nEvaluation Summary:")
        print(f"Recall: {overall_metrics['overall_recall']:.4f}")
        print(f"F4 Score: {overall_metrics['overall_f4_score']:.4f}")
        print(f"IoU: {overall_metrics['mean_iou']:.4f}")
        print(f"AP: {overall_metrics['mean_ap_score']:.4f}")
        
        return overall_metrics
        
    except ImportError:
        print("Error: Evaluation modules not found.")
        print("Please ensure model_evaluation.py and evaluation_visualizer.py are available.")
        return None
    except Exception as e:
        print(f"Evaluation failed: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(
        description='CryoET Model Main Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
  # Run inference only (default)
  python main.py
  
  # Run evaluation only
  python main.py --evaluate --dataset_path ./data/10445
  
  # Run both inference and evaluation
  python main.py --evaluate --dataset_path ./data/10445 --max_runs 5
        """
    )
    
    parser.add_argument('--evaluate', action='store_true',
                       help='Run model evaluation')
    parser.add_argument('--dataset_path', type=str, default='./data/10445',
                       help='Path to evaluation dataset (default: ./data/10445)')
    parser.add_argument('--max_runs', type=int, default=10,
                       help='Maximum runs for evaluation (default: 10)')
    parser.add_argument('--inference_only', action='store_true',
                       help='Run inference only, skip evaluation')
    
    args = parser.parse_args()
    
    print("="*60)
    print("🧬 CryoET Model Main Pipeline")
    print("="*60)
    
    results = {}
    
    # 运行推理
    if not args.evaluate or not args.inference_only:
        print("\n🔮 Running Model Inference...")
        inference_results = run_inference()
        results['inference'] = inference_results
    
    # 运行评估
    if args.evaluate:
        print("\n📊 Running Model Evaluation...")
        evaluation_results = run_evaluation(args.dataset_path, args.max_runs)
        results['evaluation'] = evaluation_results
    
    print("\n✅ Pipeline completed!")
    
    if 'evaluation' in results and results['evaluation'] is not None:
        eval_results = results['evaluation']
        print(f"\n🎯 Final Performance:")
        print(f"   Recall: {eval_results['overall_recall']:.4f}")
        print(f"   F4 Score: {eval_results['overall_f4_score']:.4f}")
    
    return results

if __name__ == "__main__":
    main() 