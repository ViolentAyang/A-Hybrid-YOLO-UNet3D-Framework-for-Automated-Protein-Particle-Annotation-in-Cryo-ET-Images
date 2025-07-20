import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List

class EvaluationVisualizer:
    def __init__(self, output_dir: str = "./evaluation_plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def plot_metric_comparison(self, aggregated_results: Dict, save: bool = True):
        proteins = list(aggregated_results.keys())
        metrics = ['overall_recall', 'overall_f4_score', 'mean_iou', 'mean_ap_score']
        metric_names = ['Recall', 'F4 Score', 'IoU', 'Average Precision']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            values = [aggregated_results[protein][metric] for protein in proteins]
            
            bars = axes[i].bar(proteins, values, alpha=0.7)
            axes[i].set_title(f'{metric_name} by Protein Type', fontsize=14, fontweight='bold')
            axes[i].set_ylabel(metric_name, fontsize=12)
            axes[i].tick_params(axis='x', rotation=45)
            
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontsize=10)
            
            axes[i].set_ylim(0, 1.0)
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'metric_comparison.png', dpi=300, bbox_inches='tight')
            print(f"Metric comparison plot saved to {self.output_dir / 'metric_comparison.png'}")
        
        plt.show()
    
    def plot_error_rates(self, aggregated_results: Dict, save: bool = True):
        proteins = list(aggregated_results.keys())
        fdr = [aggregated_results[protein]['mean_false_discovery_rate'] for protein in proteins]
        miss_rate = [aggregated_results[protein]['mean_miss_rate'] for protein in proteins]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        bars1 = ax1.bar(proteins, fdr, alpha=0.7, color='red')
        ax1.set_title('False Discovery Rate by Protein Type', fontsize=14, fontweight='bold')
        ax1.set_ylabel('False Discovery Rate', fontsize=12)
        ax1.tick_params(axis='x', rotation=45)
        ax1.set_ylim(0, 1.0)
        ax1.grid(True, alpha=0.3)
        
        for bar, value in zip(bars1, fdr):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=10)
        
        bars2 = ax2.bar(proteins, miss_rate, alpha=0.7, color='orange')
        ax2.set_title('Miss Rate by Protein Type', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Miss Rate', fontsize=12)
        ax2.tick_params(axis='x', rotation=45)
        ax2.set_ylim(0, 1.0)
        ax2.grid(True, alpha=0.3)
        
        for bar, value in zip(bars2, miss_rate):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'error_rates.png', dpi=300, bbox_inches='tight')
            print(f"Error rates plot saved to {self.output_dir / 'error_rates.png'}")
        
        plt.show()
    
    def plot_localization_errors(self, aggregated_results: Dict, save: bool = True):
        proteins = list(aggregated_results.keys())
        mean_errors = [aggregated_results[protein]['mean_localization_error'] for protein in proteins]
        std_errors = [aggregated_results[protein]['std_localization_error'] for protein in proteins]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars = ax.bar(proteins, mean_errors, yerr=std_errors, alpha=0.7, 
                     capsize=5, error_kw={'linewidth': 2})
        ax.set_title('Localization Error by Protein Type', fontsize=14, fontweight='bold')
        ax.set_ylabel('Localization Error (Angstroms)', fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        for bar, mean_val, std_val in zip(bars, mean_errors, std_errors):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std_val + 1,
                   f'{mean_val:.1f}±{std_val:.1f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'localization_errors.png', dpi=300, bbox_inches='tight')
            print(f"Localization errors plot saved to {self.output_dir / 'localization_errors.png'}")
        
        plt.show()
    
    def plot_detection_statistics(self, aggregated_results: Dict, save: bool = True):
        proteins = list(aggregated_results.keys())
        ground_truth = [aggregated_results[protein]['total_ground_truth'] for protein in proteins]
        predictions = [aggregated_results[protein]['total_predictions'] for protein in proteins]
        true_positives = [aggregated_results[protein]['total_true_positives'] for protein in proteins]
        
        x = np.arange(len(proteins))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        bars1 = ax.bar(x - width, ground_truth, width, label='Ground Truth', alpha=0.8)
        bars2 = ax.bar(x, predictions, width, label='Predictions', alpha=0.8)
        bars3 = ax.bar(x + width, true_positives, width, label='True Positives', alpha=0.8)
        
        ax.set_title('Detection Statistics by Protein Type', fontsize=14, fontweight='bold')
        ax.set_ylabel('Count', fontsize=12)
        ax.set_xlabel('Protein Type', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(proteins, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{int(height)}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'detection_statistics.png', dpi=300, bbox_inches='tight')
            print(f"Detection statistics plot saved to {self.output_dir / 'detection_statistics.png'}")
        
        plt.show()
    
    def plot_confusion_matrix_style(self, aggregated_results: Dict, save: bool = True):
        proteins = list(aggregated_results.keys())
        metrics = ['overall_recall', 'overall_precision', 'overall_f1_score', 'overall_f4_score', 'mean_iou', 'mean_ap_score']
        metric_labels = ['Recall', 'Precision', 'F1', 'F4', 'IoU', 'AP']
        
        data = []
        for protein in proteins:
            row = [aggregated_results[protein][metric] for metric in metrics]
            data.append(row)
        
        data = np.array(data)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        ax.set_xticks(np.arange(len(metric_labels)))
        ax.set_yticks(np.arange(len(proteins)))
        ax.set_xticklabels(metric_labels)
        ax.set_yticklabels(proteins)
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        for i in range(len(proteins)):
            for j in range(len(metrics)):
                text = ax.text(j, i, f'{data[i, j]:.3f}', 
                             ha="center", va="center", color="black", fontweight='bold')
        
        ax.set_title("Performance Matrix Heatmap", fontsize=14, fontweight='bold')
        
        cbar = plt.colorbar(im)
        cbar.set_label('Score', rotation=270, labelpad=20)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'performance_matrix.png', dpi=300, bbox_inches='tight')
            print(f"Performance matrix plot saved to {self.output_dir / 'performance_matrix.png'}")
        
        plt.show()
    
    def create_summary_report(self, overall_metrics: Dict, aggregated_results: Dict, save: bool = True):
        report = []
        report.append("="*80)
        report.append("CRYOET MODEL EVALUATION SUMMARY REPORT")
        report.append("="*80)
        report.append("")
        
        report.append("OVERALL PERFORMANCE:")
        report.append("-" * 40)
        report.append(f"Average Recall:              {overall_metrics['overall_recall']:.4f}")
        report.append(f"Average F4 Score:            {overall_metrics['overall_f4_score']:.4f}")
        report.append(f"Average IoU:                 {overall_metrics['mean_iou']:.4f}")
        report.append(f"Average AP:                  {overall_metrics['mean_ap_score']:.4f}")
        report.append(f"Average Localization Error:  {overall_metrics['mean_localization_error']:.2f} Å")
        report.append(f"Average False Discovery Rate: {overall_metrics['mean_false_discovery_rate']:.4f}")
        report.append(f"Average Miss Rate:           {overall_metrics['mean_miss_rate']:.4f}")
        report.append("")
        
        report.append("DETAILED RESULTS BY PROTEIN TYPE:")
        report.append("-" * 40)
        
        for protein_name in aggregated_results.keys():
            result = aggregated_results[protein_name]
            report.append(f"\n{protein_name.upper()}:")
            report.append(f"  Total Objects (GT/Pred): {result['total_ground_truth']}/{result['total_predictions']}")
            report.append(f"  True Positives: {result['total_true_positives']}")
            report.append(f"  Recall: {result['overall_recall']:.4f}")
            report.append(f"  F4 Score: {result['overall_f4_score']:.4f}")
            report.append(f"  IoU: {result['mean_iou']:.4f}")
            report.append(f"  AP: {result['mean_ap_score']:.4f}")
            report.append(f"  Localization Error: {result['mean_localization_error']:.2f} ± {result['std_localization_error']:.2f} Å")
        
        report.append("\n" + "="*50)
        report.append("PERFORMANCE ASSESSMENT:")
        report.append("="*50)
        
        def assess_performance(value, thresholds):
            if value >= thresholds[0]:
                return "Excellent"
            elif value >= thresholds[1]:
                return "Good"
            elif value >= thresholds[2]:
                return "Fair"
            else:
                return "Poor"
        
        recall_assessment = assess_performance(overall_metrics['overall_recall'], [0.8, 0.6, 0.4])
        f4_assessment = assess_performance(overall_metrics['overall_f4_score'], [0.7, 0.5, 0.3])
        
        report.append(f"Recall Performance:    {recall_assessment} ({overall_metrics['overall_recall']:.3f})")
        report.append(f"F4 Score Performance:  {f4_assessment} ({overall_metrics['overall_f4_score']:.3f})")
        
        if overall_metrics['mean_localization_error'] < 30:
            loc_assessment = "Excellent"
        elif overall_metrics['mean_localization_error'] < 50:
            loc_assessment = "Good"
        elif overall_metrics['mean_localization_error'] < 100:
            loc_assessment = "Fair"
        else:
            loc_assessment = "Poor"
        
        report.append(f"Localization Accuracy: {loc_assessment} ({overall_metrics['mean_localization_error']:.1f} Å)")
        
        report_text = "\n".join(report)
        
        if save:
            with open(self.output_dir / 'evaluation_summary_report.txt', 'w') as f:
                f.write(report_text)
        
        print(report_text)
        return report_text
    
    def generate_all_plots(self, overall_metrics: Dict, aggregated_results: Dict):
        print(f"Generating evaluation visualizations in {self.output_dir}")
        
        self.plot_metric_comparison(aggregated_results)
        self.plot_error_rates(aggregated_results)
        self.plot_localization_errors(aggregated_results)
        self.plot_detection_statistics(aggregated_results)
        self.plot_confusion_matrix_style(aggregated_results)
        self.create_summary_report(overall_metrics, aggregated_results)
        
        print(f"\nAll evaluation plots and reports saved to: {self.output_dir}")
        print("Generated files:")
        for file in self.output_dir.glob("*"):
            print(f"  - {file.name}")

def visualize_evaluation_results(overall_metrics: Dict, aggregated_results: Dict, 
                               output_dir: str = "./evaluation_plots"):
    visualizer = EvaluationVisualizer(output_dir)
    visualizer.generate_all_plots(overall_metrics, aggregated_results)