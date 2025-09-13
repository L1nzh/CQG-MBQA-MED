#!/usr/bin/env python3
"""
Plot 4 clustering evaluation metrics in one figure
python plot_clustering_metrics.py clustering_evaluation_results_mimic.json mimic_metrics.png
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any

def plot_clustering_metrics(results_file: str, output_file: str = None):
    """
    Plot all 4 clustering metrics in a single figure with subplots
    
    Args:
        results_file: Path to JSON file with clustering evaluation results
        output_file: Optional output filename (defaults to metrics_plot.png)
    """
    if output_file is None:
        output_file = results_file.replace('.json', '_metrics_plot.png')
    
    # Load results
    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # Extract data
    k_values = []
    inertias = []
    silhouette_scores = []
    db_scores = []
    ch_scores = []
    
    for result in results['evaluation_results']:
        if 'error' not in result:
            k_values.append(result['k'])
            inertias.append(result['mse_inertia'])
            silhouette_scores.append(result['silhouette_coefficient'])
            db_scores.append(result['davies_bouldin_index'])
            ch_scores.append(result['calinski_harabasz_index'])
    
    # Create 2x2 subplot figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    dataset_name = results.get('dataset', 'Dataset')
    
    # 1. Inertia (Elbow Method) - Lower is better
    ax1.plot(k_values, inertias, 'bo-', linewidth=2, markersize=6)
    ax1.set_title('Inertia (MSE) - Elbow Method', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Inertia')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='both', which='major', labelsize=10)
    
    # 2. Silhouette Score - Higher is better
    ax2.plot(k_values, silhouette_scores, 'go-', linewidth=2, markersize=6)
    ax2.set_title('Silhouette Score', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Number of Clusters (k)')
    ax2.set_ylabel('Silhouette Score')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='both', which='major', labelsize=10)
    
    # 3. Davies-Bouldin Index - Lower is better
    ax3.plot(k_values, db_scores, 'ro-', linewidth=2, markersize=6)
    ax3.set_title('Davies-Bouldin Index', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Number of Clusters (k)')
    ax3.set_ylabel('Davies-Bouldin Index')
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='both', which='major', labelsize=10)
    
    # 4. Calinski-Harabasz Index - Higher is better
    ax4.plot(k_values, ch_scores, 'mo-', linewidth=2, markersize=6)
    ax4.set_title('Calinski-Harabasz Index', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Number of Clusters (k)')
    ax4.set_ylabel('Calinski-Harabasz Index')
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(axis='both', which='major', labelsize=10)
    
    # Add overall title
    fig.suptitle(f'{dataset_name}: Clustering Metrics Comparison', fontsize=14, fontweight='bold')
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    # Save figure
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Metrics plot saved to: {output_file}")
    
    # Print optimal k suggestions based on metrics
    print(f"\n📊 Metric Analysis:")
    print(f"{'Metric':<25} {'Best k':<10} {'Value':<15} {'Trend'}")
    print("-" * 60)
    
    # Silhouette (higher is better)
    best_sil_idx = np.argmax(silhouette_scores)
    print(f"{'Silhouette Score':<25} {k_values[best_sil_idx]:<10} {silhouette_scores[best_sil_idx]:<15.4f} {'Higher is better'}")
    
    # Davies-Bouldin (lower is better)
    best_db_idx = np.argmin(db_scores)
    print(f"{'Davies-Bouldin Index':<25} {k_values[best_db_idx]:<10} {db_scores[best_db_idx]:<15.4f} {'Lower is better'}")
    
    # Calinski-Harabasz (higher is better)
    best_ch_idx = np.argmax(ch_scores)
    print(f"{'Calinski-Harabasz Index':<25} {k_values[best_ch_idx]:<10} {ch_scores[best_ch_idx]:<15.2f} {'Higher is better'}")
    
    # Elbow method suggestion (find elbow point)
    # Simple elbow detection using rate of change
    if len(inertias) > 2:
        # Calculate second derivative to find elbow
        first_diff = np.diff(inertias)
        second_diff = np.diff(first_diff)
        elbow_idx = np.argmax(second_diff) + 1  # +1 because of double diff
        if elbow_idx < len(k_values):
            print(f"{'Elbow Method':<25} {k_values[elbow_idx]:<10} {inertias[elbow_idx]:<15.2e} {'Elbow point'}")
    
    return output_file

def plot_single_metric_comparison(results_files: Dict[str, str], metric: str, output_file: str = None):
    """
    Plot a single metric across multiple datasets for comparison
    
    Args:
        results_files: Dict of {dataset_name: results_file_path}
        metric: One of 'inertia', 'silhouette', 'davies_bouldin', 'calinski_harabasz'
        output_file: Output filename
    """
    if output_file is None:
        output_file = f"{metric}_comparison.png"
    
    plt.figure(figsize=(10, 6))
    
    metric_mapping = {
        'inertia': 'mse_inertia',
        'silhouette': 'silhouette_coefficient',
        'davies_bouldin': 'davies_bouldin_index',
        'calinski_harabasz': 'calinski_harabasz_index'
    }
    
    colors = ['b', 'r', 'g', 'm', 'c', 'y']
    
    for i, (dataset_name, results_file) in enumerate(results_files.items()):
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        k_values = []
        metric_values = []
        
        for result in results['evaluation_results']:
            if 'error' not in result:
                k_values.append(result['k'])
                metric_values.append(result[metric_mapping[metric]])
        
        color = colors[i % len(colors)]
        plt.plot(k_values, metric_values, f'{color}o-', linewidth=2, markersize=6, label=dataset_name)
    
    plt.title(f'{metric.replace("_", " ").title()} Comparison Across Datasets', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel(metric.replace("_", " ").title())
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Comparison plot saved to: {output_file}")
    return output_file

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python plot_clustering_metrics.py <results_file.json> [output_file.png]")
        sys.exit(1)
    
    results_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    plot_clustering_metrics(results_file, output_file)