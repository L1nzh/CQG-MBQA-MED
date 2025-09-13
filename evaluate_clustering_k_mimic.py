#!/usr/bin/env python3
"""
MIMIC dataset clustering evaluation with MiniBatchKMeans.
- All threading disabled
- Results saved as JSON
- Elbow plot saved as PNG for manual inspection
- NO automatic k selection — you decide based on the plot and metrics
"""

# --- 必须在任何导入前设置线程限制 ---
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
# -------------------------------------------

import numpy as np
import logging
import json
import gc
from datetime import datetime
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt

# 禁用警告
import warnings
warnings.filterwarnings('ignore')

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def evaluate_k_values():
    """评估多个 k 值的聚类性能，保存结果和肘部图"""
    
    embeddings_file = "temp/mimic/corpus_embeddings.npy"
    if not os.path.exists(embeddings_file):
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_file}")
    
    logger.info("Loading MIMIC corpus embeddings...")
    corpus_embeddings = np.load(embeddings_file)
    logger.info(f"Loaded MIMIC embeddings shape: {corpus_embeddings.shape}")
    
    # 固定采样 2000k (或全部数据如果少于2M)
    sample_size = min(2000000, len(corpus_embeddings))
    np.random.seed(42)
    indices = np.random.choice(len(corpus_embeddings), sample_size, replace=False)
    sampled_embeddings = corpus_embeddings[indices].copy()
    
    del corpus_embeddings
    gc.collect()
    
    logger.info(f"Sampled {sample_size:,} MIMIC embeddings for evaluation.")
    
    # 测试的 k 值 (可根据MIMIC数据特点调整)
    k_values = [1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500]
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "dataset": "MIMIC-IV",
        "algorithm": "MiniBatchKMeans",
        "sample_size": sample_size,
        "embedding_dimension": sampled_embeddings.shape[1],
        "k_values_tested": k_values,
        "evaluation_results": [],
        "settings": {
            "init": "k-means++",
            "max_iter": 100,
            "batch_size": 1000,
            "n_init": 3,
            "random_state": 42,
            "threading": "single_thread",
            "environment": {
                "OPENBLAS_NUM_THREADS": os.getenv("OPENBLAS_NUM_THREADS"),
                "MKL_NUM_THREADS": os.getenv("MKL_NUM_THREADS"),
                "OMP_NUM_THREADS": os.getenv("OMP_NUM_THREADS"),
            }
        }
    }
    
    # 用于绘图的数据
    inertias = []
    
    # 开始评估
    for i, k in enumerate(tqdm(k_values, desc="Evaluating k values for MIMIC")):
        logger.info(f"Starting k={k} ({i+1}/{len(k_values)}) for MIMIC dataset")
        
        try:
            model = MiniBatchKMeans(
                n_clusters=k,
                init='k-means++',
                n_init=3,
                max_iter=100,
                batch_size=1000,
                random_state=42,
                verbose=0
            )
            
            start_time = datetime.now()
            labels = model.fit_predict(sampled_embeddings)
            duration = (datetime.now() - start_time).total_seconds()
            
            # 获取指标
            inertia = float(model.inertia_)
            inertias.append(inertia)
            
            # 为节省时间，silhouette 使用子采样
            sil_sample_idx = np.random.choice(len(labels), 50000, replace=False)
            silhouette = float(silhouette_score(sampled_embeddings[sil_sample_idx], labels[sil_sample_idx]))
            
            db_score = float(davies_bouldin_score(sampled_embeddings[sil_sample_idx], labels[sil_sample_idx]))
            ch_score = float(calinski_harabasz_score(sampled_embeddings[sil_sample_idx], labels[sil_sample_idx]))
            
            cluster_counts = np.bincount(labels)
            min_size = int(cluster_counts.min())
            max_size = int(cluster_counts.max())
            mean_size = float(cluster_counts.mean())
            std_size = float(cluster_counts.std())
            
            result = {
                "k": int(k),
                "mse_inertia": float(inertia),
                "silhouette_coefficient": float(silhouette),
                "davies_bouldin_index": float(db_score),
                "calinski_harabasz_index": float(ch_score),
                "duration_seconds": float(duration),
                "actual_clusters": int(len(np.unique(labels))),
                "cluster_sizes": {
                    "min": int(min_size),
                    "max": int(max_size),
                    "mean": float(mean_size),
                    "std": float(std_size)
                }
            }
            
            results["evaluation_results"].append(result)
            logger.info(f"k={k} | Inertia: {inertia:.2e} | Sil: {silhouette:.4f} | DB: {db_score:.4f} | CH: {ch_score:.2f} | Time: {duration:.1f}s")
            
            del model, labels
            gc.collect()
            
        except Exception as e:
            logger.error(f"Failed at k={k}: {str(e)}")
            results["evaluation_results"].append({
                "k": k,
                "error": str(e),
                "status": "failed"
            })
            gc.collect()
            continue
    
    # === 保存结果 JSON ===
    json_path = "clustering_evaluation_results_mimic.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"✅ MIMIC evaluation results saved to {json_path}")
    
    # === 保存肘部图 ===
    plt.figure(figsize=(10, 6))
    valid_ks = []
    valid_inertias = []
    
    for r in results["evaluation_results"]:
        if "error" not in r:
            valid_ks.append(r["k"])
            valid_inertias.append(r["mse_inertia"])
    
    plt.plot(valid_ks, valid_inertias, 'ro-', linewidth=2, markersize=8)
    plt.title('MIMIC-IV: Elbow Method - Inertia vs Number of Clusters (k)', fontsize=14)
    plt.xlabel('Number of Clusters (k)', fontsize=12)
    plt.ylabel('Inertia (MSE)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = "elbow_plot_mimic.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"✅ MIMIC elbow plot saved to {plot_path}")
    
    # 打印完成信息
    print("\n" + "="*60)
    print("        MIMIC-IV CLUSTERING EVALUATION COMPLETE")
    print("="*60)
    print(f"📊 Results saved to: {json_path}")
    print(f"📈 Elbow plot saved to: {plot_path}")
    print(f"🔍 You can now inspect the plot and JSON to choose the best k.")
    print("="*60)
    
    return results


if __name__ == "__main__":
    try:
        evaluate_k_values()
    except Exception as e:
        logger.error(f"MIMIC evaluation failed: {e}")
        raise