"""
Dynamic Text Pool System

A modular system for processing time-series text data with dynamic text pool management,
clustering, scoring, and visualization capabilities.

Modules:
- models: Data structures and models
- utils: Utility functions and GPU acceleration
- embeddings: Text embedding computation
- clustering: Clustering algorithms (K-means, robust clustering)
- scoring: Candidate scoring and pool management
- visualization: Cluster visualization
- export: Data export functions
- pipeline: Main processing pipeline
"""

from .clustering import (
    kmeans_cluster,
    robust_kmeans_cluster,
    select_topk_per_cluster_dynamic,
)
from .embeddings import compute_embeddings, load_model
from .export import export_all_pools_to_csv
from .models import PoolItem
from .pipeline import main, run_update
from .scoring import score_candidates, update_pool
from .utils import (
    gpu_cosine_similarity,
    get_gpu_memory_info,
    print_gpu_status,
    time_function,
)
from .visualization import save_cluster_plot

__version__ = "1.0.0"
__author__ = "Dynamic Text Pool Team"

__all__ = [
    # Models
    "PoolItem",
    # Utils
    "time_function",
    "get_gpu_memory_info", 
    "print_gpu_status",
    "gpu_cosine_similarity",
    # Embeddings
    "load_model",
    "compute_embeddings",
    # Clustering
    "kmeans_cluster",
    "robust_kmeans_cluster", 
    "select_topk_per_cluster_dynamic",
    # Scoring
    "score_candidates",
    "update_pool",
    # Visualization
    "save_cluster_plot",
    # Export
    "export_all_pools_to_csv",
    # Pipeline
    "run_update",
    "main",
]

































































