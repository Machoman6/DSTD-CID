"""
Scoring functions for dynamic text pool system.
"""

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from utils import gpu_cosine_similarity


def score_candidates(
    cand_indices: List[Tuple[int, int]],
    embeddings: np.ndarray,
    texts: List[str],
    timestamps: List[pd.Timestamp],
    pool_embeddings: np.ndarray,
    event_prior_embeddings: np.ndarray = None,
    w1: float = 0.0,  # representativeness weight
    w2: float = 0.0,  # temporal freshness weight
    w3: float = 1.0,  # similarity weight
    beta: float = 5.0,
) -> List[Tuple[int, float]]:
    """Score candidates with representativeness, time freshness, and similarity to event prior information."""
    scores: List[Tuple[int, float]] = []
    
    if not cand_indices:
        return scores
    
    # Approximate representativeness by local density using cosine similarity to 20 nearest neighbors
    # Use smaller subset for large datasets to avoid memory issues
    if len(embeddings) >= 3:
        # At least 3 comments needed for meaningful representativeness calculation
        n_neighbors = min(20, len(embeddings) - 1)
        local_density = gpu_cosine_similarity(embeddings, embeddings)
        np.fill_diagonal(local_density, 0.0)
        local_density = np.mean(np.sort(local_density, axis=1)[:, -n_neighbors:], axis=1)
    else:
        # Too few comments for meaningful representativeness calculation
        local_density = np.ones(len(embeddings)) * 0.6

    # Vectorized computation for better performance
    candidate_indices = [idx for _, idx in cand_indices]
    if candidate_indices:
        # Extract scores for all candidates at once
        rep_scores = np.exp(-beta * (1 - local_density[candidate_indices]))
        
        # New candidates get full time freshness (1.0)
        time_freshness_scores = np.ones(len(candidate_indices))
        
        # Calculate similarity scores with event prior information
        if event_prior_embeddings is not None and len(event_prior_embeddings) > 0:
            # Compute similarity between candidates and event prior information
            candidate_embeddings = embeddings[candidate_indices]
            similarity_matrix = gpu_cosine_similarity(candidate_embeddings, event_prior_embeddings)
            # Take maximum similarity across all event prior texts
            similarity_scores = np.max(similarity_matrix, axis=1)
        else:
            # If no event prior information, give neutral similarity score
            similarity_scores = np.ones(len(candidate_indices)) * 0.5
        
        # Combine scores: representativeness + time freshness + similarity
        final_scores = w1 * rep_scores + w2 * time_freshness_scores + w3 * similarity_scores
        
        # Create result list
        scores = [(idx, float(score)) for idx, score in zip(candidate_indices, final_scores)]
    
    return scores


def update_pool(
    candidates: List[Tuple[int, float]],
    texts: List[str],
    timestamps: List[pd.Timestamp],
    old_pool: List[Dict[str, Any]],
    embeddings: np.ndarray,
    pool_max: int,
    gamma: float = 0.1,
) -> List[Dict[str, Any]]:
    # Merge old pool and candidates, then cut by score
    merged: List[Dict[str, Any]] = []
    
    # Apply time decay to old pool items (they have been in pool for 1 hour)
    for item in old_pool:
        # Calculate time decay factor for items that have been in pool
        time_decay_factor = np.exp(-gamma * 1.0)  # 1 hour decay
        # Apply decay to the score
        decayed_score = item.get("score", 0.0) * time_decay_factor
        item_copy = item.copy()
        item_copy["score"] = float(decayed_score)
        merged.append(item_copy)
    
    # Add new candidates (they get full time freshness)
    for idx, score in candidates:
        merged.append({
            "id": str(idx),
            "text": texts[idx],
            "timestamp": str(timestamps[idx]),
            "score": float(score),
        })
    
    # Sort by score descending and truncate
    merged.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return merged[:pool_max]
