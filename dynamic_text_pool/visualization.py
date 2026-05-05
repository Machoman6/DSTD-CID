"""
Visualization functions for dynamic text pool system.
"""

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def save_cluster_plot(embeddings: np.ndarray, labels: np.ndarray, kmeans: Optional[KMeans], out_path: str,
                      highlight_indices: Optional[List[int]] = None) -> None:
    """Save a 2D visualization of clusters with optional candidate highlighting."""
    try:
        n_samples = embeddings.shape[0]
        if n_samples == 0:
            return
        # Choose PCA components within valid bounds: 1..min(n_samples, n_features)-1
        max_allowed = max(1, min(embeddings.shape[0], embeddings.shape[1]) - 1)
        n_comp = min(32, max_allowed)
        if n_comp <= 1:
            # If only 1 component possible, PCA will still work; ensure downstream gets 2D
            Xr1 = PCA(n_components=1, random_state=42).fit_transform(embeddings)
            Xr = np.hstack([Xr1, np.zeros((n_samples, 1))])
        else:
            Xr = PCA(n_components=n_comp, random_state=42).fit_transform(embeddings)
        if n_samples < 5:
            # Use PCA directly when too few samples for t-SNE
            # If only 1 component, pad with zeros
            if Xr.shape[1] == 1:
                X2 = np.hstack([Xr, np.zeros((n_samples, 1))])
            else:
                X2 = Xr[:, :2]
        else:
            base_perp = max(5, min(30, max(5, n_samples // 3)))
            perp = max(2, min(base_perp, n_samples - 1))
            try:
                X2 = TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto", perplexity=perp).fit_transform(Xr)
            except Exception:
                # fallback to PCA 2D
                X2 = Xr[:, :2] if Xr.shape[1] >= 2 else np.hstack([Xr, np.zeros((n_samples, 1))])

        # sanitize values to avoid plotting errors
        X2 = np.array(X2, dtype=float)
        X2 = np.nan_to_num(X2, nan=0.0, posinf=0.0, neginf=0.0)
        if np.allclose(X2.std(axis=0), 0.0):
            X2 = X2 + np.random.normal(scale=1e-3, size=X2.shape)

        plt.figure(figsize=(10, 8))
        uniq = sorted(set(labels.tolist())) if len(labels) else [0]
        colors = plt.get_cmap('tab20', len(uniq))
        for idx, cid in enumerate(uniq):
            m = labels == cid if len(labels) else np.ones(embeddings.shape[0], dtype=bool)
            pts = X2[m]
            plt.scatter(pts[:, 0], pts[:, 1], s=20, color=colors(idx), alpha=0.7, label=f"Cluster {cid+1}")
        if kmeans is not None and len(labels) > 0:
            for cid in uniq:
                idxs = np.where(labels == cid)[0]
                if len(idxs) == 0:
                    continue
                center = kmeans.cluster_centers_[cid]
                # choose medoid by distance in original embedding space to avoid space mismatch
                dist = np.linalg.norm(embeddings[idxs] - center, axis=1)
                med = idxs[np.argmin(dist)]
                x, y = X2[med]
                plt.scatter([x], [y], s=120, color='none', edgecolors='black', linewidths=1.5)
        # overlay: highlight candidate points
        if highlight_indices:
            mask_h = np.zeros(embeddings.shape[0], dtype=bool)
            for i in highlight_indices:
                if 0 <= i < mask_h.size:
                    mask_h[i] = True
            hx, hy = X2[mask_h, 0], X2[mask_h, 1]
            plt.scatter(hx, hy, s=80, facecolors='none', edgecolors='red', linewidths=1.5, marker='o', label='Candidates')

        plt.title("Comment Clusters (2D)")
        if len(uniq) <= 20:
            plt.legend(loc='best', fontsize=9, frameon=True)
        plt.tight_layout()
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Saved cluster plot: {out_path}")
        try:
            plt.close()
        except Exception:
            pass
    except Exception as e:
        print(f"Failed to save cluster plot: {out_path} | error: {e}")

































































