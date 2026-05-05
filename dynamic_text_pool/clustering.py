"""
Clustering algorithms for dynamic text pool system.
"""

from typing import List, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_distances
from tqdm import tqdm

from utils import gpu_cosine_similarity


def _merge_small_clusters(labels: np.ndarray, centroids: np.ndarray, min_size: int) -> np.ndarray:
    if labels.size == 0:
        return labels
    unique = sorted(set(labels.tolist()))
    counts = {cid: int(np.sum(labels == cid)) for cid in unique}
    # If all clusters are large enough, return
    if all(c >= min_size for c in counts.values()):
        return labels
    # Merge too-small clusters to nearest centroid by cosine distance
    new_labels = labels.copy()
    for cid in unique:
        if counts[cid] >= min_size:
            continue
        target = None
        best = float("inf")
        for other in unique:
            if other == cid or counts[other] == 0:
                continue
            d = cosine_distances(centroids[cid].reshape(1, -1), centroids[other].reshape(1, -1))[0, 0]
            if d < best:
                best = d
                target = other
        if target is not None:
            new_labels[new_labels == cid] = target
    # Reindex labels to 0..C-1
    uniq = sorted(set(new_labels.tolist()))
    remap = {old: i for i, old in enumerate(uniq)}
    for old, new in remap.items():
        new_labels[new_labels == old] = new
    return new_labels


def kmeans_cluster(embeddings: np.ndarray, auto_k: bool = True, k: int = 8, random_state: int = 42,
                   min_cluster_size: int = 5, k_min: int | None = None, k_max: int | None = None,
                   small_cluster_penalty: float = 0.15) -> Tuple[np.ndarray, KMeans | None]:
    num = embeddings.shape[0]
    if num == 0:
        return np.array([], dtype=int), None
    if num == 1:
        km = KMeans(n_clusters=1, n_init=10, random_state=random_state)
        labels = km.fit_predict(embeddings)
        return labels, km
    if auto_k:
        # Candidate k range with overrides
        k_lo = 2 if k_min is None else max(2, k_min)
        k_hi_default = max(3, min(20, num // 3))
        k_hi = k_hi_default if k_max is None else min(k_max, k_hi_default)
        best_score = -1e9
        best = None
        
        # Create progress bar for k-means clustering
        pbar = tqdm(range(k_lo, k_hi + 1), desc="Finding optimal clusters", 
                   unit="k", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        
        for cand in pbar:
            try:
                pbar.set_postfix({'k': cand, 'best_score': f'{best_score:.3f}'})
                km = KMeans(n_clusters=cand, n_init=20, max_iter=500, random_state=random_state)
                lab = km.fit_predict(embeddings)
                sil = silhouette_score(embeddings, lab, metric='euclidean')
                # Penalize small clusters proportion
                sizes = np.array([np.sum(lab == i) for i in range(cand)])
                small_frac = float(np.sum(sizes < min_cluster_size)) / float(cand)
                score = sil - small_cluster_penalty * small_frac
                if score > best_score:
                    best_score = score
                    best = (cand, km, lab)
            except Exception:
                continue
        
        pbar.close()
        
        if best is None:
            k = 2
            km = KMeans(n_clusters=k, n_init=20, random_state=random_state)
            labels = km.fit_predict(embeddings)
        else:
            k, km, labels = best
    else:
        km = KMeans(n_clusters=max(1, k), n_init=50, max_iter=500, random_state=random_state)
        labels = km.fit_predict(embeddings)

    # Merge clusters that are too small
    labels = _merge_small_clusters(labels, km.cluster_centers_, min_cluster_size)
    return labels, km


def _compute_cluster_centers_from_labels(
    embeddings: np.ndarray,
    labels: np.ndarray
) -> np.ndarray:
    """Compute cluster centers in the original embedding space given labels.

    This is useful when clustering is performed in a reduced space (e.g., PCA),
    but we want centers in the original space for downstream tasks.
    """
    if labels.size == 0:
        return np.empty((0, embeddings.shape[1]))
    unique = sorted(set(labels.tolist()))
    centers = []
    for cid in unique:
        idxs = np.where(labels == cid)[0]
        if len(idxs) == 0:
            continue
        centers.append(np.mean(embeddings[idxs], axis=0))
    return np.vstack(centers) if len(centers) else np.empty((0, embeddings.shape[1]))


def robust_kmeans_cluster(
    embeddings: np.ndarray,
    auto_k: bool = True,
    min_cluster_size: int = 8,
    trials: int = 5,
    random_state: int = 42,
) -> Tuple[np.ndarray, KMeans | None]:
    """Run PCA-assisted K-means multiple times and pick the best clustering.

    - Uses PCA (up to 32 dims) to stabilize k selection and reduce noise.
    - Runs several trials (different PCA initializations) and selects the best
      by silhouette (on original space) with small-cluster penalty.
    - Returns labels and a KMeans-like object whose centers are in original space.
    """
    num = embeddings.shape[0]
    if num == 0:
        return np.array([], dtype=int), None

    # Prepare PCA-reduced representation (trial-dependent via random_state offset)
    def pca_reduce(x: np.ndarray, seed: int) -> np.ndarray:
        if x.shape[0] <= 2 or x.shape[1] <= 2:
            return x
        # Adaptive PCA components: cap by samples/features, with a softer upper bound
        max_allowed = max(1, min(x.shape[0], x.shape[1]) - 1)
        n_comp = min(48, max_allowed)  # allow up to 48 if feasible
        if n_comp <= 1:
            return x
        try:
            Xr = PCA(n_components=n_comp, random_state=seed).fit_transform(x)
            return Xr
        except Exception:
            return x

    best = None
    best_score = -1e9

    for t in range(max(1, trials)):
        seed = (random_state + t) if random_state is not None else None
        Xr = pca_reduce(embeddings, seed)

        # Cluster in reduced space for stability
        labels_r, _ = kmeans_cluster(
            Xr, auto_k=auto_k, random_state=seed, min_cluster_size=min_cluster_size
        )

        # If clustering failed or empty, skip
        if labels_r.size == 0:
            continue

        # Evaluate on original embeddings
        try:
            sil = silhouette_score(embeddings, labels_r, metric='euclidean') if len(set(labels_r.tolist())) > 1 else -1.0
        except Exception:
            sil = -1.0

        # Penalize tiny clusters proportion (same philosophy as kmeans_cluster)
        sizes = np.array([np.sum(labels_r == i) for i in sorted(set(labels_r.tolist()))])
        small_frac = float(np.sum(sizes < min_cluster_size)) / float(len(sizes)) if len(sizes) else 1.0
        score = sil - 0.15 * small_frac

        if score > best_score:
            best_score = score
            best = labels_r

    if best is None:
        # Fallback to single pass on original space
        return kmeans_cluster(embeddings, auto_k=auto_k, min_cluster_size=min_cluster_size)

    # Build a KMeans-like object exposing cluster_centers_ in original space
    class _MockKMeans:
        def __init__(self, centers: np.ndarray):
            self.cluster_centers_ = centers

    # Initial centers and labels in original space
    centers_orig = _compute_cluster_centers_from_labels(embeddings, best)
    km_mock = _MockKMeans(centers_orig)

    # Final small-cluster merge in original space
    labels_step1 = _merge_small_clusters(best, km_mock.cluster_centers_, min_cluster_size)
    centers_step1 = _compute_cluster_centers_from_labels(embeddings, labels_step1)

    # Outlier-trim refinement: remove farthest 10% in each cluster, recompute centers, then reassign once
    trimmed_labels = labels_step1.copy()
    unique = sorted(set(trimmed_labels.tolist()))
    for cid in unique:
        idxs = np.where(trimmed_labels == cid)[0]
        if len(idxs) < max(10, int(1.0 / 0.1)):
            continue
        center = centers_step1[cid]
        d = np.linalg.norm(embeddings[idxs] - center, axis=1)
        k = max(1, int(0.9 * len(idxs)))  # keep closest 90%
        keep_idx = idxs[np.argsort(d)[:k]]
        # Temporarily mark removed as -1 (will be reassigned)
        removed_idx = set(idxs) - set(keep_idx.tolist())
        for rid in removed_idx:
            trimmed_labels[rid] = -1

    # Recompute centers on kept points only
    centers_refined = []
    map_old_to_new = {}
    new_id = 0
    for cid in unique:
        kept = np.where(trimmed_labels == cid)[0]
        if len(kept) == 0:
            continue
        centers_refined.append(np.mean(embeddings[kept], axis=0))
        map_old_to_new[cid] = new_id
        new_id += 1
    centers_refined = np.vstack(centers_refined) if len(centers_refined) else centers_step1

    # Reassign: any -1 or existing points go to nearest refined center
    if centers_refined.size:
        dmat = np.linalg.norm(embeddings[:, None, :] - centers_refined[None, :, :], axis=2)
        reassigned = np.argmin(dmat, axis=1)
        final_labels = reassigned.astype(int)
        centers_final = centers_refined
    else:
        final_labels = labels_step1
        centers_final = centers_step1

    km_mock = _MockKMeans(centers_final)
    return final_labels, km_mock


def select_topk_per_cluster_dynamic(
    embeddings: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    desired_total: int,
) -> List[Tuple[int, int]]:
    selected: List[Tuple[int, int]] = []
    if desired_total <= 0 or labels.size == 0:
        return selected
    unique = sorted(set(labels.tolist()))
    sizes = {cid: int(np.sum(labels == cid)) for cid in unique}
    total = sum(sizes.values())
    desired_total = max(1, min(desired_total, total))
    # proportional allocation
    alloc = {cid: 0 for cid in unique}
    remainders = []
    assigned = 0
    for cid in unique:
        proportion = sizes[cid] / total if total > 0 else 0
        raw = proportion * desired_total
        base = int(np.floor(raw))
        base = max(1, min(base, sizes[cid])) if sizes[cid] > 0 else 0
        alloc[cid] = base
        assigned += base
        remainders.append((raw - base, cid))
    # distribute remainder
    remain = max(0, desired_total - assigned)
    for _, cid in sorted(remainders, key=lambda x: x[0], reverse=True):
        if remain <= 0:
            break
        if alloc[cid] < sizes[cid]:
            alloc[cid] += 1
            remain -= 1
    # select by distance to center
    for cid in unique:
        idxs = np.where(labels == cid)[0]
        if len(idxs) == 0:
            continue
        center = centers[cid]
        dists = np.linalg.norm(embeddings[idxs] - center, axis=1)
        count = alloc.get(cid, 0)
        order = np.argsort(dists)[:count]
        for j in order:
            selected.append((cid, int(idxs[j])))
    return selected

