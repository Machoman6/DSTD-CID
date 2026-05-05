#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build an event-level split dataset for early cyberbullying prediction.
Pipeline:
- Enumerate events under hourly_outputs/<event_name>/all_pools.csv
- Split events (train/val/test) stratified by event_label
- Fit TF-IDF on train texts only, reduce with SVD to dense embeddings
- For each event, per hour h, build:
  - hour-level pooled text embedding using score-weighted average over all texts in that hour
  - text time series up to h (padded later in collate)
  - numeric time series (malicious_history, total_history) up to h
  - label (event_label), current hour h
Outputs: in-memory python dicts; consumers may serialize if needed
"""

from __future__ import annotations

import json
import pickle
import random
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


@dataclass
class DatasetSplits:
    train_events: List[str]
    val_events: List[str]
    test_events: List[str]


def get_cache_path(outputs_dir: str, seed: int, max_hours: int, embed_dim: int, 
                  use_sentence_embedder: bool, unified_length: int, use_bert_embedder: bool = False,
                  include_hourly_text_series: bool = False) -> Path:
    """Generate cache file path"""
    # Create parameter hash
    params = {
        "outputs_dir": outputs_dir,
        "seed": seed,
        "max_hours": max_hours,
        "embed_dim": embed_dim,
        "use_sentence_embedder": use_sentence_embedder,
        "unified_length": unified_length,
        "use_bert_embedder": use_bert_embedder,
        "include_hourly_text_series": include_hourly_text_series,
    }
    params_str = json.dumps(params, sort_keys=True)
    params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]
    
    cache_dir = Path("./dataset_cache")
    cache_dir.mkdir(exist_ok=True)
    return cache_dir / f"dataset_{params_hash}.pkl"


def save_dataset_cache(cache_path: Path, train_samples: List[Dict], val_samples: List[Dict], 
                      test_samples: List[Dict], embed_dim: int):
    """Save dataset to cache"""
    cache_data = {
        "train_samples": train_samples,
        "val_samples": val_samples,
        "test_samples": test_samples,
        "embed_dim": embed_dim
    }
    with open(cache_path, 'wb') as f:
        pickle.dump(cache_data, f)
    print(f"[early_dataset] Dataset cached to: {cache_path}")


def load_dataset_cache(cache_path: Path) -> Tuple[List[Dict], List[Dict], List[Dict], int]:
    """Load dataset from cache"""
    with open(cache_path, 'rb') as f:
        cache_data = pickle.load(f)
    print(f"[early_dataset] Dataset loaded from cache: {cache_path}")
    return (cache_data["train_samples"], cache_data["val_samples"], 
            cache_data["test_samples"], cache_data["embed_dim"])


def list_events(outputs_dir: Path) -> List[Path]:
    return [d for d in outputs_dir.iterdir() if d.is_dir() and (d / "all_pools.csv").exists()]


def read_event_label(all_pools_csv: Path) -> int:
    df = pd.read_csv(all_pools_csv, nrows=1)
    if "event_label" not in df.columns:
        raise RuntimeError(f"event_label column missing in {all_pools_csv}")
    return int(df["event_label"].iloc[0])


def stratified_event_split(events: List[Path], seed: int = 42, train_ratio: float = 0.8, val_ratio: float = 0.1) -> DatasetSplits:
    pos, neg = [], []
    for e in events:
        y = read_event_label(e / "all_pools.csv")
        (pos if y == 1 else neg).append(e.name)

    rnd = random.Random(seed)
    rnd.shuffle(pos)
    rnd.shuffle(neg)

    def split(lst: List[str]) -> Tuple[List[str], List[str], List[str]]:
        n = len(lst)
        n_tr = int(n * train_ratio)
        n_va = int(n * val_ratio)
        return lst[:n_tr], lst[n_tr : n_tr + n_va], lst[n_tr + n_va :]

    p_tr, p_va, p_te = split(pos)
    n_tr, n_va, n_te = split(neg)

    return DatasetSplits(
        train_events=p_tr + n_tr,
        val_events=p_va + n_va,
        test_events=p_te + n_te,
    )


def stratified_k_fold_split(events: List[Path], n_folds: int = 5, seed: int = 42) -> List[Tuple[List[str], List[str]]]:
    """
    Stratified K-Fold cross-validation split at event level.
    Returns list of (train_events, test_events) tuples, one for each fold.
    
    Args:
        events: List of event paths
        n_folds: Number of folds
        seed: Random seed for reproducibility
    
    Returns:
        List of (train_events, test_events) tuples for each fold
    """
    pos, neg = [], []
    for e in events:
        y = read_event_label(e / "all_pools.csv")
        (pos if y == 1 else neg).append(e.name)
    
    rnd = random.Random(seed)
    rnd.shuffle(pos)
    rnd.shuffle(neg)
    
    folds = []
    pos_n = len(pos)
    neg_n = len(neg)
    
    # Calculate fold size for each class
    pos_fold_size = pos_n // n_folds
    neg_fold_size = neg_n // n_folds
    
    for i in range(n_folds):
        # Calculate fold boundaries
        pos_start = i * pos_fold_size
        pos_end = (i + 1) * pos_fold_size if i < n_folds - 1 else pos_n
        neg_start = i * neg_fold_size
        neg_end = (i + 1) * neg_fold_size if i < n_folds - 1 else neg_n
        
        # Test set for this fold
        pos_test = pos[pos_start:pos_end]
        neg_test = neg[neg_start:neg_end]
        test_events = pos_test + neg_test
        
        # Train set for this fold (all other events)
        train_events = pos[:pos_start] + pos[pos_end:] + neg[:neg_start] + neg[neg_end:]
        
        folds.append((train_events, test_events))
    
    return folds


def softmax(x: np.ndarray, beta: float = 1.0) -> np.ndarray:
    x = x - np.max(x)
    ex = np.exp(beta * x)
    s = ex.sum()
    return ex / (s + 1e-8)


class TextEmbedder:
    """Lightweight text embedding: TF-IDF -> SVD dense.
    Fitted on train texts only to avoid leakage.
    """

    def __init__(self, dim: int = 256, max_features: int = 50000):
        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=2)
        self.svd = TruncatedSVD(n_components=dim, random_state=42)
        self.dim = dim

    def fit(self, texts: List[str]):
        X = self.vectorizer.fit_transform(texts)
        self.svd.fit(X)

    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        X = self.vectorizer.transform(texts)
        Z = self.svd.transform(X)
        Z = Z.astype(np.float32)
        # L2 normalize
        norms = np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8
        Z = Z / norms
        return Z


class SentenceEmbedder:
    """SentenceTransformer-based embedder (e.g., Qwen3-Embedding)."""

    def __init__(self, model_path: str, device: str | None = None, batch_size: int = 64):
        print(f"[early_dataset] Loading sentence model from: {model_path} (device={device})")
        self.model = SentenceTransformer(model_path, device=device)
        self.batch_size = batch_size
        self.dim = int(self.model.get_sentence_embedding_dimension())
        print(f"[early_dataset] Sentence model loaded. embedding_dim={self.dim}")

    def fit(self, texts: List[str]):
        # No-op for sentence models
        return

    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        emb = self.model.encode(texts, batch_size=self.batch_size, show_progress_bar=False, normalize_embeddings=True)
        emb = emb.astype(np.float32)
        return emb
class BertEmbedder:
    def __init__(self, model_name: str = "hfl/chinese-bert-wwm-ext", device: str | None = None, batch_size: int = 32):
        # Try alternative models to alleviate network/mirror issues
        model_candidates = [model_name, "bert-base-chinese", "hfl/chinese-roberta-wwm-ext"]
        last_err = None
        self.tokenizer = None
        self.model = None
        for name in model_candidates:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(name)
                self.model = AutoModel.from_pretrained(name)
                break
            except Exception as e:
                last_err = e
                continue
        if self.model is None or self.tokenizer is None:
            raise RuntimeError(f"Failed to load any BERT model from candidates {model_candidates}: {last_err}")
        self.model.eval()
        self.batch_size = batch_size
        self.device = torch.device(device) if device else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.model.to(self.device)
        self.dim = int(self.model.config.hidden_size)

    def fit(self, texts: List[str]):
        return

    @torch.no_grad()
    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        embs: List[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i+self.batch_size]
            toks = self.tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(self.device)
            outputs = self.model(**toks)
            cls = outputs.last_hidden_state[:, 0, :]  # (B, dim)
            cls = torch.nn.functional.normalize(cls, p=2, dim=-1)
            embs.append(cls.detach().cpu().numpy().astype(np.float32))
        return np.concatenate(embs, axis=0) if embs else np.zeros((0, self.dim), dtype=np.float32)



def build_hour_vector(texts: List[str], scores: List[float], embedder: TextEmbedder, beta: float = 1.2) -> np.ndarray:
    if len(texts) == 0:
        return np.zeros((embedder.dim,), dtype=np.float32)
    E = embedder.encode(texts)  # (M, d)
    s = np.clip(np.asarray(scores, dtype=np.float32), 0.0, None)
    if s.sum() <= 1e-8:
        w = np.ones_like(s) / max(len(s), 1)
    else:
        w = softmax(s, beta=beta)
    h = (w[:, None] * E).sum(axis=0)
    h = h / (np.linalg.norm(h) + 1e-8)
    return h.astype(np.float32)


def parse_seq(seq_str: str) -> List[float]:
    try:
        return list(json.loads(seq_str))
    except Exception:
        # fallback for python list string
        return list(eval(seq_str))  # nosec - controlled data files


def collect_train_texts(outputs_dir: Path, event_names: List[str]) -> List[str]:
    texts: List[str] = []
    for name in event_names:
        df = pd.read_csv(outputs_dir / name / "all_pools.csv")
        if "text" in df.columns:
            texts.extend(df["text"].fillna("").astype(str).tolist())
    return texts


def build_samples_for_event(outputs_dir: Path, event_name: str, embedder: TextEmbedder, 
                           max_hours: int = 72, unified_length: int = None, 
                           max_samples_per_event: int = None,
                           include_hourly_text_series: bool = False) -> List[Dict]:
    df = pd.read_csv(outputs_dir / event_name / "all_pools.csv")
    label = int(df["event_label"].iloc[0])
    samples: List[Dict] = []

    # Precompute hour vectors
    hour_vectors: Dict[int, np.ndarray] = {}
    for h, g in df.groupby("hour_idx"):
        if max_hours is not None and int(h) > max_hours:
            continue
        texts = g["text"].fillna("").astype(str).tolist()
        scores = g["score"].astype(float).tolist() if "score" in g.columns else [1.0] * len(texts)
        hour_vectors[int(h)] = build_hour_vector(texts, scores, embedder)

    hours = sorted(hour_vectors.keys())
    if not hours:
        return samples
    
    # Determine actual maximum length
    actual_max_hour = max(hours)
    # If unified length is set, use it; otherwise use actual maximum length
    target_length = unified_length if unified_length is not None else actual_max_hour
    
    # Unified sampling strategy: use max_hours as target sample count
    # If max_hours is None, use actual maximum length; otherwise use max_hours (allows padding virtual hours)
    target_samples = max_hours if max_hours is not None else actual_max_hour

    if len(hours) >= target_samples:
        # Long events: directly take first target_samples hours
        selected_hours = hours[:target_samples]
    else:
        # Short events: keep real hours, then append "virtual hours" until target_samples
        selected_hours = list(hours)
        pad_needed = target_samples - len(hours)
        if pad_needed > 0:
            last_h = hours[-1]
            selected_hours.extend([last_h + i for i in range(1, pad_needed + 1)])

    # Pre-fetch last real hour's history to support virtual hour extension
    last_real_h = hours[-1]
    last_row = df[df["hour_idx"] == last_real_h].iloc[0]
    last_mh_full = parse_seq(last_row["hourly_malicious_history"]) if "hourly_malicious_history" in last_row else []
    last_th_full = parse_seq(last_row["hourly_total_history"]) if "hourly_total_history" in last_row else []

    def pad_zero(values: List[float], target_len: int) -> List[float]:
        if len(values) >= target_len:
            return values[:target_len]
        if len(values) == 0:
            return [0.0] * target_len
        return values + [0.0] * (target_len - len(values))

    for h in selected_hours:
        L = int(h)
        # Text: use corresponding vector for real hours; use zero vector for virtual hours
        if h in hour_vectors:
            X_text = hour_vectors[h].reshape(1, -1).astype(np.float32)
        else:
            X_text = np.zeros((1, embedder.dim), dtype=np.float32)

        text_series = None
        if include_hourly_text_series:
            seq_vecs = []
            zero_vec = np.zeros((embedder.dim,), dtype=np.float32)
            for t in range(1, int(h) + 1):
                vec = hour_vectors.get(t)
                if vec is None:
                    seq_vecs.append(zero_vec.copy())
                else:
                    seq_vecs.append(vec.astype(np.float32))
            if seq_vecs:
                text_series = np.stack(seq_vecs, axis=0).astype(np.float32)
            else:
                text_series = np.zeros((0, embedder.dim), dtype=np.float32)

        # Numeric: use corresponding row's history for real hours; use last real history extended to L for virtual hours
        if h in hours:
            row = df[df["hour_idx"] == h].iloc[0]
            mh_full = parse_seq(row["hourly_malicious_history"]) if "hourly_malicious_history" in row else []
            th_full = parse_seq(row["hourly_total_history"]) if "hourly_total_history" in row else []
            mh = mh_full[:L]
            th = th_full[:L]
        else:
            mh = pad_zero(list(last_mh_full), L)
            th = pad_zero(list(last_th_full), L)
        
        # Do not use unified length padding: keep sequence length equal to current sample hour h
        # Dynamic alignment and pad_mask handling within batch by collate_fn

        sample = {
            "event": event_name,
            "hour": int(h),
            "label": label,
            "text_seq": X_text,           # (target_length, d)
            "mal_seq": np.asarray(mh, dtype=np.float32),  # (target_length,)
            "tot_seq": np.asarray(th, dtype=np.float32),  # (target_length,)
        }
        if include_hourly_text_series:
            sample["text_series"] = text_series
        samples.append(sample)

    return samples


def build_dataset(outputs_dir: str,
                  seed: int = 42,
                  max_hours: int = 72,
                  embed_dim: int = 256,
                  use_sentence_embedder: bool = False,
                  sentence_model_path: str = "~/.cache/modelscope/hub/models/Qwen/Qwen3-Embedding-0___6B",
                  device: str | None = None,
                  batch_size: int = 64,
                  unified_length: int = None,
                  max_samples_per_event: int = None,
                  use_cache: bool = True,
                  use_bert_embedder: bool = True,
                  bert_model_path: str | None = None,
                  include_hourly_text_series: bool = False,
                  ) -> Tuple[List[Dict], List[Dict], List[Dict], int]:
    
    # Check cache
    if use_cache:
        cache_path = get_cache_path(outputs_dir, seed, max_hours, embed_dim, use_sentence_embedder,
                                    unified_length, use_bert_embedder, include_hourly_text_series)
        if cache_path.exists():
            try:
                return load_dataset_cache(cache_path)
            except Exception as e:
                print(f"[early_dataset] Cache loading failed: {e}, rebuilding dataset...")
    
    # Build dataset
    out_dir = Path(outputs_dir)
    events = list_events(out_dir)
    print(f"[early_dataset] Found {len(events)} events under {outputs_dir}")

    splits = stratified_event_split(events, seed=seed)
    print(f"[early_dataset] Split -> train: {len(splits.train_events)}, val: {len(splits.val_events)}, test: {len(splits.test_events)}")

    # Create embedder
    if use_bert_embedder:
        model_name = bert_model_path if bert_model_path else "bert-base-chinese"
        embedder = BertEmbedder(model_name=model_name, device=device, batch_size=batch_size)
        embed_dim = embedder.dim
    elif use_sentence_embedder:
        embedder = SentenceEmbedder(sentence_model_path, device=device, batch_size=batch_size)
        embed_dim = embedder.dim
    else:
        embedder = TextEmbedder(dim=embed_dim)
        train_texts = collect_train_texts(out_dir, splits.train_events)
        embedder.fit(train_texts)

    def build_for(names: List[str]) -> List[Dict]:
        res: List[Dict] = []
        for name in tqdm(names, desc="build_samples", unit="event"):
            res.extend(build_samples_for_event(
                out_dir,
                name,
                embedder,
                max_hours=max_hours,
                unified_length=unified_length,
                max_samples_per_event=max_samples_per_event,
                include_hourly_text_series=include_hourly_text_series
            ))
        return res

    train_samples = build_for(splits.train_events)
    val_samples = build_for(splits.val_events)
    test_samples = build_for(splits.test_events)

    # Save cache
    if use_cache:
        save_dataset_cache(cache_path, train_samples, val_samples, test_samples, embed_dim)

    return train_samples, val_samples, test_samples, embed_dim


if __name__ == "__main__":
    tr, va, te, dim = build_dataset("../hourly_outputs")
    print(f"embed_dim: {dim}, train: {len(tr)}, val: {len(va)}, test: {len(te)}")

