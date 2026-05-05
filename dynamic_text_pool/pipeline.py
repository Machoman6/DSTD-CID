"""
Main pipeline functions for dynamic text pool system.
"""

import json
import os
import time
from pathlib import Path
import sys
import shutil
import sys
from typing import Any, Dict, List
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import deque

from clustering import robust_kmeans_cluster, select_topk_per_cluster_dynamic
from embeddings import compute_embeddings, load_model
from export import export_all_pools_to_csv
from scoring import score_candidates, update_pool
from utils import print_gpu_status
from visualization import save_cluster_plot


def load_prior_map_from_csv(csv_path: Path) -> Dict[str, str]:
    """Load mapping from event file stem -> prior text from a CSV with headers.
    Expected columns: 'name' (event filename, with or without .csv) and 'content' (prior text).
    """
    mapping: Dict[str, str] = {}
    if not csv_path.exists() or not csv_path.is_file():
        return mapping
    try:
        # Try utf-8 first, fallback to gbk
        try:
            df = pd.read_csv(csv_path.as_posix(), header=0, encoding='utf-8')
        except Exception:
            df = pd.read_csv(csv_path.as_posix(), header=0, encoding='gbk')

        expected_cols = {"name", "content"}
        if not expected_cols.issubset(set(df.columns.astype(str))):
            print(f"[WARN] Prior map CSV missing required columns {expected_cols}: {csv_path}")
            return mapping

        for _, row in df.iterrows():
            name = str(row["name"]).strip()
            text = str(row["content"]).strip()
            if not name or not text:
                continue
            stem = Path(name).stem
            mapping[stem] = text
    except Exception as e:
        print(f"[WARN] Failed to load prior map CSV {csv_path}: {e}")
    return mapping

def load_prior_texts_for_event(csv_path: Path) -> List[str] | None:
    """Load event prior text from a sidecar .txt with same stem as the CSV.
    Returns a single-element list [text] if found, otherwise None.
    """
    txt_path = csv_path.with_suffix('.txt')
    if txt_path.exists() and txt_path.is_file():
        try:
            # Try utf-8 first, fallback to gbk
            try:
                content = txt_path.read_text(encoding='utf-8').strip()
            except Exception:
                content = txt_path.read_text(encoding='gbk').strip()
            return [content] if content else None
        except Exception as e:
            print(f"[WARN] Failed to read prior text file: {txt_path} -> {e}")
            return None
    return None


def process_one_event(
    hour_csv: str,
    text_col: str,
    time_col: str,
    model_path: str,
    pool_max: int,
    ratio_candidates: float,
    event_prior_texts: List[str] | None,
    base_outputs_root: Path,
    base_cluster_root: Path,
) -> None:
    """Process a single event CSV and save outputs under a folder named by CSV stem."""
    print(f"\n=== Processing event CSV: {hour_csv} ===")
    # Try UTF-8 first, fallback to GBK for robustness
    try:
        df = pd.read_csv(hour_csv, encoding='utf-8')
    except Exception:
        df = pd.read_csv(hour_csv, encoding='gbk')
    # Robust time column detection and parsing
    detected_time_col = None
    for cand in [time_col, 'timestamp', 'time', 'create_time', 'pub_time', 'date', 'datetime']:
        if cand in df.columns:
            detected_time_col = cand
            break
    if detected_time_col is None:
        print(f"[ERROR] No time column found in {hour_csv}. Tried: {time_col}, timestamp, time, create_time, pub_time, date, datetime")
        sys.exit(1)
    # Parse datetime with mixed formats and coerce errors
    try:
        df[detected_time_col] = pd.to_datetime(df[detected_time_col], errors='coerce', format='mixed')
    except Exception:
        df[detected_time_col] = pd.to_datetime(df[detected_time_col], errors='coerce')
    before = len(df)
    df = df.dropna(subset=[detected_time_col])
    after = len(df)
    if after < before:
        print(f"[INFO] Dropped {before - after} rows with unparsable datetime from column '{detected_time_col}'")
    df = df.sort_values(detected_time_col)
    df['hour_key'] = df[detected_time_col].dt.floor('H')

    model = load_model(model_path)
    print_gpu_status()

    event_name = Path(hour_csv).stem
    outputs_dir = base_outputs_root / event_name
    outputs_dir.mkdir(parents=True, exist_ok=True)
    cluster_plots_dir = base_cluster_root / event_name
    cluster_plots_dir.mkdir(parents=True, exist_ok=True)

    # rolling pool and dynamic sizing params
    rolling_pool: List[Dict[str, Any]] = []
    recent_counts = deque(maxlen=3)
    base = 60
    alpha = 8.0
    min_pool = 10
    pool_max_cap = pool_max

    hour_groups = list(df.groupby('hour_key'))
    total_hours = len(hour_groups)
    print(f"Processing {total_hours} hours for event: {event_name}")

    hour_pbar = tqdm(hour_groups, desc=f"Processing hours[{event_name}]", unit="hour",
                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

    for hour_idx, (hour, g) in enumerate(hour_pbar, start=1):
        hour_pbar.set_postfix({
            'hour': f'{hour_idx}/{total_hours}',
            'texts': len(g),
            'pool_size': len(rolling_pool)
        })

        texts = g[text_col].fillna("").astype(str).tolist()
        times = g[time_col].tolist()

        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        start_time = time.time()
        emb = compute_embeddings(model, texts, desc=f"Hour {hour_idx} embeddings")
        print(f"Computed embeddings: {emb.shape} in {time.time() - start_time:.2f}s")

        start_time = time.time()
        labels, km = robust_kmeans_cluster(emb, auto_k=True, min_cluster_size=8, trials=5)
        print(f"Clustering completed (robust) in {time.time() - start_time:.2f}s")

        desired_total = max(10, int(ratio_candidates * len(texts)))
        desired_total = min(desired_total, len(texts))
        selected = select_topk_per_cluster_dynamic(emb, labels, km.cluster_centers_, desired_total)
        print(f"Selected {len(selected)} candidates from {len(set(labels))} clusters")

        pool_emb = np.empty((0, emb.shape[1]))
        if len(rolling_pool):
            pool_texts = [x['text'] for x in rolling_pool]
            start_time = time.time()
            pool_emb = compute_embeddings(model, pool_texts, desc=f"Hour {hour_idx} pool embeddings")
            print(f"Pool embeddings computed in {time.time() - start_time:.2f}s")

        event_prior_emb = None
        if event_prior_texts and len(event_prior_texts) > 0:
            start_time = time.time()
            event_prior_emb = compute_embeddings(model, event_prior_texts, desc=f"Hour {hour_idx} event prior embeddings")
            print(f"Event prior embeddings computed in {time.time() - start_time:.2f}s")

        start_time = time.time()
        scored = score_candidates(selected, emb, texts, times, pool_emb, event_prior_emb)
        print(f"Scoring completed in {time.time() - start_time:.2f}s")

        recent_counts.append(len(texts))
        N_eff = float(np.mean(recent_counts)) if len(recent_counts) > 0 else float(len(texts))
        target_pool = int(base + alpha * np.sqrt(max(N_eff, 0.0)))
        target_pool = max(min_pool, min(pool_max_cap, target_pool))

        rolling_pool = update_pool(scored, texts, times, rolling_pool, emb, target_pool)

        tag = pd.to_datetime(hour).strftime('%Y%m%d_%H00')
        hour_dir = outputs_dir / f"hour_{hour_idx}"
        hour_dir.mkdir(parents=True, exist_ok=True)
        # Only persist per-hour pool JSON (needed for CSV export); skip candidates JSON
        with open((hour_dir / f"pool_{tag}.json").as_posix(), 'w', encoding='utf-8') as f:
            json.dump(rolling_pool, f, ensure_ascii=False, indent=2)

        cluster_plot_path = cluster_plots_dir / f"hour_{hour_idx:03d}_{tag}_clusters.png"
        save_cluster_plot(emb, labels, km, str(cluster_plot_path), highlight_indices=[idx for _, idx in selected])
        print(f"Cluster plot saved: {cluster_plot_path}")

    hour_pbar.close()
    # Export aggregated CSV for this event
    export_all_pools_to_csv(outputs_dir, outputs_dir / 'all_pools.csv')
    # Cleanup per-hour JSON snapshots to keep only the CSV
    hour_dirs = [d for d in outputs_dir.iterdir() if d.is_dir() and d.name.startswith("hour_")]
    for d in hour_dirs:
        try:
            shutil.rmtree(d)
        except Exception as e:
            print(f"[WARN] Failed to remove temp dir {d}: {e}")
    print(f"✅ Event completed: {event_name} | CSV: {(outputs_dir / 'all_pools.csv').as_posix()}")

def run_update(
    hour_csv: str,
    text_col: str,
    time_col: str,
    model_path: str,
    pool_in: str | None,
    pool_out: str,
    cand_out: str,
    pool_max: int,
    top_k_per_cluster: int = 3,
    event_prior_texts: List[str] = None,
) -> None:
    print("Loading data...")
    df = pd.read_csv(hour_csv, encoding='gbk')
    df[time_col] = pd.to_datetime(df[time_col])
    texts = df[text_col].fillna("").astype(str).tolist()
    times = df[time_col].tolist()
    print(f"Loaded {len(texts)} texts")
    
    # Time decay is now handled in update_pool function

    model = load_model(model_path)
    emb = compute_embeddings(model, texts, desc="Computing embeddings")

    labels, km = robust_kmeans_cluster(emb, auto_k=True)
    selected = select_topk_per_cluster_dynamic(emb, labels, km.cluster_centers_, top_k_per_cluster)

    # Compute event prior embeddings if provided
    event_prior_emb = None
    if event_prior_texts and len(event_prior_texts) > 0:
        event_prior_emb = compute_embeddings(model, event_prior_texts, desc="Computing event prior embeddings")

    old_pool: List[Dict[str, Any]] = []
    pool_emb = np.empty((0, emb.shape[1]))
    if pool_in and os.path.exists(pool_in):
        with open(pool_in, "r", encoding="utf-8") as f:
            old_pool = json.load(f)
        if len(old_pool):
            pool_texts = [x["text"] for x in old_pool]
            pool_emb = compute_embeddings(model, pool_texts, desc="Computing pool embeddings")

    scored = score_candidates(selected, emb, texts, times, pool_emb, event_prior_emb)

    # Save candidates
    print("Saving results...")
    cand_rows = [{"idx": idx, "text": texts[idx], "timestamp": str(times[idx]), "score": score} for idx, score in scored]
    with open(cand_out, "w", encoding="utf-8") as f:
        json.dump(cand_rows, f, ensure_ascii=False, indent=2)

    # Dynamic pool sizing (Scheme A) for single-run path
    N_eff = float(len(texts))
    base = 60
    alpha = 8.0
    min_pool = 80
    pool_max_cap = pool_max
    target_pool = int(base + alpha * np.sqrt(max(N_eff, 0.0)))
    target_pool = max(min_pool, min(pool_max_cap, target_pool))

    new_pool = update_pool(scored, texts, times, old_pool, emb, target_pool)
    with open(pool_out, "w", encoding="utf-8") as f:
        json.dump(new_pool, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Update completed! Selected {len(scored)} candidates, pool size: {len(new_pool)}")


def main():
    # === Editable parameters ===
    # Relative paths (relative to project root)
    hour_csv = r"../cyberbullying_dataset/events"
    text_col = "content"
    time_col = "timestamp"
    model_path = r"~/.cache/modelscope/hub/models/Qwen/Qwen3-Embedding-0___6B"
    pool_in = None  # e.g., "../pool.json"
    pool_out = r"../pool.json"
    cand_out = r"../candidates.json"
    pool_max = 200
    # Optional: mapping CSV for per-event prior texts (first column: filename, second: text)
    prior_map_csv = r"../event_summary.csv"  # e.g., "../event_priors.csv"
    # dynamic selection controls
    ratio_candidates = 0.3  # 10% of hour comments as candidates

    # Global event_prior_texts removed: per-event priors are loaded via CSV map or sidecar .txt

    # If hour_csv is a directory, process each CSV inside into its own folder; otherwise process single CSV
    input_path = Path(hour_csv)
    base_outputs_root = Path("../hourly_outputs_0_0_1")
    base_outputs_root.mkdir(parents=True, exist_ok=True)
    base_cluster_root = Path("../cluster_plots_0_0_1")
    base_cluster_root.mkdir(parents=True, exist_ok=True)

    # Load optional prior mapping
    prior_map: Dict[str, str] = {}
    if prior_map_csv:
        prior_map = load_prior_map_from_csv(Path(prior_map_csv))

    if input_path.is_dir():
        csv_files = sorted([p for p in input_path.glob("*.csv") if p.is_file()])
        print(f"Found {len(csv_files)} CSV files in directory: {input_path}")
        for csv_path in csv_files:
            # Select per-event prior: use mapping CSV only; missing is fatal
            stem = csv_path.stem
            # Skip if already completed (all_pools.csv exists)
            event_outputs_dir = base_outputs_root / stem
            completed_csv = event_outputs_dir / 'all_pools.csv'
            if completed_csv.exists():
                print(f"[SKIP] Event already processed: {stem} -> {completed_csv}")
                continue
            if stem not in prior_map:
                print(f"[ERROR] Missing prior text for event: {stem} (from {csv_path}). Exiting.")
                sys.exit(1)
            per_event_priors = [prior_map[stem]]
            process_one_event(
                hour_csv=csv_path.as_posix(),
                text_col=text_col,
                time_col=time_col,
                model_path=model_path,
                pool_max=pool_max,
                ratio_candidates=ratio_candidates,
                event_prior_texts=per_event_priors,
                base_outputs_root=base_outputs_root,
                base_cluster_root=base_cluster_root,
            )
        return
    else:
        # Process the single CSV using the same per-event routine
        single_csv = input_path
        stem = single_csv.stem
        # Skip if already completed (all_pools.csv exists)
        event_outputs_dir = base_outputs_root / stem
        completed_csv = event_outputs_dir / 'all_pools.csv'
        if completed_csv.exists():
            print(f"[SKIP] Event already processed: {stem} -> {completed_csv}")
            return
        if stem not in prior_map:
            print(f"[ERROR] Missing prior text for event: {stem} (from {single_csv}). Exiting.")
            sys.exit(1)
        per_event_priors = [prior_map[stem]]
        process_one_event(
            hour_csv=input_path.as_posix(),
            text_col=text_col,
            time_col=time_col,
            model_path=model_path,
            pool_max=pool_max,
            ratio_candidates=ratio_candidates,
            event_prior_texts=per_event_priors,
            base_outputs_root=base_outputs_root,
            base_cluster_root=base_cluster_root,
        )


if __name__ == "__main__":
    main()
