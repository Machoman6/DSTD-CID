#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train loop for early cyberbullying prediction (non-LLM):
- Build dataset with event-level split
- Pad batches; forward model; compute joint loss
- Evaluate Acc/F1, Early@K, Avg Time-to-Detection (TTD)
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple
import os
import sys
import logging

import numpy as np
import torch
import random
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt

# Make script runnable via right-click (python training/train_early.py)
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from training.early_dataset import build_dataset
from training.early_model import ECTSModel, ects_loss
from training.analysis_utils import (
    get_event_real_length,
    build_event_groups_from_details,
    derive_event_results_with_threshold,
    summarize_accuracy_earliness,
    scan_thresholds_and_plot,
)
def setup_logger() -> logging.Logger:
    log_dir = Path("./training_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "early_train.log"
    logger = logging.getLogger("early_train")
    logger.setLevel(logging.DEBUG)
    # Console handler (INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch_formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S")
    ch.setFormatter(ch_formatter)
    # File handler (DEBUG)
    fh = logging.FileHandler(log_path.as_posix(), encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh_formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    fh.setFormatter(fh_formatter)
    # Avoid duplicate handlers on re-run
    if not logger.handlers:
        logger.addHandler(ch)
        logger.addHandler(fh)
    logger.debug("Logger initialized. Logs -> %s", log_path)
    return logger



def pad_batch(samples: List[Dict], text_dim: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    # Dynamic batch processing: find the maximum length in the batch
    max_len = max(s["mal_seq"].shape[0] for s in samples)
    B = len(samples)
    
    text_seq = torch.zeros((B, 1, text_dim), dtype=torch.float32)  # Text sequence has only 1 time step
    num_seq = torch.zeros((B, max_len, 2), dtype=torch.float32)
    pad_mask = torch.ones((B, max_len), dtype=torch.bool)  # True for PAD positions
    labels = torch.tensor([s["label"] for s in samples], dtype=torch.float32)
    hours = torch.tensor([s["hour"] for s in samples], dtype=torch.int64)
    events = [s["event"] for s in samples]  # Keep event names

    for i, s in enumerate(samples):
        # Text sequence: only 1 time step
        text_seq[i, 0] = torch.from_numpy(s["text_seq"]).float().squeeze(0)
        
        # Numerical sequence: dynamic length
        mal = s["mal_seq"]
        tot = s["tot_seq"]
        actual_len = len(mal)
        num_seq[i, :actual_len, 0] = torch.from_numpy(mal)
        num_seq[i, :actual_len, 1] = torch.from_numpy(tot)
        
        # Set valid positions: from hour 1 to hour h
        h = s["hour"]
        if h > 0:
            pad_mask[i, :h] = False  # First h hours are valid
        else:
            # If h=0, all positions are invalid
            pad_mask[i, :] = True

    return text_seq, num_seq, pad_mask, labels, hours, events


def evaluate_ects(model: ECTSModel, loader: DataLoader, device: torch.device,
                  tau: float = 0.95) -> Dict[str, float]:
    """
    Explicit prefix expansion evaluation: each sample is predicted only at the last time step
    This reflects decisions based on complete historical information
    """
    model.eval()
    y_true, y_pred, stop_times = [], [], []
    # New: data collection for event-level evaluation
    event_data = {}  # {event_name: {'samples': [...], 'label': int}}
    # New: for saving detailed prediction results
    detailed_results = []  # Store detailed prediction information for each sample
    # New: collect stop probabilities for all samples for event-level stop time calculation
    all_samples_data = []  # Store complete information for all samples
    
    with torch.no_grad():
        for batch in loader:
            text_seq, num_seq, pad_mask, labels, hours, events = [x.to(device) if isinstance(x, torch.Tensor) else x for x in batch]
            class_logits, stop_logits = model(text_seq, num_seq, key_padding_mask=pad_mask)
            
            # Convert to probabilities
            class_probs = torch.sigmoid(class_logits)
            stop_probs = torch.sigmoid(stop_logits)
            
            # Explicit prefix expansion evaluation: predict only at the last valid time step
            valid = ~pad_mask
            batch_size = class_probs.size(0)
            
            for i in range(batch_size):
                # Explicit prefix expansion: make decisions based on complete historical information
                valid_mask = valid[i]
                if not valid_mask.any():
                    # If no valid time steps, skip
                    continue
                
                # Find the last valid time step (core of explicit prefix expansion)
                last_valid_time = valid_mask.sum().item() - 1
                
                # Predict only at the last time step (based on complete historical information)
                class_prob_last = class_probs[i, last_valid_time].item()
                stop_prob_last = stop_probs[i, last_valid_time].item()
                
                # Prediction: based on probability at the last time step
                pred = 1 if class_prob_last > 0.5 else 0
                
                # Stop time: when the model thinks it can determine the event type
                # Here we use the sample's hour field to indicate how many hours of historical information the model uses
                sample_hour = hours[i].item() if isinstance(hours[i], torch.Tensor) else hours[i]
                stop_time = sample_hour - 1  # Convert to 0-based index
                
                y_true.append(labels[i].item())
                y_pred.append(pred)
                stop_times.append(stop_time)
                
                # Collect event-level data
                event_id = events[i]  # Use real event name
                if event_id not in event_data:
                    event_data[event_id] = {'samples': [], 'label': labels[i].item()}
                event_data[event_id]['samples'].append({
                    'stop_time': stop_time,
                    'prediction': pred,
                    'hour': hours[i].item() if isinstance(hours[i], torch.Tensor) else hours[i]
                })
                
                # Collect complete information for all samples for event-level stop time calculation
                all_samples_data.append({
                    'event_id': event_id,
                    'sample_hour': sample_hour,
                    'true_label': labels[i].item(),
                    'predicted_label': pred,
                    'stop_time': stop_time,
                    'class_prob_at_stop': class_prob_last,
                    'stop_prob_at_stop': stop_prob_last,
                    'hour': hours[i].item() if isinstance(hours[i], torch.Tensor) else hours[i]
                })
                
                # Collect detailed prediction results (explicit prefix expansion: only record probabilities at the last time step)
                detailed_results.append({
                    'event_id': event_id,
                    'sample_hour': sample_hour,  # Hour corresponding to the sample
                    'true_label': labels[i].item(),
                    'predicted_label': pred,
                    'stop_time': stop_time,  # Time when model thinks it can determine event type
                    'class_prob_at_stop': class_prob_last,  # Classification probability at the last time step
                    'stop_prob_at_stop': stop_prob_last,    # Stop probability at the last time step
                    'all_class_probs': [class_prob_last],    # Only record probability at the last time step
                    'all_stop_probs': [stop_prob_last]      # Only record probability at the last time step
                })

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred,average='macro')
    
    # Add macro precision and recall
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    
    # ECTS metric: average stop time
    avg_stop_time = float(np.mean(stop_times)) if stop_times else 0.0
    
    
    # Calculate event-level stop times and corresponding predictions
    def calculate_event_stop_times_and_predictions(all_samples_data):
        """Calculate stop time and prediction for each event: find the first sample with stop probability > 0.5"""
        event_results = {}
        
        # Group by event
        event_groups = {}
        for sample in all_samples_data:
            event_id = sample['event_id']
            if event_id not in event_groups:
                event_groups[event_id] = []
            event_groups[event_id].append(sample)
        
        # For each event, find the first sample with stop probability > 0.5
        for event_id, samples in event_groups.items():
            # Sort by hour and find the first sample with stop probability > 0.5
            samples_sorted = sorted(samples, key=lambda x: x['hour'])
            
            # Get the real length of the event (read from CSV file)
            real_length = get_event_real_length(event_id)
            real_length = max(real_length, 1)
            
            stop_sample = None
            for sample in samples_sorted:
                if sample['stop_prob_at_stop'] > 0.5:
                    stop_sample = sample
                    break
            
            # If no sample with stop probability > 0.5 is found, use the last sample
            if stop_sample is None:
                stop_sample = samples_sorted[-1]

            # Limit stop time to not exceed the real event length
            clamped_stop_time = max(0, min(stop_sample['stop_time'], real_length - 1))
            clamped_sample_hour = max(1, min(stop_sample['sample_hour'], real_length))
            
            # Use the prediction label and stop time of the stopping sample
            event_results[event_id] = {
                'stop_time': clamped_stop_time,
                'predicted_label': stop_sample['predicted_label'],
                'true_label': stop_sample['true_label'],
                'sample_hour': clamped_sample_hour,  # Keep original field
                'real_length': real_length  # Use real length
            }
        
        return event_results
    
    # Calculate event-level stop times and predictions
    event_results = calculate_event_stop_times_and_predictions(all_samples_data)
    

    
    # Calculate event-level accuracy: prediction correctness at each event's stop time
    def calculate_event_level_accuracy(event_results):
        """Calculate event-level accuracy: based on prediction correctness of stopping samples"""
        if not event_results:
            return 0.0
            
        correct_events = 0
        total_events = len(event_results)
        
        for event_id, result in event_results.items():
            predicted_label = result['predicted_label']
            true_label = result['true_label']
            
            if predicted_label == true_label:
                correct_events += 1
        
        return (correct_events / total_events) if total_events else 0.0
    
    # Calculate event-level accuracy
    event_level_accuracy = calculate_event_level_accuracy(event_results)
    
    # Calculate event-level Precision, Recall, F1
    def calculate_event_level_metrics(event_results):
        """Calculate event-level Precision, Recall, F1"""
        if not event_results:
            return 0.0, 0.0, 0.0
        
        event_y_true = []
        event_y_pred = []
        
        for event_id, result in event_results.items():
            event_y_true.append(result['true_label'])
            event_y_pred.append(result['predicted_label'])
        
        event_precision = precision_score(event_y_true, event_y_pred, average='macro', zero_division=0)
        event_recall = recall_score(event_y_true, event_y_pred, average='macro', zero_division=0)
        event_f1 = f1_score(event_y_true, event_y_pred, average='macro', zero_division=0)
        
        return event_precision, event_recall, event_f1
    
    event_level_precision, event_level_recall, event_level_f1 = calculate_event_level_metrics(event_results)
    
    # Calculate event-level Earliness metric
    def calculate_earliness(event_results) -> float:
        """
        Calculate event-level Earliness metric:
        Earliness = (1/N) × Σ(ti/Ti)
        where ti is the stop time of event i, and Ti is the actual length of event i
        """
        if not event_results:
            return 0.0
        
        N = len(event_results)
        earliness_sum = 0.0
        
        for event_id, result in event_results.items():
            
            ti = result['stop_time']  # Stop time of event i
            Ti = result['real_length']  # Real length of event i
            print(event_id, ti,Ti)
            if Ti > 0:  # Avoid division by zero
                earliness_sum += (ti+1) / Ti
        
        earliness = earliness_sum / N
        return earliness
    
    # Calculate event-level Earliness metric
    earliness = calculate_earliness(event_results)
    
    # Calculate Harmonic Mean: combining event_acc and earliness
    def calculate_harmonic_mean(accuracy, earliness) -> float:
        """
        Calculate harmonic mean:
        H = 2 * (Accuracy * (1 - Earliness)) / (Accuracy + (1 - Earliness))
        """
        if accuracy == 0 and earliness == 1:
            return 0.0
        
        # Calculate (1 - Earliness), because smaller earliness is better
        earliness_score = 1 - earliness
        
        # Harmonic mean formula
        if accuracy == 0 and earliness_score == 0:
            return 0.0
        
        harmonic_mean = 2 * (accuracy * earliness_score) / (accuracy + earliness_score)
        return harmonic_mean
    
    # Calculate Harmonic Mean metric
    harmonic_mean = calculate_harmonic_mean(event_level_accuracy, earliness)

    return {
        "acc": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "event_level_accuracy": event_level_accuracy,  # Event-level accuracy
        "event_level_precision": event_level_precision,  # Event-level precision
        "event_level_recall": event_level_recall,  # Event-level recall
        "event_level_f1": event_level_f1,  # Event-level F1 score
        "earliness": earliness,  # Event-level earliness metric
        "harmonic_mean": harmonic_mean,  # Harmonic mean metric
        "detailed_results": detailed_results,  # Detailed prediction results
        "event_results": event_results,  # Event-level stop time information
    }


def save_predictions_to_csv(detailed_results: List[Dict], event_results: Dict = None, output_path: str = "./test_predictions.csv"):
    """Save detailed prediction results to CSV file, including event-level stop time"""
    if not detailed_results:
        print("No detailed results to save")
        return
    
    # Create event-level stop time mapping
    event_stop_times = {}
    if event_results:
        for event_id, result in event_results.items():
            event_stop_times[event_id] = result['stop_time']
    
    # Create DataFrame
    df_data = []
    for result in detailed_results:
        # Basic information
        row = {
            'event_id': result['event_id'],
            'sample_hour': result['sample_hour'],
            'true_label': result['true_label'],
            'predicted_label': result['predicted_label'],
            'stop_time': result['stop_time'],
            'class_prob_at_stop': result['class_prob_at_stop'],
            'stop_prob_at_stop': result['stop_prob_at_stop'],
            'correct_prediction': 1 if result['true_label'] == result['predicted_label'] else 0
        }
        
        # Add event-level stop time
        event_id = result['event_id']
        if event_id in event_stop_times:
            row['event_stop_time'] = event_stop_times[event_id]
        else:
            row['event_stop_time'] = result['stop_time']  # If no event-level info, use sample-level stop time
        
        # Add probabilities for all time steps (convert to string for CSV storage)
        all_class_probs = result['all_class_probs']
        all_stop_probs = result['all_stop_probs']
        
        # Store as string format
        row['all_class_probs'] = str(all_class_probs) if all_class_probs else "[]"
        row['all_stop_probs'] = str(all_stop_probs) if all_stop_probs else "[]"
        
        # Add probability statistics
        if all_class_probs:
            row['max_class_prob'] = max(all_class_probs)
            row['min_class_prob'] = min(all_class_probs)
            row['mean_class_prob'] = sum(all_class_probs) / len(all_class_probs)
        else:
            row['max_class_prob'] = 0.0
            row['min_class_prob'] = 0.0
            row['mean_class_prob'] = 0.0
            
        if all_stop_probs:
            row['max_stop_prob'] = max(all_stop_probs)
            row['min_stop_prob'] = min(all_stop_probs)
            row['mean_stop_prob'] = sum(all_stop_probs) / len(all_stop_probs)
        else:
            row['max_stop_prob'] = 0.0
            row['min_stop_prob'] = 0.0
            row['mean_stop_prob'] = 0.0
        
        df_data.append(row)
    
    # Create DataFrame and save
    df = pd.DataFrame(df_data)
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Predictions saved to: {output_path}")
    print(f"Total samples: {len(df)}")
    print(f"Accuracy: {df['correct_prediction'].mean():.4f}")


def set_seed(seed: int = 42):
    """Set all random seeds to ensure reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic CUDA operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    # Set random seed
    set_seed(42)
    
    logger = setup_logger()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outputs_dir = "../hourly_outputs_0_0_1"
    # Use Qwen3-Embedding by default
    embed_dim = 256  # will be overwritten by dataset builder when using sentence embedder
    max_hours = 72
    
    # Loss weights configuration (can be adjusted for experiments)
    loss_alpha = 1   # Classification loss weight
    loss_beta = 0.0    # Delay penalty weight (disabled)
    loss_gamma = 0.5   # Stop loss weight
    loss_delta = 0.5   # Trend consistency loss weight (default 0.3, can try 0.5, 0.7, 1.0, etc.)
    
    logger.info("="*80)
    logger.info("Loss Weights: alpha=%.2f, beta=%.2f, gamma=%.2f, delta=%.2f", 
                loss_alpha, loss_beta, loss_gamma, loss_delta)
    logger.info("="*80)

    # Build dataset (event-level split)
    logger.info("Start building dataset from: %s (this may take minutes if encoding)", outputs_dir)

    train_samples, val_samples, test_samples, embed_dim = build_dataset(
        outputs_dir,
        seed=42,
        max_hours=max_hours,
        embed_dim=embed_dim,
        use_sentence_embedder=False,  # Disable TF-IDF
        device=("cuda" if torch.cuda.is_available() else None),
        batch_size=64,
        unified_length=None,  # No longer globally unified length; dynamic padding within batch
        max_samples_per_event=None,  # No sample count limit
        use_cache=True,  # Enable caching
        use_bert_embedder=True,  # Enable frozen BERT text encoding
        bert_model_path="../../models/bert-base-chinese",
    )
    logger.info("Finished dataset building.") 
    logger.info("Device: %s | embed_dim: %d", device, embed_dim)
    logger.info("Samples => train: %d | val: %d | test: %d", len(train_samples), len(val_samples), len(test_samples))

    # Dataloaders
    def collate_fn(batch):
        return pad_batch(batch, text_dim=embed_dim)

    # Set fixed random seed for DataLoader
    def worker_init_fn(worker_id):
        np.random.seed(42 + worker_id)
        random.seed(42 + worker_id)

    train_loader = DataLoader(
        train_samples, 
        batch_size=32,   
        shuffle=True, 
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn,
        num_workers=0  # Avoid multiprocessing randomness issues
    )
    val_loader = DataLoader(val_samples, batch_size=64, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_samples, batch_size=64, shuffle=False, collate_fn=collate_fn)

    # Model
    model = ECTSModel(text_dim=embed_dim, num_dim=2, d_model=128).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-1)  # Further reduce learning rate, increase weight decay
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())
    logger.info("Model initialized. Params: %.2fM", sum(p.numel() for p in model.parameters())/1e6)

    # Train
    best_f1 = 0.0
    for epoch in range(1, 21):
        model.train()
        losses: Dict[str, float] = {}
        running = {"total": 0.0, "class": 0.0, "delay": 0.0, "stop_loss": 0.0, "trend": 0.0}
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [train]", unit="batch")
        for step, batch in enumerate(pbar, start=1):
            text_seq, num_seq, pad_mask, labels, hours, events = [x.to(device) if isinstance(x, torch.Tensor) else x for x in batch]
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                class_logits, stop_logits = model(text_seq, num_seq, key_padding_mask=pad_mask)
                # Extract malicious comment count sequence (first channel of num_seq)
                mal_seq = num_seq[:, :, 0]  # (B, L) malicious comment count
                loss_dict = ects_loss(
                    class_logits=class_logits, stop_logits=stop_logits, labels=labels, 
                    key_padding_mask=pad_mask, mal_seq=mal_seq,
                    alpha=loss_alpha, beta=loss_beta, gamma=loss_gamma, delta=loss_delta
                )
                loss = loss_dict["total"]
            # Check numerical stability - check before backward
            if not torch.isfinite(loss):
                logger.error(f"Non-finite loss detected at epoch {epoch}, step {step}: {loss}")
                logger.error(f"Loss components: {loss_dict}")
                # Skip this batch, do not update parameters
                continue
                
            scaler.scale(loss).backward()
            
            # Add gradient clipping to prevent gradient explosion
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # Stricter gradient clipping
            
            scaler.step(optimizer)
            scaler.update()

            # track
            for k in ["total", "class", "delay", "stop_loss", "trend"]:
                losses.setdefault(k, 0.0)
                v = loss_dict.get(k, loss)
                losses[k] += float(v.detach().item() if hasattr(v, 'item') else v)
                running[k] += float(v.detach().item() if hasattr(v, 'item') else v)

            # moving display
            avg_total = running["total"] / step
            avg_class = running["class"] / step
            avg_delay = running["delay"] / step
            avg_stop_loss = running.get("stop_loss", 0.0) / step
            avg_trend = running.get("trend", 0.0) / step
            pbar.set_postfix({"loss": f"{avg_total:.4f}", "class": f"{avg_class:.3f}", "delay": f"{avg_delay:.3f}", "stop": f"{avg_stop_loss:.3f}", "trend": f"{avg_trend:.3f}"})

            # occasional batch insight: first 2 samples class probs and stop decisions
            if step % 100 == 0:
                with torch.no_grad():
                    valid = ~pad_mask
                    class_probs = torch.sigmoid(class_logits)
                    stop_probs = torch.sigmoid(stop_logits)
                    # Display classification probabilities and stop decisions for first 2 samples
                    for i in range(min(2, class_probs.size(0))):
                        valid_hours = valid[i].sum().item()
                        class_pred = class_probs[i, :valid_hours].mean().item()
                        stop_pred = stop_probs[i, :valid_hours].mean().item()
                        logger.debug("Sample %d: class_avg=%.3f, stop_avg=%.3f, valid_hours=%d, label=%d",
                                   i, class_pred, stop_pred, valid_hours, labels[i].item())

        n_batches = max(1, len(train_loader))
        print(f"Epoch {epoch} | loss: {[f'{k}:{v/n_batches:.4f}' for k,v in losses.items()]}")

        # Eval
        # Validation (ECTS metrics)
        metrics = evaluate_ects(model, val_loader, device)
        logger.info("Val: acc=%.4f f1=%.4f precision=%.4f recall=%.4f event_acc=%.3f event_pre=%.3f event_rec=%.3f event_f1=%.3f earliness=%.3f harmonic_mean=%.3f",
                    metrics['acc'], metrics['f1'], metrics['precision'], metrics['recall'], 
                    metrics['event_level_accuracy'], metrics['event_level_precision'], 
                    metrics['event_level_recall'], metrics['event_level_f1'],
                    metrics['earliness'], metrics['harmonic_mean'])
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            Path("./early_ckpt").mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), "./early_ckpt/best.pt")
            logger.info("Model checkpoint saved at ./early_ckpt/best.pt (best f1=%.4f)", best_f1)

    # Test
    if Path("./early_ckpt/best.pt").exists():
        model.load_state_dict(torch.load("./early_ckpt/best.pt", map_location=device))
    test_metrics = evaluate_ects(model, test_loader, device)
    logger.info("Test: acc=%.4f f1=%.4f precision=%.4f recall=%.4f event_acc=%.3f event_pre=%.3f event_rec=%.3f event_f1=%.3f earliness=%.3f harmonic_mean=%.3f",
                test_metrics['acc'], test_metrics['f1'], test_metrics['precision'], test_metrics['recall'], 
                test_metrics['event_level_accuracy'], test_metrics['event_level_precision'], 
                test_metrics['event_level_recall'], test_metrics['event_level_f1'],
                test_metrics['earliness'], test_metrics['harmonic_mean'])
    
    # Save detailed prediction results to CSV
    if 'detailed_results' in test_metrics:
        event_results = test_metrics.get('event_results', None)
        save_predictions_to_csv(test_metrics['detailed_results'], event_results, "./test_predictions.csv")
        logger.info("Detailed predictions saved to CSV")
        scan_thresholds_and_plot(
            test_metrics['detailed_results'],
            output_path="./earliness_accuracy_curve.png"
        )


if __name__ == "__main__":
    main()
 

