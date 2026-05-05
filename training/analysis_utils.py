#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility helpers for analyzing ECTS predictions, building event groups,
scanning stop-probability thresholds, and plotting Accuracy vs Earliness.
"""

from __future__ import annotations

from typing import Dict, List, Tuple
from pathlib import Path
import sys

import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def get_event_real_length(event_id: str) -> int:
    """
    Get the real length of an event from CSV file
    """
    candidates = [
        _PROJECT_ROOT / "hourly_outputs_1",
        _PROJECT_ROOT / "hourly_outputs",
        Path("./hourly_outputs_1"),
        Path("./hourly_outputs"),
    ]
    for base in candidates:
        csv_path = (base / event_id / "all_pools.csv").resolve()
        if not csv_path.exists():
            continue
        try:
            df = pd.read_csv(csv_path)
            max_idx = df["hour_idx"].max()
            if pd.isna(max_idx):
                continue
            length = int(max_idx)
            if length > 0:
                return length
        except Exception:
            continue
    return 72


def build_event_groups_from_details(detailed_results: List[Dict]) -> Dict[str, Dict]:
    """
    Group detailed_results by event, preparing data structure for threshold scanning
    """
    event_groups: Dict[str, Dict] = {}
    length_cache: Dict[str, int] = {}
    for result in detailed_results:
        event_id = result.get("event_id")
        if not event_id:
            continue
        if event_id not in event_groups:
            real_length = length_cache.get(event_id)
            if real_length is None:
                real_length = get_event_real_length(event_id)
                length_cache[event_id] = real_length
            event_groups[event_id] = {
                "samples": [],
                "real_length": real_length if real_length > 0 else 1,
            }
        event_groups[event_id]["samples"].append({
            "sample_hour": max(1, int(result.get("sample_hour", 1))),
            "stop_time": max(0, int(result.get("stop_time", max(int(result.get("sample_hour", 1)) - 1, 0)))),
            "predicted_label": int(result.get("predicted_label", 0)),
            "true_label": int(result.get("true_label", 0)),
            "stop_prob": float(result.get("stop_prob_at_stop", 0.0)),
        })
    for event_info in event_groups.values():
        event_info["samples"].sort(key=lambda x: x["sample_hour"])
    return event_groups


def derive_event_results_with_threshold(event_groups: Dict[str, Dict], threshold: float) -> Dict[str, Dict]:
    """
    Select stop samples for events based on given threshold, build event-level results
    """
    event_results: Dict[str, Dict] = {}
    for event_id, event_info in event_groups.items():
        samples = event_info.get("samples", [])
        if not samples:
            continue
        selected = None
        for sample in samples:
            if sample["stop_prob"] >= threshold:
                selected = sample
                break
        if selected is None:
            selected = samples[-1]
        real_length = max(event_info.get("real_length", 1), 1)
        stop_time = max(0, min(selected["stop_time"], real_length - 1))
        sample_hour = max(1, min(selected["sample_hour"], real_length))
        event_results[event_id] = {
            "stop_time": stop_time,
            "predicted_label": selected["predicted_label"],
            "true_label": selected["true_label"],
            "sample_hour": sample_hour,
            "real_length": real_length,
        }
    return event_results


def summarize_accuracy_earliness(event_results: Dict[str, Dict]) -> Tuple[float, float]:
    """
    Calculate Accuracy and Earliness based on event-level results
    """
    if not event_results:
        return 0.0, 0.0
    total_events = len(event_results)
    correct_events = sum(
        1 for result in event_results.values()
        if result["predicted_label"] == result["true_label"]
    )
    accuracy = correct_events / total_events if total_events else 0.0
    earliness_sum = 0.0
    for result in event_results.values():
        real_length = max(result.get("real_length", 1), 1)
        stop_time = max(0, min(result.get("stop_time", 0), real_length - 1))
        earliness_sum += (stop_time + 1) / real_length
    earliness = earliness_sum / total_events if total_events else 0.0
    return accuracy, earliness


def scan_thresholds_and_plot(detailed_results: List[Dict],
                             output_path: str = "./earliness_accuracy_curve.png",
                             thresholds: np.ndarray | None = None):
    """
    Scan different stop probability thresholds and plot Accuracy and Earliness curves
    """
    logger = logging.getLogger("early_train")
    if not detailed_results:
        logger.warning("No detailed results available for threshold scanning.")
        return
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.95, 10)
    series = collect_accuracy_earliness_points(detailed_results, thresholds, logger)
    if not series:
        logger.warning("No event groups available for threshold scanning.")
        return
    
    # Print threshold, accuracy, and earliness table
    print("\n" + "="*60)
    print("Threshold | Accuracy | Earliness")
    print("-"*60)
    for point in sorted(series, key=lambda x: x["threshold"]):
        print(f"{point['threshold']:9.2f} | {point['accuracy']:8.4f} | {point['earliness']:9.4f}")
    print("="*60 + "\n")
    
    sorted_points = sorted(series, key=lambda x: x["earliness"])
    sorted_earliness = [p["earliness"] for p in sorted_points]
    sorted_accuracy = [p["accuracy"] for p in sorted_points]
    plt.figure(figsize=(7, 5))
    plt.plot(sorted_earliness, sorted_accuracy, marker="o", linestyle="-", label="Threshold sweep")
    for idx, point in enumerate(sorted_points):
        earliness = point["earliness"]
        accuracy = point["accuracy"]
        
        # Adjust annotation position based on point location to avoid overlap
        # xytext format: (x_offset, y_offset) in points
        # Positive x: right, Positive y: up
        
        # Alternate annotation positions to reduce overlap
        if idx % 3 == 0:
            # Top-right
            xytext_offset = (8, 8)
            ha, va = 'left', 'bottom'
        elif idx % 3 == 1:
            # Top-left
            xytext_offset = (-8, 8)
            ha, va = 'right', 'bottom'
        else:
            # Bottom-right
            xytext_offset = (8, -8)
            ha, va = 'left', 'top'
        
        plt.annotate(f"{point['threshold']:.2f}", (earliness, accuracy),
                     textcoords="offset points", xytext=xytext_offset, 
                     fontsize=8, ha=ha, va=va)
    plt.xlabel("Earliness")
    plt.ylabel("Accuracy")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.xlim(0.1, 0.4)
    plt.ylim(0.6, 1)
    plt.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    logger.info("Threshold curve saved to %s", output_path)



def collect_accuracy_earliness_points(detailed_results: List[Dict],
                                      thresholds: np.ndarray | None = None,
                                      logger: logging.Logger | None = None) -> List[Dict]:
    if logger is None:
        logger = logging.getLogger("early_train")
    if not detailed_results:
        return []
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.95, 10)
    event_groups = build_event_groups_from_details(detailed_results)
    if not event_groups:
        return []
    series: List[Dict] = []
    for th in thresholds:
        event_results = derive_event_results_with_threshold(event_groups, float(th))
        acc, early = summarize_accuracy_earliness(event_results)
        series.append({
            "threshold": float(th),
            "accuracy": acc,
            "earliness": early,
        })
        logger.debug("Threshold %.2f -> acc=%.4f, earliness=%.4f", th, acc, early)
    return series
