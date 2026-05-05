"""
Data export functions for dynamic text pool system.
"""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def export_all_pools_to_csv(outputs_dir: Path, csv_path: Path) -> None:
    """Aggregate all per-hour pool_*.json into one CSV.

    Columns: hour_idx, tag, rank, text, timestamp, score
    """
    rows: List[Dict[str, Any]] = []
    # Iterate hour directories in order
    hour_dirs = sorted(
        [d for d in outputs_dir.iterdir() if d.is_dir() and d.name.startswith("hour_")],
        key=lambda p: int(p.name.replace("hour_", ""))
    )
    for d in hour_dirs:
        hour_idx = int(d.name.replace("hour_", ""))
        pool_files = sorted(d.glob("pool_*.json"))
        if not pool_files:
            continue
        pool_file = pool_files[-1]
        try:
            with open(pool_file, "r", encoding="utf-8") as f:
                pool = json.load(f)
        except Exception:
            continue
        # Extract tag from filename
        tag = pool_file.stem.replace("pool_", "")
        # Rank is the order within the pool file
        for rank, item in enumerate(pool, start=1):
            rows.append({
                "hour_idx": hour_idx,
                "tag": tag,
                "rank": rank,
                "text": item.get("text", ""),
                "timestamp": item.get("timestamp", ""),
                "score": item.get("score", 0.0),
            })
    # Write CSV
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["hour_idx", "tag", "rank", "text", "timestamp", "score"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"Saved aggregated pools CSV: {csv_path} (rows={len(rows)})")

































































