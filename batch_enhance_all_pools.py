#!/usr/bin/env python3
"""
Batch post-processing script: process all events' all_pools.csv files, add two columns:
1. hourly_total_history: hourly total comment count sequence from start to current hour
2. hourly_malicious_history: hourly malicious comment count sequence from start to current hour
"""

import pandas as pd
import json
from pathlib import Path
import sys
from enhance_all_pools import enhance_all_pools_csv


def batch_enhance_all_pools(outputs_dir: str, events_dir: str):
    """Batch process all events' all_pools.csv files"""
    
    outputs_path = Path(outputs_dir)
    events_path = Path(events_dir)
    
    if not outputs_path.exists():
        print(f"✗ Outputs directory not found: {outputs_path}")
        return False
    
    if not events_path.exists():
        print(f"✗ Events directory not found: {events_path}")
        return False
    
    # Find all event directories
    event_dirs = [d for d in outputs_path.iterdir() if d.is_dir()]
    print(f"Found {len(event_dirs)} event directories")
    
    success_count = 0
    skip_count = 0
    error_count = 0
    
    for event_dir in sorted(event_dirs):
        event_name = event_dir.name
        all_pools_csv = event_dir / 'all_pools.csv'
        original_event_csv = events_path / f'{event_name}.csv'
        
        print(f"\n{'='*60}")
        print(f"Processing event: {event_name}")
        print(f"CSV: {all_pools_csv}")
        print(f"Original: {original_event_csv}")
        
        # Check if files exist
        if not all_pools_csv.exists():
            print(f"✗ all_pools.csv not found: {all_pools_csv}")
            error_count += 1
            continue
        
        if not original_event_csv.exists():
            print(f"✗ Original event CSV not found: {original_event_csv}")
            error_count += 1
            continue
        
        # Check if already processed (check if new columns exist)
        try:
            df_check = pd.read_csv(all_pools_csv, nrows=1)
            if 'hourly_total_history' in df_check.columns and 'hourly_malicious_history' in df_check.columns:
                print(f"✓ Already enhanced, skipping: {event_name}")
                skip_count += 1
                continue
        except Exception as e:
            print(f"⚠ Warning: Could not check existing columns: {e}")
        
        # Process this event
        try:
            success = enhance_all_pools_csv(str(all_pools_csv), str(original_event_csv))
            if success:
                success_count += 1
                print(f"✓ Successfully enhanced: {event_name}")
            else:
                error_count += 1
                print(f"✗ Failed to enhance: {event_name}")
        except Exception as e:
            error_count += 1
            print(f"✗ Error processing {event_name}: {e}")
    
    # Output statistics
    print(f"\n{'='*60}")
    print("BATCH PROCESSING SUMMARY:")
    print(f"Total events: {len(event_dirs)}")
    print(f"Successfully enhanced: {success_count}")
    print(f"Skipped (already processed): {skip_count}")
    print(f"Errors: {error_count}")
    print(f"{'='*60}")
    
    return error_count == 0


def main():
    # Default paths (relative to project root)
    outputs_dir = "./hourly_outputs_0_0_1"
    events_dir = "./cyberbullying_dataset/events"
    
    print(f"Using default paths:")
    print(f"  Outputs: {outputs_dir}")
    print(f"  Events: {events_dir}")
    
    success = batch_enhance_all_pools(outputs_dir, events_dir)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
