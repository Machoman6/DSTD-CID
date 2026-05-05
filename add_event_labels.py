#!/usr/bin/env python3
"""
Add event label column to each event's all_pools.csv file
- Events in cyberbullying folder are labeled as 1 (cyberbullying events)
- Events in non_cyberbullying folder are labeled as 0 (non-cyberbullying events)
"""

import pandas as pd
import json
from pathlib import Path
import sys


def add_event_labels_to_csvs():
    """
    Add event label column to all events' all_pools.csv files
    """
    # Define paths (relative to project root)
    cyberbullying_dir = Path("./cyberbullying_dataset/cyberbullying")
    non_cyberbullying_dir = Path("./cyberbullying_dataset/non_cyberbullying")
    outputs_dir = Path("./hourly_outputs_0_0_1")
    
    # Check if directories exist
    if not cyberbullying_dir.exists():
        print(f"✗ Cyberbullying directory not found: {cyberbullying_dir}")
        return False
    
    if not non_cyberbullying_dir.exists():
        print(f"✗ Non-cyberbullying directory not found: {non_cyberbullying_dir}")
        return False
    
    if not outputs_dir.exists():
        print(f"✗ Outputs directory not found: {outputs_dir}")
        return False
    
    # Get all event files
    cyberbullying_events = set()
    non_cyberbullying_events = set()
    
    # Read events from cyberbullying folder
    for csv_file in cyberbullying_dir.glob("*.csv"):
        event_name = csv_file.stem
        cyberbullying_events.add(event_name)
    
    # Read events from non_cyberbullying folder
    for csv_file in non_cyberbullying_dir.glob("*.csv"):
        event_name = csv_file.stem
        non_cyberbullying_events.add(event_name)
    
    print(f"Found {len(cyberbullying_events)} cyberbullying events")
    print(f"Found {len(non_cyberbullying_events)} non-cyberbullying events")
    
    # Process each event's all_pools.csv
    processed_count = 0
    skipped_count = 0
    error_count = 0
    
    for event_dir in sorted(outputs_dir.iterdir()):
        if not event_dir.is_dir():
            continue
        
        event_name = event_dir.name
        all_pools_csv = event_dir / 'all_pools.csv'
        
        if not all_pools_csv.exists():
            print(f"✗ all_pools.csv not found: {event_name}")
            error_count += 1
            continue
        
        # Determine event label
        if event_name in cyberbullying_events:
            event_label = 1
            event_type = "cyberbullying"
        elif event_name in non_cyberbullying_events:
            event_label = 0
            event_type = "non_cyberbullying"
        else:
            print(f"⚠ Event not found in either folder: {event_name}")
            error_count += 1
            continue
        
        # Check if label column already exists
        try:
            df = pd.read_csv(all_pools_csv, nrows=1)
            if 'event_label' in df.columns:
                print(f"✓ Already has label: {event_name} ({event_type})")
                skipped_count += 1
                continue
        except Exception as e:
            print(f"✗ Error reading {event_name}: {e}")
            error_count += 1
            continue
        
        # Add label column
        try:
            df = pd.read_csv(all_pools_csv)
            df['event_label'] = event_label
            
            # Save modified CSV
            df.to_csv(all_pools_csv, index=False, encoding='utf-8')
            print(f"✓ Added label {event_label} to: {event_name} ({event_type})")
            processed_count += 1
            
        except Exception as e:
            print(f"✗ Error processing {event_name}: {e}")
            error_count += 1
    
    # Output statistics
    print(f"\n{'='*60}")
    print("LABEL ADDITION SUMMARY:")
    print(f"{'='*60}")
    print(f"Successfully processed: {processed_count}")
    print(f"Skipped (already labeled): {skipped_count}")
    print(f"Errors: {error_count}")
    print(f"Total events: {processed_count + skipped_count + error_count}")
    print(f"{'='*60}")
    
    return error_count == 0


def verify_labels():
    """
    Verify label addition results
    """
    outputs_dir = Path("./hourly_outputs_0_0_1")
    
    print("\n" + "="*60)
    print("LABEL VERIFICATION:")
    print("="*60)
    
    cyberbullying_count = 0
    non_cyberbullying_count = 0
    no_label_count = 0
    error_count = 0
    
    for event_dir in sorted(outputs_dir.iterdir()):
        if not event_dir.is_dir():
            continue
        
        event_name = event_dir.name
        all_pools_csv = event_dir / 'all_pools.csv'
        
        if not all_pools_csv.exists():
            continue
        
        try:
            df = pd.read_csv(all_pools_csv, nrows=1)
            
            if 'event_label' not in df.columns:
                print(f"✗ No label column: {event_name}")
                no_label_count += 1
            else:
                label = df['event_label'].iloc[0]
                if label == 1:
                    print(f"✓ Cyberbullying (1): {event_name}")
                    cyberbullying_count += 1
                elif label == 0:
                    print(f"✓ Non-cyberbullying (0): {event_name}")
                    non_cyberbullying_count += 1
                else:
                    print(f"⚠ Invalid label {label}: {event_name}")
                    error_count += 1
                    
        except Exception as e:
            print(f"✗ Error reading {event_name}: {e}")
            error_count += 1
    
    print(f"\nVERIFICATION SUMMARY:")
    print(f"Cyberbullying events (1): {cyberbullying_count}")
    print(f"Non-cyberbullying events (0): {non_cyberbullying_count}")
    print(f"No label column: {no_label_count}")
    print(f"Errors: {error_count}")
    print(f"Total: {cyberbullying_count + non_cyberbullying_count + no_label_count + error_count}")


def main():
    """
    Main function
    """
    print("Adding event labels to all_pools.csv files...")
    print("="*60)
    
    success = add_event_labels_to_csvs()
    
    if success:
        verify_labels()
        print("\n✅ Label addition completed successfully!")
    else:
        print("\n❌ Label addition completed with errors!")
        sys.exit(1)


if __name__ == "__main__":
    main()







