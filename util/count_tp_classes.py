#!/usr/bin/env python3
"""
Efficient (sub-)class distribution counter for TP* tensor files.
Counts all class occurrences from the last column of all tensors in data/new_synthetic_huge/TP*/
"""

import torch
import os
from collections import Counter
import glob
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import numpy as np

def count_classes_in_file(file_path):
    """Count classes in a single tensor file."""
    try:
        # Load tensor with weights_only for security
        tensor = torch.load(file_path, weights_only=True)
        # Extract classes from last column (index -1)
        classes = tensor[:, -1].numpy().astype(int)
        return Counter(classes)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return Counter()

def count_classes_batch(file_paths):
    """Count classes in a batch of files."""
    total_counter = Counter()
    for file_path in file_paths:
        file_counter = count_classes_in_file(file_path)
        total_counter.update(file_counter)
    return total_counter

def main(tp_dir=None, output_file=None):

    # Get all .pt files
    pt_files = sorted(glob.glob(os.path.join(tp_dir, "*.pt")))
    print(f"Found {len(pt_files)} tensor files in directory")
    
    if not pt_files:
        print("No .pt files found!")
        return
    
    # Use multiprocessing for faster processing
    num_cores = min(mp.cpu_count(), 8)  # Use up to 8 cores
    print(f"Using {num_cores} cores for parallel processing")
    
    # Split files into batches for each process
    batch_size = len(pt_files) // num_cores
    file_batches = [pt_files[i:i + batch_size] for i in range(0, len(pt_files), batch_size)]
    
    # Process files in parallel
    with mp.Pool(num_cores) as pool:
        batch_counters = list(tqdm(
            pool.imap(count_classes_batch, file_batches),
            total=len(file_batches),
            desc="Processing file batches"
        ))
    
    # Combine all counters
    total_counter = Counter()
    for counter in batch_counters:
        total_counter.update(counter)
    
    # Display results
    print("\n" + "="*60)
    print("CLASS DISTRIBUTION DATASET")
    print("="*60)
    
    total_points = sum(total_counter.values())
    unique_classes = len(total_counter)
    
    print(f"Total data points: {total_points:,}")
    print(f"Unique classes: {unique_classes}")
    print(f"Files processed: {len(pt_files)}")
    
    # Sort classes by frequency (descending)
    sorted_classes = total_counter.most_common()
    
    print("\nTop 20 most frequent classes:")
    print("-" * 40)
    print(f"{'Class':<8} {'Count':<12} {'Percentage':<12}")
    print("-" * 40)
    
    for i, (class_id, count) in enumerate(sorted_classes[:20]):
        percentage = (count / total_points) * 100
        print(f"{class_id:<8} {count:<12,} {percentage:<12.2f}%")
    
    if len(sorted_classes) > 20:
        print(f"\n... and {len(sorted_classes) - 20} more classes")
    
    # Statistics
    counts = list(total_counter.values())
    print(f"\nStatistics:")
    print(f"- Min count: {min(counts):,}")
    print(f"- Max count: {max(counts):,}")
    print(f"- Mean count: {np.mean(counts):.2f}")
    print(f"- Median count: {np.median(counts):.2f}")
    print(f"- Std deviation: {np.std(counts):.2f}")
    
    # Class range
    class_ids = list(total_counter.keys())
    print(f"\nClass range:")
    print(f"- Minimum class ID: {min(class_ids)}")
    print(f"- Maximum class ID: {max(class_ids)}")
    
    # Save detailed results to file
    with open(output_file, 'w') as f:
        f.write("Class Distribution in Dataset\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total data points: {total_points:,}\n")
        f.write(f"Unique classes: {unique_classes}\n")
        f.write(f"Files processed: {len(pt_files)}\n\n")
        
        f.write("Complete class distribution (sorted by frequency):\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Class':<8} {'Count':<12} {'Percentage':<12}\n")
        f.write("-" * 50 + "\n")
        
        for class_id, count in sorted_classes:
            percentage = (count / total_points) * 100
            f.write(f"{class_id:<8} {count:<12,} {percentage:<12.4f}%\n")
    
    print(f"\nDetailed results saved to: {output_file}")

if __name__ == "__main__":
    tp_dir = "/mnt/letitia/scratch/students/hhammoud/detr/data/new_synthetic_huge_clipped/TP2"
    output_file = "tp2_class_distribution.txt"
    main(tp_dir, output_file)
