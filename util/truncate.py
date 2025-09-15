"""
Code to remove the 'extras' features from the synthetic dataset tensors
"""

import os
import torch
import glob
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial
import time

def process_single_tensor(file_path, output_base_path, backup=False):
    """Process a single tensor file"""
    try:
        # Load tensor
        tensor = torch.load(file_path, map_location='cpu')  # Load to CPU for multiprocessing
        
        # Check if shape matches [N, 69]
        if len(tensor.shape) == 2 and tensor.shape[1] == 69:
            # Create output path by replacing base directory
            rel_path = os.path.relpath(file_path, "data/new_synthetic_huge/")
            output_path = os.path.join(output_base_path, rel_path)
            
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Create backup if requested
            if backup:
                backup_path = file_path + '.backup'
                torch.save(tensor, backup_path)
            
            # Delete columns 8-66 (keep 0-7 and 67-68)
            processed_tensor = torch.cat([
                tensor[:, :8],      # columns 0-7
                tensor[:, 67:]      # columns 67-68
            ], dim=1)
            
            # Save to new location
            torch.save(processed_tensor, output_path)
            return f"✓ Processed: {file_path} -> {output_path} - Shape: {tensor.shape} -> {processed_tensor.shape}"
        else:
            return f"⚠ Skipped: {file_path} - Shape: {tensor.shape} (not [N, 69])"
            
    except Exception as e:
        return f"✗ Error processing {file_path}: {e}"

def process_tensors_parallel(base_path="data/new_synthetic_huge/", output_path="data/new_synthetic_huge_clipped/", num_workers=None, backup=False):
    """Process tensors in parallel using multiprocessing"""
    
    # Find all tensor files in TP* directories
    patterns = [
        os.path.join(base_path, "TP*", "*.pt"),
        os.path.join(base_path, "TP*", "*.pth"),
        os.path.join(base_path, "TP*", "**", "*.pt"),  # Include subdirectories
        os.path.join(base_path, "TP*", "**", "*.pth")
    ]
    
    tensor_files = []
    for pattern in patterns:
        tensor_files.extend(glob.glob(pattern, recursive=True))
    
    # Remove duplicates
    tensor_files = list(set(tensor_files))
    
    print(f"Found {len(tensor_files)} tensor files to process")
    print(f"Input directory: {base_path}")
    print(f"Output directory: {output_path}")
    
    if not tensor_files:
        print("No tensor files found!")
        return
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Determine number of workers
    if num_workers is None:
        num_workers = min(cpu_count(), len(tensor_files))
    
    print(f"Using {num_workers} workers")
    
    # Create partial function with output_path and backup parameters
    process_func = partial(process_single_tensor, output_base_path=output_path, backup=backup)
    
    # Process files in parallel
    start_time = time.time()
    
    with Pool(num_workers) as pool:
        results = pool.map(process_func, tensor_files)
    
    end_time = time.time()
    
    # Print results
    processed_count = 0
    skipped_count = 0
    error_count = 0
    
    for result in results:
        print(result)
        if result.startswith("✓"):
            processed_count += 1
        elif result.startswith("⚠"):
            skipped_count += 1
        else:
            error_count += 1
    
    print(f"\n--- Summary ---")
    print(f"Total files: {len(tensor_files)}")
    print(f"Processed: {processed_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Errors: {error_count}")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print(f"Workers used: {num_workers}")

if __name__ == "__main__":
    # Run with automatic worker detection and backup option
    process_tensors_parallel(
        base_path="data/new_synthetic_huge/",
        output_path="data/new_synthetic_huge_clipped/",
        num_workers=None,  # Auto-detect based on CPU cores
        backup=False  # Set to False if you don't want backups
    )