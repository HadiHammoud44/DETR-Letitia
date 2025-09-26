#!/usr/bin/env python3
import os
import glob
import uuid
import argparse
from typing import Iterable, Tuple, List, Dict, Optional
import json

import torch
import numpy as np
import pandas as pd

# --- I/O helpers (adapted from calculate_skewness.py) -------------------------

def list_tensor_files(root: str,
                      patterns: Tuple[str, ...] = ("*.pt", "*.pth", "*.pkl")) -> List[str]:
    files = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(root, pat)))
    files.sort()
    if not files:
        raise FileNotFoundError(f"No tensor files found in {root} matching {patterns}.")
    return files

def extract_tensor(obj) -> torch.Tensor:
    """Return a 2D tensor from torch.load result (tensor directly or first tensor in dict)."""
    if isinstance(obj, torch.Tensor):
        t = obj
    elif isinstance(obj, dict):
        # pick first tensor-like entry
        cand = next((v for v in obj.values() if isinstance(v, torch.Tensor)), None)
        if cand is None:
            raise TypeError("Loaded dict has no torch.Tensor values.")
        t = cand
    else:
        raise TypeError(f"Unsupported object type from torch.load: {type(obj)}")

    if t.ndim != 2:
        raise ValueError(f"Expected 2D tensor, got shape {tuple(t.shape)}.")
    return t

def get_tensor_id_from_filename(filepath: str) -> int:
    """Extract tensor ID from filename like 'ID.pt'."""
    basename = os.path.basename(filepath)
    name_without_ext = os.path.splitext(basename)[0]
    try:
        return int(name_without_ext)
    except ValueError:
        raise ValueError(f"Cannot extract integer ID from filename: {basename}")

def pass1_count_rows_and_get_file_info(files: Iterable[str], validation_ids: List[int]) -> Tuple[int, int, Dict]:
    """Pass 1: count rows, check consistency, and collect file information."""
    total_rows = 0
    train_rows = 0
    num_features = None
    file_info = {}
    validation_ids_set = set(validation_ids)
    
    for f in files:
        try:
            tensor_id = get_tensor_id_from_filename(f)
        except ValueError as e:
            print(f"Warning: {e} - skipping file")
            continue
            
        t = extract_tensor(torch.load(f, map_location="cpu", weights_only=True))
        if num_features is None:
            num_features = t.shape[1]
        elif t.shape[1] != num_features:
            raise ValueError(f"Inconsistent num_features: {f} has {t.shape[1]} vs {num_features}.")
        
        n_rows = t.shape[0]
        is_validation = tensor_id in validation_ids_set
        
        file_info[f] = {
            'tensor_id': tensor_id,
            'n_rows': n_rows,
            'is_validation': is_validation
        }
        
        total_rows += n_rows
        if not is_validation:
            train_rows += n_rows
        
        del t
    
    return total_rows, num_features, file_info, train_rows

def pass2_fill_memmap(files: Iterable[str], mm: np.memmap, file_info: Dict) -> Dict[str, Tuple[int, int]]:
    """Pass 2: fill the memmap and return file offset information."""
    offset = 0
    file_offsets = {}
    
    for f in files:
        if f not in file_info:
            continue  # Skip files that couldn't be processed in pass1
            
        t = extract_tensor(torch.load(f, map_location="cpu", weights_only=True)).to(dtype=torch.float64)
        n = t.shape[0]
        mm[offset:offset+n, :] = t.numpy()
        
        file_offsets[f] = (offset, offset + n)
        offset += n
        del t
    
    return file_offsets

def create_valid_feature_mask(mm: np.memmap, missing_feature_range: Tuple[int, int] = (57, 110)) -> np.ndarray:
    """Create mask for rows where features 57-110 are not all zeros."""
    start_idx, end_idx = missing_feature_range
    valid_mask = np.any(mm[:, start_idx:end_idx+1] != 0, axis=1)
    return valid_mask

# --- Standardization logic ---------------------------------------------------

def compute_standardization_stats(
    mm: np.memmap,
    file_info: Dict,
    file_offsets: Dict[str, Tuple[int, int]],
    missing_feature_range: Tuple[int, int] = (57, 110),
    class_columns: Tuple[int, int] = (111, 112)
) -> Dict:
    """
    Compute mean and std for standardization grouped by class/subclass.
    
    Returns:
        Dictionary with standardization statistics for each group
    """
    print("Computing standardization statistics...")
    
    # Extract class information from training data only
    train_mask = np.zeros(mm.shape[0], dtype=bool)
    for filepath, (start_offset, end_offset) in file_offsets.items():
        if not file_info[filepath]['is_validation']:
            train_mask[start_offset:end_offset] = True
    
    if not np.any(train_mask):
        raise ValueError("No training data found!")
    
    train_data = mm[train_mask]
    class_id_col, subclass_id_col = class_columns
    
    # Extract class and subclass IDs from training data
    class_ids = train_data[:, class_id_col].astype(int)
    subclass_ids = train_data[:, subclass_id_col].astype(int)
    
    # Create mask for valid features (not all-zero in range 57-110)
    valid_feature_mask = create_valid_feature_mask(train_data, missing_feature_range)
    
    print(f"Training samples: {len(train_data)}")
    print(f"Valid feature samples (non-all-zero 57-110): {np.sum(valid_feature_mask)}")
    
    # Identify groups
    unique_class_1 = np.sum(class_ids == 1)
    unique_subclasses_0 = np.unique(subclass_ids[class_ids == 0])
    
    print(f"Class 1 samples: {unique_class_1}")
    print(f"Class 0 subclasses: {unique_subclasses_0}")
    
    stats = {}
    start_idx, end_idx = missing_feature_range
    num_features = train_data.shape[1] - 2  # Exclude class columns
    
    # Group 1: Class ID = 1 (all together)
    class_1_mask = (class_ids == 1)
    if np.any(class_1_mask):
        group_data = train_data[class_1_mask]
        group_valid_mask = create_valid_feature_mask(group_data, missing_feature_range)
        
        means = np.zeros(num_features)
        stds = np.ones(num_features)  # Default std=1 to avoid division by zero
        
        # Features 0 to start_idx-1: use all samples in this group
        if start_idx > 0:
            means[:start_idx] = np.mean(group_data[:, :start_idx], axis=0)
            stds[:start_idx] = np.std(group_data[:, :start_idx], axis=0, ddof=1)
            stds[:start_idx] = np.where(stds[:start_idx] < 1e-8, 1, stds[:start_idx])  # Avoid division by zero
        
        # Features start_idx to end_idx: use only valid samples
        if np.any(group_valid_mask):
            valid_group_data = group_data[group_valid_mask]
            if valid_group_data.shape[0] > 1:
                means[start_idx:end_idx+1] = np.mean(valid_group_data[:, start_idx:end_idx+1], axis=0)
                stds[start_idx:end_idx+1] = np.std(valid_group_data[:, start_idx:end_idx+1], axis=0, ddof=1)
                stds[start_idx:end_idx+1] = np.where(stds[start_idx:end_idx+1] < 1e-8, 1, stds[start_idx:end_idx+1])
        
        stats['class_1'] = {
            'mean': means,
            'std': stds,
            'n_samples': np.sum(class_1_mask),
            'n_valid_samples': np.sum(class_1_mask & valid_feature_mask)
        }
        print(f"  Class 1: {stats['class_1']['n_samples']} total, {stats['class_1']['n_valid_samples']} valid")
    
    # Groups for Class ID = 0 (by subclass)
    for subclass in unique_subclasses_0:
        subclass_mask = (class_ids == 0) & (subclass_ids == subclass)
        if np.any(subclass_mask):
            group_data = train_data[subclass_mask]
            group_valid_mask = create_valid_feature_mask(group_data, missing_feature_range)
            
            means = np.zeros(num_features)
            stds = np.ones(num_features)
            
            # Features 0 to start_idx-1
            if start_idx > 0:
                means[:start_idx] = np.mean(group_data[:, :start_idx], axis=0)
                stds[:start_idx] = np.std(group_data[:, :start_idx], axis=0, ddof=1)
                stds[:start_idx] = np.where(stds[:start_idx] < 1e-8, 1, stds[:start_idx])
            
            # Features start_idx to end_idx
            if np.any(group_valid_mask):
                valid_group_data = group_data[group_valid_mask]
                if valid_group_data.shape[0] > 1:
                    means[start_idx:end_idx+1] = np.mean(valid_group_data[:, start_idx:end_idx+1], axis=0)
                    stds[start_idx:end_idx+1] = np.std(valid_group_data[:, start_idx:end_idx+1], axis=0, ddof=1)
                    stds[start_idx:end_idx+1] = np.where(stds[start_idx:end_idx+1] < 1e-8, 1, stds[start_idx:end_idx+1])
            
            stats[f'class_0_subclass_{subclass}'] = {
                'mean': means,
                'std': stds,
                'n_samples': np.sum(subclass_mask),
                'n_valid_samples': np.sum(subclass_mask & valid_feature_mask)
            }
            print(f"  Class 0, Subclass {subclass}: {stats[f'class_0_subclass_{subclass}']['n_samples']} total, {stats[f'class_0_subclass_{subclass}']['n_valid_samples']} valid")
    
    return stats

def apply_standardization_to_tensor(
    tensor: torch.Tensor,
    stats: Dict,
    missing_feature_range: Tuple[int, int] = (57, 110),
    class_columns: Tuple[int, int] = (111, 112)
) -> torch.Tensor:
    """Apply group-wise standardization to a tensor."""
    
    # Convert to float64 for computation
    orig_dtype = tensor.dtype
    if not tensor.is_floating_point():
        tensor = tensor.float()
    tensor = tensor.double()
    
    class_id_col, subclass_id_col = class_columns
    class_ids = tensor[:, class_id_col].long()
    subclass_ids = tensor[:, subclass_id_col].long()
    
    start_idx, end_idx = missing_feature_range
    num_features = tensor.shape[1] - 2  # Exclude class columns
    
    # Create mask for all-zero features 57-110
    range_features = tensor[:, start_idx:end_idx+1]
    all_zero_mask = torch.all(range_features == 0, dim=1)
    
    # Apply standardization group by group
    for i in range(tensor.shape[0]):
        class_id = class_ids[i].item()
        subclass_id = subclass_ids[i].item()
        
        # Determine which stats to use
        if class_id == 1:
            group_key = 'class_1'
        else:  # class_id == 0
            group_key = f'class_0_subclass_{subclass_id}'
        
        if group_key not in stats:
            print(f"Warning: No stats found for {group_key}, skipping standardization for this sample")
            continue
        
        group_mean = torch.from_numpy(stats[group_key]['mean']).double()
        group_std = torch.from_numpy(stats[group_key]['std']).double()
        
        # Standardize features 0 to start_idx-1 (always)
        if start_idx > 0:
            tensor[i, :start_idx] = (tensor[i, :start_idx] - group_mean[:start_idx]) / group_std[:start_idx]
        
        # Standardize features start_idx to end_idx (only if not all-zero)
        if not all_zero_mask[i]:
            tensor[i, start_idx:end_idx+1] = (tensor[i, start_idx:end_idx+1] - group_mean[start_idx:end_idx+1]) / group_std[start_idx:end_idx+1]
        # else: keep all-zero features as 0.0
    
    return tensor.to(orig_dtype)

def standardize_features_dir(
    in_dir: str,
    out_dir: str,
    validation_ids: List[int] = None,
    patterns: Tuple[str, ...] = ("*.pt", "*.pth", "*.pkl"),
    memmap_dir: str = None,
    missing_feature_range: Tuple[int, int] = (57, 110),
    class_columns: Tuple[int, int] = (111, 112),
    stats_csv: str = None
) -> Dict:
    """
    Standardize features in a directory of tensors with group-wise statistics.
    
    Args:
        in_dir: Input directory
        out_dir: Output directory  
        validation_ids: List of tensor IDs to exclude from stats computation
        patterns: File patterns to match
        memmap_dir: Directory for temporary memmap file
        missing_feature_range: Range of potentially missing features
        class_columns: Indices of class and subclass columns
        stats_csv: Optional path to save standardization statistics
    
    Returns:
        Dictionary with standardization statistics
    """
    validation_ids = validation_ids or []
    memmap_dir = memmap_dir or in_dir
    
    print(f"Standardizing features from {in_dir}")
    print(f"Validation IDs to exclude from stats: {validation_ids}")
    
    # Phase 1: Collect file information
    files = list_tensor_files(in_dir, patterns)
    total_rows, num_features, file_info, train_rows = pass1_count_rows_and_get_file_info(files, validation_ids)
    
    print(f"Found {len(files)} tensor files")
    print(f"Total samples: {total_rows}, Training samples: {train_rows}")
    print(f"Features per sample: {num_features}")
    
    # Phase 2: Create memmap and load all data
    os.makedirs(memmap_dir, exist_ok=True)
    mm_path = os.path.join(memmap_dir, f"__standardize_{uuid.uuid4().hex}.dat")
    mm = np.memmap(mm_path, dtype=np.float64, mode="w+", shape=(total_rows, num_features))
    
    file_offsets = pass2_fill_memmap(files, mm, file_info)
    mm.flush()
    
    # Phase 3: Compute standardization statistics
    stats = compute_standardization_stats(mm, file_info, file_offsets, missing_feature_range, class_columns)
    
    # Save statistics if requested
    if stats_csv:
        stats_df = []
        for group_name, group_stats in stats.items():
            for feature_idx in range(len(group_stats['mean'])):
                stats_df.append({
                    'group': group_name,
                    'feature': feature_idx,
                    'mean': group_stats['mean'][feature_idx],
                    'std': group_stats['std'][feature_idx],
                    'n_samples': group_stats['n_samples'],
                    'n_valid_samples': group_stats['n_valid_samples']
                })
        pd.DataFrame(stats_df).to_csv(stats_csv, index=False)
        print(f"Standardization statistics saved to: {stats_csv}")
    
    # Phase 4: Apply standardization and save results
    os.makedirs(out_dir, exist_ok=True)
    
    print("\nApplying standardization to individual files...")
    with torch.no_grad():
        for i, filepath in enumerate(files):
            if filepath not in file_info:
                continue
                
            print(f"Processing {i+1}/{len(files)}: {os.path.basename(filepath)}")
            
            # Load original tensor
            obj = torch.load(filepath, map_location="cpu", weights_only=True)
            tensor = extract_tensor(obj)
            
            # Apply standardization
            standardized_tensor = apply_standardization_to_tensor(
                tensor, stats, missing_feature_range, class_columns
            )
            
            # Save standardized tensor
            out_path = os.path.join(out_dir, os.path.basename(filepath))
            torch.save(standardized_tensor, out_path)
    
    # Cleanup memmap
    try:
        del mm
        os.remove(mm_path)
    except Exception:
        pass
    
    print(f"\nStandardization complete. Output saved to: {out_dir}")
    return stats

# --- CLI ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Standardize features with group-wise statistics.")
    parser.add_argument("input_dir", help="Directory containing input tensor files")
    parser.add_argument("output_dir", help="Directory for standardized output files")
    parser.add_argument("--validation_ids", nargs="*", type=int, default=[18, 50, 68, 78, 91, 95, 102, 106, 111, 133, 142, 150, 177, 210, 220, 238, 245],
                       help="Tensor IDs to exclude from statistics computation")
    parser.add_argument("--patterns", nargs="+", default=["*.pt", "*.pth", "*.pkl"],
                       help="File patterns to match")
    parser.add_argument("--memmap_dir", default=None,
                       help="Directory for temporary memmap file")
    parser.add_argument("--missing_feature_range", nargs=2, type=int, default=[57, 110],
                       help="Range of potentially missing features [start, end]")
    parser.add_argument("--class_columns", nargs=2, type=int, default=[111, 112],
                       help="Indices of class and subclass columns")
    parser.add_argument("--stats_csv", default=None,
                       help="Optional path to save standardization statistics")
    
    args = parser.parse_args()
    
    standardize_features_dir(
        in_dir=args.input_dir,
        out_dir=args.output_dir,
        validation_ids=args.validation_ids,
        patterns=tuple(args.patterns),
        memmap_dir=args.memmap_dir,
        missing_feature_range=tuple(args.missing_feature_range),
        class_columns=tuple(args.class_columns),
        stats_csv=args.stats_csv
    )

if __name__ == "__main__":
    main()
