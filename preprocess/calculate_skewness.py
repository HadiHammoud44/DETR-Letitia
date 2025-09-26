#!/usr/bin/env python3
import os
import glob
import uuid
import argparse
from typing import Iterable, Tuple, List

import torch
import numpy as np
import pandas as pd
from statsmodels.stats.stattools import robust_skewness, medcouple  # sk1..sk4 + MC

# --- I/O helpers --------------------------------------------------------------

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

def pass1_count_rows(files: Iterable[str]) -> Tuple[int, int]:
    """Pass 1: sum rows and confirm consistent num_features."""
    total_rows = 0
    num_features = None
    for f in files:
        t = extract_tensor(torch.load(f, map_location="cpu"))
        if num_features is None:
            num_features = t.shape[1]
        elif t.shape[1] != num_features:
            raise ValueError(f"Inconsistent num_features: {f} has {t.shape[1]} vs {num_features}.")
        total_rows += t.shape[0]
        del t
    return total_rows, num_features

def pass2_fill_memmap(files: Iterable[str], mm: np.memmap) -> None:
    """Pass 2: fill the memmap row-wise (float64)."""
    offset = 0
    for f in files:
        t = extract_tensor(torch.load(f, map_location="cpu")).to(dtype=torch.float64)
        n = t.shape[0]
        mm[offset:offset+n, :] = t.numpy()  # safe since on CPU
        offset += n
        del t

def create_valid_feature_mask(mm: np.memmap, missing_feature_range: Tuple[int, int] = (57, 110)) -> np.ndarray:
    """
    Create a boolean mask indicating which rows have valid (non-zero) features in the specified range.
    Returns mask where True means the row should be included for features in missing_feature_range.
    
    Args:
        mm: Memory-mapped array of shape [n_samples, n_features]
        missing_feature_range: Tuple (start_idx, end_idx) - inclusive range of potentially missing features
    
    Returns:
        Boolean array of shape [n_samples] where True means features in range are not all zeros
    """
    start_idx, end_idx = missing_feature_range
    # Check if ALL features in the range are zero for each sample
    # Use np.any to check if there's at least one non-zero value in the range for each row
    valid_mask = np.any(mm[:, start_idx:end_idx+1] != 0, axis=1)
    return valid_mask

# --- Core computation ---------------------------------------------------------

def compute_skewness_for_dir(
    root: str,
    patterns: Tuple[str, ...] = ("*.pt", "*.pth", "*.pkl"),
    out_csv: str = None,
    memmap_dir: str = None,
    include_medcouple: bool = True,
    medcouple_max_n: int = 10_000,  # MC is O(N^2) memory; skip above this
    missing_feature_range: Tuple[int, int] = (57, 110),  # range of potentially missing features
) -> pd.DataFrame:
    """
    Builds a [N_total, num_features] memmap and computes per-feature skewness via statsmodels:
      - sk1: standard moment skewness (same sign convention as Fisher–Pearson)
      - sk2: quartile (Bowley) skewness (robust)
      - sk3: mean–median / MAD (robust)
      - sk4: mean–median / std (robust)
      - medcouple (optional, robust; computed only if N_total <= medcouple_max_n)

    For features in missing_feature_range, only includes samples where those features are not all zeros.

    Returns a DataFrame with one row per feature.
    """
    files = list_tensor_files(root, patterns)
    total_rows, num_features = pass1_count_rows(files)

    # Prepare on-disk matrix to avoid loading everything into RAM
    memmap_dir = memmap_dir or root
    os.makedirs(memmap_dir, exist_ok=True)
    mm_path = os.path.join(memmap_dir, f"__stack_{uuid.uuid4().hex}.dat")

    mm = np.memmap(mm_path, dtype=np.float64, mode="w+", shape=(total_rows, num_features))
    pass2_fill_memmap(files, mm)
    mm.flush()

    # Create mask for rows with valid features in the missing range
    valid_mask = create_valid_feature_mask(mm, missing_feature_range)
    start_idx, end_idx = missing_feature_range

    # Initialize result arrays
    sk1 = np.full(num_features, np.nan)
    sk2 = np.full(num_features, np.nan) 
    sk3 = np.full(num_features, np.nan)
    sk4 = np.full(num_features, np.nan)
    mc = np.full(num_features, np.nan)
    n_samples_per_feature = np.zeros(num_features, dtype=int)

    # Compute skewness for features 0 to start_idx-1 (always include all samples)
    if start_idx > 0:
        sk1_part1, sk2_part1, sk3_part1, sk4_part1 = robust_skewness(mm[:, :start_idx], axis=0)
        sk1[:start_idx] = sk1_part1
        sk2[:start_idx] = sk2_part1  
        sk3[:start_idx] = sk3_part1
        sk4[:start_idx] = sk4_part1
        n_samples_per_feature[:start_idx] = total_rows
        
        if include_medcouple and total_rows <= medcouple_max_n:
            mc_part1 = medcouple(mm[:, :start_idx], axis=0)
            mc[:start_idx] = mc_part1

    # Compute skewness for features in missing_feature_range (only valid samples)
    if np.any(valid_mask):
        valid_data = mm[valid_mask, start_idx:end_idx+1]
        n_valid = np.sum(valid_mask)
        
        if valid_data.size > 0 and valid_data.shape[0] > 1:  # Need at least 2 samples for skewness
            sk1_part2, sk2_part2, sk3_part2, sk4_part2 = robust_skewness(valid_data, axis=0)
            sk1[start_idx:end_idx+1] = sk1_part2
            sk2[start_idx:end_idx+1] = sk2_part2
            sk3[start_idx:end_idx+1] = sk3_part2  
            sk4[start_idx:end_idx+1] = sk4_part2
            n_samples_per_feature[start_idx:end_idx+1] = n_valid
            
            if include_medcouple and n_valid <= medcouple_max_n:
                mc_part2 = medcouple(valid_data, axis=0)
                mc[start_idx:end_idx+1] = mc_part2
    
    # Compute skewness for features after end_idx (always include all samples)
    if end_idx + 1 < num_features:
        sk1_part3, sk2_part3, sk3_part3, sk4_part3 = robust_skewness(mm[:, end_idx+1:], axis=0)
        sk1[end_idx+1:] = sk1_part3
        sk2[end_idx+1:] = sk2_part3
        sk3[end_idx+1:] = sk3_part3
        sk4[end_idx+1:] = sk4_part3
        n_samples_per_feature[end_idx+1:] = total_rows
        
        if include_medcouple and total_rows <= medcouple_max_n:
            mc_part3 = medcouple(mm[:, end_idx+1:], axis=0)
            mc[end_idx+1:] = mc_part3

    # Build results
    df = pd.DataFrame(
        {
            "feature": np.arange(num_features, dtype=int),
            "n_samples": n_samples_per_feature,  # actual number of samples used for each feature
            "n_total": total_rows,  # total available samples
            "sk1_moment": sk1,              # standard (non-robust) skewness
            "sk2_bowley": sk2,              # quartile-based robust skewness
            "sk3_med_absdev": sk3,          # mean–median normalized by MAD
            "sk4_med_std": sk4,             # mean–median normalized by std
            "medcouple": mc,                # robust skewness in [-1, 1]
        }
    )

    if out_csv:
        df.to_csv(out_csv, index=False)

    # Clean up memmap backing file
    try:
        del mm  # close memmap
        os.remove(mm_path)
    except Exception:
        pass

    return df

# --- CLI ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Compute per-feature skewness from a directory of torch tensors.")
    p.add_argument("dir", help="Directory containing tensors shaped [num_points, num_features].")
    p.add_argument("--patterns", nargs="+", default=["*.pt", "*.pth", "*.pkl"],
                   help="Glob patterns to include (default: *.pt *.pth *.pkl).")
    p.add_argument("--out_csv", default=None, help="Optional path to write a CSV of results.")
    p.add_argument("--memmap-dir", default=None, help="Where to place the temporary memmap file (defaults to --dir).")
    p.add_argument("--no-medcouple", action="store_true", help="Skip medcouple computation.")
    p.add_argument("--medcouple-max-n", type=int, default=20_000,
                   help="Max rows to allow for medcouple (default 20000).")
    p.add_argument("--missing-feature-range", nargs=2, type=int, default=[57, 110],
                   help="Range of potentially missing features [start, end] inclusive (default: 57 110).")
    args = p.parse_args()

    df = compute_skewness_for_dir(
        root=args.dir,
        patterns=tuple(args.patterns),
        out_csv=args.out_csv,
        memmap_dir=args.memmap_dir,
        include_medcouple=not args.no_medcouple,
        medcouple_max_n=args.medcouple_max_n,
        missing_feature_range=tuple(args.missing_feature_range),
    )
    
    # Print a compact summary with additional info about excluded samples
    print(f"\nSkewness computation completed.")
    print(f"Total samples: {df['n_total'].iloc[0]}")
    missing_start, missing_end = args.missing_feature_range
    if missing_start < len(df) and missing_end < len(df):
        n_valid_missing = df.loc[missing_start, 'n_samples']
        n_excluded = df['n_total'].iloc[0] - n_valid_missing
        print(f"For features {missing_start}-{missing_end}: {n_valid_missing} samples used, {n_excluded} excluded (all-zero)")
    
    with pd.option_context("display.max_rows", 20, "display.width", 140):
        print(df)

if __name__ == "__main__":
    main()
