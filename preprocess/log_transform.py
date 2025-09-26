# Signed-log1p transform on selected (skewed) feature columns, per tensor file.
# Formula: sgn(x) * log1p(|x|)
import os, glob, torch
from typing import List, Union, Tuple, Optional

def signed_log1p_inplace(t: torch.Tensor, cols):
    if t.ndim != 2:
        raise ValueError(f"Expected 2D [num_points, num_features], got {tuple(t.shape)}")
    if max(cols, default=-1) >= t.shape[1]:
        raise IndexError(f"Feature index out of bounds for shape {tuple(t.shape)}")

    # ensure float dtype and CPU for math; keep original dtype to restore
    orig_dtype = t.dtype
    if not t.is_floating_point():
        t = t.float()
    if t.device.type != "cpu":
        t = t.cpu()

    c = cols if isinstance(cols, (list, tuple)) else list(cols)
    x = t[:, c]
    t[:, c] = torch.sign(x) * torch.log1p(torch.abs(x))
    return t.to(orig_dtype)

def signed_log1p_smart(
    t: torch.Tensor, 
    cols: Union[List[int], Tuple[int, ...]], 
    missing_feature_range: Tuple[int, int] = (57, 110)
):
    """
    Apply signed log1p transform intelligently:
    - For features outside missing_feature_range: transform all rows
    - For features inside missing_feature_range: only transform rows where 
      features in that range are not all zeros
    
    Args:
        t: Input tensor of shape [num_points, num_features]
        cols: Column indices to transform
        missing_feature_range: Tuple (start_idx, end_idx) for conditional logic
    
    Returns:
        Transformed tensor
    """
    if t.ndim != 2:
        raise ValueError(f"Expected 2D [num_points, num_features], got {tuple(t.shape)}")
    if max(cols, default=-1) >= t.shape[1]:
        raise IndexError(f"Feature index out of bounds for shape {tuple(t.shape)}")

    # ensure float dtype and CPU for math; keep original dtype to restore
    orig_dtype = t.dtype
    if not t.is_floating_point():
        t = t.float()
    if t.device.type != "cpu":
        t = t.cpu()

    c = cols if isinstance(cols, (list, tuple)) else list(cols)
    start_idx, end_idx = missing_feature_range
    
    # Separate columns by whether they're in the missing feature range
    cols_outside_range = [col for col in c if not (start_idx <= col <= end_idx)]
    cols_inside_range = [col for col in c if start_idx <= col <= end_idx]
    
    # Transform columns outside the range for all rows
    if cols_outside_range:
        x_outside = t[:, cols_outside_range]
        t[:, cols_outside_range] = torch.sign(x_outside) * torch.log1p(torch.abs(x_outside))
    
    # Transform columns inside the range only for valid rows
    if cols_inside_range:
        # Create mask for rows with non-zero features in the specified range
        range_features = t[:, start_idx:end_idx+1]
        valid_mask = torch.any(range_features != 0, dim=1)  # [num_points] boolean mask
        
        if torch.any(valid_mask):
            # Apply transform only to valid rows and specified columns
            valid_indices = torch.where(valid_mask)[0]
            x_inside = t[valid_indices][:, cols_inside_range]
            transformed = torch.sign(x_inside) * torch.log1p(torch.abs(x_inside))
            t[valid_indices[:, None], cols_inside_range] = transformed
        
        n_transformed = torch.sum(valid_mask).item()
        n_skipped = torch.sum(~valid_mask).item()
        print(f"  Features {start_idx}-{end_idx}: transformed {n_transformed} rows, skipped {n_skipped} all-zero rows")
    
    return t.to(orig_dtype)

def transform_dir(
    in_dir: str, 
    out_dir: str, 
    cols: Union[List[int], Tuple[int, ...]], 
    missing_feature_range: Tuple[int, int] = (57, 110),
    use_conditional: bool = True
):
    """
    Transform tensors in a directory with optional conditional logic.
    
    Args:
        in_dir: Input directory containing tensor files
        out_dir: Output directory for transformed tensors
        cols: Column indices to transform
        missing_feature_range: Range for conditional logic (default: 57, 110)
        use_conditional: If True, use smart conditional logic; if False, transform all rows
    """
    os.makedirs(out_dir, exist_ok=True)
    patterns = ("*.pt", "*.pth", "*.pkl")
    files = sorted([p for pat in patterns for p in glob.glob(os.path.join(in_dir, pat))])
    if not files:
        raise FileNotFoundError(f"No tensors found in {in_dir} (looked for {patterns})")

    total_rows_processed = 0
    total_rows_transformed = 0
    
    with torch.no_grad():
        for i, src in enumerate(files):
            print(f"Processing file {i+1}/{len(files)}: {os.path.basename(src)}")
            obj = torch.load(src, map_location="cpu")
            if not isinstance(obj, torch.Tensor):
                raise TypeError(f"{src} does not contain a torch.Tensor")
            
            original_rows = obj.shape[0]
            
            if use_conditional:
                t = signed_log1p_smart(obj, cols, missing_feature_range)
                # Count transformed rows for features in the missing range
                start_idx, end_idx = missing_feature_range
                range_features = obj[:, start_idx:end_idx+1]
                valid_mask = torch.any(range_features != 0, dim=1)
                rows_transformed = torch.sum(valid_mask).item()
                # Add rows for features outside the range (all rows transformed)
                cols_outside = [c for c in cols if not (start_idx <= c <= end_idx)]
                if cols_outside:
                    rows_transformed = original_rows  # All rows get some transformation
            else:
                t = signed_log1p_inplace(obj, cols)
                rows_transformed = original_rows
            
            total_rows_processed += original_rows
            total_rows_transformed += rows_transformed
            
            dst = os.path.join(out_dir, os.path.basename(src))
            torch.save(t, dst)
    
    print(f"\nSummary:")
    print(f"Total files processed: {len(files)}")
    print(f"Total rows processed: {total_rows_processed}")
    if use_conditional:
        print(f"Conditional transformation applied for range {missing_feature_range}")

def transform_dir_legacy(in_dir: str, out_dir: str, cols):
    """Legacy function for backward compatibility - transforms all rows."""
    return transform_dir(in_dir, out_dir, cols, use_conditional=False)

if __name__ == "__main__":
    # Example configuration
    in_dir = '/mnt/letitia/scratch/H_data/SAME/tensor_features/TP0'
    out_dir = '/mnt/letitia/scratch/H_data/SAME/tensor_features_transformed/TP0'
    
    # Features identified as skewed from your analysis
    skewed_features = [11, 16, 17, 18, 21, 22, 24, 25, 26, 27, 29, 30, 34, 35, 37, 38, 39, 40, 42, 44, 45, 50, 56, 65, 68, 70, 73, 86, 88, 89, 91, 92, 93, 94, 98, 99, 108, 110]
    
    print("Applying smart signed-log1p transformation...")
    print(f"Features to transform: {skewed_features}")
    print("Conditional logic: Features 57-110 will only be transformed for rows where they are not all zeros")
    print("Other features will be transformed for all rows")
    
    # Apply smart transformation
    transform_dir(
        in_dir=in_dir,
        out_dir=out_dir, 
        cols=skewed_features,
        missing_feature_range=(57, 110),
        use_conditional=True
    )
    
    print(f"\nTransformation complete. Output saved to: {out_dir}")
    
    # Alternative: Use legacy approach (transform all rows for all features)
    # transform_dir_legacy(in_dir, out_dir, skewed_features)
