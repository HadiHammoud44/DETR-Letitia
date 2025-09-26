# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Letitia Dataset for dynamic organ/lesion forecasting.
Each sample contains point clouds at 3 time steps (TP0, TP1, TP2) with radiomics features.

Data format per point: [x, y, z, radiomics_features..., superclass_label, subclass_label]
- Coordinates: 3D spatial position (x, y, z)
- Radiomics: Feature vector extracted from medical imaging
- Superclass: High-level anatomical classification (e.g., healthy, benign, malignant)
- Subclass: Fine-grained classification within superclass (e.g., specific lesion types)
"""
import os
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional


class LetitiaDataset(Dataset):
    """
    Dataset for anatomical organ/lesion forecasting task.
    
    Args:
        data_root: Root directory containing timestep folders
        split: 'train', 'val', or 'test'
        split_ratio: Tuple of ratios for train/val/test splits
        transforms: Optional transforms to apply to the data
        seed: Random seed for reproducibility
    """
    
    def __init__(self, data_root: str, split: str = 'train', split_ratio=(0.7, 0.2, 0.1), transforms=None, seed=0):
        self.data_root = Path(data_root)
        self.split = split
        self.split_ratio = split_ratio
        self.transforms = transforms
        self.seed = seed
        
        # Timestep folders
        self.TP0_dir = self.data_root / 'TP0'
        self.TP1_dir = self.data_root / 'TP1' 
        self.TP2_dir = self.data_root / 'TP2'
        
        # Verify directories exist
        for dir_path in [self.TP0_dir, self.TP1_dir, self.TP2_dir]:
            if not dir_path.exists():
                raise ValueError(f"Directory {dir_path} does not exist")
        
        # Get list of sample (patient) IDs
        self.sample_ids = self._get_sample_ids()
        
    def _get_sample_ids(self) -> List[str]:
        """Get list of sample IDs from TP0 directory. Patients are same across timesteps"""
        TP0_files = list(self.TP0_dir.glob('*.pt'))  
        
        sample_ids = [f.stem for f in TP0_files]
                
        # Assign based on split and seed
        np.random.seed(self.seed)
        np.random.shuffle(sample_ids)
        num_train = int(len(sample_ids) * self.split_ratio[0])
        num_val = int(len(sample_ids) * self.split_ratio[1])
        if self.split == 'train':
            sample_ids = sample_ids[:num_train]
        elif self.split == 'val':
            sample_ids = sample_ids[num_train:num_train + num_val]
        elif self.split == 'test':
            sample_ids = sample_ids[num_train + num_val:]
        
        return sample_ids
            
    def __len__(self) -> int:
        return len(self.sample_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dict with keys:
                - 'TP0': Tensor of shape [num_points_TP0, num_features]
                - 'TP1': Tensor of shape [num_points_TP1, num_features] 
                - 'TP2': Tensor of shape [num_points_TP2, num_features]
        """
        sample_id = self.sample_ids[idx]
        
        # Load data for each timestep
        TP0_data = self._load_timestep_data(self.TP0_dir, sample_id)
        TP1_data = self._load_timestep_data(self.TP1_dir, sample_id)
        TP2_data = self._load_timestep_data(self.TP2_dir, sample_id)
        
        sample = {
            'TP0': TP0_data,
            'TP1': TP1_data, 
            'TP2': TP2_data,
            'sample_id': sample_id
        }
        
        if self.transforms:
            sample = self.transforms(sample)
            
        return sample
    
    def _load_timestep_data(self, timestep_dir: Path, sample_id: str) -> torch.Tensor:
        """Load point cloud data for a specific timestep and sample."""

        pt_file = timestep_dir / f"{sample_id}.pt"
        if pt_file.exists():
            data = torch.load(pt_file, weights_only=True)
            return data.float()
        
        raise FileNotFoundError(f"No data file found for sample {sample_id} in {timestep_dir}")


def pad_and_mask(batch_data: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pad a batch of variable-length sequences and create a mask.
    
    Args:
        batch_data: List of tensors, each of shape [num_points, num_features]
        
    Returns:
        padded_tensor: Tensor of shape [batch_size, max_points, num_features]
        mask: Boolean tensor of shape [batch_size, max_points] (True = padded)
    """
    batch_size = len(batch_data)
    if batch_size == 0:
        return torch.empty(0), torch.empty(0, dtype=torch.bool)
    
    # Get dimensions
    num_features = batch_data[0].shape[-1]
    lengths = [tensor.shape[0] for tensor in batch_data]
    max_length = max(lengths)
    
    # Create padded tensor and mask
    padded_tensor = torch.zeros(batch_size, max_length, num_features, dtype=batch_data[0].dtype)
    mask = torch.ones(batch_size, max_length, dtype=torch.bool)  # True = padded
    
    # Fill in the data and update mask
    for i, (tensor, length) in enumerate(zip(batch_data, lengths)):
        padded_tensor[i, :length] = tensor
        mask[i, :length] = False  # False = real data
    
    return padded_tensor, mask


def custom_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict:
    """
    Custom collate function for LetitiaDataset.
    
    Args:
        batch: List of samples from LetitiaDataset.__getitem__
        
    Returns:
        Dict containing:
            - 'inputs': Dict with:
                - 'T0': NestedTensor (padded T0 data for T0→T1 prediction)
                - 'T1': NestedTensor (padded T1 data for T1→T2 prediction)
            - 'targets': Dict with:
                - 'T1': List of target dicts for T0→T1 prediction
                - 'T2': List of target dicts for T1→T2 prediction
            - 'sample_ids': List of sample IDs
    """
    from util.misc import NestedTensor
    
    # Extract data for each timestep
    TP0_data = [sample['TP0'] for sample in batch]
    TP1_data = [sample['TP1'] for sample in batch]
    TP2_data = [sample['TP2'] for sample in batch]
    sample_ids = [sample['sample_id'] for sample in batch]
    
    # Pad T0 and T1 data for inputs
    padded_T0, mask_T0 = pad_and_mask(TP0_data)
    padded_T1, mask_T1 = pad_and_mask(TP1_data)
    
    # Create NestedTensors for inputs
    inputs = {
        'T0': NestedTensor(padded_T0, mask_T0),
        'T1': NestedTensor(padded_T1, mask_T1)
    }
    
    # Create targets in DETR format
    targets = {
        'T0': [_format_target(tp0) for tp0 in TP0_data],  # T0 targets with all keys for temporal dynamics
        'T1': [_format_target(tp1) for tp1 in TP1_data],  # T1 targets for T0→T1 prediction
        'T2': [_format_target(tp2) for tp2 in TP2_data]   # T2 targets for T1→T2 prediction
    }
    
    return {
        'inputs': inputs,
        'targets': targets,
        'sample_ids': sample_ids
    }


def _format_target(point_cloud: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Format a point cloud tensor into DETR target format.
    
    Args:
        point_cloud: Tensor of shape [num_points, num_features]
                    Features: [x, y, z, radiomics..., superclass_label, subclass_label]
    
    Returns:
        Dict with:
            - 'superclass': Superclass labels [num_points]
            - 'subclass': Subclass labels [num_points]
            - 'coordinates': 3D coordinates [num_points, 3] 
            - 'radiomics': Radiomics features [num_points, num_radiomics_features]
    """
    # Extract components from point cloud
    coordinates = point_cloud[:, :3]                    # First 3: x, y, z coordinates
    radiomics = point_cloud[:, 3:-2]                   # Middle: radiomics features
    superclass = point_cloud[:, -2].long()             # Second to last: superclass labels
    subclass = point_cloud[:, -1].long()               # Last: subclass labels
    
    return {
        'superclass': superclass,
        'subclass': subclass,
        'coordinates': coordinates,
        'radiomics': radiomics
    }


def build_letitia_dataset(data_root: str, split: str = 'train', split_ratio=(0.7, 0.2, 0.1), transforms=None, seed=0) -> LetitiaDataset:
    """
    Builder function for LetitiaDataset.
    
    Args:
        data_root: Root directory containing timestep folders
        split: 'train', 'val', or 'test'
        split_ratio: Tuple of ratios for train/val/test splits
        transforms: Optional transforms to apply
        seed: Random seed for reproducibility
        
    Returns:
        LetitiaDataset instance
    """
    return LetitiaDataset(data_root=data_root, split=split, split_ratio=split_ratio, transforms=transforms, seed=seed)


def get_feature_info(data_root: str) -> Dict[str, int]:
    """
    Get information about the feature format from a sample file.
    
    Args:
        data_root: Root directory containing timestep folders
        
    Returns:
        Dict with feature information:
            - 'total_features': Total number of features per point
            - 'num_coordinates': Number of coordinate features (should be 3)
            - 'num_radiomics': Number of radiomics features
            - 'num_superclasses': Number of unique superclass labels (approximate)
            - 'num_subclasses': Number of unique subclass labels (approximate)
    """
    data_root = Path(data_root)
    TP0_dir = data_root / 'TP0'
    
    # Load a sample file to inspect feature format
    sample_files = list(TP0_dir.glob('*.pt'))
    if not sample_files:
        raise ValueError(f"No .pt files found in {TP0_dir}")
    
    sample_data = torch.load(sample_files[0])
    
    total_features = sample_data.shape[1]
    num_coordinates = 3  # x, y, z
    num_radiomics = total_features - 5  # exclude x,y,z, superclass, and subclass
    
    # Estimate number of classes from unique labels in sample
    unique_superclasses = torch.unique(sample_data[:, -2])
    unique_subclasses = torch.unique(sample_data[:, -1])
    num_superclasses = len(unique_superclasses)
    num_subclasses = len(unique_subclasses)
    
    return {
        'total_features': total_features,
        'num_coordinates': num_coordinates, 
        'num_radiomics': num_radiomics,
        'num_superclasses': num_superclasses,
        'num_subclasses': num_subclasses
    }
