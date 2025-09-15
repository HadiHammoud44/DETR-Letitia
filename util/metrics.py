# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Metrics for temporal point cloud forecasting evaluation
"""
import torch
import numpy as np
from typing import Dict, List, Tuple


def compute_metrics_from_confusion_matrix(tp: int, fp: int, fn: int, tn: int) -> Dict[str, float]:
    """
    Compute classification metrics from confusion matrix values.
    
    Args:
        tp: True positives
        fp: False positives  
        fn: False negatives
        tn: True negatives
        
    Returns:
        Dict with accuracy, precision, recall, f1 scores
    """
    total = tp + fp + fn + tn
    if total == 0:
        return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def compute_class_metrics_only(pred_superclass: torch.Tensor,
                              tgt_superclass: torch.Tensor,
                              no_obj_superclass: int) -> Dict[str, float]:
    """
    Compute classification metrics: superclass accuracy, precision, recall, f1
    
    Args:
        pred_superclass: [N] tensor of predicted superclass labels
        tgt_superclass: [N] tensor of target superclass labels
        no_obj_superclass: superclass index representing "no object"
        
    Returns:
        Dict with superclass metrics
    """
    if len(pred_superclass) == 0:
        return {
            'superclass_accuracy': 0.0, 'superclass_precision': 0.0, 'superclass_recall': 0.0, 'superclass_f1': 0.0
        }
    
    # Compute superclass metrics
    superclass_metrics = compute_class_metrics(pred_superclass, tgt_superclass)
    
    return {
        'superclass_accuracy': superclass_metrics['accuracy'],
        'superclass_precision': superclass_metrics['precision'],
        'superclass_recall': superclass_metrics['recall'],
        'superclass_f1': superclass_metrics['f1']
    }


def compute_class_metrics(pred_classes: torch.Tensor, tgt_classes: torch.Tensor) -> Dict[str, float]:
    """
    Compute classification metrics: accuracy, precision, recall, f1
    
    Args:
        pred_classes: [N] tensor of predicted class labels
        tgt_classes: [N] tensor of target class labels
        
    Returns:
        Dict with accuracy, precision, recall, f1 scores
    """
    if len(pred_classes) == 0:
        return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    pred_np = pred_classes.cpu().numpy()
    tgt_np = tgt_classes.cpu().numpy()
    
    # Accuracy
    accuracy = np.mean(pred_np == tgt_np)
    
    # Get unique classes
    unique_classes = np.unique(np.concatenate([pred_np, tgt_np]))
    
    precisions = []
    recalls = []
    f1s = []
    
    for cls in unique_classes:
        tp = np.sum((pred_np == cls) & (tgt_np == cls))
        fp = np.sum((pred_np == cls) & (tgt_np != cls))
        fn = np.sum((pred_np != cls) & (tgt_np == cls))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    
    # Weighted average by class frequency
    class_weights = []
    for cls in unique_classes:
        class_weights.append(np.sum(tgt_np == cls))
    
    total_weight = sum(class_weights)
    if total_weight == 0:
        return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    weighted_precision = sum(p * w for p, w in zip(precisions, class_weights)) / total_weight
    weighted_recall = sum(r * w for r, w in zip(recalls, class_weights)) / total_weight
    weighted_f1 = sum(f * w for f, w in zip(f1s, class_weights)) / total_weight
    
    return {
        'accuracy': accuracy,
        'precision': weighted_precision,
        'recall': weighted_recall,
        'f1': weighted_f1
    }


def compute_coordinate_error(pred_coords: torch.Tensor, tgt_coords: torch.Tensor) -> Dict[str, float]:
    """
    Compute coordinate prediction errors (L2 distance)
    
    Args:
        pred_coords: [N, 3] tensor of predicted coordinates
        tgt_coords: [N, 3] tensor of target coordinates
        
    Returns:
        Dict with mean and std of L2 errors
    """
    l2_errors = torch.norm(pred_coords - tgt_coords, dim=-1)  # [N]
    
    return {
        'coord_error_mean': l2_errors.mean().item(),
        'coord_error_std': l2_errors.std().item() if len(l2_errors) > 1 else 0.0
    }


def compute_feature_error(pred_features: torch.Tensor, tgt_features: torch.Tensor, empty_pt: torch.Tensor = None) -> Dict[str, float]:
    """
    Compute selective radiomics feature prediction errors (MSE)
    
    Args:
        pred_features: [N, num_radiomics] tensor of predicted features
        tgt_features: [N, num_radiomics] tensor of target features
        empty_pt: [N] tensor of binary flags indicating empty second half (optional)
        
    Returns:
        Dict with mean and std of MSE errors
    """
    if empty_pt is not None:
        # Same logic as loss_radiomics: selective computation with masking
        radiomics_dim = tgt_features.shape[-1]
        half_dim = radiomics_dim // 2
        
        # First half: always compute error
        pred_first_half = pred_features[..., :half_dim]
        tgt_first_half = tgt_features[..., :half_dim]
        mse_first_half = torch.nn.functional.mse_loss(pred_first_half, tgt_first_half, reduction='none')
        
        # Second half: compute error only where empty_pt = 0 (not empty)
        pred_second_half = pred_features[..., half_dim:]
        tgt_second_half = tgt_features[..., half_dim:]
        mse_second_half = torch.nn.functional.mse_loss(pred_second_half, tgt_second_half, reduction='none')
        
        # Create mask for non-empty points (empty_pt = 0 means valid data)
        non_empty_mask = (empty_pt == 0).float().unsqueeze(-1)  # [N, 1]
        
        # Apply mask to second half error
        masked_mse_second_half = mse_second_half * non_empty_mask
        
        # Combine errors - exactly like loss_radiomics
        total_error = mse_first_half.sum() + masked_mse_second_half.sum()
        
        # Normalize by total number of valid elements - exactly like loss_radiomics
        num_first_half_elements = mse_first_half.numel()
        num_second_half_elements = non_empty_mask.sum().item() * (radiomics_dim - half_dim)
        total_elements = num_first_half_elements + num_second_half_elements
        
        mean_error = total_error / total_elements
        
        # std ignored for masked computation
        std_error = -1.0
        
    else:
        # Standard computation without masking
        mse_errors = torch.nn.functional.mse_loss(pred_features, tgt_features, reduction='none')
        mean_error = mse_errors.mean().item()
        std_error = mse_errors.std().item() if mse_errors.numel() > 1 else 0.0
    
    return {
        'feature_error_mean': mean_error.item() if torch.is_tensor(mean_error) else mean_error,
        'feature_error_std': std_error
    }


def compute_comprehensive_evaluation_metrics(outputs_1: Dict, outputs_2: Dict, 
                                           targets_t0: List[Dict], targets_t1: List[Dict], targets_t2: List[Dict],
                                           indices_1: List[Tuple], indices_2: List[Tuple],
                                           no_obj_class: int, max_queries: int) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics for temporal forecasting.
    Focuses on superclass classification, coordinate prediction, and radiomics prediction.
    
    Args:
        outputs_1: Model outputs for T0->T1 prediction
        outputs_2: Model outputs for T1->T2 prediction  
        targets_t0: Ground truth at T0 (unused since no temporal dynamics)
        targets_t1: Ground truth at T1
        targets_t2: Ground truth at T2
        indices_1: Hungarian matching indices for T0->T1
        indices_2: Hungarian matching indices for T1->T2
        no_obj_class: No-object class index
        max_queries: Maximum number of queries (unused)
        
    Returns:
        Comprehensive metrics dictionary focusing on prediction accuracy
    """
    
    def _get_src_permutation_idx(indices):
        """Helper function from DETR - get flattened indices for predictions"""
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
    
    metrics = {}
    
    # === T0->T1 METRICS ===
    if any(len(pred_idx) > 0 for pred_idx, _ in indices_1):
        batch_idx_1, src_idx_1 = _get_src_permutation_idx(indices_1)
        
        # Classification
        matched_pred_1_super_logits = outputs_1['pred_superclass'][batch_idx_1, src_idx_1]
        matched_pred_1_super_classes = torch.argmax(matched_pred_1_super_logits, dim=-1)
        
        matched_tgt_1_super_classes = torch.cat([t['superclass'][i] for t, (_, i) in zip(targets_t1, indices_1)], dim=0)
        
        # Compute classification metrics
        if len(matched_pred_1_super_classes) > 0:
            t1_class_metrics = compute_class_metrics_only(
                matched_pred_1_super_classes,
                matched_tgt_1_super_classes, no_obj_class
            )
            metrics.update({f"t1_{k}": v for k, v in t1_class_metrics.items()})
        
        # Spatial and feature metrics
        matched_pred_1_coords = outputs_1['pred_coordinates'][batch_idx_1, src_idx_1]
        matched_pred_1_radiomics = outputs_1['pred_radiomics'][batch_idx_1, src_idx_1]
        matched_tgt_1_coords = torch.cat([t['coordinates'][i] for t, (_, i) in zip(targets_t1, indices_1)], dim=0)
        matched_tgt_1_radiomics = torch.cat([t['radiomics'][i] for t, (_, i) in zip(targets_t1, indices_1)], dim=0)
        matched_tgt_1_empty_pt = torch.cat([t['empty_pt'][i] for t, (_, i) in zip(targets_t1, indices_1)], dim=0)
        
        if len(matched_pred_1_coords) > 0:
            t1_coord_metrics = compute_coordinate_error(matched_pred_1_coords, matched_tgt_1_coords)
            t1_feature_metrics = compute_feature_error(matched_pred_1_radiomics, matched_tgt_1_radiomics, matched_tgt_1_empty_pt)
            metrics.update({f"t1_{k}": v for k, v in t1_coord_metrics.items()})
            metrics.update({f"t1_{k}": v for k, v in t1_feature_metrics.items()})
    
    # === T1->T2 METRICS ===
    if any(len(pred_idx) > 0 for pred_idx, _ in indices_2):
        batch_idx_2, src_idx_2 = _get_src_permutation_idx(indices_2)
        
        # Hierarchical classification
        matched_pred_2_super_logits = outputs_2['pred_superclass'][batch_idx_2, src_idx_2]
        matched_pred_2_super_classes = torch.argmax(matched_pred_2_super_logits, dim=-1)
        
        matched_tgt_2_super_classes = torch.cat([t['superclass'][i] for t, (_, i) in zip(targets_t2, indices_2)], dim=0)
        
        # Compute classification metrics
        if len(matched_pred_2_super_classes) > 0:
            t2_class_metrics = compute_class_metrics_only(
                matched_pred_2_super_classes,
                matched_tgt_2_super_classes, no_obj_class
            )
            metrics.update({f"t2_{k}": v for k, v in t2_class_metrics.items()})
        
        # Spatial and feature metrics
        matched_pred_2_coords = outputs_2['pred_coordinates'][batch_idx_2, src_idx_2]
        matched_pred_2_radiomics = outputs_2['pred_radiomics'][batch_idx_2, src_idx_2]
        matched_tgt_2_coords = torch.cat([t['coordinates'][i] for t, (_, i) in zip(targets_t2, indices_2)], dim=0)
        matched_tgt_2_radiomics = torch.cat([t['radiomics'][i] for t, (_, i) in zip(targets_t2, indices_2)], dim=0)
        matched_tgt_2_empty_pt = torch.cat([t['empty_pt'][i] for t, (_, i) in zip(targets_t2, indices_2)], dim=0)
        
        if len(matched_pred_2_coords) > 0:
            t2_coord_metrics = compute_coordinate_error(matched_pred_2_coords, matched_tgt_2_coords)
            t2_feature_metrics = compute_feature_error(matched_pred_2_radiomics, matched_tgt_2_radiomics, matched_tgt_2_empty_pt)
            metrics.update({f"t2_{k}": v for k, v in t2_coord_metrics.items()})
            metrics.update({f"t2_{k}": v for k, v in t2_feature_metrics.items()})
    
    return metrics
