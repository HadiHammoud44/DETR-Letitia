# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Metrics for temporal point cloud forecasting evaluation with hierarchical classification support
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


def compute_hierarchical_class_metrics(pred_superclass: torch.Tensor, pred_subclass: torch.Tensor,
                                     tgt_superclass: torch.Tensor, tgt_subclass: torch.Tensor,
                                     no_obj_superclass: int) -> Dict[str, float]:
    """
    Compute hierarchical classification metrics: superclass and subclass accuracy, precision, recall, f1
    
    Args:
        pred_superclass: [N] tensor of predicted superclass labels
        pred_subclass: [N] tensor of predicted subclass labels
        tgt_superclass: [N] tensor of target superclass labels
        tgt_subclass: [N] tensor of target subclass labels
        no_obj_superclass: superclass index representing "no object"
        
    Returns:
        Dict with superclass and subclass metrics, plus combined hierarchical accuracy
    """
    if len(pred_superclass) == 0:
        return {
            'superclass_accuracy': 0.0, 'superclass_precision': 0.0, 'superclass_recall': 0.0, 'superclass_f1': 0.0,
            'subclass_accuracy': 0.0, 'subclass_precision': 0.0, 'subclass_recall': 0.0, 'subclass_f1': 0.0,
            'hierarchical_accuracy': 0.0
        }
    
    # Compute superclass metrics
    superclass_metrics = compute_class_metrics(pred_superclass, tgt_superclass)
    
    # Compute subclass metrics only for objects (where superclass < no_obj_superclass)
    object_mask = tgt_superclass < no_obj_superclass
    if object_mask.sum() > 0:
        object_pred_subclass = pred_subclass[object_mask]
        object_tgt_subclass = tgt_subclass[object_mask]
        subclass_metrics = compute_class_metrics(object_pred_subclass, object_tgt_subclass)
    else:
        subclass_metrics = {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    # Hierarchical accuracy: both superclass and subclass must be correct (only for objects)
    if object_mask.sum() > 0:
        object_pred_superclass = pred_superclass[object_mask]
        object_tgt_superclass = tgt_superclass[object_mask]
        hierarchical_correct = (object_pred_superclass == object_tgt_superclass) & (object_pred_subclass == object_tgt_subclass)
        hierarchical_accuracy = hierarchical_correct.float().mean().item()
    else:
        hierarchical_accuracy = 0.0
    
    return {
        'superclass_accuracy': superclass_metrics['accuracy'],
        'superclass_precision': superclass_metrics['precision'],
        'superclass_recall': superclass_metrics['recall'],
        'superclass_f1': superclass_metrics['f1'],
        'subclass_accuracy': subclass_metrics['accuracy'],
        'subclass_precision': subclass_metrics['precision'],
        'subclass_recall': subclass_metrics['recall'],
        'subclass_f1': subclass_metrics['f1'],
        'hierarchical_accuracy': hierarchical_accuracy
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
    Compute coordinate prediction errors (mean L2 distance and MSE)
    
    Args:
        pred_coords: [N, 3] tensor of predicted coordinates
        tgt_coords: [N, 3] tensor of target coordinates
        
    Returns:
        Dict with mean L2 error and MSE value
    """
    l2_errors = torch.norm(pred_coords - tgt_coords, dim=-1)  # [N]
    mse_error = torch.mean(l2_errors ** 2)
    
    return {
        'coord_l2': l2_errors.mean().item(),
        'coord_mse': mse_error.item()
    }


def compute_feature_error(pred_features: torch.Tensor, tgt_features: torch.Tensor) -> Dict[str, float]:
    """
    Compute radiomics feature prediction errors (MSE)
    
    Args:
        pred_features: [N, num_radiomics] tensor of predicted features
        tgt_features: [N, num_radiomics] tensor of target features
        
    Returns:
        Dict with MSE value
    """
    mse = torch.sum((pred_features - tgt_features) ** 2) / pred_features.shape[0]
    
    return {
        'radiomics_mse': mse.item(),
    }


def compute_lesion_disappearance_metrics(t0_classes: torch.Tensor, t1_classes: torch.Tensor, 
                                       t1_pred_classes: torch.Tensor, t2_pred_classes: torch.Tensor,
                                       t2_tgt_classes: torch.Tensor, no_obj_class: int, 
                                       transition: str = "T1_T2") -> Dict[str, float]:
    """
    Compute lesion disappearance metrics for temporal transitions.
    
    Args:
        t0_classes: [N0] tensor of actual classes at T0 (for T0->T1 transition)
        t1_classes: [N1] tensor of actual classes at T1
        t1_pred_classes: [N1] tensor of predicted classes at T1 (for T0->T1 transition)
        t2_pred_classes: [N2] tensor of predicted classes at T2  
        t2_tgt_classes: [N2] tensor of target classes at T2
        no_obj_class: class index representing "no object"
        transition: Either "T0_T1" or "T1_T2" to specify which transition to analyze
        
    Returns:
        Dict with disappearance confusion matrix: tp, fp, fn, tn
    """
    if transition == "T0_T1":
        # Analyze T0 -> T1 transition
        source_classes = t0_classes
        pred_classes = t1_pred_classes
        tgt_classes = t1_classes
    elif transition == "T1_T2":
        # Analyze T1 -> T2 transition (original functionality)
        source_classes = t1_classes
        pred_classes = t2_pred_classes
        tgt_classes = t2_tgt_classes
    else:
        raise ValueError(f"Unknown transition: {transition}. Must be 'T0_T1' or 'T1_T2'")
    
    if len(source_classes) == 0:
        return {'disappear_tp': 0, 'disappear_fp': 0, 'disappear_fn': 0, 'disappear_tn': 0}
    
    tp = fp = fn = tn = 0
    
    for lesion_class in source_classes:
        # Check if this lesion class truly disappears (not present in target)
        truly_disappears = lesion_class not in tgt_classes
        
        # Check if model predicts disappearance (lesion class not in predictions)
        predicted_disappears = lesion_class not in pred_classes
        
        if truly_disappears and predicted_disappears:
            tp += 1  # Correctly predicted disappearance
        elif not truly_disappears and predicted_disappears:
            fp += 1  # Falsely predicted disappearance (lesion actually remains)
        elif truly_disappears and not predicted_disappears:
            fn += 1  # Failed to predict disappearance (lesion actually disappears)
        else:  # not truly_disappears and not predicted_disappears
            tn += 1  # Correctly predicted persistence
    
    return {
        'disappear_tp': tp,
        'disappear_fp': fp, 
        'disappear_fn': fn,
        'disappear_tn': tn
    }


def compute_lesion_appearance_metrics(t0_classes: torch.Tensor, t1_classes: torch.Tensor,
                                    t1_pred_classes: torch.Tensor, t2_pred_classes: torch.Tensor,
                                    t2_tgt_classes: torch.Tensor, no_obj_class: int, 
                                    max_queries: int, transition: str = "T1_T2") -> Dict[str, float]:
    """
    Compute lesion appearance metrics for temporal transitions.
    
    Args:
        t0_classes: [N0] tensor of actual classes at T0 (for T0->T1 transition)
        t1_classes: [N1] tensor of actual classes at T1
        t1_pred_classes: [N1] tensor of predicted classes at T1 (for T0->T1 transition)
        t2_pred_classes: [N2] tensor of predicted classes at T2
        t2_tgt_classes: [N2] tensor of target classes at T2  
        no_obj_class: class index representing "no object"
        max_queries: maximum number of possible lesion slots
        transition: Either "T0_T1" or "T1_T2" to specify which transition to analyze
        
    Returns:
        Dict with appearance confusion matrix: tp, fp, fn, tn
    """
    if transition == "T0_T1":
        # Analyze T0 -> T1 transition
        source_lesions = set(t0_classes.tolist())
        tgt_lesions = set(t1_classes.tolist())
        pred_lesions = set(t1_pred_classes.tolist())
        num_source_lesions = len(source_lesions)
    elif transition == "T1_T2":
        # Analyze T1 -> T2 transition (original functionality)
        source_lesions = set(t1_classes.tolist())
        tgt_lesions = set(t2_tgt_classes.tolist())
        pred_lesions = set(t2_pred_classes.tolist())
        num_source_lesions = len(source_lesions)
    else:
        raise ValueError(f"Unknown transition: {transition}. Must be 'T0_T1' or 'T1_T2'")
    
    # Find new lesions that appear
    truly_new_lesions = tgt_lesions - source_lesions
    predicted_new_lesions = pred_lesions - source_lesions
    
    # Calculate empty slots (potential for new lesions)
    empty_slots = max_queries - num_source_lesions
    
    if empty_slots <= 0:
        return {'appear_tp': 0, 'appear_fp': 0, 'appear_fn': 0, 'appear_tn': 0}
    
    tp = len(truly_new_lesions & predicted_new_lesions)  # Correctly predicted new lesions
    fp = len(predicted_new_lesions - truly_new_lesions)  # Falsely predicted new lesions
    fn = len(truly_new_lesions - predicted_new_lesions)  # Missed new lesions
    tn = max(0, empty_slots - len(truly_new_lesions) - fp)  # Correctly predicted no new lesion (fixed: prevent negative)
    
    return {
        'appear_tp': tp,
        'appear_fp': fp,
        'appear_fn': fn,
        'appear_tn': tn
    }


def compute_comprehensive_evaluation_metrics(outputs_1: Dict, outputs_2: Dict, 
                                           targets_t0: List[Dict], targets_t1: List[Dict], targets_t2: List[Dict],
                                           indices_1: List[Tuple], indices_2: List[Tuple],
                                           no_obj_class: int, max_queries: int) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics for hierarchical temporal forecasting.
    Integrates all metrics computation in a single function for efficiency.
    
    Args:
        outputs_1: Model outputs for T0->T1 prediction
        outputs_2: Model outputs for T1->T2 prediction  
        targets_t0: Ground truth at T0 (for T0->T1 dynamics)
        targets_t1: Ground truth at T1
        targets_t2: Ground truth at T2
        indices_1: Hungarian matching indices for T0->T1
        indices_2: Hungarian matching indices for T1->T2
        no_obj_class: No-object class index
        max_queries: Maximum number of queries
        
    Returns:
        Comprehensive metrics dictionary
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
        
        # Hierarchical classification
        matched_pred_1_super_logits = outputs_1['pred_superclass'][batch_idx_1, src_idx_1]
        matched_pred_1_sub_logits = outputs_1['pred_subclass'][batch_idx_1, src_idx_1]
        matched_pred_1_super_classes = torch.argmax(matched_pred_1_super_logits, dim=-1)
        matched_pred_1_sub_classes = torch.argmax(matched_pred_1_sub_logits, dim=-1)
        
        matched_tgt_1_super_classes = torch.cat([t['superclass'][i] for t, (_, i) in zip(targets_t1, indices_1)], dim=0)
        matched_tgt_1_sub_classes = torch.cat([t['subclass'][i] for t, (_, i) in zip(targets_t1, indices_1)], dim=0)
        
        # Compute hierarchical classification metrics
        if len(matched_pred_1_super_classes) > 0:
            t1_class_metrics = compute_hierarchical_class_metrics(
                matched_pred_1_super_classes, matched_pred_1_sub_classes,
                matched_tgt_1_super_classes, matched_tgt_1_sub_classes, no_obj_class
            )
            metrics.update({f"t1_{k}": v for k, v in t1_class_metrics.items()})
        
        # Spatial and feature metrics
        matched_pred_1_coords = outputs_1['pred_coordinates'][batch_idx_1, src_idx_1]
        matched_pred_1_radiomics = outputs_1['pred_radiomics'][batch_idx_1, src_idx_1]
        matched_tgt_1_coords = torch.cat([t['coordinates'][i] for t, (_, i) in zip(targets_t1, indices_1)], dim=0)
        matched_tgt_1_radiomics = torch.cat([t['radiomics'][i] for t, (_, i) in zip(targets_t1, indices_1)], dim=0)
        
        if len(matched_pred_1_coords) > 0:
            t1_coord_metrics = compute_coordinate_error(matched_pred_1_coords, matched_tgt_1_coords)
            t1_feature_metrics = compute_feature_error(matched_pred_1_radiomics, matched_tgt_1_radiomics)
            metrics.update({f"t1_{k}": v for k, v in t1_coord_metrics.items()})
            metrics.update({f"t1_{k}": v for k, v in t1_feature_metrics.items()})
    
    # === T1->T2 METRICS ===
    if any(len(pred_idx) > 0 for pred_idx, _ in indices_2):
        batch_idx_2, src_idx_2 = _get_src_permutation_idx(indices_2)
        
        # Hierarchical classification
        matched_pred_2_super_logits = outputs_2['pred_superclass'][batch_idx_2, src_idx_2]
        matched_pred_2_sub_logits = outputs_2['pred_subclass'][batch_idx_2, src_idx_2]
        matched_pred_2_super_classes = torch.argmax(matched_pred_2_super_logits, dim=-1)
        matched_pred_2_sub_classes = torch.argmax(matched_pred_2_sub_logits, dim=-1)
        
        matched_tgt_2_super_classes = torch.cat([t['superclass'][i] for t, (_, i) in zip(targets_t2, indices_2)], dim=0)
        matched_tgt_2_sub_classes = torch.cat([t['subclass'][i] for t, (_, i) in zip(targets_t2, indices_2)], dim=0)
        
        # Compute hierarchical classification metrics
        if len(matched_pred_2_super_classes) > 0:
            t2_class_metrics = compute_hierarchical_class_metrics(
                matched_pred_2_super_classes, matched_pred_2_sub_classes,
                matched_tgt_2_super_classes, matched_tgt_2_sub_classes, no_obj_class
            )
            metrics.update({f"t2_{k}": v for k, v in t2_class_metrics.items()})
        
        # Spatial and feature metrics
        matched_pred_2_coords = outputs_2['pred_coordinates'][batch_idx_2, src_idx_2]
        matched_pred_2_radiomics = outputs_2['pred_radiomics'][batch_idx_2, src_idx_2]
        matched_tgt_2_coords = torch.cat([t['coordinates'][i] for t, (_, i) in zip(targets_t2, indices_2)], dim=0)
        matched_tgt_2_radiomics = torch.cat([t['radiomics'][i] for t, (_, i) in zip(targets_t2, indices_2)], dim=0)
        
        if len(matched_pred_2_coords) > 0:
            t2_coord_metrics = compute_coordinate_error(matched_pred_2_coords, matched_tgt_2_coords)
            t2_feature_metrics = compute_feature_error(matched_pred_2_radiomics, matched_tgt_2_radiomics)
            metrics.update({f"t2_{k}": v for k, v in t2_coord_metrics.items()})
            metrics.update({f"t2_{k}": v for k, v in t2_feature_metrics.items()})
    
    return metrics


def compute_temporal_dynamics_metrics(outputs_1: Dict, outputs_2: Dict,
                                    targets_t0: List[Dict], targets_t1: List[Dict], targets_t2: List[Dict],
                                    no_obj_class: int, max_queries: int) -> Dict[str, float]:
    """
    Compute lesion dynamics metrics for both T0->T1 and T1->T2 transitions.
    
    Args:
        outputs_1: Model outputs for T0->T1 prediction
        outputs_2: Model outputs for T1->T2 prediction
        targets_t0: Ground truth at T0
        targets_t1: Ground truth at T1
        targets_t2: Ground truth at T2
        no_obj_class: No-object class index
        max_queries: Maximum number of queries
        
    Returns:
        Dictionary with temporal dynamics metrics
    """
    batch_size = len(targets_t1)
    
    # Collect metrics for both transitions
    t0_t1_disappear_metrics = []
    t0_t1_appear_metrics = []
    t1_t2_disappear_metrics = []
    t1_t2_appear_metrics = []
    
    for i in range(batch_size):
        # Extract class predictions and targets - use superclass for object filtering, subclass for temporal tracking
        pred_1_super_logits = outputs_1['pred_superclass'][i]
        pred_2_super_logits = outputs_2['pred_superclass'][i]
        pred_1_sub_logits = outputs_1['pred_subclass'][i]
        pred_2_sub_logits = outputs_2['pred_subclass'][i]
        
        # Use subclass for individual lesion tracking (temporal dynamics)
        tgt_0_classes = targets_t0[i]['subclass']
        tgt_1_classes = targets_t1[i]['subclass']
        tgt_2_classes = targets_t2[i]['subclass']
        
        # Use superclass to filter valid predictions (non-objects)
        pred_1_super_classes = torch.argmax(pred_1_super_logits, dim=-1)
        pred_2_super_classes = torch.argmax(pred_2_super_logits, dim=-1)
        valid_pred_1 = pred_1_super_classes != no_obj_class
        valid_pred_2 = pred_2_super_classes != no_obj_class
        
        # Use subclass for actual temporal dynamics tracking
        pred_1_classes = torch.argmax(pred_1_sub_logits, dim=-1)
        pred_2_classes = torch.argmax(pred_2_sub_logits, dim=-1)
        
        # T0->T1 dynamics
        t0_t1_disappear = compute_lesion_disappearance_metrics(
            tgt_0_classes, tgt_1_classes, pred_1_classes[valid_pred_1], 
            pred_2_classes[valid_pred_2], tgt_2_classes, no_obj_class, transition="T0_T1"
        )
        t0_t1_disappear_metrics.append(t0_t1_disappear)
        
        t0_t1_appear = compute_lesion_appearance_metrics(
            tgt_0_classes, tgt_1_classes, pred_1_classes[valid_pred_1],
            pred_2_classes[valid_pred_2], tgt_2_classes, no_obj_class, 
            max_queries, transition="T0_T1"
        )
        t0_t1_appear_metrics.append(t0_t1_appear)
        
        # T1->T2 dynamics (original functionality)
        t1_t2_disappear = compute_lesion_disappearance_metrics(
            tgt_0_classes, tgt_1_classes, pred_1_classes[valid_pred_1],
            pred_2_classes[valid_pred_2], tgt_2_classes, no_obj_class, transition="T1_T2"
        )
        t1_t2_disappear_metrics.append(t1_t2_disappear)
        
        t1_t2_appear = compute_lesion_appearance_metrics(
            tgt_0_classes, tgt_1_classes, pred_1_classes[valid_pred_1],
            pred_2_classes[valid_pred_2], tgt_2_classes, no_obj_class,
            max_queries, transition="T1_T2"
        )
        t1_t2_appear_metrics.append(t1_t2_appear)
    
    # Aggregate metrics
    def aggregate_confusion_matrix_metrics(metric_list, prefix):
        """Aggregate TP/TN/FP/FN counts and compute final metrics"""
        if not metric_list:
            return {
                f'{prefix}_accuracy': 0.0, f'{prefix}_precision': 0.0, 
                f'{prefix}_recall': 0.0, f'{prefix}_f1': 0.0
            }
        
        # Sum up all confusion matrix values
        total_tp = sum(m[f'{prefix}_tp'] for m in metric_list if f'{prefix}_tp' in m)
        total_fp = sum(m[f'{prefix}_fp'] for m in metric_list if f'{prefix}_fp' in m)
        total_fn = sum(m[f'{prefix}_fn'] for m in metric_list if f'{prefix}_fn' in m)
        total_tn = sum(m[f'{prefix}_tn'] for m in metric_list if f'{prefix}_tn' in m)
        
        total = total_tp + total_fp + total_fn + total_tn
        if total == 0:
            return {
                f'{prefix}_accuracy': 0.0, f'{prefix}_precision': 0.0,
                f'{prefix}_recall': 0.0, f'{prefix}_f1': 0.0
            }
        
        # Compute metrics from aggregated counts
        accuracy = (total_tp + total_tn) / total
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            f'{prefix}_accuracy': accuracy,
            f'{prefix}_precision': precision, 
            f'{prefix}_recall': recall,
            f'{prefix}_f1': f1
        }
    
    def aggregate_metrics(metric_list, metric_names):
        if not metric_list:
            return {name: 0.0 for name in metric_names}
        
        aggregated = {}
        for name in metric_names:
            values = [m[name] for m in metric_list if name in m]
            aggregated[name] = float(np.mean(values)) if values else 0.0
        return aggregated
    
    # Use new aggregation method for confusion matrix based metrics
    t0_t1_disappear_agg = aggregate_confusion_matrix_metrics(t0_t1_disappear_metrics, 'disappear')
    t0_t1_appear_agg = aggregate_confusion_matrix_metrics(t0_t1_appear_metrics, 'appear')
    t1_t2_disappear_agg = aggregate_confusion_matrix_metrics(t1_t2_disappear_metrics, 'disappear')
    t1_t2_appear_agg = aggregate_confusion_matrix_metrics(t1_t2_appear_metrics, 'appear')
    
    return {
        # T0->T1 dynamics
        'lesion_disappear_t0_t1_accuracy': t0_t1_disappear_agg['disappear_accuracy'],
        'lesion_disappear_t0_t1_precision': t0_t1_disappear_agg['disappear_precision'],
        'lesion_disappear_t0_t1_recall': t0_t1_disappear_agg['disappear_recall'],
        'lesion_disappear_t0_t1_f1': t0_t1_disappear_agg['disappear_f1'],
        
        'lesion_appear_t0_t1_accuracy': t0_t1_appear_agg['appear_accuracy'],
        'lesion_appear_t0_t1_precision': t0_t1_appear_agg['appear_precision'],
        'lesion_appear_t0_t1_recall': t0_t1_appear_agg['appear_recall'],
        'lesion_appear_t0_t1_f1': t0_t1_appear_agg['appear_f1'],
        
        # T1->T2 dynamics
        'lesion_disappear_t1_t2_accuracy': t1_t2_disappear_agg['disappear_accuracy'],
        'lesion_disappear_t1_t2_precision': t1_t2_disappear_agg['disappear_precision'],
        'lesion_disappear_t1_t2_recall': t1_t2_disappear_agg['disappear_recall'],
        'lesion_disappear_t1_t2_f1': t1_t2_disappear_agg['disappear_f1'],
        
        'lesion_appear_t1_t2_accuracy': t1_t2_appear_agg['appear_accuracy'],
        'lesion_appear_t1_t2_precision': t1_t2_appear_agg['appear_precision'],
        'lesion_appear_t1_t2_recall': t1_t2_appear_agg['appear_recall'],
        'lesion_appear_t1_t2_f1': t1_t2_appear_agg['appear_f1'],
    }
