# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable

import torch
import numpy as np

import util.misc as utils
from util.misc import NestedTensor


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_epochs: int, max_norm: float = 0, use_schedule_sampling: bool = True):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    
    # Classification metrics
    metric_logger.add_meter('superclass_error_step1', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('superclass_error_step2', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    
    # Spatial and feature metrics
    metric_logger.add_meter('coord_error_step1', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('coord_error_step2', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('radiomics_error_step1', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('radiomics_error_step2', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))

    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50

    for batch in metric_logger.log_every(data_loader, print_freq, header):
        # Extract inputs and targets from batch
        # batch = {'inputs': {'T0': NestedTensor, 'T1': NestedTensor}, 
        #          'targets': {'T1': [target_dicts], 'T2': [target_dicts]}}
        inputs = batch['inputs']
        targets = batch['targets']
        
        # Move inputs to device
        baseline = inputs['T0'].to(device)  # T0 -> T1 prediction
        gt_t1 = inputs['T1'].to(device)     # Ground truth T1 for teacher forcing
        
        # Move targets to device  
        targets_t1 = [{k: v.to(device) for k, v in t.items()} for t in targets['T1']]
        targets_t2 = [{k: v.to(device) for k, v in t.items()} for t in targets['T2']]

        # Forward Step 1: T0 -> T1 prediction
        outputs_1 = model(baseline)
        loss_dict_1 = criterion(outputs_1, targets_t1)
        
        # Scheduled sampling probability: linearly increase from 0 to 1
        scheduled_sampling_prob = min(1.0, 2 * epoch / max_epochs) if use_schedule_sampling else 0.0
        use_prediction = torch.rand(1).item() < scheduled_sampling_prob
        
        # Choose input for second forward pass
        if use_prediction:
            # Use model's prediction from step 1
            # ToDo: this was implemented for the synthetic dataset but not yet for letitia
            # since the missing PET features pose a challenge 
            input2 = make_pointcloud_from(outputs_1, mask_thresh=0.5)
        else:
            # Use ground truth T1 (teacher forcing)
            input2 = gt_t1
            
        # Forward Step 2: T1 -> T2 prediction
        outputs_2 = model(input2)
        loss_dict_2 = criterion(outputs_2, targets_t2)
        
        # Simple loss calculation - single GPU only
        weight_dict = criterion.weight_dict
        
        # Calculate weighted losses for each step
        loss_1 = sum(loss_dict_1[k] * weight_dict[k] for k in loss_dict_1.keys() if k in weight_dict)
        loss_2 = sum(loss_dict_2[k] * weight_dict[k] for k in loss_dict_2.keys() if k in weight_dict)
        
        # Total loss (gradients flow through both steps)
        total_loss = loss_1 + loss_2
        
        # Extract classification errors
        superclass_error_1 = loss_dict_1.get('superclass_error', torch.tensor(0.0))
        superclass_error_2 = loss_dict_2.get('superclass_error', torch.tensor(0.0))
        
        # Extract spatial and feature errors
        coord_error_1 = loss_dict_1.get('loss_coordinates', torch.tensor(0.0))
        coord_error_2 = loss_dict_2.get('loss_coordinates', torch.tensor(0.0))
        radiomics_error_1 = loss_dict_1.get('loss_radiomics', torch.tensor(0.0))
        radiomics_error_2 = loss_dict_2.get('loss_radiomics', torch.tensor(0.0))

        # Check for invalid loss
        if not math.isfinite(total_loss.item()):
            print("Loss is {}, stopping training".format(total_loss.item()))
            print("Step 1 losses:", {k: v.item() for k, v in loss_dict_1.items()})
            print("Step 2 losses:", {k: v.item() for k, v in loss_dict_2.items()})
            sys.exit(1)

        # Backpropagation
        optimizer.zero_grad()
        total_loss.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        # Classification logging
        metric_logger.update(
            loss=total_loss.item(),
            loss_step1=loss_1.item(),
            loss_step2=loss_2.item(),
            superclass_error_step1=superclass_error_1.item(),
            superclass_error_step2=superclass_error_2.item(),
            coord_error_step1=coord_error_1.item(),
            coord_error_step2=coord_error_2.item(),
            radiomics_error_step1=radiomics_error_1.item(),
            radiomics_error_step2=radiomics_error_2.item(),
            scheduled_sampling_prob=scheduled_sampling_prob,
            using_prediction=float(use_prediction),
            lr=optimizer.param_groups[0]["lr"]
        )
    # Single GPU - no need for synchronization
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, data_loader, device, args):
    """
    Evaluate temporal point cloud forecasting model using integrated metrics.
    
    Args:
        model: DETR model with classification
        criterion: Loss criterion
        data_loader: Validation data loader
        device: Device to run evaluation on
        
    Returns:
        Dict with comprehensive evaluation metrics
    """
    model.eval()
    criterion.eval()
    
    from util.metrics import compute_comprehensive_evaluation_metrics
    from models.matcher import HungarianMatcher
    
    # Create matcher for evaluation
    matcher = HungarianMatcher(
        cost_superclass=args.set_cost_superclass, 
        cost_coordinates=args.set_cost_coordinates, cost_radiomics=args.set_cost_radiomics
    ) 
    
    # Track all metrics across batches
    all_static_metrics = []

    no_obj_superclass = args.num_superclasses  # Assuming last superclass is no_object

    print("Evaluating temporal point cloud forecasting...")
    
    for batch_idx, batch in enumerate(data_loader):
        # Extract inputs and targets from batch
        inputs = batch['inputs']
        targets = batch['targets']
        
        # Move inputs to device
        baseline = inputs['T0'].to(device)  # T0 -> T1 prediction

        # Move targets to device
        targets_t0 = [{k: v.to(device) for k, v in t.items()} for t in targets['T0']]
        targets_t1 = [{k: v.to(device) for k, v in t.items()} for t in targets['T1']]
        targets_t2 = [{k: v.to(device) for k, v in t.items()} for t in targets['T2']]

        # Stage 1: T0 -> T1 prediction
        outputs_1 = model(baseline)
        
        # Stage 2: T1 -> T2 prediction (always use model output, no teacher forcing)
        # input2 = make_pointcloud_from(outputs_1, mask_thresh=0.5)
        input2 = inputs['T1'].to(device)
        outputs_2 = model(input2)
        
        # Get matching indices using HungarianMatcher
        indices_1 = matcher(outputs_1, targets_t1)
        indices_2 = matcher(outputs_2, targets_t2)
        
        # Compute comprehensive static metrics (classification, coordinates, features)
        static_metrics = compute_comprehensive_evaluation_metrics(
            outputs_1, outputs_2, targets_t0, targets_t1, targets_t2,
            indices_1, indices_2, no_obj_superclass, 
            max_queries=outputs_1['pred_coordinates'].shape[1]
        )
        all_static_metrics.append(static_metrics)
        
        if batch_idx % 10 == 0:
            print(f"Processed {batch_idx + 1} batches")
    
    # Aggregate all metrics across batches
    def aggregate_batch_metrics(metrics_list):
        if not metrics_list:
            return {}
        
        aggregated = {}
        # Get all unique metric keys
        all_keys = set()
        for metrics in metrics_list:
            all_keys.update(metrics.keys())
        
        # Aggregate each metric
        for key in all_keys:
            values = [metrics.get(key, 0.0) for metrics in metrics_list]
            aggregated[key] = float(np.mean(values))
        
        return aggregated
    
    # Aggregate static metrics
    static_agg = aggregate_batch_metrics(all_static_metrics)
    
    # Use static metrics as results
    results = {}
    
    # Static metrics (classification, coordinates, features)
    for key, value in static_agg.items():
        results[key] = value
    
    # Print comprehensive results
    print("Evaluation Results:")
    print("=" * 60)
    
    # T0->T1 Prediction Results
    if any(key.startswith('t1_') for key in results.keys()):
        print("T0 -> T1 Prediction:")
        if 't1_coord_error_mean' in results:
            print(f"  Coordinate Error: {results['t1_coord_error_mean']:.4f}")
        if 't1_feature_error_mean' in results:
            print(f"  Feature Error: {results['t1_feature_error_mean']:.4f}")
        if 't1_superclass_accuracy' in results:
            print(f"  Superclass: Acc={results['t1_superclass_accuracy']:.3f}, F1={results.get('t1_superclass_f1', 0.0):.3f}")
        print()
    
    # T1->T2 Prediction Results
    if any(key.startswith('t2_') for key in results.keys()):
        print("T1 -> T2 Prediction:")
        if 't2_coord_error_mean' in results:
            print(f"  Coordinate Error: {results['t2_coord_error_mean']:.4f}")
        if 't2_feature_error_mean' in results:
            print(f"  Feature Error: {results['t2_feature_error_mean']:.4f}")
        if 't2_superclass_accuracy' in results:
            print(f"  Superclass: Acc={results['t2_superclass_accuracy']:.3f}, F1={results.get('t2_superclass_f1', 0.0):.3f}")
        print()
    
    return results


def make_pointcloud_from(outputs, mask_thresh=0.5):
    """
    Reconstruct point cloud format from model outputs.
    
    Args:
        outputs: Dict containing:
            - 'pred_superclass': [batch_size, num_queries, num_superclasses + 1]
            - 'pred_coordinates': [batch_size, num_queries, 3] 
            - 'pred_radiomics': [batch_size, num_queries, num_radiomics]
        mask_thresh: Threshold for determining valid points based on no_object probability
        
    Returns:
        NestedTensor with:
            - tensors: [batch_size, max_points, total_features] where features = [x,y,z,radiomics...,superclass]
            - mask: [batch_size, max_points] - True for padding, False for valid points
    """
    
    # Classification
    pred_super_probs = torch.softmax(outputs['pred_superclass'], dim=-1)  # [batch, queries, num_superclasses + 1]
    
    # Assuming last superclass is no_object, mask where no_object prob <= threshold
    no_obj_probs = pred_super_probs[..., -1]  # [batch, queries]
    valid_mask = no_obj_probs <= mask_thresh  # [batch, queries] - True for valid points
    
    # Get predicted classes (excluding no_object class for superclass)
    pred_superclasses = torch.argmax(pred_super_probs[..., :-1], dim=-1).float()  # [batch, queries]
    
    # Prepare class features
    superclasses = pred_superclasses.unsqueeze(-1)  # [batch, queries, 1]
    
    # Concatenate coordinates + radiomics + class features to form full feature vector
    coordinates = outputs['pred_coordinates']  # [batch, queries, 3]
    radiomics = outputs['pred_radiomics']     # [batch, queries, num_radiomics]
    
    # Combine all features: [x, y, z, radiomics..., superclass]
    point_features = torch.cat([coordinates, radiomics, superclasses], dim=-1)  # [batch, queries, total_features]
    
    # Create NestedTensor - mask follows NestedTensor convention (True = padding)
    return NestedTensor(point_features, ~valid_mask)  # ~valid_mask: True=padding, False=valid
