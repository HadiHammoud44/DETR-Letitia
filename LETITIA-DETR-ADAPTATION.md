# DETR Adaptation for Temporal Point Cloud Forecasting

**Date**: September 14, 2025  
**Project**: Adapting Facebook's DETR for anatomical organ/lesion forecasting using point clouds  
**Goal**: Transform image-based object detection to temporal point cloud prediction (T0→T1→T2)

## Overview

This document provides the complete technical specification of Facebook's DETR adaptation from image-based object detection to temporal point cloud forecasting for anatomical organs and lesions. The adaptation enables autoregressive prediction over three timesteps (T0→T1→T2) using point clouds with 3D coordinates, radiomics features, and anatomical classifications.

## Key Implementation Details

## Key Implementation Details

### 1. Architecture Adaptations

#### **Backbone Transformation** (`models/backbone.py`)
- **Original**: CNN-based feature extraction for 2D images
- **Adapted**: `PointBackbone` with `PointEmbed` MLP for point cloud processing
- **Key Changes**: 
  - Replaced CNN with MLP that processes point cloud features directly
  - Input: `[batch, max_points, original_feature_size]`
  - Output: `[batch, max_points, hidden_dim]`
  - Maintains DETR's NestedTensor interface for variable-length sequences

#### **Position Encoding Simplification** (`models/position_encoding.py`)
- **Original**: Complex 2D sinusoidal positional encodings for image patches
- **Adapted**: `PositionEmbeddingZero` class returning zeros
- **Rationale**: Point clouds inherently contain 3D spatial information (x,y,z coordinates)
- **Note**: Can be extended to use 3D positional encodings if needed

#### **DETR Model Restructuring** (`models/detr.py`)
- **Original**: Classification + bounding box regression heads
- **Adapted**: Multi-head prediction system:
  - `superclass_embed`: Superclass classification (num_superclasses + 1 for no-object)
  - `radiomics_embed`: Combined coordinates + radiomics prediction
- **Output Structure**:
  ```python
  {
      'pred_superclass': [batch, num_queries, num_superclasses + 1],
      'pred_coordinates': [batch, num_queries, 3],  # x, y, z
      'pred_radiomics': [batch, num_queries, radiomics_dim]
  }
  ```

#### **Hungarian Matcher Update** (`models/matcher.py`)
- **Original**: Box-based matching with GIoU and classification costs
- **Adapted**: Point cloud matching with three cost components:
  - `cost_superclass`: Cross-entropy classification cost
  - `cost_coordinates`: L2 distance for 3D spatial coordinates
  - `cost_radiomics`: L2 distance for radiomics features
- **Matching Algorithm**: Bipartite assignment using Hungarian algorithm

### 2. Data Format and Processing

#### **Point Cloud Data Structure**
Each point contains 113 features organized as:
```
[x, y, z, radiomics_features(108), empty_pt, superclass_label]
```
- **Coordinates**: 3D spatial position (x, y, z)
- **Radiomics**: 108-dimensional radiomics features (54 from CT, 54 from PET)
- **Empty_pt**: Binary flag (0/1) indicating missing PET features in second half
- **Superclass**: Anatomical classification (0=organ, 1=lesion)

#### **Dataset Implementation** (`datasets/letitia_ds.py`)
- **LetitiaDataset**: Handles temporal point cloud sequences (T0, T1, T2)
- **Custom Data Splitting**: Supports either ratio-based or predefined validation sets
- **NestedTensor Support**: Automatic padding with mask handling for variable-length sequences
- **Target Formatting**: Converts point clouds to DETR-compatible target dictionaries

#### **Collate Function**
```python
def custom_collate_fn(batch) -> Dict:
    return {
        'inputs': {
            'T0': NestedTensor(padded_T0, mask_T0),  # For T0→T1 prediction
            'T1': NestedTensor(padded_T1, mask_T1)   # For T1→T2 prediction
        },
        'targets': {
            'T0': [target_dicts], 'T1': [target_dicts], 'T2': [target_dicts]
        }
    }
```

### 3. Training Pipeline (`engine.py`)

#### **Autoregressive Two-Step Training**
The training follows a sequential prediction approach:

1. **Step 1**: T0 → T1 prediction
   - Input: T0 point cloud
   - Target: T1 ground truth
   - Output: Predicted T1 state

2. **Step 2**: T1 → T2 prediction  
   - Input: Ground truth T1 (teacher forcing for LETITIA dataset)
   - Target: T2 ground truth
   - Output: Predicted T2 state

#### **Scheduled Sampling (Currently for Synthetic Data)**
- **Implementation**: Gradual transition from teacher forcing to model predictions
- **Formula**: `scheduled_sampling_prob = min(1.0, 2 * epoch / max_epochs)`
- **Current Status**: Teacher forcing used for LETITIA dataset due to missing PET feature challenges

#### **Loss Computation**
```python
# Individual step losses
loss_1 = sum(loss_dict_1[k] * weight_dict[k] for k in loss_dict_1.keys())
loss_2 = sum(loss_dict_2[k] * weight_dict[k] for k in loss_dict_2.keys()) 

# Combined loss with gradient flow through both steps
total_loss = loss_1 + loss_2
```

#### **Loss Components**
- **Superclass Classification**: Cross-entropy with class imbalance weights
- **Coordinate Regression**: MSE loss for 3D spatial coordinates
- **Radiomics Regression**: Selective MSE with empty_pt masking (first half always, second when empty_pt=0)

### 4. Evaluation Framework (`util/metrics.py`)

#### **Comprehensive Metrics System**
The evaluation framework provides detailed assessment across multiple dimensions:

**Classification Metrics** (per timestep):
- Superclass accuracy, precision, recall, F1-score

**Spatial Metrics** (per timestep):
- Coordinate error: L2 distance between predicted and target 3D coordinates
- Mean and standard deviation of spatial errors

**Feature Metrics** (per timestep):
- Radiomics error: MSE with selective computation
- First half features: Always computed
- Second half features: Only computed where `empty_pt = 0`

**Evaluation Output Structure**:
```python
results = {
    # T0→T1 Prediction
    't1_superclass_accuracy', 't1_superclass_precision', 't1_superclass_recall', 't1_superclass_f1',
    't1_coord_error_mean', 't1_coord_error_std',
    't1_feature_error_mean', 't1_feature_error_std',
    
    # T1→T2 Prediction  
    't2_superclass_accuracy', 't2_superclass_precision', 't2_superclass_recall', 't2_superclass_f1',
    't2_coord_error_mean', 't2_coord_error_std',
    't2_feature_error_mean', 't2_feature_error_std'
}
```

#### **Matching-Based Evaluation**
- Uses Hungarian algorithm to match predictions with ground truth
- Evaluates only successfully matched pairs
- Provides realistic assessment of model performance

## Technical Specifications

### Model Architecture
- **Encoder**: 3-layer Transformer encoder (configurable via `--enc_layers`)
- **Decoder**: 3-layer Transformer decoder (configurable via `--dec_layers`)  
- **Hidden Dimension**: 128 (configurable via `--hidden_dim`)
- **Query Slots**: 63 object queries, as the maximum size of present pointclouds (configurable via `--num_queries`)
- **Attention Heads**: 8 heads (configurable via `--nheads`)


### Loss Configuration
```python
# Default loss weights (configurable via command line)
loss_weights = {
    'superclass_loss_coef': 1.0,     # Classification weight
    'coordinates_loss_coef': 5.0,    # Spatial error weight  
    'radiomics_loss_coef': 3.0       # Feature error weight
}

# Class imbalance handling
superclass_coef = [0.75, 2.0, 0.15]  # [organ, lesion, no-object] weights. no-object is internally created by the model 
```

## Key Design Decisions

### 1. **Zero Positional Embeddings**
Point clouds already contain explicit 3D spatial information (x,y,z coordinates), making additional positional encodings redundant. The `PositionEmbeddingZero` class maintains interface compatibility while returning zeros.

### 2. **Teacher Forcing for LETITIA Dataset**
While scheduled sampling is implemented and functional for synthetic datasets, the LETITIA dataset uses teacher forcing due to the complexity of handling missing PET features (`empty_pt` flags) in the reconstruction process.

### 3. **Selective Radiomics Loss**
The radiomics loss computation respects the `empty_pt` flag:
- First half of radiomics: Always computed
- Second half of radiomics: Only computed where `empty_pt = 0`
This ensures the model learns appropriate feature representations for incomplete medical data.

### 4. **Autoregressive Training Without Detaching**
Gradients flow through both prediction steps (T0→T1 and T1→T2) enabling the model to learn temporal dependencies end-to-end.

### 5. **Hungarian Matching for Evaluation**
Maintains DETR's core set prediction philosophy by using optimal bipartite matching between predictions and ground truth, ensuring fair evaluation of the model's set prediction capabilities.

### 6. **Class Imbalance Handling**
Custom class weights address the natural imbalance between organs and lesions in medical data, with higher weights for the less frequent lesion class.

## Command Line Interface

### Key Arguments
```bash
# Model architecture
--hidden_dim 128              # Transformer hidden dimension
--num_queries 63              # Number of object queries
--enc_layers 3                # Encoder layers
--dec_layers 3                # Decoder layers

# Data configuration  
--data_root /path/to/data     # Root directory with TP0, TP1, TP2 folders
--val_ids 18 50 68 ...        # Specific validation patient IDs
--original_feature_size 113    # Features per point

# Training parameters
--lr 1e-4                     # Learning rate
--batch_size 8               # Batch size
--epochs 1000                # Training epochs

# Loss weights
--superclass_loss_coef 1.0    # Classification loss weight
--coordinates_loss_coef 5.0   # Spatial loss weight  
--radiomics_loss_coef 3.0     # Feature loss weight
```

This adaptation successfully transforms DETR from 2D image object detection to 3D temporal point cloud forecasting while maintaining the core transformer architecture and set prediction paradigm.
