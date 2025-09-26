# Modified DETR-Letitia for Hierarchical Classification

## Overview

This document covers the extension of DETR-Letitia to dual-level classification: **superclass** (0/1 organ vs lesion) and **subclass** (fine-grained **semantic** label).

## Key Hierarchical Features
- **Dual Classification Heads**: Separate prediction heads for superclass and subclass
- **Learnable Subclass Embeddings**: Since subclasses will be used as input features, we create learnable embeddings (prototypes) for each subclass
- **Prototype-Based Classification**: Since subclasses are represented as embeddings, we classify outputs using cosine similarity to the prototypes
- **FiLM Conditioning Support**: Optional feature-wise linear modulation (FiLM) using prototypes instead of concatenating them to input features

*Note: The condition of having **semantic** labels is not met in the original LETITIA dataset (its lesion labels are instance-level rather than semantic) and is only satisfied by the Synthetic dataset*

## 1. DATA LAYER ENHANCEMENTS

### Input Structure
Point clouds now include both superclass and subclass labels:

```
[3D coordinates, features, superclass_label, subclass_label]
```

### Target Structure (`datasets/letitia_ds.py`)
```python
{
    'superclass': [num_points],      # High-level classification  
    'subclass': [num_points],        # Fine-grained classification
    'coordinates': [num_points, 3],  # 3D spatial coordinates
    'radiomics': [num_points, N]     # Feature predictions
}
```

## 2. BACKBONE ENHANCEMENTS (`models/backbone.py`)

### Learnable Subclass Embeddings
```python
# Output prototypes to be used for classification (via cosine similarity)
self.subclass_embeddings = nn.Embedding(num_subclasses, subclass_dim)

# Frozen EMA embeddings to be used as input (by concatenating or FiLM)
self.ema_embeddings = nn.Embedding(num_subclasses, subclass_dim)
```

### Key Features
- **Orthogonal Initialization**: Initializes subclass embeddings to be orthogonal for better separation
- **Frozen EMA Embeddings**: Input embeddings updated with exponential moving average (EMA) of the output 
- **L2 Normalization**: Both output and input prototypes are L2-normalized
- **FiLM Conditioning**: Optional mode that processes subclass embeddings as conditioning signals instead of concatenation

### Processing Pipeline
1. Extract subclass labels from input: `point_data[:, :, -1].long()`
2. Replace discrete labels with L2-normalized embeddings (Frozen EMA prototypes)
3. Concatenate with other features or use as FiLM conditioning
4. Pass the unfrozen prototypes through the transformer for subclass classification

## 3. DETR MODEL ENHANCEMENTS (`models/detr.py`)

### Dual Classification Architecture
```python
# Standard MLP for superclass -- +1 is a no-object class for the unmatched queries
self.superclass_embed = MLP(hidden_dim, 2*hidden_dim, num_superclasses + 1, num_layers=3) 

# Prototype-based classifier for subclass
self.subclass_embed = PrototypeClassifier(hidden_dim, backbone.subclass_dim, 
                                         num_subclasses, backbone.subclass_embeddings)
```

### PrototypeClassifier Features
- **Shared Prototypes**: Uses backbone's passed subclass embeddings (unfrozen)
- **Cosine Similarity**: L2-normalized feature similarity classification
- **Per-Class Bias**: Individual bias terms for each subclass

### Model Outputs
```python
{
    'pred_superclass': [batch, queries, num_superclasses + 1],  # Includes no-object
    'pred_subclass': [batch, queries, num_subclasses],          # Objects only
    'pred_coordinates': [batch, queries, 3],
    'pred_radiomics': [batch, queries, radiomics_dim]
}
```

### Loss Functions
- **Superclass Loss**: Standard cross-entropy with no-object class
- **Subclass Loss**: Cross-entropy for matched objects only
- **Class Imbalance**: Weighted losses for minority subclasses

## 4. SCHEDULED AUTOREGRESSIVE TRAINING (`engine.py`)

### Enhanced Training Strategy
The training now supports **scheduled sampling** instead of fixed teacher forcing:

```python
# Scheduled sampling probability: linearly increase from 0 to 1
scheduled_sampling_prob = min(1.0, 2 * epoch / max_epochs)
use_prediction = torch.rand(1).item() < scheduled_sampling_prob

# Choose input for second forward pass (T1 -> T2)
if use_prediction:
    # Use model's prediction from step 1 
    input2 = make_pointcloud_from(outputs_1, mask_thresh=0.5)
else:
    # Use ground truth T1 (teacher forcing)
    input2 = gt_t1
```

### Key Features
- **Progressive Training**: Starts with teacher forcing (0% prediction) and gradually increases model prediction usage
- **Linear Schedule**: Prediction probability increases linearly with training epochs
- **Autoregressive Capability**: Model learns to use its own predictions for multi-step forecasting
- **Improved Generalization**: Reduces exposure bias compared to fixed teacher forcing

## 5. HIERARCHICAL METRICS (`util/metrics.py`)

### New Hierarchical Classification Metrics
```python
def compute_hierarchical_class_metrics(pred_superclass, pred_subclass, 
                                     tgt_superclass, tgt_subclass, no_obj_superclass):
```

### Key Metrics
- **Superclass Metrics**: Accuracy, precision, recall, F1-score (all predictions)
- **Subclass Metrics**: Accuracy, precision, recall, F1-score (objects only) 
- **Hierarchical Accuracy**: Both superclass AND subclass must be correct

### Temporal Dynamics Enhancement
- **T0→T1 Dynamics**: Performance metrics for lesion changes from TP0 to TP1 (death and duplication)
- **T1→T2 Dynamics**: Performance metrics for lesion changes from TP1 to TP2 (death and duplication)

## 6. COMMAND LINE INTERFACE (`main.py`)

### New Hierarchical Arguments
```bash
# Model Architecture
--num_superclasses         # High-level classes (excluding no-object)
--num_subclasses           # Fine-grained classes  
--subclass_dim             # Embedding dimension

# Matching Costs
--set_cost_superclass      # Superclass weight in matching
--set_cost_subclass        # Subclass weight in matching

# Loss Coefficients  
--superclass_loss_coef     # Superclass loss weight
--subclass_loss_coef       # Subclass loss weight
```

## SUMMARY

This modification of DETR-Letitia introduces the notion of subclasses: **semantic** labels that provide fine-grained classification within the broader 'superclass' categories of organs and lesions. It assumes that subclasses are conditioning signals that are useful to be used as input features, and hence creates learnable embeddings (prototypes) for each subclass. The conditioning can be done either by concatenation to the input features or via FiLM. The subclass classification is performed using a prototype-based classifier that computes cosine similarity between the output features and the learnable subclass embeddings. The model is trained with dual losses for superclass and subclass, and new hierarchical metrics are introduced to evaluate performance at both levels.