# LETITIA-DETR: Hierarchical Temporal Point Cloud Forecasting

A PyTorch implementation of Facebook's DETR adapted for hierarchical temporal point cloud forecasting of anatomical organs and lesions. This model predicts the evolution of medical point clouds across three timesteps (T0 → T1 → T2) using transformer-based set prediction with dual-level classification: **superclass** (organ vs lesion) and **subclass** (fine-grained semantic labels).

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- PyTorch 2.5+

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd detr-letitia

# Install dependencies
pip install torch torchvision numpy pathlib

# Verify installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
```

## 📊 Data Preparation

### Data Format

Your data should be organized in the following structure:
```
data_root/
├── TP0/           # Baseline timestep
│   ├── 1.pt
│   ├── 2.pt
│   └── ...
├── TP1/           # First follow-up
│   ├── 1.pt
│   ├── 2.pt
│   └── ...
└── TP2/           # Second follow-up
    ├── 1.pt
    ├── 2.pt
    └── ...
```

### Point Cloud Format

Each `.pt` file should contain a PyTorch tensor of shape `[num_points, feature_size]` with features:
```python
# Feature organization per point:
[x, y, z,                    # 3D coordinates
 other_features,             # Radiomics features or synthetic features
 superclass_label,           # High-level classification (0=organ, 1=lesion)
 subclass_label]             # Fine-grained semantic classification (0 to num_subclasses-1)
```

### Data Validation

Check your data format using the built-in utility:
```python
from datasets.letitia_ds import get_feature_info

# Inspect your data format
info = get_feature_info('/path/to/your/data_root')
print(info)
# Expected output:
# {
#     'total_features': <feature_size>,
#     'num_coordinates': 3,
#     'num_radiomics': <radiomics_dim>,
#     'num_superclasses': 2,
#     'num_subclasses': <num_subclasses>
# }
```

## 🎯 Training

### Basic Training

```bash
python main.py \
    --data_root /path/to/your/data \
    --output_dir ./outputs \
    --epochs 1000 \
    --lr 1e-4 \
    --split_ratio 0.8 0.15 0.05 \
    --batch_size 8
```

### Detailed Training Configuration

```bash
python main.py \
    --data_root /path/to/your/data \
    --output_dir ./outputs \
    --epochs 1000 \
    --lr 1e-4 \
    --lr_backbone 1e-4 \
    --weight_decay 1e-4 \
    --batch_size 8 \
    --num_queries 92 \
    --hidden_dim 256 \
    --enc_layers 3 \
    --dec_layers 3 \
    --num_superclasses 2 \
    --num_subclasses 92 \
    --subclass_dim 92 \
    --superclass_loss_coef 2.0 \
    --subclass_loss_coef 2.0 \
    --coordinates_loss_coef 5.0 \
    --radiomics_loss_coef 3.0 \
    --set_cost_superclass 1 \
    --set_cost_subclass 1 \
    --use_film
```

### Key Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data_root` | - | Root directory containing TP0, TP1, TP2 folders |
| `--output_dir` | - | Directory to save checkpoints and logs |
| `--epochs` | 1000 | Number of training epochs |
| `--lr` | 1e-4 | Learning rate |
| `--batch_size` | 8 | Batch size |
| `--num_queries` | 92 | Number of object queries |
| `--hidden_dim` | 256 | Transformer hidden dimension |
| `--num_superclasses` | 2 | Number of superclasses (excluding no-object) |
| `--num_subclasses` | 92 | Number of subclasses |
| `--subclass_dim` | 92 | Dimension of subclass embeddings |
| `--use_film` | False | Enable FiLM conditioning with subclass embeddings |

### Loss Weights Configuration

The model uses multiple hierarchical loss components that can be weighted:

```bash
# Hierarchical classification loss weights
--superclass_loss_coef 2.0     # Superclass (organ/lesion) classification weight
--subclass_loss_coef 2.0       # Subclass (fine-grained) classification weight

# Spatial and feature loss weights
--coordinates_loss_coef 5.0    # 3D coordinate prediction weight  
--radiomics_loss_coef 3.0      # Radiomics feature prediction weight

# Hungarian matching cost weights
--set_cost_superclass 1        # Superclass weight in bipartite matching
--set_cost_subclass 1          # Subclass weight in bipartite matching
--set_cost_coordinates 5       # Coordinate weight in bipartite matching
--set_cost_radiomics 0         # Radiomics weight in bipartite matching
```

### Monitoring Training

Training progress is logged to `output_dir/log.txt`:
```bash
# View training progress
tail -f ./outputs/log.txt

# Monitor GPU usage
nvidia-smi -l 1
```

## 📈 Evaluation

### Evaluation During Training

The model automatically evaluates on the validation set after each epoch, reporting:
- **Hierarchical Classification metrics**: 
  - Superclass accuracy, precision, recall, F1-score
  - Subclass accuracy, precision, recall, F1-score (objects only)
  - Combined hierarchical accuracy (both levels must be correct)
- **Spatial metrics**: Coordinate metrics (L2 distance and MSE)
- **Feature metrics**: Radiomics prediction error (MSE)
- **Temporal dynamics**: Lesion appearance/disappearance analysis (T0→T1 and T1→T2)

### Standalone Evaluation

```bash
# Evaluate a trained model
python main.py \
    --data_root /path/to/your/data \
    --resume ./outputs/checkpoint.pth \
    --eval
```

### Evaluation Metrics

The evaluation provides comprehensive hierarchical metrics for both prediction steps:

**T0 → T1 Prediction:**
- `t1_superclass_accuracy`: Superclass classification accuracy
- `t1_subclass_accuracy`: Subclass classification accuracy  
- `t1_hierarchical_accuracy`: Combined hierarchical accuracy
- `t1_coord_l2`: Average L2 spatial error
- `t1_coord_mse`: Mean squared coordinate error
- `t1_radiomics_mse`: Radiomics feature MSE

**T1 → T2 Prediction:**
- `t2_superclass_accuracy`: Superclass classification accuracy
- `t2_subclass_accuracy`: Subclass classification accuracy
- `t2_hierarchical_accuracy`: Combined hierarchical accuracy  
- `t2_coord_l2`: Average L2 spatial error
- `t2_coord_mse`: Mean squared coordinate error
- `t2_radiomics_mse`: Radiomics feature MSE

**Temporal Dynamics:**
- `lesion_disappear_t0_t1_accuracy`: Lesion disappearance T0→T1
- `lesion_appear_t0_t1_accuracy`: Lesion appearance T0→T1
- `lesion_disappear_t1_t2_accuracy`: Lesion disappearance T1→T2
- `lesion_appear_t1_t2_accuracy`: Lesion appearance T1→T2

## ⚙️ Model Configuration

### Architecture Parameters

```bash
# Transformer configuration
--enc_layers 3              # Encoder layers
--dec_layers 3              # Decoder layers  
--nheads 8                  # Attention heads
--dim_feedforward 1024      # Feedforward dimension
--dropout 0.0               # Dropout rate

# Model capacity
--num_queries 92            # Object queries (max predictions)
--hidden_dim 256            # Hidden dimension

# Hierarchical classification
--num_superclasses 2        # High-level classes (organ/lesion) excluding no-object
--num_subclasses 92         # Fine-grained semantic classes
--subclass_dim 92           # Subclass embedding dimension

# Advanced features
--use_film                  # Enable FiLM conditioning with subclass embeddings
--ema_momentum 0.995        # EMA momentum for subclass embedding updates
```

## 📁 Output Files

Training generates several output files:

```
output_dir/
├── checkpoint.pth          # Latest model checkpoint
├── checkpoint0099.pth      # Periodic checkpoint (every 100 epochs)
├── log.txt                 # Training/validation metrics log
└── eval.pth                # Evaluation results (if --eval used)
```

### Checkpoint Contents

```python
checkpoint = torch.load('checkpoint.pth')
# Contains:
# - 'model': Model state dict
# - 'optimizer': Optimizer state
# - 'lr_scheduler': Learning rate scheduler state  
# - 'epoch': Current epoch
# - 'args': Training arguments
```


## 📚 Code Structure

```
detr/
├── main.py                         # Training/evaluation entry point
├── engine.py                       # Training and evaluation loops with scheduled sampling
├── models/
│   ├── detr.py                    # Main DETR model with hierarchical classification
│   ├── backbone.py                # Point cloud backbone with subclass embeddings
│   ├── transformer.py             # Transformer implementation with FiLM support
│   └── matcher.py                 # Hungarian matching
├── datasets/
│   ├── letitia_ds.py              # LETITIA dataset with hierarchical target formatting
│   └── __init__.py                # Dataset builder
├── util/
│   ├── metrics.py                 # Hierarchical evaluation metrics
│   ├── misc.py                    # Utility functions
├── LETITIA-DETR-ADAPTATION.md     # Original technical documentation
└── LETITIA-DETR-HIERARCHICAL.md   # Hierarchical classification documentation
```

## 🔬 Technical Details

For detailed technical information about the DETR adaptation, model architecture, and implementation decisions, see:
- [LETITIA-DETR-ADAPTATION.md](LETITIA-DETR-ADAPTATION.md) - Original temporal forecasting implementation
- [LETITIA-DETR-HIERARCHICAL.md](LETITIA-DETR-HIERARCHICAL.md) - Hierarchical classification extensions

## 📄 License

This project is based on Facebook's DETR implementation. See [LICENSE](LICENSE) for details.