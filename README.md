# LETITIA-DETR: Temporal Point Cloud Forecasting

A PyTorch implementation of Facebook's DETR adapted for temporal point cloud forecasting of anatomical organs and lesions. This model predicts the evolution of medical point clouds across three timesteps (T0 → T1 → T2) using transformer-based set prediction.

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

Each `.pt` file would contain a PyTorch tensor of shape `[num_points, num_features]` with features:
```python
# Feature organization per point:
[x, y, z,                    # 3D coordinates
 radiomics_features,         # radiomics features from medical imaging, in this case 54 CT + 54 PET radiomics
 empty_pt,                   # Binary flag (0/1) for missing PET radiomics features
 superclass_label]           # Anatomical classification (0=organ, 1=lesion)
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
#     'total_features': 113,
#     'num_coordinates': 3,
#     'num_radiomics': 108,
#     'num_superclasses': 2
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
    --batch_size 8
```

### Training with Custom Validation Split

```bash
python main.py \
    --data_root /path/to/your/data \
    --output_dir ./outputs \
    --val_ids 18 50 68 78 91 95 102 106 111 133 142 150 177 210 220 238 245 \
    --epochs 1000 \
    --lr 1e-4
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
    --batch_size 1 \
    --num_queries 63 \
    --hidden_dim 128 \
    --enc_layers 3 \
    --dec_layers 3 \
    --superclass_loss_coef 1.0 \
    --coordinates_loss_coef 5.0 \
    --radiomics_loss_coef 3.0 \
    --superclass_coef 0.75 2.0 0.15
```

### Key Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data_root` | - | Root directory containing TP0, TP1, TP2 folders |
| `--output_dir` | - | Directory to save checkpoints and logs |
| `--epochs` | 1000 | Number of training epochs |
| `--lr` | 1e-4 | Learning rate |
| `--batch_size` | 8 | Batch size |
| `--num_queries` | 63 | Number of object queries |
| `--hidden_dim` | 128 | Transformer hidden dimension |
| `--val_ids` | None | Specific patient IDs for validation |

### Loss Weights Configuration

The model uses multiple loss components that can be weighted:

```bash
# Classification loss weight
--superclass_loss_coef 1.0

# Spatial coordinate loss weight  
--coordinates_loss_coef 5.0

# Radiomics feature loss weight
--radiomics_loss_coef 3.0

# Class imbalance weights [organ, lesion, no-object]
--superclass_coef 0.75 2.0 0.15
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
- **Classification metrics**: Accuracy, precision, recall, F1-score
- **Spatial metrics**: Coordinate error metrics (L2 distance and MSE)
- **Feature metrics**: Radiomics prediction error (MSE)

### Standalone Evaluation

```bash
# Evaluate a trained model
python main.py \
    --data_root /path/to/your/data \
    --resume ./outputs/checkpoint.pth \
    --eval
```

### Evaluation Metrics

The evaluation provides comprehensive metrics for both prediction steps:

**T0 → T1 Prediction:**
- `t1_superclass_accuracy`: Classification accuracy
- `t1_coord_l2`: Average L2 spatial error
- `t1_coord_mse`: Mean squared coordinate error
- `t1_radiomics_mse`: Radiomics feature MSE

**T1 → T2 Prediction:**
- `t2_superclass_accuracy`: Classification accuracy  
- `t2_coord_l2`: Average L2 spatial error
- `t2_coord_mse`: Mean squared coordinate error
- `t2_radiomics_mse`: Radiomics feature MSE

## ⚙️ Model Configuration

### Architecture Parameters

```bash
# Transformer configuration
--enc_layers 3              # Encoder layers
--dec_layers 3              # Decoder layers  
--nheads 8                  # Attention heads
--dim_feedforward 512       # Feedforward dimension
--dropout 0.0               # Dropout rate

# Model capacity
--num_queries 63            # Object queries (max predictions)
--hidden_dim 128            # Hidden dimension
```

## 🔍 Understanding the Model

### Model Architecture

1. **Backbone**: Independent MLP for per-point feature extraction
2. **Encoder**: Multi-head self-attention over point sequences
3. **Decoder**: Cross-attention between queries and encoded features
4. **Output Heads**: Separate prediction heads for classification, coordinates, and radiomics

### Training Process

1. **Step 1**: Predict T1 from T0
   - Input: T0 point cloud
   - Target: T1 ground truth
   
2. **Step 2**: Predict T2 from T1  
   - Input: T1 ground truth (teacher forcing)
   - Target: T2 ground truth

3. **Loss**: Combined loss from both steps

### Data Handling

- **Variable Sequences**: Automatic padding with attention masks
- **Missing Features**: Selective loss computation for incomplete PET data
- **Class Imbalance**: Weighted loss for organ/lesion classification

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
├── main.py                 # Training/evaluation entry point
├── engine.py               # Training and evaluation loops
├── models/
│   ├── detr.py            # Main DETR model
│   ├── backbone.py        # Point cloud backbone
│   ├── transformer.py     # Transformer implementation
│   └── matcher.py         # Hungarian matching
├── datasets/
│   ├── letitia_ds.py      # LETITIA dataset implementation
│   └── __init__.py        # Dataset builder
├── util/
│   ├── metrics.py         # Evaluation metrics
│   ├── misc.py            # Utility functions
└── LETITIA-DETR-ADAPTATION.md  # Technical documentation
```

## 🔬 Technical Details

For detailed technical information about the DETR adaptation, model architecture, and implementation decisions, see [LETITIA-DETR-ADAPTATION.md](LETITIA-DETR-ADAPTATION.md).

## 📄 License

This project is based on Facebook's DETR implementation. See [LICENSE](LICENSE) for details.