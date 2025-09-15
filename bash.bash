#!/bin/bash

# RunAI Job Script for Adapted DETR Training

set -e  # Exit on any error
set -u  # Exit on undefined variables

# Print job information
echo "=== RunAI DETR Training Job Started ==="
echo "Job started at: $(date)"
echo "Running on host: $(hostname)"
echo "Working directory: $(pwd)"
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits

# Environment setup
echo "=== Setting up environment ==="
source /opt/conda/etc/profile.d/conda.sh
conda activate conda_env  # Replace with your conda environment name/path

# Verify conda environment
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"

# Change to project directory
cd /path/to/your/project  # Replace with your project path

# Create output directory if it doesn't exist
OUTPUT_DIR='path/to/output/directory'
mkdir -p "$OUTPUT_DIR"

# Log the command being executed
echo "=== Starting training ==="
echo "Output directory: $OUTPUT_DIR"

# Run the training with error handling
python main.py \
    --device='cuda' \
    --output_dir="$OUTPUT_DIR" \
    --enc_layers=3 \
    --dec_layers=3 \
    --original_feature_size=113 \
    --hidden_dim=128 \
    --dim_feedforward=512 \
    --nheads=8 \
    --num_queries=63 \
    --epochs=1000 \
    --batch_size=8 \
    --weight_decay=0.001 \
    --clip_max_norm=1 \
    --dropout=0.1 \
    --num_workers=4 \
    --lr_drop=100000 \ 
    --data_root='path/to/data/dir' \
    --superclass_loss_coef=1.0 \
    --coordinates_loss_coef=4.0 \
    --radiomics_loss_coef=5.0 \
    --set_cost_superclass=2.0 \
    --set_cost_coordinates=1.0 \
    --set_cost_radiomics=0.03 \
    --NO_schedule_sampling \
    2>&1 | tee "$OUTPUT_DIR/training_log.txt"

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo "=== Training completed successfully ==="
    echo "Training finished at: $(date)"
    echo "Output saved to: $OUTPUT_DIR"
    echo "Log saved to: $OUTPUT_DIR/training_log.txt"
    
    # Show final output directory contents
    echo "=== Output directory contents ==="
    ls -la "$OUTPUT_DIR"
else
    echo "=== Training failed ==="
    echo "Check logs for details: $OUTPUT_DIR/training_log.txt"
    exit 1
fi

echo "=== Job completed ==="