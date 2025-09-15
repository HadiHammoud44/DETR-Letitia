# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules for point cloud processing.
"""
import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple

from util.misc import NestedTensor


class PointEmbed(nn.Module):
    """
    Simplified point cloud embedding module for anatomical organs/lesions.
    
    Processes point cloud features: [coordinates, radiomics, empty_pt, superclass] 
    
    Input: [Batch, max_tokens, num_features] - point clouds with radiomics features
    Output: [Batch, max_tokens, hidden_dim] - embedded representations
    """
    
    def __init__(self, num_features: int, hidden_dim: int = 256):
        """
        Initialize simplified point embedding module.
        
        Args:
            num_features: Number of input features per point (3D coords + radiomics + empty_pt + superclass)
            hidden_dim: Output embedding dimension (default: 256)
        """
        super().__init__()
        self.num_channels = hidden_dim  # For compatibility with downstream code
        
        # Simple embedding transformation: input features → hidden representation
        self.embed = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """
        Initialize weights and zero out biases.
        """
        # Zero out biases in embedding layers
        for layer in self.embed:
            if isinstance(layer, nn.Linear) and layer.bias is not None:
                nn.init.constant_(layer.bias, 0)
    
    def forward(self, tensor_list: NestedTensor):
        """
        Forward pass through simplified point embedding.
        
        Args:
            tensor_list: NestedTensor containing:
                - tensors: [Batch, max_tokens, num_features] - padded point clouds
                  Features: [x, y, z, radiomics..., empty_pt, superclass_label]
                - mask: [Batch, max_tokens] - padding mask (True = padded)
        
        Returns:
            Tuple of (embedded_features, None) where:
            - embedded_features: NestedTensor with embedded point representations
            - None: No conditioning/positional encoding
        """
        # Extract point cloud data and mask
        point_data = tensor_list.tensors  # [Batch, max_tokens, num_features]
        mask = tensor_list.mask          # [Batch, max_tokens]
        
        # Apply embedding transformation
        embedded = self.embed(point_data)  # [Batch, max_tokens, hidden_dim]
        
        # Return as NestedTensor with original mask preserved
        return NestedTensor(embedded, mask), None


class PointBackbone(nn.Module):
    """
    Simplified point cloud backbone that wraps PointEmbed for DETR integration.
    """
    
    def __init__(self, num_features: int, hidden_dim: int = 256):
        """
        Initialize simplified point cloud backbone.
        
        Args:
            num_features: Number of input features per point
            hidden_dim: Embedding dimension
        """
        super().__init__()
        self.point_embed = PointEmbed(num_features, hidden_dim)
        self.num_channels = hidden_dim
    
    def forward(self, tensor_list: NestedTensor) -> Tuple[NestedTensor, None]:
        """
        Forward pass through point backbone.
        
        Args:
            tensor_list: NestedTensor with point cloud data
            
        Returns:
            Tuple of (embedded_features, None)
            - embedded_features: NestedTensor with processed features
            - None: No conditioning/positional encoding
        """
        return self.point_embed(tensor_list)
